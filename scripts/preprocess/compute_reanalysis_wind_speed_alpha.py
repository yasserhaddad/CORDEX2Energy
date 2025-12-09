import os
import sys

sys.path.append("../..")

import json
import time
from pathlib import Path

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from src.conversion_energy.wind import compute_power_law_exponent
from src.data_processing.load_data import ClimateReanalysisData
from src.utils.constants import REFERENCE_PERIOD
import argparse


def configure_dask():
    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
    os.environ["MKL_NUM_THREADS"] = "1"  # MKL
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

    dask.config.set({"logging.distributed": "error"})
    dask.config.set(num_workers=16, threads_per_worker=16)
    dask.config.set({"array.chunk-size": "400MiB"})
    dask.config.set({'distributed.worker.memory.target': 0.75})  # Spill to disk at 75% usage
    dask.config.set({'distributed.worker.memory.spill': 0.8})   # Start spilling at 80%
    dask.config.set({'distributed.worker.memory.pause': 0.9})   # Pause worker at 90%



def load_paths(paths_file):
    with open(paths_file) as f:
        paths = json.load(f)
    return Path(paths["path_data_processed"])


def load_wind_speed_data(ws_path):
    ds_ws = ClimateReanalysisData(
        ws_path,
        chunks="auto",
        extent=None,
        load=False,
        drop_leap_days=True,
    )
    ds_ws.ds_data = ds_ws.ds_data.sel(time=slice(*REFERENCE_PERIOD))
    return ds_ws


def calculate_alpha(ds_ws, heights):
    return compute_power_law_exponent(
        ds_ws.ds_data.ws.sel(height=heights[0]),
        ds_ws.ds_data.ws.sel(height=heights[1]),
        heights[0],
        heights[1],
    ).clip(min=0.0, max=1.0)


# -------------------- Group by Synthetic Time --------------------
def group_by_synthetic_time_and_aggregate(da_alpha, aggregation_method="mean"):
    month = da_alpha.time.dt.month
    day = da_alpha.time.dt.day
    hour = da_alpha.time.dt.hour

    time_key = xr.DataArray(
        [
            f"{m:02d}-{d:02d} {h:02d}:00"
            for m, d, h in zip(month.values, day.values, hour.values)
        ],
        dims="time",
    )
    da_alpha = (
        da_alpha.assign_coords(time_key=("time", time_key.data))
        .groupby("time_key")
    )
    if aggregation_method == "median":
        da_alpha = da_alpha.median(dim="time", skipna=True)
    else:
        da_alpha = da_alpha.mean(dim="time", skipna=True)
    synthetic_times = pd.date_range("2001-01-01", periods=365 * 8, freq="3h")
    da_alpha = da_alpha.assign_coords(time_key=("time_key", synthetic_times)).rename(
        {"time_key": "time"}
    )
    return da_alpha


# -------------------- Smoothing Functions --------------------
def temporal_smooth(da, sigma=3):
    def smooth_func(arr):
        return gaussian_filter1d(arr, sigma=sigma, axis=0, mode="wrap")

    return xr.apply_ufunc(
        smooth_func,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    ).transpose("time", "lat", "lon")


def spatial_smooth(da, sigma=1):
    def smooth_func(arr):
        return gaussian_filter(arr, sigma=(0, sigma, sigma), mode="nearest")

    return xr.apply_ufunc(
        smooth_func,
        da,
        input_core_dims=[["time", "lat", "lon"]],
        output_core_dims=[["time", "lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )


def save_dataarrays(output_path, dataarrays, filenames):
    for output_file in filenames:
        output_path_file = output_path / output_file
        if output_path_file.exists():
            output_path_file.unlink()
    for da, fname in zip(dataarrays, filenames):
        da.compute().to_netcdf(output_path / fname, mode="w")


# -------------------- Main Pipeline --------------------
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Compute wind speed alpha with optional smoothing")
    parser.add_argument("--paths_file", type=str, required=True, help="Path to the JSON file containing paths.")
    parser.add_argument("--aggregation_method", type=str, default="median", help="Aggregation method for alpha calculation.")
    parser.add_argument(
        "--do_temporal", action="store_true", help="Apply temporal smoothing."
    )
    parser.add_argument(
        "--do_spatial", action="store_true", help="Apply spatial smoothing, on top of temporal smoothing."
    )
    args = parser.parse_args()
    if args.aggregation_method not in ["mean", "median"]:
        raise ValueError("aggregation_method must be either 'mean' or 'median'")

    configure_dask()
    path_data_processed = load_paths(args.paths_file)
    ws_cerra_path = path_data_processed / "cerra_processed" / "regridded" / "ws.zarr"
    ws_10m_cerra_path = (
        path_data_processed / "cerra_processed" / "regridded" / "10si.zarr"
    )
    output_path = path_data_processed / "cerra_processed" / "alpha"
    output_path.mkdir(parents=True, exist_ok=True)

    ds_ws = load_wind_speed_data(ws_cerra_path)
    ds_ws.ds_data = xr.combine_by_coords(
        [
            ds_ws.ds_data,
            load_wind_speed_data(ws_10m_cerra_path).ds_data.rename({"10si": "ws"}),
        ],
        combine_attrs="drop_conflicts",
    )

    heights = [10, 100]  # Heights in meters for wind speed data
    output_daarrays = []
    with ProgressBar(dt=10):
        print("Starting wind speed alpha computation...")
        da_alpha = calculate_alpha(ds_ws, heights).rename("alpha")
        output_daarrays.append(group_by_synthetic_time_and_aggregate(da_alpha, aggregation_method=args.aggregation_method).persist())
        print("Calculating temporal smoothing...")
        if args.do_temporal:
            da_alpha_temp_smoothed = temporal_smooth(da_alpha)
            output_daarrays.append(
                xr.DataArray(
                    da_alpha_temp_smoothed,
                    dims=da_alpha.dims,
                    coords=da_alpha.coords,
                    attrs=da_alpha.attrs,
                    name=da_alpha.name,
                )
            )
        print("Calculating spatial smoothing...")
        if args.do_temporal and args.do_spatial:
            da_alpha_tempspatial_smoothed = spatial_smooth(
                da_alpha_temp_smoothed.chunk({"time": -1, "lat": -1, "lon": -1})
            )
            output_daarrays.append(
                xr.DataArray(
                    da_alpha_tempspatial_smoothed,
                    dims=da_alpha.dims,
                    coords=da_alpha.coords,
                    attrs=da_alpha.attrs,
                    name=da_alpha.name,
                ).compute()
            )

        print("Saving results...")
        output_files = [
            f"da_alpha_cerra_{args.aggregation_method}_heights_{heights[0]}_{heights[1]}.nc",
            f"da_alpha_temp_smoothed_cerra_{args.aggregation_method}_heights_{heights[0]}_{heights[1]}.nc",
            f"da_alpha_tempspatial_smoothed_cerra_{args.aggregation_method}_heights_{heights[0]}_{heights[1]}.nc",
        ]
        save_dataarrays(
            output_path,
            output_daarrays,
            output_files[:len(output_daarrays)],
        )
        elapsed_minutes = (time.time() - start_time) / 60
        print(
            f"Wind speed alpha computation completed in {elapsed_minutes:.2f} minutes."
        )


if __name__ == "__main__":
    main()
