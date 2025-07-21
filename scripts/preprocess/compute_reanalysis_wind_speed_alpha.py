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


def configure_dask():
    dask.config.set({"distributed.worker.multiprocessing-method": "forkserver"})
    dask.config.set({"distributed.worker.use-file-locking": "False"})
    dask.config.set({"logging.distributed": "error"})
    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})


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
def group_by_synthetic_time(da_alpha):
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
        .mean("time")
    )
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
    configure_dask()
    path_data_processed = load_paths("../../paths_so4.json")
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

    with ProgressBar(dt=10):
        print("Starting wind speed alpha computation...")
        da_alpha = calculate_alpha(ds_ws, heights).rename("alpha")
        da_alpha = group_by_synthetic_time(da_alpha).persist()

        print("Calculating temporal smoothing...")
        da_alpha_temp_smoothed = temporal_smooth(da_alpha)
        da_alpha_temp_smoothed = xr.DataArray(
            da_alpha_temp_smoothed,
            dims=da_alpha.dims,
            coords=da_alpha.coords,
            attrs=da_alpha.attrs,
            name=da_alpha.name,
        )
        print("Calculating spatial smoothing...")
        da_alpha_tempspatial_smoothed = spatial_smooth(
            da_alpha_temp_smoothed.chunk({"time": -1, "lat": -1, "lon": -1})
        )
        da_alpha_tempspatial_smoothed = xr.DataArray(
            da_alpha_tempspatial_smoothed,
            dims=da_alpha.dims,
            coords=da_alpha.coords,
            attrs=da_alpha.attrs,
            name=da_alpha.name,
        ).compute()

        print("Saving results...")
        output_files = [
            f"da_alpha_cerra_heights_{heights[0]}_{heights[1]}.nc",
            f"da_alpha_temp_smoothed_cerra_heights_{heights[0]}_{heights[1]}.nc",
            f"da_alpha_tempspatial_smoothed_cerra_heights_{heights[0]}_{heights[1]}.nc",
        ]
        save_dataarrays(
            output_path,
            [da_alpha, da_alpha_temp_smoothed, da_alpha_tempspatial_smoothed],
            output_files,
        )
        elapsed_minutes = (time.time() - start_time) / 60
        print(
            f"Wind speed alpha computation completed in {elapsed_minutes:.2f} minutes."
        )


if __name__ == "__main__":
    main()
