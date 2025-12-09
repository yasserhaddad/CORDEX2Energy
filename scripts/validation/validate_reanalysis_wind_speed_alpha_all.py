import os
import sys

sys.path.append("../..")

import json
import pprint
import time
from pathlib import Path

import dask
import numpy as np
import seaborn as sns
import xarray as xr
from dask.diagnostics import ProgressBar
from src.conversion_energy.wind import extrapolate_wind_speed
from src.data_processing.load_data import ClimateModelData, ClimateReanalysisData
from src.utils.utils_xarray import sel_extent, tile_dataset_to_target_time

sns.set_style("whitegrid", {"grid.color": ".93"})


os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")

dask.config.set({"logging.distributed": "error"})
dask.config.set(num_workers=15, threads_per_worker=16)
dask.config.set({
    "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
    "distributed.worker.memory.spill": 0.8,   # Start spilling at 80%
    "distributed.worker.memory.pause": 0.9,   # Pause worker at 90%
    "array.chunk-size": "512MiB",
    "logging.distributed": "error",           # Less verbose logs
})


# Compute metrics
def compute_metrics(ref, pred):
    diff = pred - ref
    mse = float(((diff) ** 2).mean().compute().item())
    rmse = np.sqrt(mse)
    mae = float(np.abs(diff).mean().compute().item())
    bias = float(diff.mean().compute().item())
    results = {"mse": mse, "rmse": rmse, "mae": mae, "bias": bias}
    return {metric: round(result, 4) for metric, result in results.items()}

def compute_metrics_disaggregated(ref, pred, dim):
    diff = pred - ref
    mse = ((diff) ** 2).mean(dim=dim).round(4).rename("mse")
    rmse = np.sqrt(mse).round(4).rename("rmse")
    mae = np.abs(diff).mean(dim=dim).round(4).rename("mae")
    bias = diff.mean(dim=dim).round(4).rename("bias")
    return xr.merge([mse, rmse, mae, bias])

def extrapolate_ws_to_target_height(
    ds_ws_reference,
    height_reference,
    height_target,
    da_alpha_cerra,
    da_alpha_temp_smoothed_cerra,
    da_alpha_tempspatial_smoothed_cerra,
):
    print(f"Extrapolating wind speed to {height_target}m using alpha.")
    ds_ws_target_alpha = extrapolate_wind_speed(
        ds_ws_reference,
        height_reference,
        height_target,
        tile_dataset_to_target_time(da_alpha_cerra.alpha, ds_ws_reference.time),
    )

    print(f"Extrapolating wind speed to {height_target}m using alpha with temporal smoothing.")
    ds_ws_target_alpha_temp_smoothed = extrapolate_wind_speed(
        ds_ws_reference,
        height_reference,
        height_target,
        tile_dataset_to_target_time(
            da_alpha_temp_smoothed_cerra.alpha, ds_ws_reference.time
        ),
    )

    print(
        f"Extrapolating wind speed to {height_target}m using alpha with temporal and spatial smoothing."
    )
    ds_ws_target_alpha_tempspatial_smoothed = extrapolate_wind_speed(
        ds_ws_reference,
        height_reference,
        height_target,
        tile_dataset_to_target_time(
            da_alpha_tempspatial_smoothed_cerra.alpha, ds_ws_reference.time
        ),
    )

    return (
        ds_ws_target_alpha,
        ds_ws_target_alpha_temp_smoothed,
        ds_ws_target_alpha_tempspatial_smoothed,
    )


def compute_and_print_metrics_for_extrapolation(
    ds_ws_target_height_ref,
    ds_ws_target_height_alpha,
    ds_ws_target_height_alpha_temp_smoothed,
    ds_ws_target_height_alpha_tempspatial_smoothed,
):
    metrics_alpha = compute_metrics(ds_ws_target_height_ref.ws, ds_ws_target_height_alpha.ws)
    metrics_alpha_temp_smoothed = compute_metrics(
        ds_ws_target_height_ref.ws, ds_ws_target_height_alpha_temp_smoothed.ws
    )
    metrics_alpha_tempspatial_smoothed = compute_metrics(
        ds_ws_target_height_ref.ws, ds_ws_target_height_alpha_tempspatial_smoothed.ws
    )

    pp = pprint.PrettyPrinter(indent=4)
    print("Metrics for alpha:")
    pp.pprint(metrics_alpha)
    print("Metrics for alpha_temp_smoothed:")
    pp.pprint(metrics_alpha_temp_smoothed)
    print("Metrics for alpha_tempspatial_smoothed:")
    pp.pprint(metrics_alpha_tempspatial_smoothed)

if __name__ == "__main__":
    time_start = time.time()
    climate_model_chain = "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    climate_scenario = "rcp85"

    with open("../../paths_so4.json") as f:
        paths = json.load(f)

    path_data_cerra = Path(paths["path_data_cerra"])
    path_data_processed = Path(paths["path_data_processed"])

    ws_cerra_path = path_data_processed / "cerra_processed" / "regridded" / "ws.zarr"
    ws_10m_cerra_path = path_data_processed / "cerra_processed" / "regridded" / "10si.zarr"
    ws_alpha_cerra_path = path_data_processed / "cerra_processed" / "alpha"
    climate_model_bc_path = (
        path_data_processed
        / "cordex_processed"
        / f"{climate_model_chain}_{climate_scenario}"
        / "bias_corrected"
        / "3hr"
    )

    # Load CERRA wind speed data
    print("Loading CERRA wind speed data...")
    ds_ws = ClimateReanalysisData(
        ws_cerra_path,
        chunks="auto",
        extent=None,
        load=False,
        drop_leap_days=True,
    )

    ds_ws_10m = ClimateReanalysisData(
        ws_10m_cerra_path,
        chunks="auto",
        extent=None,
        load=False,
        drop_leap_days=True,
    )
    ds_ws_10m.ds_data = ds_ws_10m.ds_data.isel(height=0).rename({"10si": "ws"})

    # Load CORDEX wind speed data
    print("Loading CORDEX wind speed data...")
    ds_sfcWind_climate_model = ClimateModelData(
        climate_model_bc_path / "sfcWind.zarr",
        chunks="auto",
        extent=None,
        load=False,
        drop_leap_days=True,
    )
    ds_sfcWind_climate_model.ds_data = ds_sfcWind_climate_model.ds_data.rename(
        {"sfcWind": "ws"}
    )

    ds_ws_100m_climate_model = ClimateModelData(
        climate_model_bc_path / "wind100m.zarr",
        chunks="auto",
        extent=None,
        load=False,
        drop_leap_days=True,
    )

    ds_ws_100m_climate_model.ds_data = ds_ws_100m_climate_model.ds_data.rename(
        {"wind100m": "ws"}
    )

    ds_ws_150m_climate_model = ClimateModelData(
        climate_model_bc_path / "wind150m.zarr",
        chunks="auto",
        extent=None,
        load=False,
        drop_leap_days=True,
    )

    ds_ws_150m_climate_model.ds_data = ds_ws_150m_climate_model.ds_data.rename(
        {"wind150m": "ws"}
    )
    extent_bc = {
        "lat_min": ds_sfcWind_climate_model.ds_data.lat.min(),
        "lat_max": ds_sfcWind_climate_model.ds_data.lat.max(),
        "lon_min": ds_sfcWind_climate_model.ds_data.lon.min(),
        "lon_max": ds_sfcWind_climate_model.ds_data.lon.max(),
    }

    # Load CERRA alpha data
    alpha_heights = [10, 100]
    print(f"Loading CERRA alpha data, using alpha heights: {alpha_heights}")

    da_alpha_cerra = xr.open_dataset(
        ws_alpha_cerra_path
        / f"da_alpha_cerra_heights_{alpha_heights[0]}_{alpha_heights[1]}.nc"
    )
    da_alpha_temp_smoothed_cerra = xr.open_dataset(
        ws_alpha_cerra_path
        / f"da_alpha_temp_smoothed_cerra_heights_{alpha_heights[0]}_{alpha_heights[1]}.nc"
    )
    da_alpha_tempspatial_smoothed_cerra = xr.open_dataset(
        ws_alpha_cerra_path
        / f"da_alpha_tempspatial_smoothed_cerra_heights_{alpha_heights[0]}_{alpha_heights[1]}.nc"
    )

    with ProgressBar(dt=10):
        (path_data_processed / "validation" / "alpha_wind_speed").mkdir(
            parents=True, exist_ok=True
        )

        # CERRA output paths
        p_ws_100m_alpha_cerra = (
            path_data_processed
            / "validation"
            / "alpha_wind_speed"
            / "ds_ws_100m_alpha_cerra_10m.zarr"
        )
        p_ws_100m_alpha_temp_smoothed_cerra = (
            path_data_processed
            / "validation"
            / "alpha_wind_speed"
            / "ds_ws_100m_alpha_temp_smoothed_cerra_10m.zarr"
        )
        p_ws_100m_alpha_tempspatial_smoothed_cerra = (
            path_data_processed
            / "validation"
            / "alpha_wind_speed"
            / "ds_ws_100m_alpha_tempspatial_smoothed_cerra_10m.zarr"
        )

        # CORDEX output paths
        p_ws_100m_alpha_cm_10m = (
            path_data_processed
            / "validation"
            / "alpha_wind_speed"
            / f"ds_ws_100m_alpha_{climate_model_chain}_10m.zarr"
        )
        p_ws_100m_alpha_temp_smoothed_cm_10m = (
            path_data_processed
            / "validation"
            / "alpha_wind_speed"
            / f"ds_ws_100m_alpha_temp_smoothed_{climate_model_chain}_10m.zarr"
        )
        p_ws_100m_alpha_tempspatial_smoothed_cm_10m = (
            path_data_processed
            / "validation"
            / "alpha_wind_speed"
            / f"ds_ws_100m_alpha_tempspatial_smoothed_{climate_model_chain}_10m.zarr"
        )

        def _all_exist(*paths):
            return all(p.exists() for p in paths)

        # print("----------------------------------------------------")
        print("CERRA 10m -> 100m extrapolation: checking for existing files...")
        if _all_exist(
            p_ws_100m_alpha_cerra,
            p_ws_100m_alpha_temp_smoothed_cerra,
            p_ws_100m_alpha_tempspatial_smoothed_cerra,
        ):
            print("Found existing CERRA extrapolation files. Loading from disk...")
            ds_ws_100m_alpha = xr.open_zarr(str(p_ws_100m_alpha_cerra)).load()
            ds_ws_100m_alpha_temp_smoothed = xr.open_zarr(
                str(p_ws_100m_alpha_temp_smoothed_cerra)
            ).load()
            ds_ws_100m_alpha_tempspatial_smoothed = xr.open_zarr(
                str(p_ws_100m_alpha_tempspatial_smoothed_cerra)
            ).load()
        else:
            print("Computing wind extrapolation to 100m from CERRA 10m wind...")
            ds_ws_10m_2021 = ds_ws_10m.ds_data.sel(time="2021")
            (
                ds_ws_100m_alpha,
                ds_ws_100m_alpha_temp_smoothed,
                ds_ws_100m_alpha_tempspatial_smoothed,
            ) = extrapolate_ws_to_target_height(
                ds_ws_10m_2021,
                10,
                100,
                da_alpha_cerra,
                da_alpha_temp_smoothed_cerra,
                da_alpha_tempspatial_smoothed_cerra,
            )

            ds_ws_100m_alpha.to_zarr(p_ws_100m_alpha_cerra, mode="w")
            ds_ws_100m_alpha_temp_smoothed.to_zarr(
                p_ws_100m_alpha_temp_smoothed_cerra, mode="w"
            )
            ds_ws_100m_alpha_tempspatial_smoothed.to_zarr(
                p_ws_100m_alpha_tempspatial_smoothed_cerra, mode="w"
            )

        print("----------------------------------------------------")
        print("CORDEX 10m -> 100m extrapolation: checking for existing files...")
        if _all_exist(
            p_ws_100m_alpha_cm_10m,
            p_ws_100m_alpha_temp_smoothed_cm_10m,
            p_ws_100m_alpha_tempspatial_smoothed_cm_10m,
        ):
            print("Found existing CORDEX extrapolation files. Loading from disk...")
            ds_ws_100m_alpha_climate_model_10m = xr.open_zarr(str(p_ws_100m_alpha_cm_10m)).load()
            ds_ws_100m_alpha_temp_smoothed_climate_model_10m = xr.open_zarr(
                str(p_ws_100m_alpha_temp_smoothed_cm_10m)
            ).load()
            ds_ws_100m_alpha_tempspatial_smoothed_climate_model_10m = xr.open_zarr(
                str(p_ws_100m_alpha_tempspatial_smoothed_cm_10m)
            ).load()
        else:
            print("Computing wind extrapolation to 100m from CORDEX 10m wind...")
            ds_sfcWind_climate_model.ds_data = ds_sfcWind_climate_model.ds_data.sel(
                time=slice("2021", "2030")
            )
            (
                ds_ws_100m_alpha_climate_model_10m,
                ds_ws_100m_alpha_temp_smoothed_climate_model_10m,
                ds_ws_100m_alpha_tempspatial_smoothed_climate_model_10m,
            ) = extrapolate_ws_to_target_height(
                ds_sfcWind_climate_model.ds_data,
                10,
                100,
                sel_extent(da_alpha_cerra, extent_bc),
                sel_extent(da_alpha_temp_smoothed_cerra, extent_bc),
                sel_extent(da_alpha_tempspatial_smoothed_cerra, extent_bc),
            )

            ds_ws_100m_alpha_climate_model_10m.to_zarr(p_ws_100m_alpha_cm_10m, mode="w")
            ds_ws_100m_alpha_temp_smoothed_climate_model_10m.to_zarr(
                p_ws_100m_alpha_temp_smoothed_cm_10m, mode="w"
            )
            ds_ws_100m_alpha_tempspatial_smoothed_climate_model_10m.to_zarr(
                p_ws_100m_alpha_tempspatial_smoothed_cm_10m, mode="w"
            )

        print("----------------------------------------------------")
        print(
            "Computing metrics for CERRA alpha extrapolation from 10m to 100m, based on year 2021..."
        )
        # Reference 100m wind speed
        ds_ws_100m_ref = ds_ws.ds_data.sel(height=100, time="2021").load()

        compute_and_print_metrics_for_extrapolation(
            ds_ws_100m_ref,
            ds_ws_100m_alpha,
            ds_ws_100m_alpha_temp_smoothed,
            ds_ws_100m_alpha_tempspatial_smoothed,
        )
        print("----------------------------------------------------")

        print(
            "Computing metrics for CORDEX alpha extrapolation from 10m to 100m, based on year 2021-2030..."
        )
        ds_ws_100m_climate_model.ds_data = ds_ws_100m_climate_model.ds_data.sel(
            time=slice("2021", "2030")
        ).load()
        compute_and_print_metrics_for_extrapolation(
            ds_ws_100m_climate_model.ds_data,
            ds_ws_100m_alpha_climate_model_10m,
            ds_ws_100m_alpha_temp_smoothed_climate_model_10m,
            ds_ws_100m_alpha_tempspatial_smoothed_climate_model_10m,
        )
        compute_metrics_disaggregated(ds_ws_100m_climate_model.ds_data.ws, ds_ws_100m_alpha_climate_model_10m.ws, dim="time").to_netcdf(path_data_processed / "validation" / "alpha_wind_speed" / f"metrics_alpha_{climate_model_chain}_10m.nc", mode="w")
        compute_metrics_disaggregated(
            ds_ws_100m_climate_model.ds_data.ws, ds_ws_100m_alpha_temp_smoothed_climate_model_10m.ws, dim="time"
        ).to_netcdf(path_data_processed / "validation" / "alpha_wind_speed" / f"metrics_alpha_temp_smoothed_{climate_model_chain}_10m.nc", mode="w")
        compute_metrics_disaggregated(
            ds_ws_100m_climate_model.ds_data.ws, ds_ws_100m_alpha_tempspatial_smoothed_climate_model_10m.ws, dim="time"
        ).to_netcdf(path_data_processed / "validation" / "alpha_wind_speed" / f"metrics_alpha_tempspatial_smoothed_{climate_model_chain}_10m.nc", mode="w")


        print("----------------------------------------------------")


    print(f"Total processing time: {(time.time() - time_start) / 60:.2f} minutes.")
