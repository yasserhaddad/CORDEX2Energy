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


dask.config.set({"distributed.worker.multiprocessing-method": "forkserver"})
dask.config.set({"distributed.worker.use-file-locking": "False"})
dask.config.set({"logging.distributed": "error"})
dask.config.set(num_workers=15)
dask.config.set({"array.chunk-size": "500MiB"})


# Compute metrics
def compute_metrics(ref, pred):
    diff = pred - ref
    mse = float(((diff) ** 2).mean().item())
    rmse = np.sqrt(mse)
    mae = float(np.abs(diff).mean().item())
    bias = float(diff.mean().item())
    results = {"mse": mse, "rmse": rmse, "mae": mae, "bias": bias}
    return {metric: round(result, 4) for metric, result in results.items()}


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
    ).compute()

    print(f"Extrapolating wind speed to {height_target}m using alpha with temporal smoothing.")
    ds_ws_target_alpha_temp_smoothed = extrapolate_wind_speed(
        ds_ws_reference,
        height_reference,
        height_target,
        tile_dataset_to_target_time(
            da_alpha_temp_smoothed_cerra.alpha, ds_ws_reference.time
        ),
    ).compute()

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
    ).compute()

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
    print("----------------------------------------------------")
    print("Computing wind extrapolation to 100m from CERRA 10m wind...")
    ds_ws_10m_2021 = ds_ws_10m.ds_data.sel(time="2021").load()
    # ds_ws_100m_2021 = ds_ws.ds_data.sel(height=100, time="2021").load()
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

    print("Computing wind extrapolation to 150m from CERRA 10m wind...")
    (
        ds_ws_150m_alpha,
        ds_ws_150m_alpha_temp_smoothed,
        ds_ws_150m_alpha_tempspatial_smoothed,
    ) = extrapolate_ws_to_target_height(
        ds_ws_10m_2021,
        10,
        150,
        da_alpha_cerra,
        da_alpha_temp_smoothed_cerra,
        da_alpha_tempspatial_smoothed_cerra,
    )

    print(
        f"\tProcessing time for CERRA wind extrapolation computation: {(time.time() - time_start) / 60:.2f} minutes."
    )

    print("----------------------------------------------------")
    print("Computing wind extrapolation to 100m from CORDEX 10m wind...")
    ds_sfcWind_climate_model.ds_data = ds_sfcWind_climate_model.ds_data.sel(
        time=slice("2021", "2030")
    ).load()
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

    print("Computing wind extrapolation to 150m from CORDEX 10m wind...")
    (
        ds_ws_150m_alpha_climate_model_10m,
        ds_ws_150m_alpha_temp_smoothed_climate_model_10m,
        ds_ws_150m_alpha_tempspatial_smoothed_climate_model_10m,
    ) = extrapolate_ws_to_target_height(
        ds_sfcWind_climate_model.ds_data,
        10,
        150,
        sel_extent(da_alpha_cerra, extent_bc),
        sel_extent(da_alpha_temp_smoothed_cerra, extent_bc),
        sel_extent(da_alpha_tempspatial_smoothed_cerra, extent_bc),
    )

    print(
        f"\tProcessing time for CORDEX wind extrapolation computation: {(time.time() - time_start) / 60:.2f} minutes."
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
        "Computing metrics for CERRA alpha extrapolation from 10m to 150m, based on year 2021..."
    )
    # Reference 150m wind speed
    ds_ws_150m_ref = ds_ws.ds_data.sel(height=150, time="2021").load()

    compute_and_print_metrics_for_extrapolation(
        ds_ws_150m_ref,
        ds_ws_150m_alpha,
        ds_ws_150m_alpha_temp_smoothed,
        ds_ws_150m_alpha_tempspatial_smoothed,
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
    del ds_ws_100m_climate_model

    print(
        "Computing metrics for CORDEX alpha extrapolation from 10m to 150m, based on year 2021-2030..."
    )
    ds_ws_150m_climate_model.ds_data = ds_ws_150m_climate_model.ds_data.sel(
        time=slice("2021", "2030")
    ).load()
    compute_and_print_metrics_for_extrapolation(
        ds_ws_150m_climate_model.ds_data,
        ds_ws_150m_alpha_climate_model_10m,
        ds_ws_150m_alpha_temp_smoothed_climate_model_10m,
        ds_ws_150m_alpha_tempspatial_smoothed_climate_model_10m,
    )

    print("----------------------------------------------------")


print(f"Total processing time: {(time.time() - time_start) / 60:.2f} minutes.")
