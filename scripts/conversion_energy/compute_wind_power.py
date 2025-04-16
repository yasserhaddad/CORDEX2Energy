import argparse
import json
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.wind import (
    CH_HUB_HEIGHTS,
    CH_TURBINES,
    EUROPE_HUB_HEIGHTS,
    EUROPE_TURBINES,
    compute_power_law_exponent,
    compute_smooth_power_curve,
    compute_wind_power,
    extrapolate_wind_speed,
)
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import COMPRESSED_ZARR_ENCODING
from src.utils.utils_xarray import sel_extent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    path_data_analysis = Path(paths["path_data_analysis"])
    path_data_processed = Path(paths["path_data_processed"])

    path_energy = path_data_processed / "energy"

    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})

    time_start = time.time()
    print("Computing smoothed power curves...")
    power_curves_europe_smoothed = {
        turbine: compute_smooth_power_curve(turbine) for turbine in EUROPE_TURBINES
    }
    power_curves_ch_smoothed = {
        turbine: compute_smooth_power_curve(turbine) for turbine in CH_TURBINES
    }

    print("Loading wind data...")
    extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}
    extent_ch = {"lat_min": 45.8, "lat_max": 47.9, "lon_min": 5.9, "lon_max": 10.5}
    chunks_bc = {
        "time": 2920,
        "lat": 128,
        "lon": 128,
    }

    ds_wind_100m = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp85"
        / "bias_corrected"
        / "3hr"
        / "wind100m.zarr",
        extent=extent_ch,
        chunks=chunks_bc,
    )

    ds_wind_150m = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp85"
        / "bias_corrected"
        / "3hr"
        / "wind150m.zarr",
        extent=extent_ch,
        chunks=chunks_bc,
    )

    print("Computing wind speed exponent...")
    with ProgressBar(dt=60):
        da_alpha = compute_power_law_exponent(
            ds_wind_100m.ds_data.wind100m, ds_wind_150m.ds_data.wind150m, 100, 150
        ).compute()

    ds_wind_power = []

    for turbine_type in CH_TURBINES:
        print(f"Computing wind power for {turbine_type}")
        da_wind_hub_height = extrapolate_wind_speed(
            ds_wind_100m.ds_data.wind100m, 100, CH_HUB_HEIGHTS[turbine_type], da_alpha
        )
        ds_wind_power.append(
            compute_wind_power(
                da_wind_hub_height,
                power_curves_ch_smoothed[turbine_type][
                    turbine_type.replace("/", "_")
                ],
                turbine_type.replace("/", "_"),
                step=0.01,
                parallelized=True,
            )
        )

    # Combine all wind power data arrays into a single dataset with each turbine type as a separate data variable
    print("Computing all wind power outputs...")
    with ProgressBar(dt=60):
        ds_wind_power = xr.merge(ds_wind_power).compute()

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(chunks_bc.values()),
        }
        for var in ds_wind_power.data_vars
    }

    # Save the combined dataset to a file
    output_path = (
        path_energy
        / "ch"
        / f"wind_power_{args.climate_model_chain}_{args.climate_scenario}.zarr"
    )
    ds_wind_power.to_zarr(output_path, mode="w", encoding=zarr_encoding)
    print("Time to compute wind power for all turbines: ", (time.time() - time_start) / 60, " minutes.")
