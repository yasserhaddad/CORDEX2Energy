import argparse
import json
import sys
import time
from pathlib import Path

import dask
import pandas as pd
import xarray as xr

sys.path.append("../..")

from src.conversion_energy.pv import (
    NORMALISED_ROOFTOP_PV_CONFIGURATION,
    compute_pv_for_grid,
)
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import COMPRESSED_ZARR_ENCODING, EXTENT_CH
from src.utils.utils_xarray import sel_extent, match_instaneous_to_mean_values

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

    print("Loading data")
    chunks_bc = {
        "time": 2920,
        "lat": 128,
        "lon": 128,
    }

    ds_rsds_bc = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "rsds.zarr",
        extent=EXTENT_CH,
    )

    ds_tas_bc = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "tas.zarr",
        extent=EXTENT_CH,
    )

    ds_sfcWind_bc = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "sfcWind.zarr",
        extent=EXTENT_CH,
    )

    ds_orog = sel_extent(
        xr.open_dataset(
            sorted(
                (
                    path_data_processed
                    / "cordex_processed"
                    / f"{args.climate_model_chain}_{args.climate_scenario}"
                    / "regridded"
                    / "fx"
                ).glob("*.nc")
            )[0]
        ),
        extent=EXTENT_CH,
    )

    print("Matching instantaneous to mean values")
    ds_tas_bc.ds_data = match_instaneous_to_mean_values(
        ds_tas_bc.ds_data - 273.15, skip_leap_days=True, time_res="3h"
    )
    ds_tas_bc.ds_data["time"] = ds_tas_bc.ds_data["time"] + pd.Timedelta(
        hours=1, minutes=30
    )

    ds_sfcWind_bc.ds_data = match_instaneous_to_mean_values(
        ds_sfcWind_bc.ds_data, skip_leap_days=True, time_res="3h"
    )
    ds_sfcWind_bc.ds_data["time"] = ds_sfcWind_bc.ds_data["time"] + pd.Timedelta(
        hours=1, minutes=30
    )

    print(f"Data matching time: {(time.time() - time_start) / 60:.2f} minutes")

    step_start = time.time()
    ds_rsds_bc.ds_data = ds_rsds_bc.ds_data.load()
    ds_tas_bc.ds_data = ds_tas_bc.ds_data.load()
    ds_sfcWind_bc.ds_data = ds_sfcWind_bc.ds_data.load()
    print(f"Data loading time: {(time.time() - step_start) / 60:.2f} minutes")

    step_start = time.time()
    # azimuths = {"E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "N": 0}
    # tilts = [3, 8, 15, 25, 35, 45, 60, 80, 90]

    azimuths = {"E": 90, "S": 180, "W": 270, "N": 0}
    tilts = [25, 35, 45]
    datasets = []

    pv_step_start = time.time()
    for direction, azimuth in azimuths.items():
        for tilt in tilts:
            step_start = time.time()
            datasets.append(
                compute_pv_for_grid(
                    ds_rsds_bc.ds_data.rsds,
                    ds_tas_bc.ds_data.tas,
                    ds_sfcWind_bc.ds_data.sfcWind,
                    tilt=tilt,
                    azimuth=azimuth,
                    pv_configuration=NORMALISED_ROOFTOP_PV_CONFIGURATION,
                    da_altitude=ds_orog.orog,
                    num_workers=15,
                ).rename({"ac_power": f"tilt_{tilt}_azimuth_{direction}"})
                / NORMALISED_ROOFTOP_PV_CONFIGURATION["module_parameters"]["pdc0"]
            )
            print(
                f"Computation time for tilt {tilt}, azimuth {direction}: {(time.time() - step_start) / 60:.2f} minutes"
            )
    print(
        f"Total PV computation time: {(time.time() - pv_step_start) / 60:.2f} minutes"
    )

    step_start = time.time()
    ds_pv = xr.merge(datasets)
    print(f"Dataset merging time: {(time.time() - step_start) / 60:.2f} minutes")

    # Save the combined dataset to a file
    step_start = time.time()
    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(chunks_bc.values()),
        }
        for var in ds_pv.data_vars
    }
    output_path = (
        path_energy
        / "ch"
        / "pv"
        / f"pv_{args.climate_model_chain}_{args.climate_scenario}.zarr"
    )
    ds_pv.to_zarr(output_path, mode="w", encoding=zarr_encoding)
    print(f"PV data saved to {output_path}")
    print(f"PV data saving time: {(time.time() - step_start) / 60:.2f} minutes")

    print("Total time: ", (time.time() - time_start) / 60, "minutes")
