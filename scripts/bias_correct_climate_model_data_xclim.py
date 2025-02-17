import sys

sys.path.append("..")

from src.load_data import ClimateModelData, ClimateReanalysisData
from src.constants import COMPRESSED_ZARR_ENCODING
from src.utils_xarray import get_time_resolution

import time
import json
import shutil
import argparse
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("variable", type=str, help="Variable to process.")
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")
    parser.add_argument(
        "variable_ref",
        type=str,
        help="Variable to use as reference for bias correction.",
    )
    parser.add_argument(
        "bias_correction_method", type=str, help="Bias correction method to use."
    )
    parser.add_argument(
        "--height_ref",
        type=int,
        default=None,
        help="Height to select from the reference dataset.",
    )
    parser.add_argument(
        "--nr_processes",
        type=int,
        default=8,
        help="Height to select from the reference dataset.",
    )
    parser.add_argument(
        "--default_parameters",
        action="store_true",
        help="Use default settings for bias correction.",
    )

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])

    time_start = time.time()

    # Define input and output paths
    regridded_cordex_var_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "regridded"
        / args.temp_res
    )
    regridded_cerra_var_path = path_data_processed / "cerra_processed" / "regridded"

    output_bias_corrected_cordex_var_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / args.temp_res
    )
    output_bias_corrected_cordex_var_path.mkdir(parents=True, exist_ok=True)

    # Load data
    # chunks = {"time": -1}
    chunks = {"time": -1, "lat": 20, "lon": 20}
    extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}
    climate_data = ClimateModelData(
        regridded_cordex_var_path / f"{args.variable}.zarr",
        chunks=chunks,
        extent=extent,
        load=False,
    )
    reanalysis_data = ClimateReanalysisData(
        regridded_cerra_var_path / f"{args.variable_ref}.zarr",
        chunks=chunks,
        extent=extent,
        load=False,
    )

    if args.height_ref is not None:
        reanalysis_data.ds_data = reanalysis_data.ds_data.sel(
            height=args.height_ref
        ).drop_vars("height")
        if (
            "height" in climate_data.ds_data.data_vars
            and "height" not in climate_data.ds_data[args.variable].dims
        ):
            climate_data.ds_data = climate_data.ds_data.drop_vars("height")

    if args.variable == "pr":
        if climate_data.ds_data[args.variable].units == "kg m-2 s-1":
            climate_data.ds_data = climate_data.ds_data * 86400
            climate_data.ds_data[args.variable].attrs["units"] = "mm d-1"
        if reanalysis_data.ds_data[args.variable_ref].units == "kg m**-2":
            reanalysis_data.ds_data[args.variable_ref].attrs["units"] = "mm d-1"

    if args.variable == "rsds":
        if reanalysis_data.ds_data[args.variable_ref].units == "J m**-2":
            time_res = get_time_resolution(reanalysis_data.ds_data).total_seconds() / 3600
            reanalysis_data.ds_data = reanalysis_data.ds_data / (3600 * time_res)
            reanalysis_data.ds_data[args.variable_ref].attrs["units"] = "W m**-2"

        mask_ssrd_land = ~np.isnan(reanalysis_data.ds_data[args.variable_ref].isel(time=0))
        climate_data.ds_data = climate_data.ds_data.where(mask_ssrd_land).fillna(-10)
        reanalysis_data.ds_data = reanalysis_data.ds_data.fillna(-10)


    # Perform bias correction
    ds_bias_corrected = climate_data.bias_correction_xclim(
        reanalysis_data.ds_data.rename({args.variable_ref: args.variable}),
        data_var=args.variable,
        reference_period=("1991", "2017"),
        future_period=("2018", "2100"),
        bias_correction_method=args.bias_correction_method,
        running_window_length=91,
        default_parameters=args.default_parameters,
        print_time=True,
        load_data=False,
        nr_processes=args.nr_processes,
    )

    # Limit relative humidity to [0, 100]
    if args.variable == "hurs":
        ds_bias_corrected = ds_bias_corrected.clip(0, 100)

    if args.variable == "rsds":
        ds_bias_corrected = ds_bias_corrected.where(ds_bias_corrected > 0)

    # Save the bias-corrected data
    target_chunks = {
        "time": 365 if args.temp_res == "day" else 500,
        "lat": 256,
        "lon": 256,
    }

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in climate_data.ds_data.data_vars
    }

    if (output_bias_corrected_cordex_var_path / f"{args.variable}.zarr").exists():
        shutil.rmtree(output_bias_corrected_cordex_var_path / f"{args.variable}.zarr")

    ds_bias_corrected.chunk(target_chunks).to_zarr(
        output_bias_corrected_cordex_var_path / f"{args.variable}.zarr",
        mode="w",
        encoding=zarr_encoding,
    )

    print(f"\tProcessing time: {(time.time() - time_start) / 60:.2f} minutes.")
