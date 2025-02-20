import sys

sys.path.append("..")

import time
import json
import shutil
import argparse
import xarray as xr
from pathlib import Path

from src.convert_climate_vars import (
    convert_wind_components_to_speed,
)
from src.utils_load_data import (
    get_raw_cordex_data_paths,
    get_components_climate_model_chain,
)

from src.load_data import ClimateModelData, ClimateReanalysisData
from src.constants import COMPRESSED_ZARR_ENCODING

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
        "--parametric_existing_settings",
        action="store_true",
        help="Use existing parametric settings for bias correction.",
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
    chunks = {"time": 500, "lat": 256, "lon": 256}
    extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}
    climate_data = ClimateModelData(
        regridded_cordex_var_path / f"{args.variable}.zarr",
        chunks=chunks,
        extent=extent,
        load=False
    )
    reanalysis_data = ClimateReanalysisData(
        regridded_cerra_var_path / f"{args.variable_ref}.zarr",
        chunks=chunks,
        extent=extent,
        load=False,
    )

    if args.height_ref is not None:
        reanalysis_data.ds_data = reanalysis_data.ds_data.sel(height=args.height_ref)

    if args.variable == "pr":
        # Convert from kg m-2 (total precipitation) to kg m-2 s-1 (precipitation rate)
        reanalysis_data.ds_data = reanalysis_data.ds_data / (60 * 60 * 24)

    # Perform bias correction
    ds_bias_corrected = climate_data.bias_correction(
        reanalysis_data.ds_data.rename({args.variable_ref: args.variable}),
        data_var=args.variable,
        reference_period=("1991", "2017"),
        future_period=("2018", "2100"),
        bias_correction_method=args.bias_correction_method,
        parametric_existing_settings=args.parametric_existing_settings,
        running_window_mode=True,
        running_window_length=91,
        print_time=True,
        parallel=True,
        nr_processes=args.nr_processes
    )

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
