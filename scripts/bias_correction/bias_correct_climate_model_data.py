import sys

import pandas as pd

sys.path.append("../..")

from src.data_processing.load_data import ClimateModelData, ClimateReanalysisData
from src.utils.constants import (
    COMPRESSED_ZARR_ENCODING,
    EXTENT_BC,
    EXTENT_BC_MRRO,
    REFERENCE_PERIOD,
    FUTURE_PERIOD,
)
from src.utils.utils_xarray import get_time_resolution, get_time_offset
from src.data_processing.bias_correction import BiasCorrection

import dask
from dask.diagnostics import ProgressBar
import time
import json
import shutil
import argparse
from pathlib import Path

dask.config.set({"distributed.worker.multiprocessing-method": "forkserver"})
dask.config.set({"distributed.worker.use-file-locking": "False"})
dask.config.set({"logging.distributed": "error"})

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
        help="Number of cores to use for parallel processing.",
    )
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="auto",
        help="Memory limit for dask workers.",
    )
    parser.add_argument(
        "--default_parameters",
        action="store_true",
        help="Use default settings for bias correction.",
    )

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    time_start = time.time()
    client = None

    if args.nr_processes > 1:
        dask.config.set(num_workers=args.nr_processes, scheduler="processes")
        dask.config.set({"array.chunk-size": "500MiB"})

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
    chunks = {"time": -1, "lat": 20, "lon": 20}

    extent = EXTENT_BC_MRRO if args.variable == "mrro" else EXTENT_BC
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
        drop_leap_days=(args.variable != "rsds"),
    )

    climate_data.ds_data = climate_data.ds_data.chunk(chunks)
    reanalysis_data.ds_data = reanalysis_data.ds_data.chunk(chunks)

    if args.height_ref is not None:
        reanalysis_data.ds_data = reanalysis_data.ds_data.sel(
            height=args.height_ref
        ).drop_vars("height")
        print("Reference data height selected.")
        if (
            "height" in climate_data.ds_data.coords
            and "height" not in climate_data.ds_data[args.variable].dims
        ):
            climate_data.ds_data = climate_data.ds_data.drop_vars("height")
            print("Climate data height dropped.")

    if args.variable == "pr":
        if climate_data.ds_data[args.variable].units == "kg m-2 s-1":
            climate_data.ds_data = climate_data.ds_data * 86400
            climate_data.ds_data[args.variable].attrs["units"] = "mm d-1"
            print("Climate data units converted to mm d-1.")
        if reanalysis_data.ds_data[args.variable_ref].units == "kg m**-2":
            reanalysis_data.ds_data[args.variable_ref].attrs["units"] = "mm d-1"
            print("Reference data units converted to mm d-1.")

    if args.variable == "rsds":
        if reanalysis_data.ds_data[args.variable_ref].units == "J m**-2":
            time_res = get_time_resolution(reanalysis_data.ds_data).total_seconds()
            reanalysis_data.ds_data = reanalysis_data.ds_data / time_res
            reanalysis_data.ds_data[args.variable_ref].attrs["units"] = "W m**-2"
            print("Reference data units converted to W m-2.")

        # Check if the first time step of the climate data has hour and minutes as 1h30
        most_common_offset_between = get_time_offset(
            climate_data.ds_data, reanalysis_data.ds_data
        )
        if most_common_offset_between != pd.Timedelta("0 days 00:00:00"):
            # Subtract 1 hour and 30 minutes from all timesteps using a time offset
            reanalysis_data.ds_data = reanalysis_data.ds_data.assign_coords(
                time=reanalysis_data.ds_data.time - most_common_offset_between
            )
            print(
                f"Subtracted the time offset {most_common_offset_between} from all timesteps of the reanalysis dataset."
            )

        # Remove leap days from the reanalysis dataset
        reanalysis_data.ds_data = reanalysis_data.ds_data.sel(
            time=~(
                (reanalysis_data.ds_data.time.dt.month == 2)
                & (reanalysis_data.ds_data.time.dt.day == 29)
            )
        )

    if args.variable == "mrro":
        if climate_data.ds_data[args.variable].units == "kg m-2 s-1":
            climate_data.ds_data = climate_data.ds_data * 86400
            climate_data.ds_data[args.variable].attrs["units"] = "kg m-2"
            print("Climate data units converted to kg m-2.")
        if "lev" in reanalysis_data.ds_data.coords:
            if len(reanalysis_data.ds_data.lev) > 1:
                raise ValueError("Reference data has multiple levels.")

            reanalysis_data.ds_data = reanalysis_data.ds_data.isel(lev=0)
            print("Reference data level selected.")

        mask_nan = climate_data.ds_data[args.variable].notnull()
        climate_data.ds_data = climate_data.ds_data.fillna(0)
        reanalysis_data.ds_data = reanalysis_data.ds_data.fillna(0)

    # Quick check for NaN values along borders in the first time step
    has_nan = (
        climate_data.ds_data[args.variable].isel(time=0).isel(lat=0).isnull().any() |
        climate_data.ds_data[args.variable].isel(time=0).isel(lat=-1).isnull().any() |
        climate_data.ds_data[args.variable].isel(time=0).isel(lon=0).isnull().any() |
        climate_data.ds_data[args.variable].isel(time=0).isel(lon=-1).isnull().any()
    ).compute().item()
    if has_nan:
        # fill NaN values along spatial dimensions
        print(
            f"NaN values found in the borders of the first time step of {args.variable}. Filling NaN values along spatial dimensions."
        )
        # Forward fill and backward fill along lat and lon dimensions
        climate_data.ds_data = (
            climate_data.ds_data.ffill(dim="lat")
            .ffill(dim="lon")
            .bfill(dim="lat")
            .bfill(dim="lon")
        )
        reanalysis_data.ds_data = (
            reanalysis_data.ds_data.ffill(dim="lat")
            .ffill(dim="lon")
            .bfill(dim="lat")
            .bfill(dim="lon")
        )

    temp_output_dir = (
        path_data_temporary
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "temp"
        / "bias_corrected"
        / args.variable
    )
    temp_output_dir.mkdir(exist_ok=True, parents=True)

    # Perform bias correction
    bias_correction = BiasCorrection(
        climate_data,
        reanalysis_data,
        reference_period=REFERENCE_PERIOD,
        future_period=FUTURE_PERIOD,
        bias_correction_method=args.bias_correction_method,
        client=client,
    )

    ds_bias_corrected = bias_correction.bias_correction_xclim(
        data_var=args.variable,
        reference_data_var=args.variable_ref,
        running_window_length=91,
        default_parameters=args.default_parameters,
        output_path=temp_output_dir,
        load_data=False,
        print_time=True,
    )

    # Limit relative humidity and cloud cover fraction to [0, 100]
    if args.variable == "hurs" or args.variable == "clt":
        ds_bias_corrected = ds_bias_corrected.clip(0, 100)
        print("Values clipped to [0, 100].")

    print("Bias correction completed. Saving data...")
    # Save the bias-corrected data
    target_chunks = {
        "time": 365 if args.temp_res == "day" else 2920,
        "lat": 256 if args.temp_res == "day" else 128,
        "lon": 256 if args.temp_res == "day" else 128,
    }

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in ds_bias_corrected.data_vars
    }

    if (output_bias_corrected_cordex_var_path / f"{args.variable}.zarr").exists():
        shutil.rmtree(output_bias_corrected_cordex_var_path / f"{args.variable}.zarr")

    with ProgressBar(dt=60):
        ds_bias_corrected.to_zarr(
            output_bias_corrected_cordex_var_path / f"{args.variable}.zarr",
            mode="w",
            encoding=zarr_encoding,
        )
    print(
        f"Data saved to {output_bias_corrected_cordex_var_path / f'{args.variable}.zarr'}."
    )

    shutil.rmtree(temp_output_dir)
    if client is not None:
        client.close()

    print(f"\tProcessing time: {(time.time() - time_start) / 60:.2f} minutes.")
