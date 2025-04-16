# Description: Apply a mask to the data to set NaNs present in the dataset
# prior to bias correction back to NaN.

import argparse
import shutil
import sys
import time

sys.path.append("../..")

import json
from pathlib import Path

import xarray as xr

from src.data_processing.load_data import ClimateModelData
from src.utils.cdo_commands import cdo_remap
from src.utils.constants import COMPRESSED_ZARR_ENCODING

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    t = time.time()

    climate_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    )

    tmp_climate_data_dir = (
        path_data_temporary
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    )

    temp_res = "day"
    variable = "mrro"

    regridded_chunks = {"time": -1, "lat": 20, "lon": 20}
    bc_chunks = {"time": 365, "lat": 256, "lon": 256}

    extent={'lat_min': 20.8, 'lat_max': 71.8, 'lon_min': -51.7, 'lon_max': 72.7}

    # Load the regridded and bias-corrected datasets
    ds_regridded = ClimateModelData(
        climate_data_dir / "regridded" / temp_res / f"{variable}.zarr",
        extent=extent,
        chunks=regridded_chunks
    )

    ds_bc = ClimateModelData(
        climate_data_dir / "bias_corrected" / temp_res / f"{variable}.zarr",
        extent=extent,
        chunks=bc_chunks
    )

    # Apply the mask to the bias-corrected dataset
    # to set NaNs present in the dataset prior to bias correction back to NaN
    print("Applying mask to bias-corrected dataset...")
    mask_nan = ds_regridded.ds_data[variable].isel(time=0).isnull()
    ds_bc.ds_data = ds_bc.ds_data.where(~mask_nan)

    temp_output_dir = tmp_climate_data_dir / "temp" / "mrro"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    if not (temp_output_dir / f"{variable}_masked.nc").exists():
        ds_bc.ds_data.to_netcdf(temp_output_dir / f"{variable}_masked.nc")
    print("\tTime to write masked dataset: ", (time.time() - t) / 60, " minutes.")

    print("Regridding masked dataset...")
    if not (temp_output_dir / f"{variable}_masked_regridded.nc").exists():
        cdo_remap(
            str(temp_output_dir / f"{variable}_masked.nc"),
            str(temp_output_dir / f"{variable}_masked_regridded.nc"),
            "conservative",
            path_data_processed / "climate_grids" / "g010_MERIT_2.txt"
        )
    print("\tTime to regrid masked dataset: ", (time.time() - t) / 60, " minutes.")
    # Load the regridded masked dataset
    print("Write the masked dataset as zarr...")
    ds_bc = xr.open_dataset(
        temp_output_dir / f"{variable}_masked_regridded.nc",
    )

    target_chunks = {
        "time": 365,
        "lat": 256,
        "lon": 256,
    }

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in ds_bc.data_vars
    }

    ds_bc.to_zarr(
        climate_data_dir / "bias_corrected" / temp_res / f"{variable}_masked.zarr",
        mode="w",
        encoding=zarr_encoding,
    )

    # Clean up temporary files
    shutil.rmtree(temp_output_dir)

    print(f"Completed. Time elapsed: {(time.time() - t) / 60} minutes.")
