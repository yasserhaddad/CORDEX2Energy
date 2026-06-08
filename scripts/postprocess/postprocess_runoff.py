# Description: Apply a mask to the data to set NaNs present in the dataset
# prior to bias correction back to NaN.

import argparse
import shutil
import sys
import time

import os
import dask
from dask.diagnostics import ProgressBar



import json
from pathlib import Path

from cordex2energy.data_processing.load_data import ClimateModelData
from cordex2energy.utils.cdo_commands import cdo_remap
from cordex2energy.utils.constants import EXTENT_BC_MRRO, REGRIDDED_CHUNKS, BC_DAILY_CHUNKS

dask.config.set(num_workers=10)
dask.config.set({"array.chunk-size": "500MiB"})


os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

dask.config.set({"logging.distributed": "error"})
dask.config.set(num_workers=16, threads_per_worker=16)
dask.config.set({"array.chunk-size": "400MiB"})

dask.config.set(
    {"distributed.worker.memory.target": 0.75}
)  # Spill to disk at 75% usage
dask.config.set({"distributed.worker.memory.spill": 0.8})  # Start spilling at 80%
dask.config.set({"distributed.worker.memory.pause": 0.9})  # Pause worker at 90%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument(
        "--climate_model_chain", type=str, default=None, help="Climate model chain."
    )
    parser.add_argument(
        "--climate_scenario", type=str, default=None, help="Climate scenario."
    )

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    t = time.time()

    climate_data_dir = (
        (
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
        )
        if args.climate_model_chain and args.climate_scenario
        else (path_data_processed / "cerra_processed")
    )

    tmp_climate_data_dir = (
        (
            path_data_temporary
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
        )
        if args.climate_model_chain and args.climate_scenario
        else (path_data_temporary / "cerra_processed")
    )

    temp_res = "day"
    variable = "mrro" if args.climate_model_chain and args.climate_scenario else "ro"

    # Load the regridded and bias-corrected datasets
    regridded_path = (
        climate_data_dir / "regridded"
        if args.climate_model_chain and args.climate_scenario
        else (
            path_data_processed
            / "cordex_processed"
            / "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp85"
            / "regridded"
        )
    )

    ds_regridded = ClimateModelData(
        regridded_path / temp_res / "mrro.zarr",
        extent=EXTENT_BC_MRRO,
        chunks=REGRIDDED_CHUNKS,
    )

    bc_file_path = (
        climate_data_dir / "bias_corrected" / temp_res / f"{variable}.zarr"
        if args.climate_model_chain and args.climate_scenario
        else climate_data_dir / "regridded" / f"{variable}.zarr"
    )
    ds_bc = ClimateModelData(
        bc_file_path,
        extent=EXTENT_BC_MRRO,
        chunks=BC_DAILY_CHUNKS if args.climate_model_chain and args.climate_scenario else REGRIDDED_CHUNKS,
    )

    # convert from kg m-2 to kg m-2 s-1
    ds_bc.ds_data = ds_bc.ds_data / 86400
    ds_bc.ds_data[variable].attrs["units"] = "kg m-2 s-1"
    # Apply the mask to the bias-corrected dataset
    # to set NaNs present in the dataset prior to bias correction back to NaN
    print("Applying mask to bias-corrected dataset...")
    with ProgressBar(dt=60):
        mask_nan = (
            ds_regridded.ds_data["mrro"]
            .sel(time="1991-01-01")
            .isnull()
            .compute()
            .drop_vars("time")
            .squeeze()
        )
        ds_bc.ds_data = ds_bc.ds_data.where(~mask_nan)

    temp_output_dir = tmp_climate_data_dir / "temp" / variable / "blocks"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    print("Writing and regridding masked dataset in 20-year blocks...")
    start_year = ds_bc.ds_data["time"].dt.year.min().item()
    end_year = ds_bc.ds_data["time"].dt.year.max().item()

    for year_start in range(start_year, end_year + 1, 20):
        year_end = min(year_start + 19, end_year)
        block_file = temp_output_dir / f"{variable}_masked_{year_start}_{year_end}.nc"
        block_file_regridded = (
            temp_output_dir / f"{variable}_masked_regridded_{year_start}_{year_end}.nc"
        )
        if not block_file.exists() and not block_file_regridded.exists():
            print(f"\tProcessing years {year_start}-{year_end}...")
            ds_block = ds_bc.ds_data.sel(
                time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
            )
            ds_block.to_netcdf(block_file)
            cdo_remap(
                str(block_file),
                str(block_file_regridded),
                "conservative",
                path_data_processed / "climate_grids" / "g010_MERIT_2.txt",
            )
            block_file.unlink()

    print(
        "\tTime to write and regrid masked dataset: ",
        (time.time() - t) / 60,
        " minutes.",
    )

    print(f"Completed. Time elapsed: {(time.time() - t) / 60} minutes.")
