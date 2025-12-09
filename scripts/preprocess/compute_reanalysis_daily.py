#!/usr/bin/env python3
"""
Compute daily means from regridded reanalysis data.

This script takes the finalized regridded dataset, computes daily means,
renames variables to {variable}_daily, and saves to zarr format.
"""

import sys

sys.path.append("../..")

import json
from pathlib import Path
import argparse
from time import time
import shutil
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import os

from src.utils.constants import COMPRESSED_ZARR_ENCODING


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute daily means from regridded reanalysis data."
    )
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("variable", type=str, help="Variable to process.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])

    time_start = time()

    # Configure dask
    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
    os.environ["MKL_NUM_THREADS"] = "1"  # MKL
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

    dask.config.set({"logging.distributed": "error"})
    dask.config.set(num_workers=16, threads_per_worker=16)
    dask.config.set({"array.chunk-size": "400MiB"})

    dask.config.set({"distributed.worker.memory.target": 0.75})
    dask.config.set({"distributed.worker.memory.spill": 0.8})
    dask.config.set({"distributed.worker.memory.pause": 0.9})

    # Load the regridded data
    input_path = (
        path_data_processed / "cerra_processed" / "regridded" / f"{args.variable}.zarr"
    )

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print(f"Loading regridded data from {input_path}...")
    ds = xr.open_zarr(input_path)

    print(f"Computing daily mean for {args.variable}...")
    # Compute daily mean
    ds_daily = ds.resample(time="1D").mean()

    # Rename variable to {variable}_daily
    var_daily_name = f"{args.variable}_daily"
    ds_daily = ds_daily.rename({args.variable: var_daily_name})

    print(f"Renamed {args.variable} to {var_daily_name}")

    # Prepare output
    # Use same chunks as regridded data
    target_chunks = {"time": -1, "lat": 20, "lon": 20}
    zarr_encoding = {
        var_daily_name: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
    }

    output_zarr = (
        path_data_processed
        / "cerra_processed"
        / "regridded"
        / f"{args.variable}_daily.zarr"
    )

    if output_zarr.exists():
        print(f"Removing existing {output_zarr}")
        shutil.rmtree(output_zarr)

    print(f"Saving to {output_zarr}...")
    with ProgressBar(dt=10):
        ds_daily.chunk(target_chunks).to_zarr(
            output_zarr, mode="w", encoding=zarr_encoding
        )

    print(f"✓ Successfully saved {var_daily_name} to {output_zarr}")
    print(f"  Processing time: {(time() - time_start) / 60:.2f} minutes.")
