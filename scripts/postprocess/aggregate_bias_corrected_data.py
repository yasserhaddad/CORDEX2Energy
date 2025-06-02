import sys

import argparse
import json
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import xarray as xr

sys.path.append("../..")
from src.utils.constants import COMPRESSED_ZARR_ENCODING

dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})
dask.config.set(num_workers=20, scheduler="processes")
dask.config.set({"array.chunk-size": "500MiB"})

def aggregate_to_daily(ds: xr.Dataset) -> xr.Dataset:
    return ds.resample(time='1D').mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()

    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
    )

    bc_chunks = {
        "time": 2920,
        "lat": 128,
        "lon": 128,
    }

    for file in (bias_corrected_data_dir / args.temp_res).glob("*.zarr"):
        print(f"Processing {file.name}")
        with ProgressBar(dt=10):
            ds = aggregate_to_daily(
                xr.open_zarr(file, chunks=bc_chunks)
            ).compute()

            output_path = bias_corrected_data_dir / "day" / file.name
            target_chunks = {
                "time": 365,
                "lat": 256,
                "lon": 256
            }

            zarr_encoding = {
                var: {
                    "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                    "chunks": tuple(target_chunks.values()),
                }
                for var in ds.data_vars
            }
            ds.to_zarr(output_path, mode="w")
