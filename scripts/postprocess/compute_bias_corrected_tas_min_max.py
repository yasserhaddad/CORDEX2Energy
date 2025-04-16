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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])

    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
    )

    ds_tas = xr.open_zarr(bias_corrected_data_dir / args.temp_res / "tas.zarr")

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
        for var in ds_tas.data_vars
    }

    print("Computing daily min")
    with ProgressBar(dt=10):
        ds_tas_min = ds_tas.resample(time="1D").min().compute()
        ds_tas_min.to_zarr(bias_corrected_data_dir / "day" / "tas_min.zarr", encoding=zarr_encoding, mode="w")
    print("Computing daily max")
    with ProgressBar(dt=10):
        ds_tas_max = ds_tas.resample(time="1D").max().compute()
        ds_tas_max.to_zarr(bias_corrected_data_dir / "day" / "tas_max.zarr", encoding=zarr_encoding, mode="w")