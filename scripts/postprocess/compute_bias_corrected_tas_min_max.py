import sys
import argparse
import json
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import xarray as xr


from cordex2energy.utils.constants import COMPRESSED_ZARR_ENCODING, BC_DAILY_CHUNKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")
    args = parser.parse_args()

    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
            "distributed.worker.memory.spill": 0.8,  # Start spilling at 80%
            "distributed.worker.memory.pause": 0.9,  # Pause worker at 90%
            "array.chunk-size": "512MiB",
            "logging.distributed": "error",  # Less verbose logs
        }
    )

    # Load paths
    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    # Paths and chunking
    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
    )
    ds_tas = xr.open_zarr(
        bias_corrected_data_dir / args.temp_res / "tas.zarr", chunks="auto"
    )

    # Compute daily min and max in one resample pass
    ds_daily = xr.Dataset(
        {
            "tas_min": ds_tas.tas.resample(time="1D").min(),
            "tas_max": ds_tas.tas.resample(time="1D").max(),
        }
    )
    ds_daily = ds_daily.sel(
        time=~((ds_daily["time"].dt.month == 2) & (ds_daily["time"].dt.day == 29))
    )
    zarr_encoding = {
        "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
        "chunks": tuple(BC_DAILY_CHUNKS.values()),
    }

    # Save results lazily (single DAG execution per write)
    out_dir = bias_corrected_data_dir / "day"
    out_dir.mkdir(exist_ok=True)

    with ProgressBar(dt=30):
        ds_daily[["tas_min"]].chunk(BC_DAILY_CHUNKS).to_zarr(
            out_dir / "tas_min.zarr", encoding={"tas_min": zarr_encoding}, mode="w"
        )
        ds_daily[["tas_max"]].chunk(BC_DAILY_CHUNKS).to_zarr(
            out_dir / "tas_max.zarr", encoding={"tas_max": zarr_encoding}, mode="w"
        )
