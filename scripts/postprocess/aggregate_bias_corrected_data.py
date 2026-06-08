import sys

import argparse
import json
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import xarray as xr


from cordex2energy.utils.constants import COMPRESSED_ZARR_ENCODING, BC_DAILY_CHUNKS

def aggregate_to_daily(ds: xr.Dataset, drop_leap_days: bool = False, function: str = "mean") -> xr.Dataset:
    """Aggregate dataset to daily means, mins or max and optionally remove Feb 29 entries."""

    if function == "mean":
        daily = ds.resample(time="1D").mean()
    elif function == "min":
        daily = ds.resample(time="1D").min()
    elif function == "max":
        daily = ds.resample(time="1D").max()
    else:
        raise ValueError(f"Unsupported aggregation function: {function}")

    if drop_leap_days:
        daily = daily.sel(
            time=~((daily["time"].dt.month == 2) & (daily["time"].dt.day == 29))
        )
    return daily

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if output exists.")

    args = parser.parse_args()

    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set({
        "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
        "distributed.worker.memory.spill": 0.8,   # Start spilling at 80%
        "distributed.worker.memory.pause": 0.9,   # Pause worker at 90%
        "array.chunk-size": "512MiB",
        "logging.distributed": "error",           # Less verbose logs
    })

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
    )

    for file in (bias_corrected_data_dir / args.temp_res).glob("*.zarr"):
        if file.stem in ["wind100m", "wind150m"]:
            print(f"Skipping {file.name}.")
            continue
        print(f"Processing {file.name}")
        with ProgressBar(dt=10):
            output_path = bias_corrected_data_dir / "day" / file.name
            if output_path.exists() and not args.force:
                print(f"  Output {output_path} already exists. Skipping.")
                continue
            ds = aggregate_to_daily(
                xr.open_zarr(file, chunks="auto"),
                drop_leap_days=True
            )


            zarr_encoding = {
                var: {
                    "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                    "chunks": tuple(BC_DAILY_CHUNKS.values()),
                }
                for var in ds.data_vars
            }
            ds.chunk(BC_DAILY_CHUNKS).to_zarr(output_path, mode="w", encoding=zarr_encoding)
