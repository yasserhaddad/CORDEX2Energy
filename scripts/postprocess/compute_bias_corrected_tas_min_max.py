import sys
import argparse
import json
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import pandas as pd
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

    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    bc_chunks = {
        "time": 2920,
        "lat": 128,
        "lon": 128,
    }
    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
    )

    ds_tas = xr.open_zarr(bias_corrected_data_dir / args.temp_res / "tas.zarr", chunks=bc_chunks)
    time_range = pd.date_range(
        start=ds_tas.time.values[0],
        end=ds_tas.time.values[-1],
        freq="1h",
    )

    # Split the time range into three equal parts without splitting days
    total_days = (time_range[-1] - time_range[0]).days + 1
    split_size = total_days // 3
    split_points = [
        time_range[0] + pd.Timedelta(days=split_size * i) for i in range(3)
    ]
    split_points.append(time_range[-1].floor("D") + pd.Timedelta(days=1))  # Ensure the last split point is the last timestep

    time_ranges = [
        time_range[(time_range >= split_points[i]) & (time_range <= split_points[i + 1])]
        for i in range(len(split_points) - 1)
    ]
    for time_range_split in time_ranges:
        print(f"Split: {time_range_split[0]} to {time_range_split[-1]}")

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

    # Function to process each split
    def process_split(time_range_split):
        ds_tas_split = ds_tas.sel(time=slice(time_range_split[0], time_range_split[-1]))
        ds_tas_split = ds_tas.interp(time=time_range_split)
        if not time_range_split[-1] == ds_tas.time.isel(time=-1).item():
            ds_tas_split.sel(time=slice(time_range_split[0], time_range_split[-2]))
        print(f"Computing daily min for split {time_range_split[0]} to {time_range_split[-1]}")
        with ProgressBar(dt=10):
            ds_tas_min_split = ds_tas_split.resample(time="1D").min().compute()
        print(f"Computing daily max for split {time_range_split[0]} to {time_range_split[-1]}")
        with ProgressBar(dt=10):
            ds_tas_max_split = ds_tas_split.resample(time="1D").max().compute()
        return ds_tas_min_split, ds_tas_max_split

    # Process all splits
    results = [process_split(tr) for tr in time_ranges]

    # Combine the results
    ds_tas_min = xr.concat([res[0] for res in results], dim="time").rename({"tas": "tas_min"})
    ds_tas_max = xr.concat([res[1] for res in results], dim="time").rename({"tas": "tas_max"})

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in ds_tas_min.data_vars
    }
    # Save the combined results
    ds_tas_min.to_zarr(bias_corrected_data_dir / "day" / "tas_min.zarr", encoding=zarr_encoding, mode="w")

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in ds_tas_max.data_vars
    }
    ds_tas_max.to_zarr(bias_corrected_data_dir / "day" / "tas_max.zarr", encoding=zarr_encoding, mode="w")
