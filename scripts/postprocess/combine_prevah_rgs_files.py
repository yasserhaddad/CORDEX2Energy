import argparse
import json
import xarray as xr
import dask
import sys

from dask.diagnostics import ProgressBar
from pathlib import Path

sys.path.append("../..")
from src.utils.constants import (
    COMPRESSED_ZARR_ENCODING
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_rgs = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "prevah"
        / "rgs"
    )
    target_chunks = {
        "time": 365, "x": 256, "y": 256
    }
    with ProgressBar(dt=1):
        ds_rgs = xr.open_mfdataset(path_rgs.glob("*.nc"), combine="by_coords").rename({
            "lon2": "x", "lat2": "y"
        })

        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(target_chunks.values()),
            }
            for var in ds_rgs.data_vars
        }
        ds_rgs.chunk(target_chunks).to_zarr(
            path_rgs.parent / "rgs.zarr",
            encoding=zarr_encoding,
            mode="w",
        )