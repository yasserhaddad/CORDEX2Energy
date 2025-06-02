import xarray as xr
import sys

import argparse
import json
from dask.diagnostics import ProgressBar
from pathlib import Path

sys.path.append("../..")
from src.utils.utils_xarray import sel_extent

VARIABLES = [
    "huss",
    "hurs",
    "pr",
    "rsds",
    "sfcWind",
    "tas",
    "tas_max",
    "tas_min",
    "clt",
    "psl",
]

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
    climate_model_chain_dir =(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    )
    bias_corrected_data_dir = (
        climate_model_chain_dir
        / "bias_corrected"
    )

    extent = {
        "lat_min": 44.4,
        "lat_max": 48.6,
        "lon_min": 4.5,
        "lon_max": 16.5,
    }
    (climate_model_chain_dir / "prevah").mkdir(parents=True, exist_ok=True)

    for file in (bias_corrected_data_dir / "day").glob("*.zarr"):
        if file.stem not in VARIABLES:
            continue
        print(f"Processing {file.name}")
        with ProgressBar(dt=2):
            ds = sel_extent(xr.open_zarr(file, chunks="auto"), extent=extent)
            output_path = climate_model_chain_dir / "prevah" / f"{file.stem}.nc"
            ds.to_netcdf(output_path, mode="w")