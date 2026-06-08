import xarray as xr
import sys

import argparse
import json
from dask.diagnostics import ProgressBar
from pathlib import Path


from cordex2energy.data_processing.load_data import ClimateModelData

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
    output_path_dir = (climate_model_chain_dir / "prevah" / "input_data")
    output_path_dir.mkdir(parents=True, exist_ok=True)

    # First pass: find common time period and load all datasets
    datasets = {}
    time_ranges = []

    for file in (bias_corrected_data_dir / "day").glob("*.zarr"):
        if file.stem not in VARIABLES:
            continue
        print(f"Loading {file.name}")
        ds = ClimateModelData(
            file,
            chunks="auto",
            drop_leap_days=True,
            extent=extent
        ).ds_data
        ds = ds.assign_coords(time=ds.time.dt.floor('D'))
        ds["time"].encoding["units"] = ds["time"].encoding.get("units", "days since 1951-01-01T00:00:00").replace("12:00:00", "00:00:00")

        datasets[file.stem] = ds
        time_ranges.append((ds.time.min().values, ds.time.max().values))

    # Get the intersection of all time periods
    common_start = max(t[0] for t in time_ranges)
    common_end = min(t[1] for t in time_ranges)
    print(f"Common time period: {common_start} to {common_end}")

    # Second pass: slice to common period and save
    for var_name, ds in datasets.items():
        print(f"Processing {var_name}")
        with ProgressBar(dt=10):
            ds_subset = ds.sel(time=slice(common_start, common_end))
            # Convert to float32
            for var in ds_subset.data_vars:
                ds_subset[var] = ds_subset[var].astype('float32')
            output_path = output_path_dir / f"{var_name}.nc"
            if output_path.exists():
                output_path.unlink()
            ds_subset.ffill(dim="lon").ffill(dim="lat").to_netcdf(output_path, mode="w")
        ds.close()

    # for file in (bias_corrected_data_dir / "day").glob("*.zarr"):
    #     if file.stem not in VARIABLES:
    #         continue
    #     print(f"Processing {file.name}")
    #     with ProgressBar(dt=10):
    #         ds = ClimateModelData(
    #             file,
    #             chunks="auto",
    #             drop_leap_days=True,
    #             extent=extent
    #         ).ds_data
    #         ds = ds.assign_coords(time=ds.time.dt.floor('D'))
    #         ds["time"].encoding["units"] = ds["time"].encoding.get("units", "days since 1951-01-01T00:00:00").replace("12:00:00", "00:00:00")
    #         output_path = output_path_dir / f"{file.stem}.nc"
    #         if output_path.exists():
    #             output_path.unlink()
    #         ds.ffill(dim="lon").ffill(dim="lat").to_netcdf(output_path, mode="w")