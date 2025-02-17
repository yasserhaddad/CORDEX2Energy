import sys

sys.path.append("..")

import json
from pathlib import Path
import argparse
from time import time
from src.constants import COMPRESSED_ZARR_ENCODING
from src.load_data import ClimateReanalysisData, cdo_remap
import shutil


def get_paths_cerra_data(
    path_data_cerra: Path,
    path_data_cerra_land: Path,
    path_data_processed: Path,
    variable: str,
) -> Path:
    if variable == "ssrd" or variable == "skt" or variable == "tp":
        cerra_path = path_data_cerra_land / "original" / variable
    elif variable == "q":
        cerra_path = path_data_processed / "cerra_processed" / variable
    else:
        cerra_path = path_data_cerra / "original" / variable
    return cerra_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("variable", type=str, help="Variable to process.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_cerra = Path(paths["path_data_cerra"])
    path_data_cerra_land = Path(paths["path_data_cerra_land"])

    path_data_cerra_var = get_paths_cerra_data(
        path_data_cerra, path_data_cerra_land, path_data_processed, args.variable
    )
    time_start = time()

    # Remap the data to the target grid
    temp_output_dir = path_data_processed / "cerra_processed" / "temp" / args.variable
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    for sim_path in path_data_cerra_var.glob("*.nc"):
        cdo_remap(
            sim_path,
            temp_output_dir / sim_path.name,
            "conservative",
            path_data_processed / "climate_grids" / "grid_EUR11_lonlat_griddes.txt",
        )

    # Process the remapped data
    extent = None
    chunks = None
    reanalysis_data = ClimateReanalysisData(
        temp_output_dir,
        subset_data_vars=[args.variable],
        chunks=chunks,
        extent=extent,
        reordered_coords=["time", "lat", "lon"],
        load=False,
    )

    output_path = path_data_processed / "cerra_processed" / "regridded"

    output_path.mkdir(parents=True, exist_ok=True)

    target_chunks = {"time": -1, "lat": 20, "lon": 20}
    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values())
        }
        for var in reanalysis_data.ds_data.data_vars
    }

    if (output_path / f"{args.variable}.zarr").exists():
        shutil.rmtree(output_path / f"{args.variable}.zarr")

    reanalysis_data.ds_data.chunk(target_chunks).to_zarr(
        output_path / f"{args.variable}.zarr", mode="w", encoding=zarr_encoding
    )

    shutil.rmtree(temp_output_dir)

    print(
        f"\tProcessing time for variable {args.variable}: {(time() - time_start) / 60:.2f} minutes."
    )
