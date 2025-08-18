import os
import sys

sys.path.append("../..")

import argparse
import ast
import json
import shutil
from pathlib import Path
from time import time

import dask
import pandas as pd
from src.data_processing.load_data import ClimateModelData
from src.data_processing.test_climate_data import (
    has_duplicate_coords,
    has_identical_random_timesteps,
    is_time_sorted,
)
from src.data_processing.timesteps_adjustments import extract_adjusted_timesteps
from src.utils.cdo_commands import cdo_remap
from src.utils.constants import COMPRESSED_ZARR_ENCODING, EXTENT_BC_MRRO
from src.utils.utils_load_data import get_raw_cordex_data_paths


def extract_year(path_cordex: Path) -> int:
    return int(path_cordex.stem.split("_")[-1][:4])


def get_adjusted_timesteps(
    path_data_adjusted_cordex: Path, climate_model_chain: str, climate_scenario: str
):
    df_climate_models_mapping = pd.read_csv(
        path_data_adjusted_cordex / "year_lists.csv"
    )
    df_climate_models_mapping["year_list"] = [
        ast.literal_eval(l) for l in df_climate_models_mapping["year_list"]
    ]

    return extract_adjusted_timesteps(
        df_climate_models_mapping, climate_model_chain, climate_scenario
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("variable", type=str, help="Variable to process.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
    os.environ["MKL_NUM_THREADS"] = "1"  # MKL
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

    dask.config.set({"logging.distributed": "error"})
    dask.config.set(num_workers=16, scheduler="processes")
    dask.config.set({"array.chunk-size": "400MiB"})

    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_cordex = Path(paths["path_data_cordex"])
    path_data_adjusted_cordex = Path(paths["path_data_adjusted_cordex"])
    path_data_personal_cordex = Path(paths["path_data_personal"]) / "cordex" / "EUR-11"
    path_data_temporary = Path(paths["path_data_temporary"])
    climate_grid_file = (
        "g010_MERIT_2.txt"
        if args.variable == "mrro"
        else "grid_EUR11_lonlat_griddes.txt"
    )

    time_start = time()
    cordex_var, cordex_var_hist = get_raw_cordex_data_paths(
        path_data_personal_cordex
        if args.variable.startswith("wind")
        else path_data_cordex,
        args.climate_model_chain,
        args.climate_scenario,
        args.variable,
        args.temp_res,
    )

    temp_output_dir = (
        path_data_temporary
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "temp"
        / args.variable
    )

    temp_output_dir.mkdir(parents=True, exist_ok=True)
    (temp_output_dir / "hist").mkdir(parents=True, exist_ok=True)
    (temp_output_dir / "future").mkdir(parents=True, exist_ok=True)
    print("Remapping historical data.")
    for sim_path in cordex_var_hist.glob("*.nc"):
        if (temp_output_dir / "hist" / sim_path.name).exists():
            continue
        print(sim_path.name)
        if extract_year(sim_path) > 2006:
            print("There's future data in historical directory. Skipping...")
            continue
        cdo_remap(
            str(sim_path),
            str(temp_output_dir / "hist" / sim_path.name),
            "conservative",
            path_data_processed / "climate_grids" / climate_grid_file,
            path_data_processed / "climate_grids" / "grid_cnrm_453x453.txt"
            if "CNRM-ALADIN63" in args.climate_model_chain
            else None,
        )
    print("Historical data remapped.")
    print("Remapping future data.")
    for sim_path in cordex_var.glob("*.nc"):
        if (temp_output_dir / "future" / sim_path.name).exists():
            continue
        print(sim_path.name)
        cdo_remap(
            str(sim_path),
            str(temp_output_dir / "future" / sim_path.name),
            "conservative",
            path_data_processed / "climate_grids" / climate_grid_file,
            path_data_processed / "climate_grids" / "grid_cnrm_453x453.txt"
            if "CNRM-ALADIN63" in args.climate_model_chain
            else None,
        )

    print(
        f"\tRemapping time for variable {args.variable}: {(time() - time_start) / 60:.2f} minutes."
    )

    # adjusted_timesteps = get_adjusted_timesteps(
    #     path_data_adjusted_cordex, args.climate_model_chain, args.climate_scenario
    # )

    # adjusted_timesteps = {
    #     "original_timesteps": pd.date_range("01-01-1991 00:00", "12-31-2100 00:00", freq="D"),
    #     "start_year": 1991,
    # }
    adjusted_timesteps = None

    chunks = None
    extent = EXTENT_BC_MRRO if args.variable == "mrro" else None

    climate_data = ClimateModelData(
        temp_output_dir / "future",
        temp_output_dir / "hist",
        adjusted_timesteps,
        subset_data_vars=[args.variable],
        chunks=chunks,
        extent=extent,
        load=False,
    )

    print("\tChecking time coordinates and duplicates...")
    if not is_time_sorted(climate_data.ds_data):
        print("Time coordinate is not sorted. Sorting now...")
        climate_data.ds_data = climate_data.ds_data.sortby("time")

    if has_duplicate_coords(climate_data.ds_data)["time"]:
        print("Dataset contains duplicate time coordinates. Removing duplicates...")
        climate_data.ds_data = climate_data.ds_data.drop_duplicates(dim="time")

    if has_identical_random_timesteps(climate_data.ds_data):
        print("Dataset contains identical random timesteps.")

    print("\tSaving processed dataset...")
    output_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "regridded"
        / args.temp_res
    )

    output_path.mkdir(parents=True, exist_ok=True)

    target_chunks = {"time": -1, "lat": 20, "lon": 20}

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in climate_data.ds_data.data_vars
    }

    if (output_path / f"{args.variable}.zarr").exists():
        shutil.rmtree(output_path / f"{args.variable}.zarr")

    climate_data.ds_data.chunk(target_chunks).to_zarr(
        output_path / f"{args.variable}.zarr", mode="w", encoding=zarr_encoding
    )

    shutil.rmtree(temp_output_dir, ignore_errors=True)

    print(
        f"\tProcessing time for variable {args.variable}: {(time() - time_start) / 60:.2f} minutes."
    )
