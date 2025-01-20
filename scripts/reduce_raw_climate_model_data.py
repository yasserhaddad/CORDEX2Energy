import sys
from typing import Tuple

sys.path.append("..")

import ast
import json
from pathlib import Path
import pandas as pd
import argparse
from time import time
from src.encodings import REDUCED_NETCDF_ENCODINGS
from src.load_data import ClimateModelData
from src.timesteps_adjustments import extract_adjusted_timesteps


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


def get_raw_cordex_data_paths(
    path_data_cordex: Path,
    climate_model_chain: str,
    climate_scenario: str,
    variable: str,
    temp_res: str,
) -> Tuple[Path, Path]:
    climate_model_chain = "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    climate_scenario = "rcp85"

    institute = climate_model_chain.split("-")[0]
    rcm = climate_model_chain.split("_")[0]
    gcm = climate_model_chain.split("_")[1]
    ensemble = climate_model_chain.split("_")[-1]

    cordex_tas = (
        path_data_cordex
        / climate_scenario
        / temp_res
        / variable
        / institute
        / gcm
        / rcm
        / ensemble
    )
    cordex_tas_hist = (
        path_data_cordex
        / "historical"
        / temp_res
        / variable
        / institute
        / gcm
        / rcm
        / ensemble
    )

    return cordex_tas, cordex_tas_hist


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
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_cordex = Path(paths["path_data_cordex"])
    path_data_adjusted_cordex = Path(paths["path_data_adjusted_cordex"])

    extent = {"lat_min": 45, "lat_max": 48, "lon_min": 5.8, "lon_max": 10.8}

    time_start = time()
    cordex_tas, cordex_tas_hist = get_raw_cordex_data_paths(
        path_data_cordex,
        args.climate_model_chain,
        args.climate_scenario,
        args.variable,
        args.temp_res,
    )
    adjusted_timesteps = get_adjusted_timesteps(
        path_data_adjusted_cordex, args.climate_model_chain, args.climate_scenario
    )

    chunks = {"time": 2000, "lat": 100, "lon": 100}
    climate_data = ClimateModelData(
        cordex_tas,
        cordex_tas_hist,
        adjusted_timesteps,
        subset_data_vars=[args.variable],
        chunks=chunks,
        extent=extent,
        load=False,
    )

    output_path = (
        path_data_processed
        / "cordex-processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "raw"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    climate_data.ds_data.to_netcdf(
        output_path / f"{args.variable}_ch.nc",
        encoding={
            var: REDUCED_NETCDF_ENCODINGS for var in climate_data.ds_data.data_vars
        },
    )
    print(
        f"\tProcessing time for variable {args.variable}: {(time() - time_start)/60:.2f} minutes."
    )
