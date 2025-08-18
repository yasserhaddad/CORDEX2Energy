import sys

sys.path.append("../..")

import json
import argparse
from pathlib import Path
from src.utils.cdo_commands import cdo_remap
from src.utils.utils_load_data import get_orog_cordex_data_path

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
    path_data_cordex = Path(paths["path_data_cordex"])
    path_data_adjusted_cordex = Path(paths["path_data_adjusted_cordex"])
    path_data_personal_cordex = Path(paths["path_data_personal"]) / "cordex" / "EUR-11"

    orog_dir = get_orog_cordex_data_path(
        path_data_cordex,
        args.climate_model_chain,
        args.climate_scenario,
    )
    orog_path = sorted(orog_dir.glob("*.nc"))[0]
    output_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "regridded"
        / "fx"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    climate_grid_file = "grid_EUR11_lonlat_griddes.txt"

    (output_dir / orog_path.name).unlink(missing_ok=True)
    cdo_remap(
        str(orog_path),
        str(output_dir / orog_path.name),
        "conservative",
        path_data_processed / "climate_grids" / climate_grid_file,
        path_data_processed / "climate_grids" / "grid_cnrm_453x453.txt"
        if "CNRM-ALADIN63" in args.climate_model_chain
        else None
    )