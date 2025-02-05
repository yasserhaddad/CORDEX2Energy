import sys

sys.path.append("..")

import time
import json
import xarray as xr
from pathlib import Path

from src.convert_climate_vars import (
    convert_wind_components_to_speed,
)
from src.utils_load_data import get_raw_cerra_data_paths, get_components_climate_model_chain

def convert_cordex_wind_components_to_speed(paths: dict, climate_model_chain: str, climate_scenario: str, temp_res: str):
    institute, rcm, gcm, ensemble = get_components_climate_model_chain(climate_model_chain)

    path_data_personal = Path(paths["path_data_personal"])
    path_data_personal_cordex = path_data_personal / "cordex"

    get_raw_cerra_data_paths(Path(paths["path_data_cordex"]),
                            climate_model_chain,
                            climate_scenario,
                            "variable",
                            temp_res)

    output_path = path_data_personal / "cordex" / "q"
    output_path.mkdir(parents=True, exist_ok=True)
    # Load the data paths
    u_path = sorted((path_data_cerra / "original" / "u").glob("*.nc"))
    v_path = sorted((path_data_cerra / "original" / "v").glob("*.nc"))

    t_start = time.time()
    print("Calculating wind speed...")
    # Iterate over each year of data
    for u_file, v_file in zip(u_path, v_path):
        year = u_file.stem.split("_")[-1]
        print(year)
        # Load the data for the current year
        chunks = {"time": 2920, "lat": 40, "lon": 40}
        ds_tas = xr.open_dataset(tas_file, chunks=chunks).sortby("time")
        ds_rh = xr.open_dataset(rh_file, chunks=chunks).sortby("time")
        ds_pressure = xr.open_dataset(pressure_file, chunks=chunks).sortby("time")

        # Calculate specific humidity
        ds_sh = calculate_specific_humidity(ds_tas["2t"], ds_rh["2r"], ds_pressure["sp"])

        # Save the specific humidity data
        output_path = (
            path_data_processed / "cerra_processed" / "q" / f"q_3hr_cerra_{year}.nc"
        )
        ds_sh.to_netcdf(output_path)
        print(f"\tProcessing time: {(time() - t_start) / 60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("height", type=str, help="Variable to process.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()
    with open("../paths.json") as f:
        paths = json.load(f)

    # /CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/CNRM-ALADIN63/v2/3hr/va150m