import sys

sys.path.append("..")

import time
import json
import argparse
import xarray as xr
from pathlib import Path

from src.convert_climate_vars import (
    convert_wind_components_to_speed,
)
from src.utils_load_data import (
    get_raw_cordex_data_paths,
    get_components_climate_model_chain,
)


def convert_cordex_wind_components_to_speed(
    paths: dict,
    climate_model_chain: str,
    climate_scenario: str,
    height: int,
    output_dir: Path,
):
    institute, _, gcm, _ = get_components_climate_model_chain(climate_model_chain)

    path_data_personal = Path(paths["path_data_personal"])
    path_data_personal_cordex = path_data_personal / "cordex" / "EUR-11"

    # Load the data paths
    u_path = sorted(
        list(
            (path_data_personal_cordex / institute / gcm / climate_scenario).rglob(
                f"*/ua{height}m"
            )
        )[0].glob("*/*.nc")
    )
    v_path = sorted(
        list(
            (path_data_personal_cordex / institute / gcm / climate_scenario).rglob(
                f"*/va{height}m"
            )
        )[0].glob("*/*.nc")
    )

    print("Calculating wind speed...")
    # Iterate over each year of data
    for u_file, v_file in zip(u_path, v_path):
        t_start = time.time()
        print(f"\tProcessing {u_file.stem.split('_')[-1]}")
        # Load the data for the current year
        # chunks = {"time": 2920, "y": 40, "x": 40}
        ds_u = xr.open_dataset(u_file).sortby("time").compute()
        ds_v = xr.open_dataset(v_file).sortby("time").compute()

        # Calculate wind speed
        ds_wind = ds_u.copy(
            {
                f"ua{height}m": convert_wind_components_to_speed(
                    ds_u[f"ua{height}m"], ds_v[f"va{height}m"]
                )
            }
        ).rename({f"ua{height}m": f"wind{height}m"})
        ds_wind[f"wind{height}m"].attrs.update({
            "long_name": "Wind speed",
            "standard_name": "wind_speed",
            "units": "degrees_east"
        })

        # Save the wind speed data
        filename = u_file.name.replace(f"ua{height}m", f"wind{height}m")
        ds_wind.to_netcdf(output_dir / filename)
        print(f"\tProcessing time: {(time.time() - t_start) / 60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("height", type=int, help="Height to process.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()
    with open("../paths.json") as f:
        paths = json.load(f)

    cordex_wind, cordex_wind_hist = get_raw_cordex_data_paths(
        Path(paths["path_data_personal"]) / "cordex" / "EUR-11",
        args.climate_model_chain,
        args.climate_scenario,
        f"wind{args.height}m",
        args.temp_res,
    )
    cordex_wind.mkdir(parents=True, exist_ok=True)
    cordex_wind_hist.mkdir(parents=True, exist_ok=True)

    for scenario in ["historical", args.climate_scenario]:
        convert_cordex_wind_components_to_speed(
            paths,
            args.climate_model_chain,
            scenario,
            args.height,
            cordex_wind if scenario != "historical" else cordex_wind_hist,
        )
