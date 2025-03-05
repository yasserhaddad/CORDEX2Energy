import sys

sys.path.append("..")

import time
import json
import xarray as xr
from pathlib import Path

from src.convert_climate_vars import (
    calculate_specific_humidity,
)
from src.cdo_commands import cdo_setgrid

with open("../paths_so4.json") as f:
    paths = json.load(f)

path_data_processed = Path(paths["path_data_processed"])
path_data_cerra = Path(paths["path_data_cerra"])

output_path = path_data_processed / "cerra_processed" / "q"
# output_path.mkdir(parents=True, exist_ok=True)
# Load the data paths
tas_cerra_path = sorted((path_data_cerra / "original" / "2t").glob("*.nc"))[1:]
relative_humidity_cerra_path = sorted(
    (path_data_cerra / "original" / "2r").glob("*.nc")
)
pressure_cerra_path = sorted((path_data_cerra / "original" / "sp").glob("*.nc"))
grid_path = path_data_processed / "climate_grids" / "grid_cerra_griddes.txt"

t_start = time.time()
print("Calculating specific humidity")
# Iterate over each year of data
for tas_file, rh_file, pressure_file in zip(
    tas_cerra_path, relative_humidity_cerra_path, pressure_cerra_path
):
    year = tas_file.stem.split("_")[-1]
    print(year)
    # Load the data for the current year
    chunks = {"time": -1, "y": 40, "x": 40}
    ds_tas = xr.open_dataset(tas_file).sortby("time")
    ds_rh = xr.open_dataset(rh_file).sortby("time")
    ds_pressure = xr.open_dataset(pressure_file).sortby("time")

    if ds_tas["2t"].attrs["units"] == "K":
        ds_tas["2t"] -= 273.15

    if ds_pressure["sp"].attrs["units"] == "Pa":
        ds_pressure["sp"] /= 100

    # Calculate specific humidity
    specific_humidity = calculate_specific_humidity(
        ds_tas["2t"], ds_rh["2r"], ds_pressure["sp"]
    ).compute()

    # Create a new dataset for specific humidity
    ds_sh = xr.Dataset(
        {
            "q": specific_humidity
        },
        coords=ds_tas.coords,
    )
    ds_sh["q"].attrs = {
        "long_name": "Specific humidity",
        "units": "kg kg-1",
        "standard_name": "specific_humidity",
    }

    # Save the specific humidity data
    output_path = (
        path_data_processed / "cerra_processed" / "q" / f"q_3hr_cerra_{year}.nc"
    )
    temp_output_path = output_path.with_name(f"temp_{output_path.name}")
    ds_sh.to_netcdf(temp_output_path)

    # Set the grid of the output file
    cdo_setgrid(temp_output_path, grid_path, output_path)
    temp_output_path.unlink()

    print(f"\tProcessing time: {(time.time() - t_start) / 60:.2f} minutes.")
