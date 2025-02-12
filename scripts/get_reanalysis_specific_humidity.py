import sys

sys.path.append("..")

import time
import json
import xarray as xr
from pathlib import Path

from src.convert_climate_vars import (
    calculate_specific_humidity,
)

with open("../paths.json") as f:
    paths = json.load(f)

path_data_processed = Path(paths["path_data_processed"])
path_data_cerra = Path(paths["path_data_cerra"])

output_path = path_data_processed / "cerra_processed" / "q"
output_path.mkdir(parents=True, exist_ok=True)
# Load the data paths
tas_cerra_path = sorted((path_data_cerra / "original" / "2t").glob("*.nc"))[1:]
relative_humidity_cerra_path = sorted(
    (path_data_cerra / "original" / "2r").glob("*.nc")
)
pressure_cerra_path = sorted((path_data_cerra / "original" / "sp").glob("*.nc"))

t_start = time.time()
print("Calculating specific humidity")
# Iterate over each year of data
for tas_file, rh_file, pressure_file in zip(
    tas_cerra_path, relative_humidity_cerra_path, pressure_cerra_path
):
    year = tas_file.stem.split("_")[-1]
    print(year)
    # Load the data for the current year
    chunks = {"time": 2920, "y": 40, "x": 40}
    ds_tas = xr.open_dataset(tas_file).chunk(chunks=chunks).sortby("time")
    ds_rh = xr.open_dataset(rh_file).chunk(chunks=chunks).sortby("time")
    ds_pressure = xr.open_dataset(pressure_file).chunk(chunks=chunks).sortby("time")

    # Calculate specific humidity
    ds_sh = ds_tas.copy({"2t":
        calculate_specific_humidity(ds_tas["2t"], ds_rh["2r"], ds_pressure["sp"])
    }).rename({"2t": "q"})

    # Save the specific humidity data
    output_path = (
        path_data_processed / "cerra_processed" / "q" / f"q_3hr_cerra_{year}.nc"
    )
    ds_sh.to_netcdf(output_path)
    print(f"\tProcessing time: {(time.time() - t_start) / 60:.2f} minutes.")
