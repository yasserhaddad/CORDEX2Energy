import sys

sys.path.append("../..")

import json
import time
from pathlib import Path

import xarray as xr
import numpy as np
from src.cdo_commands import cdo_setgrid
from src.utils_xarray import check_contiguous_years, fill_missing_years_with_climatology

with open("../../paths_so4.json") as f:
    paths = json.load(f)

path_data_processed = Path(paths["path_data_processed"])
path_data_cerra_land = Path(paths["path_data_cerra_land"])

sro_cerra_path = sorted((path_data_cerra_land / "original" / "sro").glob("*.nc"))[:-1]
perc_cerra_path = sorted(
    (path_data_cerra_land / "original" / "perc").glob("*.nc")
)[:-1]
grid_path = path_data_processed / "climate_grids" / "grid_cerra_griddes.txt"

ro_cerra_path = path_data_processed / "cerra_processed" / "ro"
ro_cerra_path.mkdir(parents=True, exist_ok=True)

t_start = time.time()
print("Loading data")

ds_sro = xr.open_mfdataset(sro_cerra_path, combine="by_coords")
ds_perc = xr.open_mfdataset(perc_cerra_path, combine="by_coords")

ds_sro = ds_sro.resample(time='1D').sum().compute()
ds_perc = ds_perc.resample(time='1D').sum().compute()

print(f"\Time to load: {(time.time() - t_start) / 60:.2f} minutes.")
time_interm = time.time()

missing_years_sro = check_contiguous_years(ds_sro)
missing_years_perc = check_contiguous_years(ds_perc)

if len(missing_years_sro) > 0:
    print(f"Missing years in sro dataset: {sorted(missing_years_sro)}")
    ds_sro = fill_missing_years_with_climatology(ds_sro)
else:
    print("sro dataset is contiguous")

if len(missing_years_perc) > 0:
    print(f"Missing years in perc dataset: {sorted(missing_years_perc)}")
    ds_perc = fill_missing_years_with_climatology(ds_perc)
else:
    print("perc dataset is contiguous")

print(f"Time to check missing values: {(time.time() - time_interm) / 60:.2f} minutes.")
time_interm = time.time()

print("Calculating runoff")
ro = (ds_sro["sro"] + ds_perc["perc"]).compute()
ds_ro = xr.Dataset(
    {
        "ro": ro
    },
    coords=ds_sro.coords,
)
ds_ro["Lambert_Conformal"] = xr.open_dataset(sro_cerra_path[0])["Lambert_Conformal"]
ds_ro["ro"].attrs = {
    "long_name": "Runoff",
    "units": "kg m**-2",
    "grid_mapping": "Lambert_Conformal",
}

print(f"\Time to compute runoff: {(time.time() - time_interm) / 60:.2f} minutes.")

time_interm = time.time()
print("Saving runoff data")
for year in np.unique(ds_ro["time.year"].values):
    output_path = ro_cerra_path / f"ro_day_cerra-land_{year}.nc"
    temp_output_path = output_path.with_name(f"temp_{output_path.name}")
    ds_ro.sel(time=str(year)).to_netcdf(temp_output_path)

    cdo_setgrid(temp_output_path, grid_path, output_path)
    temp_output_path.unlink()

    print("\t", year, f"\tProcessing time: {(time.time() - time_interm) / 60:.2f} minutes.")
    time_interm = time.time()

print(f"Total time: {(time.time() - time_interm) / 60:.2f} minutes")
