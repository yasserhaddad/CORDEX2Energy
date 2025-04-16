import sys

sys.path.append("../..")

import json
import time
from pathlib import Path

import xarray as xr

start_time = time.time()
paths = json.load(open("../../paths_so4.json"))

path_data_processed = Path(paths["path_data_processed"])

extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}
extent_ch = {"lat_min": 45.8, "lat_max": 47.9, "lon_min": 5.9, "lon_max": 10.5}

path_era5_land = Path("/net/atmos/data/ERA5-Land/recent/0.25deg_lat-lon_1h/original")
file_paths = []
for year in range(1985, 2022):
    file_paths.extend(path_era5_land.glob(f"era5-land_recent.ssrd.*{year}.nc"))

ds_ssrd_era5_subset = xr.open_mfdataset(file_paths)
ds_ssrd_era5_subset = ds_ssrd_era5_subset.sel(
    latitude=slice(extent_ch["lat_max"], extent_ch["lat_min"]),
    longitude=slice(extent_ch["lon_min"], extent_ch["lon_max"]),
)

ds_ssrd_era5_subset.to_netcdf(
    path_data_processed / "era5_processed" / "era5_ssrd_1h_1985-2021_ch.nc"
)

print(f"Processing time: {(time.time() - start_time)/60:.2f} minutes")