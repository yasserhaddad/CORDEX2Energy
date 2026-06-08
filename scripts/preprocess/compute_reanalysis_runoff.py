import json
import os
import re
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.utils.cdo_commands import cdo_setgrid
from cordex2energy.utils.utils_xarray import (
    check_contiguous_years,
    fill_missing_years_with_climatology,
)

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

dask.config.set({"logging.distributed": "error"})
# dask.config.set(num_workers=16, threads_per_worker=16, scheduler="processes")
dask.config.set(num_workers=16, threads_per_worker=16)
dask.config.set({"array.chunk-size": "400MiB"})

dask.config.set(
    {"distributed.worker.memory.target": 0.75}
)  # Spill to disk at 75% usage
dask.config.set({"distributed.worker.memory.spill": 0.8})  # Start spilling at 80%
dask.config.set({"distributed.worker.memory.pause": 0.9})  # Pause worker at 90%

with open("../../paths_so4.json") as f:
    paths = json.load(f)

path_data_processed = Path(paths["path_data_processed"])
path_data_cerra_land = Path(paths["path_data_cerra_land"])


# Get all files and filter to only include years <= 2020
def filter_files_by_year(files, max_year=2020):
    filtered = []
    for f in files:
        match = re.search(r"(\d{4})", f.name)
        if match:
            year = int(match.group(1))
            if year <= max_year:
                filtered.append(f)
    return sorted(filtered)


sro_files = filter_files_by_year(
    list((path_data_cerra_land / "original" / "sro").glob("*.nc"))
)
perc_files = filter_files_by_year(
    list((path_data_cerra_land / "original" / "perc").glob("*.nc"))
)
print(
    f"Found {len(sro_files)} sro files and {len(perc_files)} perc files up to 2020 in the original directory."
)

sro_files += list((path_data_processed / "cerra_processed" / "sro").glob("*.nc"))
perc_files += list((path_data_processed / "cerra_processed" / "perc").glob("*.nc"))

print(
    f"Total {len(sro_files)} sro files and {len(perc_files)} perc files after including processed data."
)


grid_path = path_data_processed / "climate_grids" / "grid_cerra_griddes.txt"

ro_cerra_path = path_data_processed / "cerra_processed" / "ro"
ro_cerra_path.mkdir(parents=True, exist_ok=True)

t_start = time.time()
print("Loading data lazily (keeping as dask arrays)")

# Open datasets with chunking - keep lazy, don't compute yet
ds_sro = xr.open_mfdataset(
    sro_files,
    combine="by_coords",
    parallel=True,
)
ds_perc = xr.open_mfdataset(
    perc_files,
    combine="by_coords",
    parallel=True,
)

print("Resampling to daily data (lazy operation)")
ds_sro_daily = ds_sro.resample(time="1D").sum()
ds_perc_daily = ds_perc.resample(time="1D").sum()

print(f"Time to set up lazy operations: {(time.time() - t_start) / 60:.2f} minutes.")

# Check for missing years (this requires some computation but minimal)
time_interm = time.time()
missing_years_sro = check_contiguous_years(ds_sro_daily)
missing_years_perc = check_contiguous_years(ds_perc_daily)

if len(missing_years_sro) > 0:
    print(f"Missing years in sro dataset: {sorted(missing_years_sro)}")
    ds_sro_daily = fill_missing_years_with_climatology(ds_sro_daily)
else:
    print("sro dataset is contiguous")

if len(missing_years_perc) > 0:
    print(f"Missing years in perc dataset: {sorted(missing_years_perc)}")
    ds_perc_daily = fill_missing_years_with_climatology(ds_perc_daily)
else:
    print("perc dataset is contiguous")

print(f"Time to check missing values: {(time.time() - time_interm) / 60:.2f} minutes.")

# Get Lambert_Conformal metadata
lambert_conformal = xr.open_dataset(list(sro_files)[0])["Lambert_Conformal"]

# Get all years to process
print("Getting years to process...")
years = np.unique(ds_sro_daily["time.year"].values)
print(f"Processing {len(years)} years")

time_interm = time.time()
print("Processing and saving runoff data year by year")

for year in years:
    print(f"Processing year {year}...")
    year_start = time.time()

    output_path = ro_cerra_path / f"ro_day_cerra-land_{year}.nc"
    if output_path.exists():
        # output_path.unlink()
        print(f"\tSkipping {year} - output file already exists")
        continue

    # Select year and compute only this year's data
    sro_year = ds_sro_daily.sel(time=str(year))["sro"]
    perc_year = ds_perc_daily.sel(time=str(year))["perc"]

    # Calculate runoff for this year only and compute
    print(f"\tComputing runoff for {year}...")
    with ProgressBar(dt=1):
        ro_year = (sro_year + perc_year).compute()

    # Create dataset for this year
    ds_ro_year = xr.Dataset(
        {"ro": ro_year},
        coords=sro_year.coords,
    )
    ds_ro_year["Lambert_Conformal"] = lambert_conformal
    ds_ro_year["ro"].attrs = {
        "long_name": "Runoff",
        "units": "kg m**-2",
        "grid_mapping": "Lambert_Conformal",
    }

    # Save to temporary file
    temp_output_path = output_path.with_name(f"temp_{output_path.name}")
    ds_ro_year.to_netcdf(temp_output_path)

    # Apply grid correction
    cdo_setgrid(temp_output_path, grid_path, output_path)
    temp_output_path.unlink()

    print(f"\t{year} completed in {(time.time() - year_start) / 60:.2f} minutes.")

print(f"Total time: {(time.time() - t_start) / 60:.2f} minutes")
