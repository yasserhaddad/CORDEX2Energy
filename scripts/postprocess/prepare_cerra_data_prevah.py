import pandas as pd
import xarray as xr
import sys

import argparse
import json
from dask.diagnostics import ProgressBar
from pathlib import Path


from cordex2energy.data_processing.load_data import ClimateModelData
from cordex2energy.utils.constants import EXTENT_PREVAH
from cordex2energy.utils.utils_xarray import get_time_resolution
from scripts.postprocess.aggregate_bias_corrected_data import aggregate_to_daily

VARIABLES_REF_MAPPING = {
    "2t": "tas",
    "2r": "hurs",
    "q": "huss",
    "10si": "sfcWind",
    "ssrd": "rsds",
    "tp": "pr",
    "tcc": "clt",
    "msl": "psl",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CERRA data for PREVAH.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )

    args = parser.parse_args()

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])
    cerra_data_dir = path_data_processed / "cerra_processed"

    output_path_dir = cerra_data_dir / "prevah" / "input_data"
    output_path_dir.mkdir(parents=True, exist_ok=True)

    for var_cerra, var_prevah in VARIABLES_REF_MAPPING.items():
        print(f"Processing variable: {var_cerra} -> {var_prevah}")
        if var_cerra == "tp":
            daily_file = True
        else:
            daily_file = False
        cerra_file = cerra_data_dir / "regridded" / f"{var_cerra}.zarr"
        ds = ClimateModelData(
            cerra_file,
            extent=EXTENT_PREVAH,
            drop_leap_days=True,
            check_calendar=False,
            chunks="auto",
        ).ds_data

        if "height" in ds.coords:
            ds = ds.isel(height=0).drop_vars("height")
            print("Reference data height selected.")

        if var_cerra == "tp":
            if ds[var_cerra].units == "kg m**-2":
                ds[var_cerra].attrs["units"] = "mm d-1"
                print("Reference data units converted to mm d-1.")

        if var_cerra == "ssrd" or var_cerra == "ssrd_daily":
            if ds[var_cerra].units == "J m**-2":
                time_res = get_time_resolution(ds).total_seconds()
                ds = ds / time_res
                ds[var_cerra].attrs["units"] = "W m**-2"
            print("Reference data units converted to W m-2.")
            ds = ds.assign_coords(time=ds.time - pd.Timedelta(hours=1, minutes=30))

            # Remove leap days from the reanalysis dataset
            ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

        ds = ds.sel(time=slice("1991", "2020"))
        for var in ds.data_vars:
            ds[var] = ds[var].astype("float32")
        output_file = output_path_dir / f"{var_prevah}.nc"

        with ProgressBar(dt=1):
            if var_cerra == "2t":
                print("Computing tas_max and tas_min")
                ds_min = aggregate_to_daily(
                    ds, drop_leap_days=True, function="min"
                ).rename({var_cerra: var_prevah + "_min"})
                ds_min = ds_min.assign_coords(time=ds_min.time.dt.floor("D"))
                ds_max = aggregate_to_daily(
                    ds, drop_leap_days=True, function="max"
                ).rename({var_cerra: var_prevah + "_max"})
                ds_max = ds_max.assign_coords(time=ds_max.time.dt.floor("D"))
                print(
                    f"Saving to {output_path_dir / f'{var_prevah}_min.nc'} and {output_path_dir / f'{var_prevah}_max.nc'}"
                )
                if (output_path_dir / f"{var_prevah}_min.nc").exists():
                    (output_path_dir / f"{var_prevah}_min.nc").unlink()
                if (output_path_dir / f"{var_prevah}_max.nc").exists():
                    (output_path_dir / f"{var_prevah}_max.nc").unlink()
                ds_min.to_netcdf(output_path_dir / f"{var_prevah}_min.nc", mode="w")
                ds_max.to_netcdf(output_path_dir / f"{var_prevah}_max.nc", mode="w")
            if not daily_file:
                ds = aggregate_to_daily(ds, drop_leap_days=True, function="mean")
            print(f"Saving to {output_file}")
            ds = ds.assign_coords(time=ds.time.dt.floor("D")).rename(
                {var_cerra: var_prevah}
            )
            if output_file.exists():
                output_file.unlink()
            ds.to_netcdf(output_file, mode="w")
