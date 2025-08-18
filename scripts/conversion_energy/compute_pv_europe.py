import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.pv import (
    NORMALISED_ROOFTOP_PV_CONFIGURATION,
    compute_pv_for_grid,
    PVOutput,
)
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import COMPRESSED_ZARR_ENCODING, BC_SUBDAILY_CHUNKS
from src.utils.utils_xarray import sel_extent, match_instaneous_to_mean_values
from src.conversion_energy.aggregate_data import aggregate_data_per_country

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    path_data_analysis = Path(paths["path_data_analysis"])
    path_data_processed = Path(paths["path_data_processed"])

    path_energy = path_data_processed / "energy"

    output_path_solar = (
        path_energy
        / "europe"
        / "pv"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    )
    output_path_solar.mkdir(parents=True, exist_ok=True)

    output_path_temp_solar = (
        path_data_temporary
        / "energy"
        / "europe"
        / "pv"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    )
    output_path_temp_solar.mkdir(parents=True, exist_ok=True)

    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})

    time_start = time.time()

    print("Loading data")
    extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 34}
    step = 15
    time_slices = [(start, min(start + step - 1, 2100)) for start in range(1991, 2101, step)]
    for start_year, end_year in time_slices:
        print(f"Processing data for {start_year}-{end_year}")
        time_slice = slice(str(start_year), str(end_year))

        output_path_zarr = (
            output_path_temp_solar
            / f"pv_{args.climate_model_chain}_{args.climate_scenario}_{start_year}_{end_year}.zarr"
        )

        output_path_agg = (
            output_path_temp_solar
            / f"pv_agg_{args.climate_model_chain}_{args.climate_scenario}_{start_year}_{end_year}.nc"
        )

        if output_path_zarr.exists() and output_path_agg.exists():
            print(f"Output files already exist for {start_year}-{end_year}. Skipping.")
            continue

        ds_rsds_bc = ClimateModelData(
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
            / "bias_corrected"
            / "3hr"
            / "rsds.zarr",
            extent=extent,
            chunks=BC_SUBDAILY_CHUNKS,
        )

        ds_tas_bc = ClimateModelData(
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
            / "bias_corrected"
            / "3hr"
            / "tas.zarr",
            extent=extent,
            chunks=BC_SUBDAILY_CHUNKS,
        )

        ds_sfcWind_bc = ClimateModelData(
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
            / "bias_corrected"
            / "3hr"
            / "sfcWind.zarr",
            extent=extent,
            chunks=BC_SUBDAILY_CHUNKS,
        )

        ds_orog = sel_extent(
            xr.open_dataset(
                sorted(
                    (
                        path_data_processed
                        / "cordex_processed"
                        / f"{args.climate_model_chain}_{args.climate_scenario}"
                        / "regridded"
                        / "fx"
                    ).glob("*.nc")
                )[0]
            ),
            extent,
        )

        step_start = time.time()
        ds_tas_bc.ds_data = ds_tas_bc.ds_data.sel(time=time_slice)
        ds_sfcWind_bc.ds_data = ds_sfcWind_bc.ds_data.sel(time=time_slice)
        ds_rsds_bc.ds_data = ds_rsds_bc.ds_data.sel(time=time_slice)
        # print("Data loading time: ", (time.time() - step_start) / 60, "minutes")

        print("Matching instantaneous to mean values")
        step_start = time.time()

        ds_tas_bc.ds_data = match_instaneous_to_mean_values(
            ds_tas_bc.ds_data - 273.15, skip_leap_days=True, time_res="3h"
        )
        ds_tas_bc.ds_data["time"] = ds_tas_bc.ds_data["time"] + pd.Timedelta(
            hours=1, minutes=30
        )

        ds_sfcWind_bc.ds_data = match_instaneous_to_mean_values(
            ds_sfcWind_bc.ds_data, skip_leap_days=True, time_res="3h"
        )
        ds_sfcWind_bc.ds_data["time"] = ds_sfcWind_bc.ds_data["time"] + pd.Timedelta(
            hours=1, minutes=30
        )
        with ProgressBar(dt=60):
            ds_rsds_bc.ds_data = ds_rsds_bc.ds_data.compute()
            ds_sfcWind_bc.ds_data = ds_sfcWind_bc.ds_data.compute()
            ds_tas_bc.ds_data = ds_tas_bc.ds_data.compute()

        print(f"Data matching time: {(time.time() - step_start) / 60:.2f} minutes")

        step_start = time.time()
        azimuth = 180
        direction = "S"
        tilt = 35

        pv_step_start = time.time()
        step_start = time.time()

        pv = PVOutput(pv_configuration=NORMALISED_ROOFTOP_PV_CONFIGURATION)
        print(f"Computing PV data for {direction} direction and {tilt} degrees tilt")
        ds_pv = (
            compute_pv_for_grid(
                ds_rsds_bc.ds_data.rsds,
                ds_tas_bc.ds_data.tas,
                ds_sfcWind_bc.ds_data.sfcWind,
                tilt=tilt,
                azimuth=azimuth,
                pv_configuration=NORMALISED_ROOFTOP_PV_CONFIGURATION,
                da_altitude=ds_orog.orog,
                num_workers=15,
            ).rename({"ac_power": f"tilt_{tilt}_azimuth_{direction}"})
            / NORMALISED_ROOFTOP_PV_CONFIGURATION["module_parameters"]["pdc0"]
        )
        print("PV data computation time: ", (time.time() - step_start) / 60, "minutes")

        step_start = time.time()
        print("Aggregating PV data")
        ds_pv_agg = aggregate_data_per_country(
            ds_pv,
            method="median_and_better",
            offshore=False,
        ).compute()
        print("PV data aggregation time: ", (time.time() - step_start) / 60, "minutes")

        # Save the combined dataset to a file
        step_start = time.time()
        ds_pv_agg.to_netcdf(output_path_agg, mode="w")

        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
            }
            for var in ds_pv.data_vars
        }
        ds_pv.to_zarr(output_path_zarr, mode="w", encoding=zarr_encoding)

        print(f"PV data saved to {output_path_zarr}")
        print(f"PV data saving time: {(time.time() - step_start) / 60:.2f} minutes")

        del ds_pv, ds_pv_agg, ds_rsds_bc, ds_tas_bc, ds_sfcWind_bc

    print("Concatenating datasets...")
    with ProgressBar(dt=60):
        ds_pv = xr.open_mfdataset(
            output_path_temp_solar.glob(
                f"pv_{args.climate_model_chain}_{args.climate_scenario}_*.zarr"
            ),
            engine="zarr",
            combine="by_coords",
            parallel=True,
        )
        output_path_zarr = (
            output_path_solar
            / f"pv_{args.climate_model_chain}_{args.climate_scenario}.zarr"
        )
        ds_pv.to_zarr(
            output_path_zarr,
            mode="w",
            encoding={
                var: {
                    "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                    "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
                }
                for var in ds_pv.data_vars
            },
        )

        del ds_pv

        ds_pv_agg = xr.open_mfdataset(
            output_path_temp_solar.glob(
                f"pv_agg_{args.climate_model_chain}_{args.climate_scenario}_*.nc"
            ),
            combine="by_coords",
        )

        output_path_agg = (
            output_path_solar
            / f"pv_agg_{args.climate_model_chain}_{args.climate_scenario}.nc"
        )
        ds_pv_agg.compute().to_netcdf(output_path_agg, mode="w")
        del ds_pv_agg

    shutil.rmtree(output_path_temp_solar)

    print("Total time: ", (time.time() - time_start) / 60, "minutes")
