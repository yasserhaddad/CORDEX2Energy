import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.pv import (
    NORMALISED_ROOFTOP_PV_CONFIGURATION,
    compute_pv_from_config,
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

    # dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set({
        "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
        "distributed.worker.memory.spill": 0.8,   # Start spilling at 80%
        "distributed.worker.memory.pause": 0.9,   # Pause worker at 90%
        "array.chunk-size": "512MiB",
        "logging.distributed": "error",           # Less verbose logs
    })

    time_start = time.time()

    print("Loading data")
    extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}
    ds_tas_bc = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "tas.zarr",
        extent=extent,
        chunks="auto",
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
    # ds_orog = ds_orog.chunk({k: v for k, v in BC_SUBDAILY_CHUNKS.items() if k in ds_orog.dims})
    ds_orog = ds_orog.chunk({"lat": 20, "lon": 20})
    step = 20
    # Determine time slices based on ds_tas_bc time range
    time_index = pd.to_datetime(ds_tas_bc.ds_data.time.values)
    start_year = time_index.year.min()
    end_year = time_index.year.max()
    time_slices = [(start, min(start + step - 1, end_year)) for start in range(start_year, end_year + 1, step)]
    for start_year, end_year in time_slices:
        print(f"Processing data for {start_year}-{end_year}")
        output_path_zarr = (
            output_path_temp_solar
            / f"pv_{args.climate_model_chain}_{args.climate_scenario}_{start_year}_{end_year}.zarr"
        )

        output_path_agg = (
            output_path_temp_solar
            / f"pv_agg_{args.climate_model_chain}_{args.climate_scenario}_{start_year}_{end_year}.nc"
        )

        # if output_path_zarr.exists() and output_path_agg.exists():
        #     print(f"Output files already exist for {start_year}-{end_year}. Skipping.")
        #     continue
        if output_path_zarr.exists():
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
            chunks="auto",
        )

        ds_tas_bc = ClimateModelData(
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
            / "bias_corrected"
            / "3hr"
            / "tas.zarr",
            extent=extent,
            chunks="auto",
        )

        ds_sfcWind_bc = ClimateModelData(
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
            / "bias_corrected"
            / "3hr"
            / "sfcWind.zarr",
            extent=extent,
            chunks="auto",
        )
        time_slice = slice(str(start_year), str(end_year))
        ds_rsds_bc.ds_data = ds_rsds_bc.ds_data.sel(time=time_slice)
        ds_tas_bc.ds_data = ds_tas_bc.ds_data.sel(time=time_slice)
        ds_sfcWind_bc.ds_data = ds_sfcWind_bc.ds_data.sel(time=time_slice)

        print("Matching instantaneous to mean values")
        with ProgressBar(dt=60):
            ds_tas_bc.ds_data = match_instaneous_to_mean_values(
                ds_tas_bc.ds_data - 273.15, skip_leap_days=True, time_res="3h"
            ).persist()
            ds_tas_bc.ds_data["time"] = ds_tas_bc.ds_data["time"] + pd.Timedelta(
                hours=1, minutes=30
            )

            ds_sfcWind_bc.ds_data = match_instaneous_to_mean_values(
                ds_sfcWind_bc.ds_data, skip_leap_days=True, time_res="3h"
            ).persist()
            ds_sfcWind_bc.ds_data["time"] = ds_sfcWind_bc.ds_data["time"] + pd.Timedelta(
                hours=1, minutes=30
            )




        step_start = time.time()
        azimuth = 180
        direction = "S"
        tilt = 35

        pv_step_start = time.time()
        step_start = time.time()

        print(f"Computing PV data for {direction} direction and {tilt} degrees tilt")

        with ProgressBar(dt=60):
            print("\tComputing spatially-resolved PV...")
            ds_pv = (
                compute_pv_from_config(
                    ds_rsds_bc.ds_data.rsds.chunk({"time": -1, "lat": 20, "lon": 20}),
                    ds_tas_bc.ds_data.tas.chunk({"time": -1, "lat": 20, "lon": 20}),
                    ds_sfcWind_bc.ds_data.sfcWind.chunk({"time": -1, "lat": 20, "lon": 20}),
                    tilt=tilt,
                    azimuth=azimuth,
                    pv_configuration=NORMALISED_ROOFTOP_PV_CONFIGURATION,
                    altitude=ds_orog.orog,
                ).rename({"ac_power": f"tilt_{tilt}_azimuth_{direction}"})
                / NORMALISED_ROOFTOP_PV_CONFIGURATION["module_parameters"]["pdc0"]
            )

            zarr_encoding = {
                var: {
                    "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                    "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
                }
                for var in ds_pv.data_vars
            }
            ds_pv.chunk(BC_SUBDAILY_CHUNKS).to_zarr(output_path_zarr, mode="w", encoding=zarr_encoding)
            print(f"PV data saved to {output_path_zarr}")

            # print("\tComputing aggregated PV...")
            # step_start = time.time()
            # print("Aggregating PV data")
            # ds_pv_agg = aggregate_data_per_country(
            #     ds_pv,
            #     method="median_and_better",
            #     offshore=False,
            # ).compute()
            # print("PV data aggregation time: ", (time.time() - step_start) / 60, "minutes")

            # # Save the combined dataset to a file
            # step_start = time.time()
            # ds_pv_agg.to_netcdf(output_path_agg, mode="w")
            # print(f"PV aggregated data saved to {output_path_agg}")
            # print(f"PV data aggregation time: {(time.time() - step_start) / 60:.2f} minutes")

        del ds_pv, ds_rsds_bc, ds_tas_bc, ds_sfcWind_bc
        # del ds_pv_agg

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

        # ds_pv_agg = xr.open_mfdataset(
        #     output_path_temp_solar.glob(
        #         f"pv_agg_{args.climate_model_chain}_{args.climate_scenario}_*.nc"
        #     ),
        #     combine="by_coords",
        # )

        # output_path_agg = (
        #     output_path_solar
        #     / f"pv_agg_{args.climate_model_chain}_{args.climate_scenario}.nc"
        # )
        # ds_pv_agg.compute().to_netcdf(output_path_agg, mode="w")
        # del ds_pv_agg

    shutil.rmtree(output_path_temp_solar)

    print("Total time: ", (time.time() - time_start) / 60, "minutes")
