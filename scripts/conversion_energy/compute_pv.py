import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import os

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar



from cordex2energy.conversion_energy.pv import (
    NORMALISED_ROOFTOP_PV_CONFIGURATION,
    compute_pv_from_config,
)
from cordex2energy.data_processing.load_data import ClimateModelData, ClimateReanalysisData
from cordex2energy.utils.constants import (
    COMPRESSED_ZARR_ENCODING,
    BC_SUBDAILY_CHUNKS,
    CORDEX_TO_CERRA_VARS,
    EXTENT_BC,
    EXTENT_CH,
)
from cordex2energy.utils.utils_xarray import (
    get_nan_mask,
    get_time_resolution,
    sel_extent,
    match_instaneous_to_mean_values,
)
from cordex2energy.conversion_energy.aggregate_data import aggregate_data_per_country

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("scope", type=str, help="Scope: europe or ch")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    if args.scope not in ["europe", "ch"]:
        raise ValueError("Scope must be 'europe' or 'ch'")

    is_cerra = args.climate_model_chain.lower() == "cerra"
    model_tag = "cerra" if is_cerra else f"{args.climate_model_chain}_{args.climate_scenario}"

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    path_data_analysis = Path(paths["path_data_analysis"])
    path_data_processed = Path(paths["path_data_processed"])

    path_energy = path_data_processed / "energy"

    output_path_solar = path_energy / args.scope / "pv" / model_tag
    output_path_solar.mkdir(parents=True, exist_ok=True)

    output_path_temp_solar = path_data_temporary / "energy" / args.scope / "pv" / model_tag
    output_path_temp_solar.mkdir(parents=True, exist_ok=True)

    # dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
            "distributed.worker.memory.spill": 0.8,  # Start spilling at 80%
            "distributed.worker.memory.pause": 0.9,  # Pause worker at 90%
            "array.chunk-size": "512MiB",
            "logging.distributed": "error",  # Less verbose logs
        }
    )

    time_start = time.time()

    print("Loading data")
    extent = EXTENT_BC if args.scope == "europe" else EXTENT_CH
    spatial_chunks = {"lat": 30, "lon": 30}

    if args.scope == "europe":
        if is_cerra:
            path_nan_mask = (
                path_data_processed / "cerra_processed" / "regridded"
                / f"{CORDEX_TO_CERRA_VARS['tas']}.zarr"
            )
            da_nan_mask = get_nan_mask(
                xr.open_zarr(path_nan_mask, chunks="auto")
                .sel(time="1991-01-01 00:00")
                .drop_vars("time")
                .squeeze(),
                var_name=CORDEX_TO_CERRA_VARS["tas"],
                climate_model_chain=args.climate_model_chain,
            ).chunk({"lat": spatial_chunks["lat"], "lon": spatial_chunks["lon"]})
        else:
            path_tas = (
                path_data_processed
                / "cordex_processed"
                / f"{args.climate_model_chain}_{args.climate_scenario}"
                / "regridded"
                / "3hr"
                / "tas.zarr"
            )
            da_nan_mask = get_nan_mask(
                xr.open_zarr(path_tas, chunks="auto").sel(time="1991-01-01 00:00").drop_vars("time").squeeze(),
                var_name="tas",
                climate_model_chain=args.climate_model_chain,
            ).drop_vars("height").chunk({"lat": spatial_chunks["lat"], "lon": spatial_chunks["lon"]})

    if is_cerra:
        cerra_reg = path_data_processed / "cerra_processed" / "regridded"
        ds_rsds_bc = ClimateReanalysisData(
            cerra_reg / f"{CORDEX_TO_CERRA_VARS['rsds']}.zarr",
            extent=extent,
            chunks="auto",
        )
        time_res_s = get_time_resolution(ds_rsds_bc.ds_data).total_seconds()
        ds_rsds_bc.ds_data = ds_rsds_bc.ds_data / time_res_s
        ds_rsds_bc.ds_data[CORDEX_TO_CERRA_VARS["rsds"]].attrs["units"] = "W m**-2"
        ds_rsds_bc.ds_data = ds_rsds_bc.ds_data.rename({CORDEX_TO_CERRA_VARS["rsds"]: "rsds"})
    else:
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

    if is_cerra:
        # CERRA has its own orography; fall back to first available CORDEX fx file if absent
        cerra_orog_path = path_data_processed / "cerra_processed" / "regridded" / "orog.zarr"
        if cerra_orog_path.exists():
            ds_orog = sel_extent(xr.open_zarr(cerra_orog_path), extent)
        else:
            first_model = sorted(
                (path_data_processed / "cordex_processed").glob("*/regridded/fx/*.nc")
            )[0]
            ds_orog = sel_extent(xr.open_dataset(first_model), extent)
    else:
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
    ds_orog = ds_orog.chunk(spatial_chunks)
    step = 30
    # Determine time slices based on ds_tas_bc time range
    time_index = pd.to_datetime(ds_rsds_bc.ds_data.time.values)
    start_year = time_index.year.min()
    end_year = time_index.year.max()
    time_slices = (
        [
            (start, min(start + step - 1, end_year))
            for start in range(start_year, end_year + 1, step)
        ]
        if args.scope == "europe"
        else [(start_year, end_year)]
    )
    for start_year, end_year in time_slices:
        print(f"Processing data for {start_year}-{end_year}")
        output_path_zarr = (
            output_path_temp_solar
            / f"pv_{model_tag}_{start_year}_{end_year}.zarr"
        )

        output_path_agg = (
            output_path_temp_solar
            / f"pv_agg_{model_tag}_{start_year}_{end_year}.nc"
        )

        if output_path_zarr.exists() and output_path_agg.exists():
            print(f"Output files already exist for {start_year}-{end_year}. Skipping.")
            continue

        if is_cerra:
            cerra_reg = path_data_processed / "cerra_processed" / "regridded"
            ds_rsds_bc = ClimateReanalysisData(
                cerra_reg / f"{CORDEX_TO_CERRA_VARS['rsds']}.zarr",
                extent=extent, chunks="auto",
            )
            time_res_s = get_time_resolution(ds_rsds_bc.ds_data).total_seconds()
            ds_rsds_bc.ds_data = ds_rsds_bc.ds_data / time_res_s
            ds_rsds_bc.ds_data[CORDEX_TO_CERRA_VARS["rsds"]].attrs["units"] = "W m**-2"
            ds_rsds_bc.ds_data = ds_rsds_bc.ds_data.rename({CORDEX_TO_CERRA_VARS["rsds"]: "rsds"})

            ds_tas_bc = ClimateReanalysisData(
                cerra_reg / f"{CORDEX_TO_CERRA_VARS['tas']}.zarr",
                extent=extent, chunks="auto",
            )
            ds_tas_bc.ds_data = ds_tas_bc.ds_data.rename({CORDEX_TO_CERRA_VARS["tas"]: "tas"})

            ds_sfcWind_bc = ClimateReanalysisData(
                cerra_reg / f"{CORDEX_TO_CERRA_VARS['sfcWind']}.zarr",
                extent=extent, chunks="auto",
            )
            ds_sfcWind_bc.ds_data = ds_sfcWind_bc.ds_data.rename(
                {CORDEX_TO_CERRA_VARS["sfcWind"]: "sfcWind"}
            )
        else:
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
        # Check if all datasets have the same lat, lon, and time
        for coord in ["lat", "lon"]:
            arrs = [
                ds_rsds_bc.ds_data[coord].values,
                ds_tas_bc.ds_data[coord].values,
                ds_sfcWind_bc.ds_data[coord].values,
            ]
            if not all((arrs[0] == arr).all() for arr in arrs):
                raise ValueError(f"Mismatch in {coord} coordinates among datasets.")
        print("Matching instantaneous to mean values")
        with ProgressBar(dt=60):
            ds_tas_bc.ds_data = match_instaneous_to_mean_values(
                ds_tas_bc.ds_data - 273.15, skip_leap_days=True, time_res="3h"
            ).persist()
            ds_tas_bc.ds_data["time"] = ds_tas_bc.ds_data["time"] + pd.Timedelta(
                hours=1, minutes=30
            )
            ds_tas_bc.ds_data = ds_tas_bc.ds_data.sel(time=ds_rsds_bc.ds_data.time)

            ds_sfcWind_bc.ds_data = match_instaneous_to_mean_values(
                ds_sfcWind_bc.ds_data, skip_leap_days=True, time_res="3h"
            ).persist()
            ds_sfcWind_bc.ds_data["time"] = ds_sfcWind_bc.ds_data[
                "time"
            ] + pd.Timedelta(hours=1, minutes=30)
            ds_sfcWind_bc.ds_data = ds_sfcWind_bc.ds_data.sel(
                time=ds_rsds_bc.ds_data.time
            )

        step_start = time.time()
        azimuths = (
            {"S": 180}
            if args.scope == "europe"
            else {"E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "N": 0}
        )
        tilts = [35] if args.scope == "europe" else [3, 8, 15, 25, 35, 45, 60, 80, 90]

        pv_step_start = time.time()
        step_start = time.time()

        with ProgressBar(dt=10):
            ds_pv = []
            ds_pv_agg = []
            for direction, azimuth in azimuths.items():
                for tilt in tilts:
                    print(
                        f"Computing PV data for {direction} direction and {tilt} degrees tilt"
                    )
                    ds_pv.append(
                        compute_pv_from_config(
                            ds_rsds_bc.ds_data.rsds.chunk(
                                {"time": -1, "lat": spatial_chunks["lat"], "lon": spatial_chunks["lon"]}
                            ),
                            ds_tas_bc.ds_data.tas.chunk(
                                {"time": -1, "lat": spatial_chunks["lat"], "lon": spatial_chunks["lon"]}
                            ),
                            ds_sfcWind_bc.ds_data.sfcWind.chunk(
                                {"time": -1, "lat": spatial_chunks["lat"], "lon": spatial_chunks["lon"]}
                            ),
                            tilt=tilt,
                            azimuth=azimuth,
                            pv_configuration=NORMALISED_ROOFTOP_PV_CONFIGURATION,
                            altitude=ds_orog.orog,
                        ).rename({"ac_power": f"tilt_{tilt}_azimuth_{direction}"})
                        / NORMALISED_ROOFTOP_PV_CONFIGURATION["module_parameters"][
                            "pdc0"
                        ]
                    )

            print("Saving data")
            ds_pv = xr.merge(ds_pv)
            if args.scope == "europe":
                ds_pv = ds_pv.where(~da_nan_mask)

            for var_name in ds_pv.data_vars:
                parts = var_name.split("_")
                tilt_val = int(parts[1])
                direction = parts[3]
                ds_pv[var_name].attrs.update({
                    "long_name": f"Normalised rooftop PV capacity factor — tilt {tilt_val}° azimuth {direction}",
                    "units": "fraction (0–1)",
                    "tilt_deg": tilt_val,
                    "azimuth_deg": azimuths[direction],
                    "azimuth_direction": direction,
                })

            zarr_encoding = {
                var: {
                    "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                    "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
                }
                for var in ds_pv.data_vars
            }
            ds_pv.chunk(BC_SUBDAILY_CHUNKS).to_zarr(
                output_path_zarr, mode="w", encoding=zarr_encoding
            )
            print(f"PV data saved to {output_path_zarr}")

            if args.scope == "europe":
                ds_pv = xr.open_zarr(output_path_zarr, chunks="auto")
                ds_pv_agg = aggregate_data_per_country(
                    ds_pv,
                    method="median_and_better",
                    offshore=False,
                )
                for var_name in ds_pv_agg.data_vars:
                    parts = var_name.split("_")
                    tilt_val = int(parts[1])
                    direction = parts[3]
                    ds_pv_agg[var_name].attrs.update({
                        "long_name": f"Country-aggregated rooftop PV capacity factor — tilt {tilt_val}° azimuth {direction}",
                        "units": "fraction (0–1)",
                        "tilt_deg": tilt_val,
                        "azimuth_deg": azimuths[direction],
                        "azimuth_direction": direction,
                        "aggregation": "median_and_better",
                        "domain": "onshore",
                    })
                ds_pv_agg.to_netcdf(output_path_agg, mode="w")
                print(f"PV aggregated data saved to {output_path_agg}")

            del ds_pv, ds_rsds_bc, ds_tas_bc, ds_sfcWind_bc, ds_pv_agg


    print("Concatenating datasets...")
    with ProgressBar(dt=60):
        ds_pv = xr.open_mfdataset(
            output_path_temp_solar.glob(f"pv_{model_tag}_*.zarr"),
            engine="zarr",
            combine="by_coords",
            parallel=True,
        )
        output_path_zarr = output_path_solar / f"pv_{model_tag}.zarr"
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

        if args.scope == "europe":
            ds_pv_agg = xr.open_mfdataset(
                output_path_temp_solar.glob(f"pv_agg_{model_tag}_*.nc"),
                combine="by_coords",
            )

            output_path_agg = output_path_solar / f"pv_agg_{model_tag}.nc"
            ds_pv_agg.compute().to_netcdf(output_path_agg, mode="w")
            del ds_pv_agg

    print("Removing temporary files...")
    shutil.rmtree(output_path_temp_solar.absolute())

    print("Total time: ", (time.time() - time_start) / 60, "minutes")
