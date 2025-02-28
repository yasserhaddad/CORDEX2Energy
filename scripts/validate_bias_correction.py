import sys

from dask.distributed import LocalCluster, Client

sys.path.append("..")

import argparse
import json
import time
from pathlib import Path

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from src.bias_correction import BiasCorrectionValidation
from src.load_data import ClimateModelData, ClimateReanalysisData
from src.utils_xarray import get_time_resolution, get_valid_hours, match_time_steps

dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})
dask.config.set({"distributed.worker.resources.CPU": 1})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("variable", type=str, help="Variable to process.")
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")
    parser.add_argument(
        "variable_ref",
        type=str,
        help="Variable to use as reference for bias correction.",
    )
    parser.add_argument(
        "--height_ref",
        type=int,
        default=None,
        help="Height to select from the reference dataset.",
    )
    parser.add_argument(
        "--nr_processes",
        type=int,
        default=8,
        help="Height to select from the reference dataset.",
    )
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="auto",
        help="Memory limit for dask workers.",
    )

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])

    time_start = time.time()

    if args.nr_processes > 1:
        # dask.config.set(num_workers=args.nr_processes,
        #                 scheduler="processes")
        # print(dask.config.config)
        # dask.config.set({
        #     "scheduler.work-stealing": True,
        #     "array.chunk-size": "500MiB",
        #     "distributed.worker.memory.target": args.memory_limit,
        #     "distributed.worker.threads": 1,
        # })
        dask.config.set({
            "array.chunk-size": "500MiB",
        })

        cluster = LocalCluster(
            processes=True,
            n_workers=args.nr_processes,
            threads_per_worker=1,
            memory_limit=args.memory_limit,
            resources={'CPU': 1}
        )
        client=Client(cluster)
        print(cluster.dashboard_link)

    # Define input and output paths
    regridded_cordex_var_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "regridded"
        / args.temp_res
    )
    regridded_cerra_var_path = path_data_processed / "cerra_processed" / "regridded"

    bias_corrected_cordex_var_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / args.temp_res
    )

    output_var_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "validation"
        / args.temp_res
        / args.variable
    )
    output_var_path.mkdir(parents=True, exist_ok=True)

    # Load data
    bias_corrected_chunks = {"time": 2920, "lat": 128, "lon": 128}
    regridded_chunks = {"time": -1, "lat": 20, "lon": 20}
    extent = {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}

    climate_model_data = ClimateModelData(
        regridded_cordex_var_path / f"{args.variable}.zarr",
        chunks=regridded_chunks,
        extent=extent,
        load=False,
    )
    climate_model_data_bc = ClimateModelData(
        bias_corrected_cordex_var_path / f"{args.variable}.zarr",
        chunks=bias_corrected_chunks,
        extent=extent,
        load=False,
    )
    reanalysis_data = ClimateReanalysisData(
        regridded_cerra_var_path / f"{args.variable_ref}.zarr",
        chunks=regridded_chunks,
        extent=extent,
        load=False,
    )

    # Process data
    first_time_step = pd.to_datetime(climate_model_data.ds_data.time.values[0])
    time_res = get_time_resolution(reanalysis_data.ds_data)
    reanalysis_data.ds_data = match_time_steps(
            climate_model_data.ds_data,
            reanalysis_data.ds_data,
            time_res,
            skip_leap_days=climate_model_data.drop_leap_days,
        )

    if args.height_ref is not None:
        reanalysis_data.ds_data = reanalysis_data.ds_data.sel(
            height=args.height_ref
        ).drop_vars("height")
        print("Reference data height selected.")
        if (
            "height" in climate_model_data.ds_data.coords
            and "height" not in climate_model_data.ds_data[args.variable].dims
        ):
            climate_model_data.ds_data = climate_model_data.ds_data.drop_vars("height")
            print("Climate data height dropped.")

    if args.variable == "pr":
        if climate_model_data.ds_data[args.variable].units == "kg m-2 s-1":
            climate_model_data.ds_data = climate_model_data.ds_data * 86400
            climate_model_data.ds_data[args.variable].attrs["units"] = "mm d-1"
            print("Climate data units converted to mm d-1.")
        if reanalysis_data.ds_data[args.variable_ref].units == "kg m**-2":
            reanalysis_data.ds_data[args.variable_ref].attrs["units"] = "mm d-1"
            print("Reference data units converted to mm d-1.")

    if args.variable == "rsds":
        if reanalysis_data.ds_data[args.variable_ref].units == "J m**-2":
            reanalysis_data.ds_data = reanalysis_data.ds_data / time_res.total_seconds()
            reanalysis_data.ds_data[args.variable_ref].attrs["units"] = "W m**-2"
            print("Reference data units converted to W m-2.")

    # Persist data to avoid recomputation
    with ProgressBar(dt=60):
        climate_model_data.ds_data = climate_model_data.ds_data.persist()
        climate_model_data_bc.ds_data = climate_model_data_bc.ds_data.persist()
        reanalysis_data.ds_data = reanalysis_data.ds_data.persist()

    # Bias correction validation
    validation = BiasCorrectionValidation(reference_period=("1991", "2020"), future_period=("2021", "2100"))

    # Compute means
    print("Computing overall yearly mean values...")
    with ProgressBar(dt=60):
        climate_model_data_mean, climate_model_data_bc_mean, reanalysis_data_mean = validation.compute_mean(
            climate_model_data.ds_data[args.variable],
            climate_model_data_bc.ds_data[args.variable],
            reanalysis_data.ds_data[args.variable_ref],
            hour=None,
            group="year",
            period=None,
            spatial_mean=False
        )
        climate_model_data_mean = climate_model_data_mean.compute()
        climate_model_data_bc_mean = climate_model_data_bc_mean.compute()
        reanalysis_data_mean = reanalysis_data_mean.compute()
        xr.Dataset({
            "climate_model_data_mean": climate_model_data_mean,
            "climate_model_data_bc_mean": climate_model_data_bc_mean,
            "reanalysis_data_mean": reanalysis_data_mean
        }).to_netcdf(output_var_path / "yearly_temporal_mean.nc")

        climate_model_data_mean = climate_model_data_mean.mean(
            dim=["lat", "lon"], skipna=True
        ).compute()
        climate_model_data_bc_mean = climate_model_data_bc_mean.mean(
            dim=["lat", "lon"], skipna=True
        ).compute()
        reanalysis_data_mean = reanalysis_data_mean.mean(
            dim=["lat", "lon"], skipna=True
        ).compute()
        xr.Dataset({
            "climate_model_data_mean": climate_model_data_mean,
            "climate_model_data_bc_mean": climate_model_data_bc_mean,
            "reanalysis_data_mean": reanalysis_data_mean
        }).to_netcdf(output_var_path / "yearly_spatial_mean.nc")
        print(f"\tTime to compute overall yearly spatial and temporal mean: {(time.time() - time_start)/60:.2f} minutes.")

    for hour in get_valid_hours(time_res, first_time_step.hour):
        print(f"Computing yearly mean values for hour {hour}...")
        with ProgressBar(dt=60):
            climate_model_data_mean, climate_model_data_bc_mean, reanalysis_data_mean = validation.compute_mean(
                climate_model_data.ds_data[args.variable],
                climate_model_data_bc.ds_data[args.variable],
                reanalysis_data.ds_data[args.variable_ref],
                hour=hour,
                group="year",
                period=None,
                spatial_mean=True
            )
            climate_model_data_mean = climate_model_data_mean.compute()
            climate_model_data_bc_mean = climate_model_data_bc_mean.compute()
            reanalysis_data_mean = reanalysis_data_mean.compute()
            xr.Dataset({
                "climate_model_data_mean": climate_model_data_mean,
                "climate_model_data_bc_mean": climate_model_data_bc_mean,
                "reanalysis_data_mean": reanalysis_data_mean
            }).to_netcdf(output_var_path / f"yearly_spatial_mean_hour_{hour}.nc")

        print(f"Computing quantiles for hour {hour}...")
        with ProgressBar(dt=60):
            (climate_model_data_quantile,
             climate_model_data_bc_quantile,
             reanalysis_data_quantile) = validation.get_quantile(
                climate_model_data.ds_data[args.variable],
                climate_model_data_bc.ds_data[args.variable],
                reanalysis_data.ds_data[args.variable_ref],
                hour=hour,
                period="reference",
                quantiles=[0.05, 0.95],
            )
            xr.Dataset({
                "climate_model_data": climate_model_data_quantile,
                "climate_model_data_bc": climate_model_data_bc_quantile,
                "reanalysis_data": reanalysis_data_quantile
            }).to_netcdf(output_var_path / f"reference_quantiles_hour_{hour}.nc")

    print(f"\tTime to compute hourly yearly statistics: {(time.time() - time_start)/60:.2f} minutes.")

    # Compute trends
    print("Computing trends for yearly values...")
    yearly_mean_trends = []
    ds_yearly_mean = xr.open_dataset(output_var_path / "yearly_spatial_mean.nc")
    yearly_mean_trends.append(validation.compute_trend(
        ds_yearly_mean["climate_model_data_mean"],
        ds_yearly_mean["climate_model_data_bc_mean"],
        ds_yearly_mean["reanalysis_data_mean"],
        hour=None,
        dim="year"
    ))
    yearly_mean_trends[-1]["selection"] = "total"

    for hour in get_valid_hours(time_res, first_time_step.hour):
        ds_hourly_mean = xr.open_dataset(output_var_path / f"yearly_mean_hour_{hour}.nc")
        yearly_mean_trends.append(validation.compute_trend(
            ds_hourly_mean["climate_model_data_mean"],
            ds_hourly_mean["climate_model_data_bc_mean"],
            ds_hourly_mean["reanalysis_data_mean"],
            hour=hour,
            dim="year"
        ))
        yearly_mean_trends[-1]["selection"] = f"hour_{hour}"

    yearly_mean_trends = pd.concat(yearly_mean_trends, axis=0, ignore_index=True)
    yearly_mean_trends.to_csv(output_var_path / "yearly_mean_trends.csv")

    # metrics.append(validation.compute_bias()
    print("Validation completed.")
    print(f"Total time: {(time.time() - time_start)/60:.2f} minutes.")