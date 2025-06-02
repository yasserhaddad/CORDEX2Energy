import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.wind import (
    CH_HUB_HEIGHTS,
    CH_TURBINES,
    EUROPE_HUB_HEIGHTS,
    EUROPE_TURBINES,
    compute_power_law_exponent,
    compute_smooth_power_curve,
    compute_wind_power,
    extrapolate_wind_speed,
)
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import COMPRESSED_ZARR_ENCODING
from src.conversion_energy.aggregate_data import aggregate_data_per_country

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("scope", type=str, help="Scope of the analysis (europe or ch).")

    args = parser.parse_args()

    if args.scope not in ["europe", "ch"]:
        raise ValueError("Scope must be either 'europe' or 'ch'.")

    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    path_data_analysis = Path(paths["path_data_analysis"])
    path_data_processed = Path(paths["path_data_processed"])

    path_energy = path_data_processed / "energy"

    output_path = (
        (path_energy / "europe") if args.scope == "europe" else (path_energy / "ch")
    )
    output_path_wind = (output_path / "wind_power" / f"{args.climate_model_chain}_{args.climate_scenario}")
    output_path_wind.mkdir(parents=True, exist_ok=True)

    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})

    time_start = time.time()
    print("Computing smoothed power curves...")
    power_curves_smoothed = (
        {turbine: compute_smooth_power_curve(turbine) for turbine in EUROPE_TURBINES}
        if args.scope == "europe"
        else {turbine: compute_smooth_power_curve(turbine) for turbine in CH_TURBINES}
    )

    print("Loading wind data...")
    extent = (
        {"lat_min": 30, "lat_max": 75, "lon_min": -12, "lon_max": 30}
        if args.scope == "europe"
        else {"lat_min": 45.8, "lat_max": 47.9, "lon_min": 5.9, "lon_max": 10.5}
    )
    chunks_bc = {
        "time": 2920,
        "lat": 128,
        "lon": 128,
    }

    ds_wind_100m = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "wind100m.zarr",
        extent=extent,
        chunks=chunks_bc,
    )

    ds_wind_150m = ClimateModelData(
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "wind150m.zarr",
        extent=extent,
        chunks=chunks_bc,
    )

    print("Computing wind speed exponent...")
    da_alpha = compute_power_law_exponent(
        ds_wind_100m.ds_data.wind100m, ds_wind_150m.ds_data.wind150m, 100, 150
    )
    with ProgressBar(dt=60):
        da_alpha = da_alpha.compute() if args.scope == "ch" else da_alpha.persist()

    turbines = EUROPE_TURBINES if args.scope == "europe" else CH_TURBINES
    hub_heights = EUROPE_HUB_HEIGHTS if args.scope == "europe" else CH_HUB_HEIGHTS

    for turbine_type in turbines:
        step_start = time.time()
        print(f"Computing wind power for {turbine_type}")
        with ProgressBar(dt=60):
            da_wind_hub_height = extrapolate_wind_speed(
                ds_wind_100m.ds_data.wind100m, 100, hub_heights[turbine_type], da_alpha
            )
            ds_wind_power = (
                (
                    compute_wind_power(
                        da_wind_hub_height,
                        power_curves_smoothed[turbine_type][
                            turbine_type.replace("/", "_")
                        ],
                        turbine_type.replace("/", "_"),
                        step=0.01,
                        parallelized=True,
                    )
                )
                .to_dataset()
                .persist()
            )

        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(chunks_bc.values()),
            }
            for var in ds_wind_power.data_vars
        }
        output_path_zarr = (
            output_path_wind
            / f"wind_power_{args.climate_model_chain}_{args.climate_scenario}_{turbine_type.replace('/', '_')}.zarr"
        )
        if output_path_zarr.exists():
            shutil.rmtree(output_path)
        ds_wind_power.to_zarr(output_path_zarr, mode="w", encoding=zarr_encoding)

        del da_wind_hub_height, ds_wind_power

        ds_wind_power = xr.open_zarr(output_path_zarr, chunks=chunks_bc)
        with ProgressBar(dt=60):
            ds_wind_power_agg = aggregate_data_per_country(
                ds_wind_power,
                method="median_and_better",
                offshore=False,
            ).compute()
            ds_wind_power_agg_offshore = aggregate_data_per_country(
                ds_wind_power,
                method="median_and_better",
                offshore=True,
                path_eez_file=path_data_processed / "maps" / "eez_v12" / "eez_v12.shp",
            ).compute()

        output_path_agg = (
            output_path_wind
            / f"wind_power_agg_{args.climate_model_chain}_{args.climate_scenario}_{turbine_type.replace('/', '_')}.nc"
        )
        if output_path_agg.exists():
            output_path_agg.unlink()
        ds_wind_power_agg.to_netcdf(output_path_agg, mode="w")

        output_path_offshore = (
            output_path_wind
            / f"wind_power_offshore_{args.climate_model_chain}_{args.climate_scenario}_{turbine_type.replace('/', '_')}.nc"
        )
        if output_path_offshore.exists():
            output_path_offshore.unlink()
        ds_wind_power_agg_offshore.to_netcdf(output_path_offshore, mode="w")
        del ds_wind_power_agg, ds_wind_power_agg_offshore

        print(
            "Time to compute wind power for turbine type ",
            turbine_type,
            ": ",
            round((time.time() - step_start) / 60, 2),
            " minutes.",
        )

    print(
        "Time to compute wind power for all turbines: ",
        round((time.time() - time_start) / 60, 2),
        " minutes.",
    )
