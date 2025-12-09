import argparse
import json
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
)
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
    print("Loading wind data...")
    chunks_bc = {
        "time": 2920,
        "lat": 128,
        "lon": 128,
    }

    turbines = EUROPE_TURBINES if args.scope == "europe" else CH_TURBINES
    hub_heights = EUROPE_HUB_HEIGHTS if args.scope == "europe" else CH_HUB_HEIGHTS

    for turbine_type in turbines:
        step_start = time.time()
        print(f"Computing wind power for {turbine_type}")
        output_path_zarr = (
            output_path_wind
            / f"wind_power_{args.climate_model_chain}_{args.climate_scenario}_{turbine_type.replace('/', '_')}.zarr"
        )

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
