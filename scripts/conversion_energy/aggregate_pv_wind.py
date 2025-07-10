import json
import sys
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.aggregate_data import aggregate_data_per_country

dask.config.set({"array.chunk-size": "500MiB"})
dask.config.set(num_workers=15)

paths = json.load(open("../../paths_so4.json"))

path_data_analysis = Path(paths["path_data_analysis"])
path_data_processed = Path(paths["path_data_processed"])
path_data_cordex = Path(paths["path_data_cordex"])
path_data_adjusted_cordex = Path(paths["path_data_adjusted_cordex"])
path_data_cerra = Path(paths["path_data_cerra"])
path_data_cerra_land = Path(paths["path_data_cerra_land"])
path_data_personal = Path(paths["path_data_personal"])
path_data_personal_cordex = path_data_personal / "cordex"
path_data_temporary = Path(paths["path_data_temporary"])

path_energy_ch = path_data_processed / "energy" / "ch"
path_energy_europe = path_data_processed / "energy" / "europe"

path_figs = Path(paths["path_figs"])


climate_model_chains_scenarios = [p.name for p in (path_energy_europe / "pv").glob("*")]
climate_model_chains = [c.split("_rcp")[0] for c in climate_model_chains_scenarios]
climate_scenarios = [c.split("_")[-1] for c in climate_model_chains_scenarios]

for climate_model_chain, climate_scenario in zip(
    climate_model_chains, climate_scenarios
):
    print(f"Processing {climate_model_chain} - {climate_scenario}")

    # Load the PV data for the current climate model chain and scenario
    path_pv = (
        path_energy_europe
        / "pv"
        / f"{climate_model_chain}_{climate_scenario}"
        / f"pv_{climate_model_chain}_{climate_scenario}.zarr"
    )

    paths_wind = sorted(
        (
            path_energy_europe
            / "wind_power"
            / f"{climate_model_chain}_{climate_scenario}"
        ).glob(f"wind_power_{climate_model_chain}_{climate_scenario}*.zarr")
    )

    # Aggregate PV data per country
    print("Aggregating PV data")
    with ProgressBar(dt=10):
        ds_pv = xr.open_zarr(path_pv, chunks="auto")
        ds_pv_agg = aggregate_data_per_country(
            ds_pv,
            method="median_and_better",
            offshore=False,
        ).compute()

        # Save aggregated data
        output_path_agg = (
            path_energy_europe
            / "pv"
            / f"{climate_model_chain}_{climate_scenario}"
            / f"pv_agg_{climate_model_chain}_{climate_scenario}.nc"
        )
        if output_path_agg.exists():
            output_path_agg.unlink()
        ds_pv_agg.to_netcdf(output_path_agg, mode="w")

    del ds_pv, ds_pv_agg

    # Aggregate Wind data per country
    for path_wind in paths_wind:
        turbine_type = path_wind.name.split(f"_{climate_scenario}_")[-1].replace(".zarr", "")
        print(
            f"Aggregating Wind data for {turbine_type}"
        )
        with ProgressBar(dt=10):
            ds_wind_power = xr.open_zarr(path_wind, chunks="auto")
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
                path_energy_europe
                / "wind_power"
                / f"{climate_model_chain}_{climate_scenario}"
                / f"wind_power_agg_{climate_model_chain}_{climate_scenario}_{turbine_type.replace('/', '_')}.nc"
            )
            if output_path_agg.exists():
                output_path_agg.unlink()
            ds_wind_power_agg.to_netcdf(output_path_agg, mode="w")

            output_path_offshore = (
                path_energy_europe
                / "wind_power"
                / f"{climate_model_chain}_{climate_scenario}"
                / f"wind_power_offshore_{climate_model_chain}_{climate_scenario}_{turbine_type}.nc"
            )
            if output_path_offshore.exists():
                output_path_offshore.unlink()
            ds_wind_power_agg_offshore.to_netcdf(output_path_offshore, mode="w")

    del ds_wind_power, ds_wind_power_agg, ds_wind_power_agg_offshore