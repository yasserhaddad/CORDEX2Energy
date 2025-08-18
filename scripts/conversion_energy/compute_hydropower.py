import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.hydropower import simple_hydropower_potential
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import (
    COMPRESSED_ZARR_ENCODING,
    EXTENT_BC,
    EXTENT_BC_MRRO,
    REGRIDDED_CHUNKS,
    BC_DAILY_CHUNKS,
    REFERENCE_PERIOD,
)
from src.conversion_energy.aggregate_data import aggregate_data_per_country

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()

    dask.config.set(num_workers=15, scheduler="processes")
    dask.config.set({"array.chunk-size": "500MiB"})

    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])
    path_routed_runoff = Path(paths["path_routed_runoff"])
    path_energy = path_data_processed / "energy"

    output_path_hydropower = (
        path_energy
        / "europe"
        / "hydropower"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    )
    output_path_hydropower.mkdir(parents=True, exist_ok=True)

    extent = {
        "lat_min": EXTENT_BC["lat_max"],
        "lat_max": EXTENT_BC["lat_min"],
        "lon_min": EXTENT_BC["lon_min"],
        "lon_max": EXTENT_BC["lon_max"],
    }

    # ds_routed = ClimateModelData(
    #     list(
    #         (
    #             path_routed_runoff
    #             / f"{args.climate_model_chain}_{args.climate_scenario}"
    #         ).glob("*/Qrouted_historical_*_m3s.zarr")
    #     ),
    #     chunks="auto",
    #     extent=extent,
    #     check_calendar=False,
    # )
    ds_routed = ClimateModelData(
        list((path_routed_runoff / "CNRM-85").glob("*/Qrouted_historical_*_m3s.zarr")),
        chunks="auto",
        extent=extent,
        check_calendar=False,
    )
    nan_mask = np.isnan(ds_routed.ds_data.isel(time=0))
    ds_routed.ds_data = ds_routed.ds_data.fillna(1).chunk(REGRIDDED_CHUNKS)

    with ProgressBar(dt=60):
        ds_hydro = simple_hydropower_potential(
            ds_routed.ds_data,
            reference_period=REFERENCE_PERIOD,
            quantile_max=0.95,
            quantile_min=0.05,
            dim="time",
        ).compute()
        ds_hydro = ds_hydro.where(~nan_mask).rename({"Qrouted": "hydropower"})

        output_path = (
            output_path_hydropower
            / f"hydropower_{args.climate_model_chain}_{args.climate_scenario}.zarr"
        )
        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(BC_DAILY_CHUNKS.values()),
            }
            for var in ds_hydro.data_vars
        }
        ds_hydro.to_zarr(output_path, mode="w", encoding=zarr_encoding)
