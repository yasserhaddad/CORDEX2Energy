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
from src.utils.cdo_commands import regrid_with_xesmf
from src.utils.utils_xarray import sel_extent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()

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

    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])
    path_routed_runoff = Path(paths["path_routed_runoff_argon"])
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
    #         ).glob("*/Qrouted_historical_*.zarr")
    #     ),
    #     chunks="auto",
    #     extent=extent,
    #     check_calendar=False,
    # )
    ds_routed = ClimateModelData(
        list(
            (
                path_routed_runoff
                / "CNRM-85"
            ).glob("*/Qrouted_historical_*m3s.zarr")
        ),
        chunks="auto",
        extent=extent,
        check_calendar=False,
    )
    nan_mask = np.isnan(ds_routed.ds_data.isel(time=0))
    ds_routed.ds_data = ds_routed.ds_data.fillna(1).chunk(REGRIDDED_CHUNKS)

    # # Build target EUR-11 grid from any regridded FX file
    # ds_target = (
    #     sel_extent(
    #         xr.open_dataset(
    #             sorted(
    #                 (
    #                     path_data_processed
    #                     / "cordex_processed"
    #                     / f"{args.climate_model_chain}_{args.climate_scenario}"
    #                     / "regridded"
    #                     / "fx"
    #                 ).glob("*.nc")
    #             )[0]
    #         ),
    #         extent=EXTENT_BC,
    #     )[["lat", "lon"]]
    #     .sortby("lat")
    #     .sortby("lon")
    # ).chunk({"lat": BC_DAILY_CHUNKS["lat"], "lon": BC_DAILY_CHUNKS["lon"]})

    # Regrid with cached weights (bilinear by default)
    weights_file = output_path_hydropower.parent / "weights_hydro_to_eur11_bilinear.nc"
    with ProgressBar(dt=60):
        ds_hydro = simple_hydropower_potential(
            ds_routed.ds_data,
            reference_period=REFERENCE_PERIOD,
            quantile_max=0.95,
            quantile_min=0.05,
            dim="time",
        )
        ds_hydro = (
            ds_hydro.where(~nan_mask)
            .rename({"Qrouted": "hydropower"})
            .convert_calendar("standard", align_on="year")
            .chunk(BC_DAILY_CHUNKS)
        )

        # ds_hydro = regrid_with_xesmf(
        #     ds_hydro,
        #     ds_target,
        #     var_names=["hydropower"],
        #     method="bilinear",
        #     weights_path=weights_file if weights_file.exists() else None,
        #     reuse_weights=weights_file.exists(),
        #     extrap_method=None,
        #     chunks=BC_DAILY_CHUNKS,
        # )
        # ds_hydro = xr.open_dataset(output_path.parent / (output_path.stem + "_regridded.nc"))
        output_path_zarr = (
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
        # .chunk(BC_DAILY_CHUNKS)
        ds_hydro.to_zarr(output_path_zarr, mode="w", encoding=zarr_encoding)
        # (output_path.parent / (output_path.stem + "_regridded.nc")).unlink()
