import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar



from cordex2energy.conversion_energy.hydropower import (
    compute_hydropower_reservoir_inflows,
    compute_ror_hydropower_potential,
    mean_discharge_weight,
    persistence_weight,
    simple_hydropower_potential,
)
from cordex2energy.data_processing.load_data import ClimateModelData
from cordex2energy.utils.constants import (
    BC_DAILY_CHUNKS,
    COMPRESSED_ZARR_ENCODING,
    EXTENT_BC,
    REFERENCE_PERIOD,
    REGRIDDED_CHUNKS,
)

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
    model_name = f"{args.climate_model_chain}_{args.climate_scenario}"
    if model_name == "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp26":
        model_name = "CNRM-26"
        filename_template = "Qrouted_historical_*_m3s.zarr"
    elif model_name == "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp85":
        model_name = "CNRM-85"
        filename_template = "Qrouted_historical_*_m3s.zarr"
    else:
        filename_template = "Qrouted_historical_*.zarr"
    nb_folders = len(list((path_routed_runoff / model_name).glob("*/")))
    list_q_routed = list(
        (path_routed_runoff / model_name).glob(f"*/{filename_template}")
    )
    if any("new.zarr" in str(p) for p in list_q_routed):
        list_q_routed = [
            p for p in list_q_routed if "new.zarr" in str(p)
        ]  # Prefer new.zarr files

    if len(list_q_routed) == 0:
        raise ValueError(
            f"No routed runoff data found for {args.climate_model_chain} {args.climate_scenario} in {path_routed_runoff}"
        )
    if len(list_q_routed) != nb_folders:
        raise ValueError(
            f"Expected {nb_folders} routed runoff files for {args.climate_model_chain} {args.climate_scenario}, but found {len(list_q_routed)} in {path_routed_runoff}"
        )

    ds_routed = ClimateModelData(
        list_q_routed,
        chunks="auto",
        extent=extent,
        check_calendar=False,
    )
    nan_mask = np.isnan(ds_routed.ds_data.isel(time=0))
    ds_routed.ds_data = ds_routed.ds_data.fillna(1).chunk(REGRIDDED_CHUNKS)

    with ProgressBar(dt=60):
        ds_hydro = compute_ror_hydropower_potential(
            ds_routed.ds_data,
            reference_period=REFERENCE_PERIOD,
            quantile_max=0.95,
            quantile_min=0.05,
            dim="time",
        )
        ds_hydro = (
            ds_hydro.where(~nan_mask)
            .rename({"Qrouted": "hydropower_ror"})
            .convert_calendar("standard", align_on="year")
            .chunk(BC_DAILY_CHUNKS)
        )

        output_path_zarr = (
            output_path_hydropower
            / f"hydropower_ror_{args.climate_model_chain}_{args.climate_scenario}.zarr"
        )
        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(BC_DAILY_CHUNKS.values()),
            }
            for var in ds_hydro.data_vars
        }

        ds_hydro.to_zarr(output_path_zarr, mode="w", encoding=zarr_encoding)

    with ProgressBar(dt=60):
        ds_hydro_reservoir = compute_hydropower_reservoir_inflows(
            ds_routed.ds_data, storage_window=7, dim="time"
        )
        ds_hydro_reservoir = (
            ds_hydro_reservoir.where(~nan_mask)
            .rename({"Qrouted": "hydropower_reservoir"})
            .convert_calendar("standard", align_on="year")
            .chunk(BC_DAILY_CHUNKS)
        )

        output_path_zarr_reservoir = (
            output_path_hydropower
            / f"hydropower_reservoir_{args.climate_model_chain}_{args.climate_scenario}.zarr"
        )
        zarr_encoding_reservoir = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(BC_DAILY_CHUNKS.values()),
            }
            for var in ds_hydro_reservoir.data_vars
        }

        ds_hydro_reservoir.to_zarr(
            output_path_zarr_reservoir, mode="w", encoding=zarr_encoding_reservoir
        )

    # # Compute weights and save to NetCDF
    # print("Computing hydropower weights...")
    # with ProgressBar(dt=60):
    #     # Compute mean discharge weight
    #     weight_mean_discharge = mean_discharge_weight(
    #         ds_routed.ds_data,
    #         reference_period=REFERENCE_PERIOD,
    #         exponent=0.5,
    #         dim="time",
    #     ).rename({"Qrouted": "mean_discharge_weight"})

    #     # Compute persistence weight
    #     weight_persistence = persistence_weight(
    #         ds_routed.ds_data,
    #         reference_period=REFERENCE_PERIOD,
    #         dim="time",
    #     ).rename({"Qrouted": "persistence_weight"})

    #     # Combine weights into a single dataset
    #     ds_weights = xr.merge([
    #         weight_mean_discharge.where(~nan_mask),
    #         weight_persistence.where(~nan_mask),
    #     ])

    #     # Compute and save
    #     ds_weights = ds_weights.compute()

    #     output_path_weights = (
    #         output_path_hydropower
    #         / f"hydropower_weights_{args.climate_model_chain}_{args.climate_scenario}.nc"
    #     )
    #     if output_path_weights.exists():
    #         output_path_weights.unlink()

    #     ds_weights.to_netcdf(output_path_weights)
    #     print(f"Weights saved to {output_path_weights}")
