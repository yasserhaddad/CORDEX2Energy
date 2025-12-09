import argparse
import json
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
import rioxarray as rxr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.demand import (
    process_population_data,
    normalize_population_data,
    compute_cdm_hdm_xr,
)
from src.conversion_energy.aggregate_data import aggregate_data_per_country
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import REDUCED_NETCDF_ENCODINGS, BC_SUBDAILY_CHUNKS, EXTENT_BC
from src.utils.utils_xarray import sel_extent

import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

dask.config.set({"logging.distributed": "error"})
dask.config.set(num_workers=15, threads_per_worker=16)
dask.config.set({"array.chunk-size": "400MiB"})


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute heating and cooling days per month per country."
    )
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument(
        "--temperature_weighted",
        action="store_true",
        help="Apply temperature weighting.",
    )
    return parser.parse_args()


def load_and_process_data(
    paths, climate_model_chain, climate_scenario, chunks_bc, extent=EXTENT_BC
):
    extent_pop = {
        "lat_min": extent["lat_min"] - 1,
        "lat_max": extent["lat_max"] + 1,
        "lon_min": extent["lon_min"] - 1,
        "lon_max": extent["lon_max"] + 1,
    }

    ds_tas_bc = ClimateModelData(
        Path(paths["path_data_processed"])
        / "cordex_processed"
        / f"{climate_model_chain}_{climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "tas.zarr",
        extent=extent,
        chunks=chunks_bc,
    )

    da_worldpop_coarse = process_population_data(
        sel_extent(
            rxr.open_rasterio(
                Path(paths["path_data_processed"])
                / "pop"
                / "ppp_2020_1km_Aggregated.tif"
            ).rename({"x": "lon", "y": "lat"}),
            extent=extent_pop,
        ),
        ds_tas_bc.ds_data,
        coarsen=True,
    )

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    mask_countries_worldpop = countries.mask(da_worldpop_coarse, flag="names")
    da_worldpop_coarse = normalize_population_data(
        da_worldpop_coarse, mask_countries_worldpop
    ).where(da_worldpop_coarse > 0, np.nan)

    da_worldpop_coarse = da_worldpop_coarse.chunk(
        {
            "lat": ds_tas_bc.ds_data.chunksizes["lat"][0],
            "lon": ds_tas_bc.ds_data.chunksizes["lon"][0],
        }
    )

    return ds_tas_bc, da_worldpop_coarse


def main():
    time_start_script = time.time()

    args = parse_arguments()
    paths = json.load(open(args.paths_file))

    print("Loading and processing data")
    ds_tas_bc, da_worldpop_coarse = load_and_process_data(
        paths, args.climate_model_chain, args.climate_scenario, BC_SUBDAILY_CHUNKS
    )

    print("\tComputing heating and cooling days...")
    with ProgressBar(dt=10):
        ds_tas_bc_daily = ds_tas_bc.ds_data.resample(time="1D").mean()
        ds_tas_bc_daily = ds_tas_bc_daily - 273.15
        if args.temperature_weighted:
            ds_tas_bc_daily = aggregate_data_per_country(
                ds_tas_bc_daily * da_worldpop_coarse,
                method="sum"
            )
        ds_demand = compute_cdm_hdm_xr(
            ds_tas_bc_daily.tas,
            da_worldpop_coarse,
            temperature_weighting=args.temperature_weighted,
            parallelized=True,
        )

        if not args.temperature_weighted:
            ds_demand = aggregate_data_per_country(ds_demand, method="sum")
        # ensure data variables use single-precision floats to reduce output size
        for var in ds_demand.data_vars:
            if np.issubdtype(ds_demand[var].dtype, np.floating):
                ds_demand[var] = ds_demand[var].astype(np.float32)

        encoding = {
            var: REDUCED_NETCDF_ENCODINGS
            for var in ds_demand.data_vars
        }
        output_path = (
            Path(paths["path_data_processed"])
            / "website_data"
            / "demand"
            / f"{args.climate_model_chain}_{args.climate_scenario}{'_temperature_weighted' if args.temperature_weighted else ''}.nc"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        print(f"Writing heating and cooling days to {output_path}")
        ds_demand.isel(band=0).to_netcdf(output_path, encoding=encoding, mode="w")

    print("Done.")
    print("Total time: {:.2f} minutes".format((time.time() - time_start_script) / 60))


if __name__ == "__main__":
    main()
