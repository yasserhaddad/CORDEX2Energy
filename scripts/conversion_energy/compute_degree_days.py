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



from cordex2energy.conversion_energy.demand import (
    process_population_data,
    normalize_population_data,
    compute_cdd_hdd_xr,
)
from cordex2energy.data_processing.load_data import ClimateModelData, ClimateReanalysisData
from cordex2energy.utils.constants import (
    COMPRESSED_ZARR_ENCODING,
    BC_SUBDAILY_CHUNKS,
    CORDEX_TO_CERRA_VARS,
    DEGREE_DAYS_DIRNAME,
    EXTENT_BC,
)
from cordex2energy.utils.utils_xarray import sel_extent

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
        description="Compute heating and cooling degree days."
    )
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument(
        "--population_weighted",
        action="store_true",
        help="Compute population-weighted demand.",
    )
    return parser.parse_args()


def setup_tmp_directories(paths, model_tag):
    path_data_temporary = Path(paths["path_data_temporary"])
    tmp_energy_data_dir = (
        path_data_temporary / "energy" / "europe" / DEGREE_DAYS_DIRNAME / model_tag
    )
    tmp_energy_data_dir.mkdir(parents=True, exist_ok=True)
    return tmp_energy_data_dir


def load_and_process_data(
    paths, climate_model_chain, climate_scenario, chunks_bc, population_weighted=False
):
    is_cerra = climate_model_chain.lower() == "cerra"

    extent_pop = {
        "lat_min": EXTENT_BC["lat_min"] - 1,
        "lat_max": EXTENT_BC["lat_max"] + 1,
        "lon_min": EXTENT_BC["lon_min"] - 1,
        "lon_max": EXTENT_BC["lon_max"] + 1,
    }

    if is_cerra:
        path_tas = (
            Path(paths["path_data_processed"])
            / "cerra_processed"
            / "regridded"
            / f"{CORDEX_TO_CERRA_VARS['tas']}.zarr"
        )
        ds_tas_bc = ClimateReanalysisData(path_tas, extent=EXTENT_BC, chunks="auto")
        ds_tas_bc.ds_data = (
            ds_tas_bc.ds_data.rename({CORDEX_TO_CERRA_VARS["tas"]: "tas"})
            .squeeze("height", drop=True)
        )
    else:
        ds_tas_bc = ClimateModelData(
            Path(paths["path_data_processed"])
            / "cordex_processed"
            / f"{climate_model_chain}_{climate_scenario}"
            / "bias_corrected"
            / "3hr"
            / "tas.zarr",
            extent=EXTENT_BC,
            chunks="auto",
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
    if not population_weighted:
        da_worldpop_coarse = xr.where(~da_worldpop_coarse.isnull(), 1, 1)

    da_worldpop_coarse = da_worldpop_coarse.chunk(
        {
            "lat": ds_tas_bc.ds_data.chunksizes["lat"][0],
            "lon": ds_tas_bc.ds_data.chunksizes["lon"][0],
        }
    )

    return ds_tas_bc, da_worldpop_coarse


def split_time_range(ds_tas_bc, nb_splits=3):
    time_range = pd.date_range(
        start=ds_tas_bc.ds_data.time.isel(time=0).item(),
        end=ds_tas_bc.ds_data.time.isel(time=-1).item(),
        freq="1h",
    )

    total_days = (time_range[-1] - time_range[0]).days + 1
    split_size = total_days // nb_splits
    split_points = [
        time_range[0] + pd.Timedelta(days=split_size * i) for i in range(nb_splits)
    ]
    split_points.append(time_range[-1].floor("D") + pd.Timedelta(days=1))

    time_ranges = [
        time_range[(time_range >= split_points[i]) & (time_range < split_points[i + 1])]
        for i in range(len(split_points) - 1)
    ]
    return time_ranges


def process_splits(
    ds_tas_bc, da_worldpop_coarse, time_ranges, tmp_energy_data_dir, chunks_bc
):
    for i, time_range_split in enumerate(time_ranges):
        print(f"Processing split {time_range_split[0]} to {time_range_split[-1]}")
        ds_split = ds_tas_bc.ds_data.sel(
            time=slice(time_range_split[0], time_range_split[-1])
        )
        ds_split = ds_split.interp(time=time_range_split)
        if not time_range_split[-1] == ds_tas_bc.ds_data.time.isel(time=-1).item():
            ds_split = ds_split.sel(
                time=slice(time_range_split[0], time_range_split[-2])
            )
        ds_split = ds_split - 273.15

        print("\tComputing heating and cooling degree days for split...")
        with ProgressBar(dt=10):
            ds_demand_split = compute_cdd_hdd_xr(
                ds_split.tas,
                da_worldpop_coarse,
                parallelized=True,
            )
            zarr_encoding = {
                var: {
                    "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                    "chunks": tuple(chunks_bc.values()),
                }
                for var in ds_demand_split.data_vars
            }
            ds_demand_split.chunk(chunks_bc).to_zarr(
                tmp_energy_data_dir / f"tmp_{i+1}.zarr",
                mode="w",
                encoding=zarr_encoding,
            )

        del ds_split, ds_demand_split


def concatenate_and_save_datasets(
    tmp_energy_data_dir, path_energy, model_tag, chunks_bc
):
    print("Concatenating datasets...")
    ds_demand = xr.open_mfdataset(
        tmp_energy_data_dir.glob("*.zarr"),
        engine="zarr",
        combine="by_coords",
        parallel=True,
    )

    print("Writing heating and cooling degree days...")
    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(chunks_bc.values()),
        }
        for var in ds_demand.data_vars
    }
    output_path = path_energy / "europe" / DEGREE_DAYS_DIRNAME / f"{model_tag}.zarr"
    output_path.mkdir(parents=True, exist_ok=True)

    ds_demand.chunk(chunks_bc).to_zarr(
        output_path,
        mode="w",
        encoding=zarr_encoding,
    )


def main():
    args = parse_arguments()
    paths = json.load(open(args.paths_file))

    model_tag = (
        "cerra"
        if args.climate_model_chain.lower() == "cerra"
        else f"{args.climate_model_chain}_{args.climate_scenario}"
    )

    tmp_energy_data_dir = setup_tmp_directories(paths, model_tag)

    time_start_script = time.time()

    print("Loading and processing data")
    ds_tas_bc, da_worldpop_coarse = load_and_process_data(
        paths,
        args.climate_model_chain,
        args.climate_scenario,
        BC_SUBDAILY_CHUNKS,
        population_weighted=args.population_weighted,
    )

    time_ranges = split_time_range(ds_tas_bc, nb_splits=2 if model_tag == "cerra" else 5)
    process_splits(
        ds_tas_bc,
        da_worldpop_coarse,
        time_ranges,
        tmp_energy_data_dir,
        BC_SUBDAILY_CHUNKS,
    )

    path_energy = Path(paths["path_data_processed"]) / "energy"
    concatenate_and_save_datasets(
        tmp_energy_data_dir,
        path_energy,
        model_tag,
        BC_SUBDAILY_CHUNKS,
    )

    print("Done.")
    print("Total time: {:.2f} minutes".format((time.time() - time_start_script) / 60))


if __name__ == "__main__":
    main()
