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
from dask.diagnostics import ProgressBar

sys.path.append("../..")

import geopandas as gpd

from src.conversion_energy.aggregate_data import aggregate_data_per_country
from src.data_processing.load_data import ClimateModelData
from src.utils.constants import (
    REDUCED_NETCDF_ENCODINGS,
    BC_SUBDAILY_CHUNKS,
    EXTENT_BC,
    EXTENT_CH,
)
from src.utils.utils_xarray import sel_extent

import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr
os.environ["USE_PYGEOS"] = "1"

dask.config.set({"logging.distributed": "error"})
dask.config.set(num_workers=15, threads_per_worker=16)
dask.config.set({"array.chunk-size": "400MiB"})


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute decadal mean temperature anomalies per country and (optionally) per Swiss canton."
    )
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument(
        "--baseline-start",
        type=int,
        default=1991,
        help="Baseline start year (inclusive).",
    )
    parser.add_argument(
        "--baseline-end", type=int, default=2020, help="Baseline end year (inclusive)."
    )
    return parser.parse_args()


def load_tas_zarr(
    paths, climate_model_chain, climate_scenario, chunks_bc, extent=EXTENT_BC
):
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

    return ds_tas_bc


def compute_decadal_anomalies(
    ds: xr.Dataset,
    baseline_start: int,
    baseline_end: int,
    centers=(2030, 2040, 2050),
) -> xr.Dataset:
    """Compute anomalies for centered 10-year windows around given center years.

    Windows are computed as [center-4, center+5] (inclusive) so that:
      - 2030 -> 2026..2035
      - 2040 -> 2036..2045
      - 2050 -> 2046..2055

    Input ds: dataset with tas variable in Kelvin and time/lat/lon dims.
    Returns: dataset with a time dimension corresponding to the center years
    (time coordinate set to YYYY-01-01 for each center).
    """
    # Convert to degC
    ds_c = ds - 273.15

    # yearly means
    ds_yearly = ds_c.resample(time="YE").mean()

    # baseline mean computed on yearly data
    years = ds_yearly["time"].dt.year
    baseline_mask = (years >= baseline_start) & (years <= baseline_end)
    if baseline_mask.sum() == 0:
        raise ValueError(
            f"No years found in baseline range {baseline_start}-{baseline_end} on dataset time axis."
        )
    ds_baseline = ds_yearly.sel(
        time=slice(str(baseline_start), str(baseline_end))
    ).mean(dim="time")

    anom_list = []
    times = []
    for center in centers:
        start = center - 9
        end = center + 10
        ds_center_mean = ds_yearly.sel(time=slice(str(start), str(end))).mean(
            dim="time"
        )
        ds_center_anom = ds_center_mean - ds_baseline
        anom_list.append(ds_center_anom)
        times.append(pd.to_datetime(f"{center}-12-31"))

    # concat along a new time dimension
    ds_anom = xr.concat(anom_list, dim="time")
    ds_anom = ds_anom.assign_coords(time=("time", times))

    return ds_anom


def aggregate_cantons(ds_anom: xr.Dataset, canton_shapefile: Path) -> xr.Dataset:
    """Aggregate decadal anomaly dataset to cantons using regionmask from the provided shapefile.

    Returns a dataset with a 'canton' dimension where values are canton names.
    """
    gdf = gpd.read_file(canton_shapefile, engine="pyogrio")
    gdf = gdf.to_crs("EPSG:4326")
    name_col = "NAME"
    regions = regionmask.from_geopandas(gdf, names=name_col)

    # use numeric mask for grouping
    mask_numbers = regions.mask_3D_frac_approx(ds_anom)
    mask_numbers = mask_numbers.fillna(-1).rename({"region": "canton"})
    weights = np.cos(np.deg2rad(ds_anom.lat))

    # group and mean over lat/lon
    ds_cantons = ds_anom.weighted(mask_numbers * weights).mean(dim=["lat", "lon"], skipna=True)

    # remove -1 (no region)
    ds_cantons = ds_cantons.sel(
        canton=ds_cantons["canton"].values[ds_cantons["canton"].values != -1]
    )

    # replace numeric ids by names using gdf index -> regions numbers map to gdf order
    canton_names = [gdf.loc[int(i), name_col] for i in ds_cantons["canton"].values]
    ds_cantons = ds_cantons.assign_coords(canton=canton_names).drop_vars(["names", "abbrevs"])

    return ds_cantons


def main():
    t0 = time.time()
    args = parse_arguments()
    paths = json.load(open(args.paths_file))

    print("Loading bias-corrected tas.zarr and computing decadal anomalies...")
    ds_tas_bc = load_tas_zarr(
        paths, args.climate_model_chain, args.climate_scenario, BC_SUBDAILY_CHUNKS
    )

    with ProgressBar(dt=10):
        ds_tas = ds_tas_bc.ds_data

        ds_anom = compute_decadal_anomalies(
            ds_tas, args.baseline_start, args.baseline_end
        )

        ds_countries = aggregate_data_per_country(ds_anom, method="mean")
        for var in ds_countries.data_vars:
            if np.issubdtype(ds_countries[var].dtype, np.floating):
                ds_countries[var] = ds_countries[var].astype(np.float32)

        output_path = (
            Path(paths["path_data_processed"])
            / "website_data"
            / "temperature"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = (
            output_path
            / f"{args.climate_model_chain}_{args.climate_scenario}_decadal_temperature_anomalies.nc"
        )
        print(f"Writing country decadal anomalies to {out_file}")
        if out_file.exists():
            out_file.unlink()
        ds_countries.to_netcdf(out_file, mode="w")

        # Canton aggregation
        canton_path = (
            Path(paths["path_data_processed"])
            / "maps"
            / "cantons_ch"
            / "swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp"
        )
        print("Computing canton-level aggregates...")
        ds_cantons = aggregate_cantons(
            sel_extent(ds_anom, extent=EXTENT_CH), canton_path
        )
        output_path_ch = (
            Path(paths["path_data_processed"])
            / "website_data"
            / "temperature"
        )
        output_path_ch.mkdir(parents=True, exist_ok=True)
        out_file_ch = (
            output_path_ch
            / f"{args.climate_model_chain}_{args.climate_scenario}_decadal_temperature_anomalies_cantons.nc"
        )
        print(f"Writing canton decadal anomalies to {out_file_ch}")
        for var in ds_cantons.data_vars:
            if np.issubdtype(ds_cantons[var].dtype, np.floating):
                ds_cantons[var] = ds_cantons[var].astype(np.float32)
        if out_file_ch.exists():
            out_file_ch.unlink()
        ds_cantons.to_netcdf(out_file_ch, mode="w")

    print("Done.")
    print("Total time: {:.2f} minutes".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
