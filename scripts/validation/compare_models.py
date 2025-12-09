import json
import xarray as xr
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import pandas as pd

import sys

sys.path.append("../..")
from src.utils.utils_xarray import sel_extent

from src.utils.constants import EXTENT_CH, EXTENT_BC

import os

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

dask.config.set(num_workers=15, threads_per_worker=16)
dask.config.set(
    {
        "distributed.worker.memory.target": 0.7,
        "distributed.worker.memory.spill": 0.8,
        "distributed.worker.memory.pause": 0.9,
        "array.chunk-size": "512MiB",
        "logging.distributed": "error",
    }
)


def compute_yearly_mean(da, climate_model_chain_scenario: str):
    """Compute spatially-averaged annual (YE) mean for a DataArray.

    Parameters
    - da: xarray.DataArray with time, lat, lon
    - climate_model_chain_scenario: string id to set in the climate_model_chain_scenario dim

    Returns
    - DataArray with dims ('time','climate_model_chain_scenario') where 'time' are YE timestamps
    """
    out = (
        da.resample(time="YE")
        .mean(dim="time")
        .mean(dim=["lat", "lon"])  # spatial average
        .expand_dims({"climate_model_chain_scenario": [climate_model_chain_scenario]})
        .compute()
    )
    return out


def compute_seasonal_means(
    da, climate_model_chain_scenario: str, seasons=("DJF", "JJA")
):
    """Compute seasonal (per-season-year) means for given seasons.

    Returns a DataArray with dims ('season_year','season','climate_model_chain_scenario').
    If no seasonal data available, returns None.
    """
    season_arrays = []
    season_names = []

    for s in seasons:
        if s == "DJF":
            da_masked = da.sel(time=da.time.dt.month.isin([12, 1, 2]))
            if da_masked.sizes.get("time", 0) == 0:
                continue
            season_year = xr.where(
                da_masked.time.dt.month == 12,
                da_masked.time.dt.year + 1,
                da_masked.time.dt.year,
            )
            arr = da_masked.groupby(season_year).mean(dim="time").rename({"group": "season_year"})
        else:  # intra-year season like JJA
            months = {
                "JJA": [6, 7, 8],
                "MAM": [3, 4, 5],
                "SON": [9, 10, 11],
            }
            da_masked = da.sel(time=da.time.dt.month.isin(months.get(s, [])))
            if da_masked.sizes.get("time", 0) == 0:
                continue
            season_year = da_masked.time.dt.year
            arr = da_masked.groupby(season_year).mean(dim="time").rename({"year": "season_year"})

        # spatial mean, keep model id dim
        arr = arr.mean(dim=["lat", "lon"]).expand_dims(
            {"climate_model_chain_scenario": [climate_model_chain_scenario],
             "season": [s]}
        )

        season_arrays.append(arr)
        season_names.append(s)

    if not season_arrays:
        return None

    # concat along a new season dim and ensure ordering
    ds_seasonly = (
        xr.concat(season_arrays, dim="season")
        .transpose("season_year", "season", "climate_model_chain_scenario")
    )
    return ds_seasonly.compute()

if __name__ == "__main__":
    paths = json.load(open("../../paths_so4.json"))
    path_data_processed = Path(paths["path_data_processed"])

    vars = ["tas", "pr", "rsds", "mrro"]
    res = {"tas": "3hr", "pr": "day", "rsds": "3hr", "mrro": "day"}

    ds_var_bs_yearly = {var: [] for var in vars}
    ds_var_bs_yearly_ch = {var: [] for var in vars}
    # new containers for seasonal yearly means
    ds_var_bs_seasonly = {var: [] for var in vars}
    ds_var_bs_seasonly_ch = {var: [] for var in vars}

    for var in vars:
        print(f"Processing variable: {var}\n")
        for path in (path_data_processed / "cordex_processed").glob(
            f"*/bias_corrected/{res[var]}/{var}.zarr"
        ):
            climate_model_chain, climate_scenario = path.parts[-4].rsplit("_", 1)
            print(f"Model chain: {climate_model_chain}, Scenario: {climate_scenario}")
            for extent in ["CH"]:
                extent_used = EXTENT_BC if extent == "BC" else EXTENT_CH
                with ProgressBar(dt=1):
                    ds = sel_extent(
                        xr.open_zarr(
                            path_data_processed
                            / "cordex_processed"
                            / f"{climate_model_chain}_{climate_scenario}"
                            / "bias_corrected"
                            / f"{res[var]}"
                            / f"{var}.zarr",
                            chunks="auto",
                        ),
                        extent=extent_used,
                    )[var].sel(time=slice("1991-01-01", None))

                    # Annual yearly mean (as in original) and seasonal means (modularized)
                    climate_model_chain_scenario = (
                        climate_model_chain + "_" + climate_scenario
                    )
                    ds_yearly = compute_yearly_mean(ds, climate_model_chain_scenario)
                    ds_seasonly = compute_seasonal_means(
                        ds, climate_model_chain_scenario, seasons=("DJF", "MAM", "JJA", "SON")
                    )

                    if extent == "BC":
                        ds_var_bs_yearly[var].append(ds_yearly)
                        ds_var_bs_seasonly[var].append(ds_seasonly)
                    else:
                        ds_var_bs_yearly_ch[var].append(ds_yearly)
                        ds_var_bs_seasonly_ch[var].append(ds_seasonly)

    for var in vars:
        print(f"Saving results for variable: {var}\n")
        # concat annual means (CH)
        if ds_var_bs_yearly_ch[var]:
            ds_var_bs_yearly_ch[var] = xr.concat(
                ds_var_bs_yearly_ch[var], dim="climate_model_chain_scenario"
            )
            (path_data_processed / "validation").mkdir(parents=True, exist_ok=True)
            ds_var_bs_yearly_ch[var].to_netcdf(
                path_data_processed / "validation" / f"ds_{var}_bs_yearly_ch.nc"
            )

        # concat seasonal yearly means (CH)
        if ds_var_bs_seasonly_ch[var]:
            # concat along the climate_model_chain_scenario dim
            ds_var_bs_seasonly_ch[var] = xr.concat(
                ds_var_bs_seasonly_ch[var], dim="climate_model_chain_scenario"
            )
            ds_var_bs_seasonly_ch[var].to_netcdf(
                path_data_processed
                / "validation"
                / f"ds_{var}_bs_yearly_season_ch.nc"
            )
