"""Compute national heating & cooling demand from degree days + JRC coefficients.

Applies the per-country coefficients from ``demand_coefficients.csv`` (produced by
``derive_demand_coefficients.py``) to the gridded degree days from ``compute_degree_days.py``:

1. national **population-weighted** daily HDD/CDD (GPW v4 weighting, matching the derivation),
2. ``demand = coefficient · degree_day`` (heating ``P_heat_elec``; cooling ``P_cool``, with an
   optional exogenous ``--ac-growth`` multiplier),
3. daily → hourly via the diurnal profile,
4. written as a national NetCDF (dims ``country``/``time``, vars ``heating_demand`` /
   ``cooling_demand`` [GW]) that the RECIPE aggregation consumes.

Usage:
    python compute_demand.py <paths_file> <climate_model_chain> <climate_scenario> [--ac-growth RATE]
"""

import argparse
import json
import os
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.conversion_energy import demand_coefficients as dc  # noqa: F401  (kept for parity)
from cordex2energy.conversion_energy import demand_model as dm
from cordex2energy.utils.constants import REDUCED_NETCDF_ENCODINGS

# Avoid BLAS/OpenMP oversubscription under the dask threaded scheduler.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
dask.config.set({"logging.distributed": "error"})
dask.config.set(num_workers=15, threads_per_worker=16)
dask.config.set({"array.chunk-size": "400MiB"})

POP_RASTER = "gpw_v4/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"


def parse_arguments():
    p = argparse.ArgumentParser(description="Compute national heating/cooling demand.")
    p.add_argument("paths_file", type=str, help="Path to the JSON file containing paths.")
    p.add_argument("climate_model_chain", type=str, help="Climate model chain (or 'cerra').")
    p.add_argument("climate_scenario", type=str, help="Climate scenario (e.g. rcp85).")
    p.add_argument("--ac-growth", type=float, default=None,
                   help="Optional exogenous cooling growth per year, e.g. 0.01 for +1%%/yr "
                        "relative to the base year (default: off, static coefficient).")
    return p.parse_args()


def load_population_on_grid(pop_path: Path, ref: xr.Dataset) -> xr.DataArray:
    """GPW v4 raster clipped to *ref* and coarsened onto its grid (same as the derivation)."""
    da = rxr.open_rasterio(pop_path, masked=True).rio.clip_box(
        minx=float(ref.lon.min()), miny=float(ref.lat.min()),
        maxx=float(ref.lon.max()), maxy=float(ref.lat.max()),
    )
    return dm.population_on_grid(da, ref)


def ac_growth_factor(rate: float | None, base_year: int):
    """Exogenous cooling multiplier ``(1 + rate)^(year - base_year)``, or None if off."""
    if rate is None:
        return None
    return lambda year: (1.0 + rate) ** (year - base_year)


def _national_df(national: xr.DataArray, names: list) -> pd.DataFrame:
    """Wrap a computed ``(time, region)`` national-mean array as a country DataFrame
    (same shape as :func:`cordex2energy.conversion_energy.demand_model.national_mean`)."""
    df = pd.DataFrame(
        national.transpose("time", "region").values,
        index=pd.to_datetime(national["time"].values),
        columns=names,
    )
    df.index.name = "time"
    return df.loc[:, ~df.columns.duplicated()]


def main():
    args = parse_arguments()
    paths = json.load(open(args.paths_file))
    root = Path(paths["path_data_processed"])
    model_tag = ("cerra" if args.climate_model_chain.lower() == "cerra"
                 else f"{args.climate_model_chain}_{args.climate_scenario}")

    dd_zarr = root / "energy" / "europe" / "degree_days" / f"{model_tag}.zarr"
    energy_inputs = root / "energy" / "inputs"
    pop_path = root / "pop" / POP_RASTER
    out_nc = root / "energy" / "europe" / "demand" / f"{model_tag}.nc"

    t0 = time.time()
    print(f"Loading degree days {dd_zarr.name} and GPW v4 population …")
    ds = xr.open_zarr(dd_zarr, chunks="auto").squeeze(drop=True)      # lazy, native zarr chunking
    pop = load_population_on_grid(pop_path, ds)
    weights, names = dm.build_country_weights(ds["HDD"].isel(time=0), pop)

    # Population-weighted national means — the spatial reduction is the bottleneck. Build the
    # HDD and CDD reductions lazily (across the zarr's native time chunks) and evaluate them in a
    # single threaded-dask pass with a progress bar, instead of two blocking .compute() calls.
    print("Aggregating national daily degree days …")
    hdd_lazy = ds["HDD"].weighted(weights).mean(dim=["lat", "lon"])
    cdd_lazy = ds["CDD"].weighted(weights).mean(dim=["lat", "lon"])
    with ProgressBar(dt=10):
        hdd_nat, cdd_nat = dask.compute(hdd_lazy, cdd_lazy)
    nat_hdd = _national_df(hdd_nat, names)
    nat_cdd = _national_df(cdd_nat, names)

    coefs = pd.read_csv(energy_inputs / "demand_coefficients.csv", index_col=0)
    diurnal = dm._load_diurnal_profiles(energy_inputs / "diurnal_profiles.csv")
    grow = ac_growth_factor(args.ac_growth, base_year=int(nat_cdd.index.year.min()))

    print("Applying coefficients + diurnal profile …")
    heat_daily = dm.degree_days_to_demand(nat_hdd, coefs["P_heat_elec"])
    cool_daily = dm.degree_days_to_demand(nat_cdd, coefs["P_cool"], ac_growth=grow)
    heat_hourly = dm.apply_diurnal_profile(heat_daily, diurnal, kind="heating")
    cool_hourly = dm.apply_diurnal_profile(cool_daily, diurnal, kind="cooling")

    countries = sorted(set(heat_hourly.columns) & set(cool_hourly.columns))
    heat_hourly, cool_hourly = heat_hourly[countries], cool_hourly[countries]
    ds_out = xr.Dataset(
        {
            "heating_demand": (("time", "country"), heat_hourly.to_numpy(np.float32)),
            "cooling_demand": (("time", "country"), cool_hourly.to_numpy(np.float32)),
        },
        coords={"time": heat_hourly.index.values, "country": countries},
    )
    ds_out["heating_demand"].attrs["units"] = ds_out["cooling_demand"].attrs["units"] = "GW"
    ds_out.attrs["description"] = (
        "National electric heating/cooling demand from the JRC-coefficient degree-day model."
    )
    ds_out.attrs["ac_growth_rate"] = "off" if args.ac_growth is None else str(args.ac_growth)

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: REDUCED_NETCDF_ENCODINGS for v in ds_out.data_vars}
    ds_out.to_netcdf(out_nc, encoding=encoding)
    print(f"Wrote {out_nc} ({len(countries)} countries, {ds_out.sizes['time']} hourly steps).")
    print("Total time: {:.1f} min".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
