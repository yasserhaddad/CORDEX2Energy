"""Compute national heating & cooling demand from degree days + JRC coefficients.

Applies the per-country coefficients from ``demand_coefficients.csv`` (produced by
``derive_demand_coefficients.py``) to the gridded degree days from ``compute_degree_days.py``:

1. national **population-weighted** daily HDD/CDD (GPW v4 weighting, matching the derivation),
2. heating ``= P_heat_elec · HDD``; cooling from the **sector-split saturating** form when the
   coefficients carry it (residential + service, each ``B_j + P_j·CDD·s_j(CDD)``, optionally
   anchored to the observed JRC level), else the older intercept / legacy forms,
3. daily → hourly via the diurnal profile,
4. written as a national NetCDF (dims ``country``/``time``, vars ``heating_demand`` /
   ``cooling_demand`` [GW]) that the RECIPE aggregation consumes.

The two cooling switches are orthogonal — ``--no-corrective-ratio`` drops the *level* anchor and
``--no-ac-growth`` freezes *adoption* — so all four combinations can be produced; non-default
variants are written to their own file.

Usage:
    python compute_demand.py <paths_file> <climate_model_chain> <climate_scenario>
        [--no-corrective-ratio] [--no-ac-growth]
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
from cordex2energy.utils.constants import DEGREE_DAYS_DIRNAME, REDUCED_NETCDF_ENCODINGS

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
    p.add_argument("--no-corrective-ratio", dest="apply_corrective_ratio", action="store_false",
                   help="Do not pin present-day cooling to the observed JRC level (drops the "
                        "per-country corrective ratio). The un-anchored form needs no benchmark, "
                        "so it is the right choice where none exists; it reproduces the historical "
                        "average level rather than today's. Sector-split cooling only.")
    p.add_argument("--no-ac-growth", dest="ac_growth", action="store_false",
                   help="Freeze air-conditioning penetration at each country's anchor climate "
                        "instead of letting the saturating adoption share follow the projected "
                        "degree days. Cooling still grows with warming through the degree days "
                        "themselves. Sector-split cooling only.")
    return p.parse_args()


def load_population_on_grid(pop_path: Path, ref: xr.Dataset) -> xr.DataArray:
    """GPW v4 raster clipped to *ref* and coarsened onto its grid (same as the derivation)."""
    da = rxr.open_rasterio(pop_path, masked=True).rio.clip_box(
        minx=float(ref.lon.min()), miny=float(ref.lat.min()),
        maxx=float(ref.lon.max()), maxy=float(ref.lat.max()),
    )
    return dm.population_on_grid(da, ref)


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

    dd_zarr = root / "energy" / "europe" / DEGREE_DAYS_DIRNAME / f"{model_tag}.zarr"
    energy_inputs = root / "energy" / "inputs"
    pop_path = root / "pop" / POP_RASTER
    out_nc = root / "energy" / "europe" / "demand" / f"{model_tag}.nc"

    # Coefficients are read (and the run configuration validated) up front: the national degree-day
    # aggregation below takes over a minute, so a bad argument combination must fail before it.
    coefs = pd.read_csv(energy_inputs / "demand_coefficients.csv", index_col=0)
    diurnal = dm._load_diurnal_profiles(energy_inputs / "diurnal_profiles.csv")

    # Cooling form, newest usable first:
    #  1. sector-split saturating (preferred) — residential + service, each with a shared per-CDD
    #     intensity and a saturating adoption share inside the level equation, optionally anchored
    #     to the observed JRC level. Bounded by construction: no per-country reference denominator.
    #  2. intercept form — one weather-insensitive baseload plus a degree-day response.
    #  3. legacy through-origin `P_cool` — unusable under the Eurostat definition (near-zero CDD
    #     gives an unbounded slope); kept only for coefficient files predating the change.
    use_split = set(dm.COOLING_SPLIT_COLUMNS) <= set(coefs.columns)
    use_intercept = {"B_cool", "P_cool_weather"} <= set(coefs.columns)
    if use_split:
        cooling_form = ("sector-split: sum_j k*[B_j + P_j*CDD*s_j(CDD)]"
                        f" (corrective_ratio={'on' if args.apply_corrective_ratio else 'off'},"
                        f" ac_growth={'climate-driven' if args.ac_growth else 'frozen'})")
    elif use_intercept:
        cooling_form = "B_cool + P_cool_weather*CDD"
    else:
        cooling_form = "legacy P_cool*CDD"
    print(f"Cooling form: {cooling_form}")

    # Non-default variants get their own file so both can be produced side by side; the default
    # keeps the plain name the RECIPE aggregation and the TYNDP scripts already read.
    suffix = ""
    if use_split:
        suffix += "" if args.apply_corrective_ratio else "_noanchor"
        suffix += "" if args.ac_growth else "_nogrowth"
    out_nc = out_nc.with_name(f"{model_tag}{suffix}.nc")

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

    print("Applying coefficients + diurnal profile …")
    heat_daily = dm.degree_days_to_demand(nat_hdd, coefs["P_heat_elec"])
    if use_split:
        cool_daily = dm.cooling_demand(
            nat_cdd, coefs,
            apply_corrective_ratio=args.apply_corrective_ratio,
            ac_growth=args.ac_growth,
        )
    else:
        cool_daily = dm.degree_days_to_demand(
            nat_cdd,
            coefs["P_cool_weather"] if use_intercept else coefs["P_cool"],
            baseload=coefs["B_cool"] if use_intercept else None,
        )
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
    ds_out.attrs["cooling_form"] = cooling_form
    ds_out.attrs["degree_day_definition"] = str(ds.attrs.get("definition", "unknown"))
    if use_split:
        ds_out.attrs["corrective_ratio"] = "applied" if args.apply_corrective_ratio else "off"
        ds_out.attrs["ac_growth"] = "climate-driven" if args.ac_growth else "frozen penetration"
        for sector in ("res", "srv"):
            theta = coefs[f"theta_cool_{sector}"].dropna()
            ds_out.attrs[f"theta_cool_{sector}"] = float(theta.iloc[0]) if len(theta) else np.nan

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: REDUCED_NETCDF_ENCODINGS for v in ds_out.data_vars}
    ds_out.to_netcdf(out_nc, encoding=encoding)
    print(f"Wrote {out_nc} ({len(countries)} countries, {ds_out.sizes['time']} hourly steps).")
    print("Total time: {:.1f} min".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
