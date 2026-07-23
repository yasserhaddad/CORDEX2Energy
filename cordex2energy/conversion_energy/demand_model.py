"""Transparent degree-day → demand model core (heating & cooling).

The reusable core of the JRC-coefficient demand model: population-weighted national
degree days, the linear ``demand = coefficient · degree_day`` mapping (with an optional
exogenous AC-growth multiplier for cooling), and daily→hourly upscaling with a diurnal
profile. Coefficients are derived separately (see :mod:`demand_coefficients`) and applied
here at run time.

This module is the **single source of truth** shared by the ``cordex2energy`` production
scripts and the ``c2e_recipe`` research package (which re-imports these functions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import regionmask
import rioxarray as rxr  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from cordex2energy.conversion_energy.demand import process_population_data


# ---------------------------------------------------------------------------
# Country-name reconciliation
# ---------------------------------------------------------------------------
# The naming schemes (demand.ninja params, demand CSVs, regionmask Natural Earth)
# disagree on a few countries. Everything is normalised to the demand.ninja names.
_NAME_ALIASES = {
    "Great Britain": "United Kingdom",
    "Czechia": "Czech Republic",
    "Macedonia": "North Macedonia",
    "Bosnia and Herz.": "Bosnia and Herzegovina",
    "Republic of Serbia": "Serbia",
}


def _canonical(name: str) -> str:
    return _NAME_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Diurnal profile (path-loaded input)
# ---------------------------------------------------------------------------

def _load_diurnal_profiles(path: Path) -> pd.DataFrame:
    """Load the hour-of-day diurnal profile, normalised so each column means 1.

    Normalising to mean 1 guarantees the daily total is preserved when the profile is
    applied (see :func:`apply_diurnal_profile`). *path* points to the vendored
    ``diurnal_profiles.csv`` (a data-tree input, e.g. ``energy/inputs/``), which has a
    ``hour`` index column plus ``heating`` / ``cooling`` columns.
    """
    prof = pd.read_csv(path, index_col="hour")
    return prof / prof.mean()


# ---------------------------------------------------------------------------
# Population-weighted national degree days
# ---------------------------------------------------------------------------

def population_on_grid(da_pop_raw: xr.DataArray, ref: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Coarsen a fine population raster onto the *ref* (climate) grid.

    *da_pop_raw* is a rioxarray raster (``y``/``x`` dims); *ref* carries the target
    ``lat``/``lon``. Reuses the demand pipeline's coarsen-then-nearest approach.
    """
    da = da_pop_raw
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    rename = {d: t for d, t in (("y", "lat"), ("x", "lon")) if d in da.dims}
    da = da.rename(rename)
    return process_population_data(da, ref, coarsen=True)


def build_country_weights(
    ref: xr.DataArray,
    da_pop: xr.DataArray,
    countries: regionmask.Regions | None = None,
) -> tuple[xr.DataArray, list[str]]:
    """Population weights per country on the *ref* grid, plus canonical names.

    Rasterising the country polygons (``mask_3D``) is the expensive step, so build the
    weights **once** and reuse them for HDD, CDD and any resampled field via
    :func:`national_mean`.
    """
    if countries is None:
        countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    mask3d = countries.mask_3D(ref)                       # (region, lat, lon) bool
    weights = (mask3d * da_pop).fillna(0.0)
    names = [_canonical(n) for n in np.atleast_1d(mask3d["names"].values)]
    return weights, names


def national_mean(da: xr.DataArray, weights: xr.DataArray, names: list[str]) -> pd.DataFrame:
    """Population-weighted country means of *da* using precomputed *weights*."""
    national = da.weighted(weights).mean(dim=["lat", "lon"]).compute()
    df = pd.DataFrame(
        national.transpose("time", "region").values,
        index=pd.to_datetime(national["time"].values),
        columns=names,
    )
    df.index.name = "time"
    return df.loc[:, ~df.columns.duplicated()]


def compute_national_degree_days(
    ds_dd: xr.Dataset,
    da_pop: xr.DataArray,
    var: str,
    countries: regionmask.Regions | None = None,
    weights: xr.DataArray | None = None,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Population-weighted national daily degree days for one variable.

    Pass precomputed *weights*/*names* (from :func:`build_country_weights`) to avoid
    re-rasterising the country polygons on every call.
    """
    if weights is None or names is None:
        weights, names = build_country_weights(ds_dd[var], da_pop, countries)
    return national_mean(ds_dd[var], weights, names)


# ---------------------------------------------------------------------------
# Degree days → demand, and daily → hourly
# ---------------------------------------------------------------------------

def degree_days_to_demand(
    national_dd: pd.DataFrame,
    coefficient: pd.Series,
    ac_growth: Callable[[int], float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Daily national demand = coefficient · degree days (× optional AC growth).

    *coefficient* is a per-country Series (GW per °C·day). *ac_growth* (cooling only) is
    an optional exogenous multiplier: a callable ``year -> factor`` or a Series indexed by
    year. Countries absent from *coefficient* are dropped.
    """
    cols = [c for c in national_dd.columns if c in coefficient.index and np.isfinite(coefficient[c])]
    demand = national_dd[cols].mul(coefficient[cols], axis=1)

    if ac_growth is not None:
        years = demand.index.year
        if callable(ac_growth):
            factor = pd.Series([ac_growth(int(y)) for y in years], index=demand.index)
        else:
            factor = pd.Series(years, index=demand.index).map(ac_growth).astype(float)
        demand = demand.mul(factor, axis=0)
    return demand


def apply_diurnal_profile(
    daily_demand: pd.DataFrame,
    profile: pd.DataFrame,
    kind: str = "heating",
) -> pd.DataFrame:
    """Upscale daily-mean demand to hourly using an hour-of-day *profile*.

    ``hourly[d, h] = daily_demand[d] · profile[h, kind]``. Because *profile* is
    mean-normalised (see :func:`_load_diurnal_profiles`), the daily mean — and hence daily
    total energy — is preserved. *profile* must expose a *kind* column indexed by hour 0–23.
    """
    factors = profile[kind]
    start = daily_demand.index.min().normalize()
    end = daily_demand.index.max().normalize() + pd.Timedelta(days=1)
    hourly_idx = pd.date_range(start, end, freq="h", inclusive="left")

    daily_broadcast = daily_demand.reindex(hourly_idx.normalize())
    daily_broadcast.index = hourly_idx
    hourly = daily_broadcast.mul(factors.loc[hourly_idx.hour].to_numpy(), axis=0)
    hourly.index.name = "time"
    return hourly
