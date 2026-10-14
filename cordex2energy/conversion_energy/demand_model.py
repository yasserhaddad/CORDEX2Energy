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
    ac_growth: Callable[[int], float] | pd.Series | pd.DataFrame | None = None,
    baseload: pd.Series | None = None,
    corrective_ratio: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily national demand = baseload + coefficient · degree days (× optional AC growth).

    *coefficient* is a per-country Series (GW per °C·day). Countries absent from *coefficient*
    are dropped. *ac_growth* (cooling only) is an optional adoption multiplier, one of:

    * a callable ``year -> factor`` or a Series indexed by year — the same factor for every
      country (an exogenous ramp);
    * a **DataFrame** indexed by year (or by time) with one column per country — a per-country
      factor, which is what a climate-driven adoption term needs, since saturation depends on each
      country's own degree days (see :func:`cooling_adoption_factor`).

    *baseload* (cooling only, GW) is the **weather-insensitive** part of the demand — commercial
    refrigeration and ventilation, which run irrespective of degree days. Carrying it separately is
    what keeps *coefficient* a genuine climate response: with a through-origin form that constant is
    forced onto the degree-day slope, which is unbounded where degree days approach zero (Ireland
    gets ~0.04 °C·d/yr under the Eurostat definition). The AC-growth multiplier is applied to the
    **weather term only**, since adoption scales the cooling response, not the refrigeration load.

    *corrective_ratio* (per country) scales the **total** — baseload *and* weather — after both are
    assembled. It is the optional level correction that pins modelled demand to an observed
    benchmark; being a per-country scalar it cancels in any ratio, so it shifts the level and leaves
    the climate response, the saturation and the inter-chain ordering untouched.
    """
    cols = [c for c in national_dd.columns if c in coefficient.index and np.isfinite(coefficient[c])]
    demand = national_dd[cols].mul(coefficient[cols], axis=1)

    if ac_growth is not None:
        years = demand.index.year
        if isinstance(ac_growth, pd.DataFrame):
            factor = ac_growth.copy()
            if isinstance(factor.index, pd.DatetimeIndex):
                factor.index = factor.index.year
            factor = factor.reindex(index=years, columns=cols).astype(float)
            factor.index = demand.index
            demand = demand.mul(factor.fillna(1.0))
        else:
            if callable(ac_growth):
                scalar = pd.Series([ac_growth(int(y)) for y in years], index=demand.index)
            else:
                scalar = pd.Series(years, index=demand.index).map(ac_growth).astype(float)
            demand = demand.mul(scalar, axis=0)

    if baseload is not None:
        shared = [c for c in cols if c in baseload.index and np.isfinite(baseload[c])]
        demand[shared] = demand[shared].add(baseload[shared], axis=1)

    if corrective_ratio is not None:
        shared = [c for c in cols if c in corrective_ratio.index
                  and np.isfinite(corrective_ratio[c])]
        demand[shared] = demand[shared].mul(corrective_ratio[shared], axis=1)
    return demand


#: Sector suffixes of the sector-split cooling coefficients.
_COOL_SECTORS = ("res", "srv")
#: Columns :func:`cooling_demand` needs; their absence means an older coefficient file.
COOLING_SPLIT_COLUMNS = (
    [f"{p}_cool_{s}" for s in _COOL_SECTORS for p in ("P", "B", "theta")]
    + ["cool_anchor_cdd", "corrective_ratio_cool"]
)


def cooling_adoption_factor(
    cdd_annual: pd.DataFrame,
    theta: float,
    smooth_years: int = 5,
    freeze_at: pd.Series | None = None,
) -> pd.DataFrame:
    """AC-adoption multiplier ``s(CDD) = 1 − exp(−θ·CDD)`` per country and year.

    *cdd_annual* is a ``(year × country)`` frame of annual degree days. The factor is evaluated on a
    **trailing mean** over *smooth_years*: an air-conditioner installed after a hot summer stays
    installed, so penetration tracks the sustained climate rather than a single year — and without
    the smoothing a steep saturation curve would amplify interannual degree-day swings into
    year-to-year noise.

    With *freeze_at* (a per-country Series of reference degree days, e.g. ``cool_anchor_cdd``) the
    factor is held constant at ``s(CDD_ref)`` — **frozen penetration**: cooling still grows with
    warming through the degree days themselves, but no *new* air-conditioning is installed. That is
    the coherent no-adoption counterfactual. (A refitted linear form is *not*: pooled across
    countries its slope is anchored by the already-saturated warm ones, so it over-states the cool
    countries and can sit spuriously above the adoption curve.)
    """
    if freeze_at is not None:
        frozen = 1.0 - np.exp(-theta * freeze_at.astype(float))
        return pd.DataFrame(
            {c: np.full(len(cdd_annual), frozen.get(c, np.nan)) for c in cdd_annual.columns},
            index=cdd_annual.index,
        )
    smoothed = cdd_annual.astype(float).rolling(smooth_years, min_periods=1).mean()
    return 1.0 - np.exp(-theta * smoothed)


def cooling_demand(
    national_cdd: pd.DataFrame,
    coefficients: pd.DataFrame,
    apply_corrective_ratio: bool = True,
    ac_growth: bool = True,
    smooth_years: int = 5,
) -> pd.DataFrame:
    """Daily national cooling demand [GW] from the sector-split saturating model.

    For each sector ``j`` (residential, service)::

        E_j(t) = k_c · [ B_j(c) + P_j(c) · CDD(t) · s_j(CDD_annual(t)) ]

    with ``P_j``/``B_j`` the per-country weather coefficient and baseload (population already folded
    in by :func:`~cordex2energy.conversion_energy.demand_coefficients.derive_cooling_coefficients`),
    ``s_j`` the saturating adoption share, and ``k_c`` the optional level anchor. The sectors are
    summed. Splitting them matters because the service sector dominates the cooling energy but is
    largely weather-insensitive and close to saturation, while the residential part carries the
    emerging climate signal — a lumped fit mixes the two and can lose the signal entirely.

    *national_cdd* is the **daily** population-weighted national CDD frame; the annual sums driving
    the adoption term are derived from it, so the model is self-consistent.

    * ``apply_corrective_ratio`` (default ``True``) applies ``corrective_ratio_cool``, pinning
      present-day demand to the observed JRC benchmark — the recommended form. Set ``False`` for the
      benchmark-free variant, which reproduces the *historical average* level instead and is the
      right choice where no benchmark exists.
    * ``ac_growth`` (default ``True``) lets the adoption share follow the projected climate. Set
      ``False`` to freeze penetration at each country's anchor climate (no new air-conditioning);
      an anchored run then still reproduces the benchmark exactly.
    """
    missing = [c for c in COOLING_SPLIT_COLUMNS if c not in coefficients.columns]
    if missing:
        raise ValueError(
            "coefficients lack the sector-split cooling columns "
            f"{missing} — regenerate them with derive_demand_coefficients.py"
        )
    annual = national_cdd.resample("YS").sum()
    annual.index = annual.index.year
    freeze = None if ac_growth else coefficients["cool_anchor_cdd"]
    ratio = coefficients["corrective_ratio_cool"] if apply_corrective_ratio else None

    total = None
    for sector in _COOL_SECTORS:
        theta = float(coefficients[f"theta_cool_{sector}"].dropna().iloc[0])
        factor = cooling_adoption_factor(annual, theta, smooth_years, freeze_at=freeze)
        part = degree_days_to_demand(
            national_cdd,
            coefficients[f"P_cool_{sector}"],
            ac_growth=factor,
            baseload=coefficients[f"B_cool_{sector}"],
            corrective_ratio=ratio,
        )
        total = part if total is None else total.add(part, fill_value=0.0)
    return total


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
