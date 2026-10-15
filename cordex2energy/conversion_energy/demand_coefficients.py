"""Derive per-country HDD/CDD → demand coefficients from measured energy data.

Instead of adopting demand.ninja's literature parameters, this fits **our own**
coefficients by regressing measured annual heating/cooling energy on
population-weighted annual degree days, following demand.ninja's regression form
(Staffell et al. 2023, eq. 11) but on annual data:

    E_heat[c, y] = P0 + s_heat[c] · HDD_annual[c, y] + β[c] · (y − y0) + ε
    E_cool[c, y] = P0 + s_cool[c] · CDD_annual[c, y] + β[c] · (y − y0) + ε

* **Energy** = annual space-heating / space-cooling final energy consumption from
  **JRC-IDEES** (residential ``RES_hh_fec`` + tertiary ``SER_hh_fec``, ktoe → GWh);
  edition ``"2015"`` (2000–2015) or ``"2023"`` (2000–2023) via *edition*. Heating is
  total across all carriers; the electric part is recovered via the JRC electrified
  share. Cooling is essentially all-electric.
* **Degree days** = population-weighted national annual HDD/CDD from the **CERRA**
  reanalysis (same grid as the CORDEX products, so the country weights are reused).

Slopes are converted to demand.ninja units ``P = slope / 24`` [GW/°C] (since
``P·HDD_daily`` is GW ⇒ annual GWh ``= P · HDD_annual · 24``). Coverage is EU-28;
countries JRC lacks (Switzerland, Norway, non-EU) are gap-filled with the same
neighbour/proxy/population-scaling as the parameter table.

This is the **single source of truth** shared with ``c2e_recipe`` (which re-exports it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cordex2energy.conversion_energy.demand_model import national_mean

#: 1 ktoe = 11.63 GWh.
KTOE_TO_GWH = 11.63

#: JRC-IDEES ISO codes that differ from the two-letter norm used elsewhere.
_JRC_ISO = {"Greece": "EL", "United Kingdom": "UK"}

#: Country name for each JRC ISO code (canonical demand.ninja names).
_ISO_TO_NAME = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France", "HR": "Croatia",
    "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden",
    "SI": "Slovenia", "SK": "Slovakia", "UK": "United Kingdom",
}


# ---------------------------------------------------------------------------
# JRC-IDEES energy series
# ---------------------------------------------------------------------------

def _row(df: pd.DataFrame, label: str) -> pd.Series:
    """Return one end-use row as a Series (first if duplicated, zeros if missing)."""
    if label not in df.index:
        return pd.Series(0.0, index=df.columns)
    sub = df.loc[label]
    return sub.iloc[0] if isinstance(sub, pd.DataFrame) else sub


def _jrc_files(jrc_dir: Path, iso: str, edition: str) -> tuple[Path, Path]:
    """Residential + Tertiary xlsx paths for one country and edition.

    Tries the per-country subfolder layout (``<dir>/<ISO>/JRC-IDEES-<ed>_…``, as
    JRC-IDEES-2023 ships and as the rearranged 2015 tree) and falls back to the flat
    layout (``<dir>/JRC-IDEES-<ed>_…``, the original 2015 dump) so both work.
    """
    sub_res = jrc_dir / iso / f"JRC-IDEES-{edition}_Residential_{iso}.xlsx"
    sub_ser = jrc_dir / iso / f"JRC-IDEES-{edition}_Tertiary_{iso}.xlsx"
    if sub_res.exists() and sub_ser.exists():
        return sub_res, sub_ser
    return (jrc_dir / f"JRC-IDEES-{edition}_Residential_{iso}.xlsx",
            jrc_dir / f"JRC-IDEES-{edition}_Tertiary_{iso}.xlsx")


def _year_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the year columns (drop trailing ``Code``/``Unnamed`` cols in 2023)."""
    return df[[c for c in df.columns if str(c).strip().isdigit()]]


def load_jrc_fec_series(jrc_dir: Path, name: str, edition: str = "2015") -> pd.DataFrame | None:
    """Annual space-heating / cooling final energy [GWh] for one country.

    *name* is a canonical country name; the JRC file uses the ISO code in
    :data:`_ISO_TO_NAME`. *edition* selects the JRC-IDEES vintage — ``"2015"``
    (years 2000–2015) or ``"2023"`` (years 2000–2023); both share the ``RES_hh_fec`` /
    ``SER_hh_fec`` sheets and row labels. Returns columns ``heat_thermal_gwh`` (all
    carriers), ``heat_electric_gwh`` (advanced + conventional electric heating,
    residential + tertiary), ``cool_gwh`` (space cooling, ≈ all-electric) and its
    per-sector parts ``cool_res_gwh`` / ``cool_srv_gwh``, indexed by year. Returns ``None``
    if the country's files are absent (e.g. UK under 2023, Switzerland, Norway, non-EU).
    """
    iso = next((i for i, n in _ISO_TO_NAME.items() if n == name), _JRC_ISO.get(name))
    if iso is None:
        return None
    res_f, ser_f = _jrc_files(jrc_dir, iso, edition)
    if not (res_f.exists() and ser_f.exists()):
        return None

    res = _year_cols(pd.read_excel(res_f, sheet_name="RES_hh_fec", index_col=0))
    ser = _year_cols(pd.read_excel(ser_f, sheet_name="SER_hh_fec", index_col=0))

    heat_thermal = _row(res, "Space heating") + _row(ser, "Space heating")
    heat_electric = (
        _row(res, "Advanced electric heating") + _row(res, "Conventional electric heating")
        + _row(ser, "Advanced electric heating") + _row(ser, "Conventional electric heating")
    )
    # Cooling is kept BOTH lumped (legacy consumers) and split by sector. The split matters because
    # the service sector dominates the cooling energy (58-97 % across the domain) but is largely
    # weather-insensitive baseload, while the residential part carries the emerging climate signal —
    # so the two saturate with degree days at very different rates and must be fitted separately.
    cool_res = _row(res, "Space cooling")
    cool_srv = _row(ser, "Space cooling")

    out = pd.DataFrame({
        "heat_thermal_gwh": heat_thermal,
        "heat_electric_gwh": heat_electric,
        "cool_gwh": cool_res + cool_srv,
        "cool_res_gwh": cool_res,
        "cool_srv_gwh": cool_srv,
    }) * KTOE_TO_GWH
    out.index = out.index.astype(int)
    out.index.name = "year"
    return out


# ---------------------------------------------------------------------------
# National annual degree days (from CERRA)
# ---------------------------------------------------------------------------

def national_annual_degree_days(
    ds_dd_daily: xr.Dataset,
    var: str,
    weights: xr.DataArray,
    names: list[str],
) -> pd.DataFrame:
    """Population-weighted national **annual-sum** degree days [°C·day/yr].

    Annual-sums the daily field first (cheap), then applies the precomputed country
    weights via :func:`cordex2energy.conversion_energy.demand_model.national_mean`.
    """
    annual = ds_dd_daily[var].resample(time="YS").sum()
    df = national_mean(annual, weights, names)
    df.index = df.index.year
    df.index.name = "year"
    return df


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regress_coefficient(
    dd_annual: pd.Series,
    energy_annual: pd.Series,
    years: set | None = None,
    method: str = "ls",
) -> dict:
    """Fit a degree-day → energy coefficient for one country, two ways.

    * **Primary (usable) coefficient** — through-origin ``P·DD`` matching
      demand.ninja's demand form (``E = P·DD``, no base). ``method="ls"`` (default)
      uses the least-squares slope ``s0 = Σ(DD·E)/Σ(DD²)``; ``method="ratio"`` uses the
      total-preserving ratio ``s0 = ΣE/ΣDD`` (matches the annual energy exactly over
      the window — the recent-window estimator). Both give ``P_gw_per_degc = s0/24``.
    * **Diagnostic** — the eq.11 trend fit ``E = P0 + s·DD + β·(year−y0)``. ``β`` is
      the population+efficiency (for cooling: adoption) growth, ``slope_trend`` the
      weather sensitivity *net of* the trend. On annual data with a strong adoption
      trend the cooling ``slope_trend`` is weakly identified and can go negative — a
      finding, not a coefficient to use.

    *years* optionally restricts the fit to a subset (e.g. a recent window). Returns
    ``P_gw_per_degc`` (primary), ``r2_origin``, ``slope_trend``, ``beta``, ``intercept``,
    ``r2_trend``, ``n``.

    ``intercept`` is the eq.11 fit's constant term, in the units of *energy_annual* (GWh/yr). For
    cooling it is the **weather-insensitive** part of the series — commercial refrigeration and
    ventilation, which do not follow degree days. It is large in cool countries (≈100 % of mean
    cooling energy for the UK and Ireland), which is precisely why the through-origin
    ``P_gw_per_degc`` degenerates there: with no intercept, that constant is forced onto the
    degree-day slope.
    """
    idx = dd_annual.index.intersection(energy_annual.index)
    if years is not None:
        idx = idx[idx.isin(set(years))]
    x = dd_annual.loc[idx].to_numpy(float)
    y = energy_annual.loc[idx].to_numpy(float)
    yr = np.asarray(idx, float) - float(min(idx)) if len(idx) else np.array([])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, yr = x[ok], y[ok], yr[ok]
    nan = {"P_gw_per_degc": np.nan, "r2_origin": np.nan, "slope_trend": np.nan,
           "beta": np.nan, "r2_trend": np.nan, "intercept": np.nan, "n": int(x.size)}
    if x.size < 3 or np.allclose(x, 0):
        return nan

    # primary: through the origin
    s0 = float(y.sum() / x.sum()) if method == "ratio" else float((x * y).sum() / (x * x).sum())
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_origin = 1.0 - ((y - s0 * x) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

    # diagnostic: eq.11 trend fit
    X = np.column_stack([np.ones_like(x), x, yr])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_trend = 1.0 - ((y - X @ coef) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

    return {"P_gw_per_degc": s0 / 24.0, "r2_origin": float(r2_origin),
            "slope_trend": float(coef[1]) / 24.0, "beta": float(coef[2]),
            "intercept": float(coef[0]), "r2_trend": float(r2_trend), "n": int(x.size)}


def predict_trend(dd_annual: pd.Series, energy_annual: pd.Series) -> pd.Series:
    """Fitted values of the eq.11 trend model ``E = P0 + s·DD + β·(year−y0)``.

    Returns the fitted energy series (same units as *energy_annual*), indexed by
    year over the overlapping, finite points — for overlaying the with-trend fit
    against the through-origin ``P·DD`` line in the fit diagnostic.
    """
    idx = dd_annual.index.intersection(energy_annual.index)
    x = dd_annual.loc[idx].to_numpy(float)
    y = energy_annual.loc[idx].to_numpy(float)
    yr = np.asarray(idx, float) - float(min(idx))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return pd.Series(np.nan, index=idx)
    X = np.column_stack([np.ones(ok.sum()), x[ok], yr[ok]])
    coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    return pd.Series(coef[0] + coef[1] * x + coef[2] * yr, index=idx)


def annual_ratio_stats(
    dd_annual: pd.Series,
    energy_annual: pd.Series,
    years: set | None = None,
) -> dict:
    """Robustness diagnostics for the total-preserving ratio coefficient.

    Computes the pooled ratio ``P_ratio = ΣE / (24·ΣDD)`` [GW/°C] over the (optionally
    windowed) years, the **interannual CV** of the per-year ratios ``E_y/(24·DD_y)``
    (how stationary the intensity is — high CV flags a weak/adoption-driven signal),
    and a **leave-one-year-out** relative RMSE (predict each held-out year's energy from
    the pooled ratio of the rest). Returns ``P_ratio``, ``cv``, ``loyo_rmse``, ``n``.
    """
    idx = dd_annual.index.intersection(energy_annual.index)
    if years is not None:
        idx = idx[idx.isin(set(years))]
    x = dd_annual.loc[idx].to_numpy(float)
    y = energy_annual.loc[idx].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return {"P_ratio": np.nan, "cv": np.nan, "loyo_rmse": np.nan, "n": int(x.size)}

    per_year = (y / x) / 24.0
    p_ratio = float(y.sum() / x.sum()) / 24.0
    cv = float(np.std(per_year) / np.mean(per_year)) if np.mean(per_year) != 0 else np.nan
    errs = []
    for i in range(x.size):
        m = np.ones(x.size, bool)
        m[i] = False
        pred = float(y[m].sum() / x[m].sum()) * x[i]
        if y[i] != 0:
            errs.append((pred - y[i]) / y[i])
    loyo = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
    return {"P_ratio": p_ratio, "cv": cv, "loyo_rmse": loyo, "n": int(x.size)}


def analog_fill_cooling(
    p_cool: pd.Series,
    mean_cdd: pd.Series,
    size: pd.Series,
    reliability: pd.Series,
    threshold: float = 0.5,
) -> tuple[pd.Series, list]:
    """Replace low-reliability cooling coefficients with a climate + size analog.

    Countries whose *reliability* metric (e.g. leave-one-year-out RMSE) exceeds
    *threshold* take the ``P_cool`` of the nearest **reliable** country in mean-daily-CDD
    space, scaled by the *size* ratio (population, or any size proxy) so the borrowed
    intensity matches the country's scale. NOTE: validated in the notebook to *distort*
    tiny-cooling states (population-scaled climatic analog is a poor cooling estimator) —
    kept as a documented utility but **not** used in the production coefficient build.
    Returns the filled series and the list of refilled countries.
    """
    out = p_cool.copy()
    unreliable = [c for c in p_cool.index
                  if c in reliability.index and np.isfinite(reliability[c])
                  and reliability[c] > threshold and np.isfinite(p_cool.get(c, np.nan))]
    reliable = [c for c in p_cool.index
                if c not in unreliable and np.isfinite(p_cool.get(c, np.nan))
                and c in mean_cdd.index and c in size.index]
    for c in unreliable:
        if c not in mean_cdd.index or c not in size.index or not reliable:
            continue
        analog = min(reliable, key=lambda r: abs(float(mean_cdd[r]) - float(mean_cdd[c])))
        out[c] = float(p_cool[analog]) * float(size[c]) / float(size[analog])
    return out, unreliable


def derive_coefficients(
    jrc_dir: Path,
    national_hdd: pd.DataFrame,
    national_cdd: pd.DataFrame,
    edition: str = "2015",
    years: set | None = None,
    method: str = "ls",
    fallback: tuple[Path, str] | None = None,
) -> pd.DataFrame:
    """Fit heating/cooling coefficients for every country JRC covers.

    Heating is regressed on **total thermal** FEC → ``P_heat_thermal`` (through-origin,
    GW/°C); the electric coefficient ``P_heat_elec = share · P_heat_thermal`` where
    ``share`` is the electrified space-heating ratio in the **latest available** JRC
    year. Cooling FEC (≈ electric) gives ``P_cool``. The columns ``beta_*`` and
    ``slope_trend_*`` carry the eq.11 trend diagnostic. *edition* picks the JRC vintage;
    *years* restricts to a window (e.g. recent); *method* is passed to
    :func:`regress_coefficient` (``"ls"`` or ``"ratio"``); *fallback* is an optional
    ``(dir, edition)`` tried per country when the primary edition lacks it (e.g. UK
    under 2023 → 2015). Returns a table indexed by country name.
    """
    rows = {}
    for name in sorted(set(national_hdd.columns) & set(_ISO_TO_NAME.values())):
        fec = load_jrc_fec_series(jrc_dir, name, edition)
        if fec is None and fallback is not None:
            fec = load_jrc_fec_series(fallback[0], name, fallback[1])
        if fec is None:
            continue
        h = regress_coefficient(national_hdd[name], fec["heat_thermal_gwh"], years, method)
        c = regress_coefficient(national_cdd[name], fec["cool_gwh"], years, method)
        base_year = int(fec.index.max())
        e_thermal = fec.loc[base_year, "heat_thermal_gwh"]
        share = float(fec.loc[base_year, "heat_electric_gwh"] / e_thermal) if e_thermal > 0 else np.nan
        rows[name] = {
            "P_heat_thermal": h["P_gw_per_degc"],
            "electrified_share": share,
            "P_heat_elec": h["P_gw_per_degc"] * share,
            "P_cool": c["P_gw_per_degc"],
            "r2_heat": h["r2_origin"], "r2_cool": c["r2_origin"],
            "beta_heat": h["beta"], "beta_cool": c["beta"],
            "slope_trend_cool": c["slope_trend"], "n": h["n"],
        }
    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# Sector-split cooling: pooled "space-for-time" saturation fit
# ---------------------------------------------------------------------------
# Cooling is fitted per sector in PER-CAPITA space, pooled across countries:
#
#     e_pc(c, y) = b_j(c) + p0_j · CDD(c,y)·s_j(CDD(c,y)) + β_j·(y − ȳ)
#     s_j(x)     = 1 − exp(−θ_j·x)
#
# `θ_j` (curvature) and `p0_j` (intensity) are **shared across countries** — the
# space-for-time assumption: warm countries today stand in for cool countries' future. This is
# what makes the curvature identifiable at all, since within a single country CDD was essentially
# flat over 2000-2020 and adoption showed up only as the year trend. `b_j(c)` is a per-country
# weather-insensitive baseload; `β_j` exists solely to de-bias the slope and is **never projected**
# (a linear year ramp is unbounded and climate-blind).
#
# The saturation sits INSIDE the level equation (the "absolute" form) rather than being applied as
# a multiplier s(CDD)/s(CDD_ref). That removes the reference denominator, which is what blows up
# where CDD_ref → 0 (Ireland ≈ 0.05 °C·d/yr) — with a shared intensity there is no per-country
# denominator anywhere, so boundedness is structural instead of patched.

#: Fit window. CERRA degree days end 2021-06-30, so 2021 is a partial year and is dropped.
COOL_FIT_YEARS = range(2000, 2021)
#: Window whose observed JRC total the anchor reproduces (see :func:`cooling_anchor_window`).
COOL_ANCHOR_WINDOW = (2015, 2020)
#: The UK is the only country on the JRC-2015 edition, which stops in 2015.
COOL_ANCHOR_OVERRIDES = {"United Kingdom": (2010, 2015)}
#: Candidate curvatures profiled when fitting ``θ``.
COOL_THETA_GRID = np.geomspace(1e-5, 0.5, 500)
#: Heat Roadmap Europe's published curvatures, for reference in diagnostics.
HRE_THETA = {"res": 0.00225, "srv": 0.020}
#: Minimum pooled R² for a candidate to be eligible in :func:`select_cooling_curvature`.
COOL_R2_MIN = 0.85
#: Trailing window (years) over which degree days are averaged before evaluating ``s(CDD)``.
COOL_SMOOTH_YEARS = 5

_COOL_SECTORS = {"res": "cool_res_gwh", "srv": "cool_srv_gwh"}


def cooling_anchor_window(country: str) -> tuple[int, int]:
    """Inclusive ``(first, last)`` year whose observed JRC cooling the anchor matches."""
    return COOL_ANCHOR_OVERRIDES.get(country, COOL_ANCHOR_WINDOW)


def _cooling_panel(cool_pc: pd.DataFrame, cdd_annual: pd.DataFrame) -> pd.DataFrame:
    """Long ``(country, year, e_pc, cdd)`` panel of the aligned, finite observations."""
    idx = cool_pc.index.intersection(cdd_annual.index)
    frames = []
    for c in [c for c in cool_pc.columns if c in cdd_annual.columns]:
        sub = pd.DataFrame({"e_pc": cool_pc.loc[idx, c], "cdd": cdd_annual.loc[idx, c]})
        sub = sub[np.isfinite(sub["e_pc"]) & np.isfinite(sub["cdd"])]
        sub["country"], sub["year"] = c, sub.index.astype(float)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_pooled_cooling(
    cool_pc: pd.DataFrame,
    cdd_annual: pd.DataFrame,
    theta: float | None = None,
    fixed_effects: bool = True,
    theta_grid: np.ndarray | None = None,
) -> dict:
    """Pooled space-for-time fit of one sector's cooling curvature and intensity.

    *cool_pc* and *cdd_annual* are ``(year × country)`` frames of per-capita cooling energy
    [GWh/yr per million capita] and annual degree days [°C·day/yr]. For a given ``θ`` the model is
    linear, so it is solved by least squares over ``[country effects | CDD·s(CDD) | year − ȳ]``.
    When *theta* is ``None`` the curvature is estimated by profiling the residual sum of squares
    over *theta_grid*; pass a value to fit the remaining terms at a fixed curvature.
    *fixed_effects* selects a per-country baseload (``True``) or a single shared constant.

    Returns ``theta``, ``p0`` (shared intensity), ``b`` (per-country baseload Series, clamped at
    ≥0 — a negative baseload is an unidentified term, not a physical result), ``r2``, ``width``
    (how well ``θ`` is identified: the ratio of the largest to the smallest curvature whose RSS is
    within 1 % of the minimum) and ``n``.

    **``θ`` is a calibrated choice, not a sharp estimate.** The RSS profile is flat — ``width`` is
    typically large, and HRE's published curvature costs only a few percent more RSS. In-sample fit
    therefore cannot discriminate between saturating and near-linear adoption, even though they
    imply very different end-century demand: a structural uncertainty to report, not something a
    better regression settles. :func:`select_cooling_curvature` turns that into a stated rule.
    """
    panel = _cooling_panel(cool_pc, cdd_annual)
    if len(panel) < 10:
        return {"theta": np.nan, "p0": np.nan, "b": pd.Series(dtype=float),
                "r2": np.nan, "width": np.inf, "n": 0}

    if fixed_effects:
        effects = pd.get_dummies(panel["country"], dtype=float)
    else:
        effects = pd.DataFrame({"const": np.ones(len(panel))}, index=panel.index)
    year_c = panel["year"].to_numpy(float) - panel["year"].to_numpy(float).mean()
    cdd = panel["cdd"].to_numpy(float)
    y = panel["e_pc"].to_numpy(float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    n_fe = effects.shape[1]

    def solve(candidate: float):
        weather = cdd * (1.0 - np.exp(-candidate * cdd))
        design = np.column_stack([effects.to_numpy(), weather, year_c])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coef, float(((y - design @ coef) ** 2).sum())

    grid = COOL_THETA_GRID if theta_grid is None else np.asarray(theta_grid, float)
    rss = np.array([solve(float(t))[1] for t in grid])
    best = int(np.argmin(rss))
    within = grid[rss <= rss[best] * 1.01]
    width = float(within.max() / max(within.min(), 1e-12))
    chosen = float(grid[best]) if theta is None else float(theta)
    coef, chosen_rss = solve(chosen)
    if fixed_effects:
        baseload = pd.Series(coef[:n_fe], index=effects.columns)
    else:
        baseload = pd.Series(float(coef[0]), index=sorted(panel["country"].unique()))
    return {
        "theta": chosen,
        "p0": float(coef[n_fe]),
        "b": baseload.clip(lower=0.0),
        "r2": 1.0 - chosen_rss / ss_tot if ss_tot > 0 else np.nan,
        "width": width,
        "n": len(panel),
    }


def cooling_country_groups(cdd_annual: pd.DataFrame, cool_twh: pd.Series) -> dict[str, list[str]]:
    """Candidate country subsets for the curvature selection.

    Curvature is only visible where a country actually cools, so alongside the full set the
    selection also considers warmer / higher-consumption subsets: countries with essentially no
    air-conditioning contribute observations clustered near ``CDD ≈ 0`` that carry no information
    about how the response bends, and can flatten the profile.
    """
    mean_cdd = cdd_annual.mean()
    everyone = [c for c in cdd_annual.columns if c in cool_twh.index]
    return {
        "all": everyone,
        "CDD>50": [c for c in everyone if mean_cdd.get(c, 0.0) > 50],
        "CDD>100": [c for c in everyone if mean_cdd.get(c, 0.0) > 100],
        "cooling>1TWh": [c for c in everyone if cool_twh.get(c, 0.0) > 1.0],
    }


def select_cooling_curvature(
    cool_pc: pd.DataFrame,
    cdd_annual: pd.DataFrame,
    groups: dict[str, list[str]],
    r2_min: float = COOL_R2_MIN,
    theta_grid: np.ndarray | None = None,
) -> dict:
    """Choose ``θ`` by a stated, reproducible identification rule.

    Because the RSS profile is flat (see :func:`fit_pooled_cooling`), the minimum alone is not a
    defensible estimate. Every combination of country *group* and specification (per-country
    baseloads vs a single constant) is profiled, and among those whose pooled ``R² ≥ r2_min`` the
    one whose curvature is **best identified** — the narrowest 1 %-RSS profile — is taken. Stated up
    front and reproducible, rather than hand-picked.

    Returns the winning ``theta`` plus its provenance (``group``, ``spec``, ``width``, ``r2``) and
    the full candidate table under ``candidates``.
    """
    candidates, best = {}, None
    for name, members in groups.items():
        present = [c for c in members if c in cool_pc.columns]
        if len(present) < 3:
            continue
        for fixed_effects in (True, False):
            fit = fit_pooled_cooling(cool_pc[present], cdd_annual[present],
                                     fixed_effects=fixed_effects, theta_grid=theta_grid)
            spec = "FE" if fixed_effects else "noFE"
            candidates[(name, spec)] = fit
            eligible = np.isfinite(fit["r2"]) and fit["r2"] >= r2_min
            if eligible and (best is None or fit["width"] < best["width"]):
                best = {"theta": fit["theta"], "group": name, "spec": spec,
                        "width": fit["width"], "r2": fit["r2"]}
    if best is None:                     # nothing clears r2_min: fall back to the full-set FE fit
        fit = fit_pooled_cooling(cool_pc, cdd_annual, theta_grid=theta_grid)
        best = {"theta": fit["theta"], "group": "all", "spec": "FE",
                "width": fit["width"], "r2": fit["r2"]}
    best["candidates"] = candidates
    return best


def derive_cooling_coefficients(
    jrc_dir: Path,
    national_cdd: pd.DataFrame,
    population: pd.Series,
    edition: str = "2023",
    fallback: tuple[Path, str] | None = None,
    countries: list[str] | None = None,
    smooth_years: int = COOL_SMOOTH_YEARS,
    diagnostics: dict | None = None,
) -> pd.DataFrame:
    """Per-country sector-split cooling coefficients, in production units.

    *national_cdd* is the ``(year × country)`` annual degree-day frame (from
    :func:`national_annual_degree_days`); *population* is a per-country head count (persons).
    Fits both sectors with :func:`fit_pooled_cooling`, gap-fills the countries JRC does not cover
    (donor means from :data:`_PROXY_GROUPS`, taken in **per-capita** space so no population scaling
    is needed), and returns the columns
    :func:`~cordex2energy.conversion_energy.demand_model.cooling_demand` consumes:

    ``P_cool_res`` / ``P_cool_srv``
        weather coefficient ``pop·p0/24`` [GW per °C·day].
    ``B_cool_res`` / ``B_cool_srv``
        weather-insensitive baseload ``pop·b/8760`` [GW].
    ``theta_cool_res`` / ``theta_cool_srv``
        the fitted curvatures — global constants, repeated on every row so the coefficient CSV
        remains the single interface.
    ``cool_anchor_cdd``
        annual CDD over the country's anchor window, needed to freeze adoption when it is off.
    ``corrective_ratio_cool``
        ``E_JRC(anchor) / E_model(anchor)`` — the optional level correction (M7 vs M7r).
    ``cool_gap_filled``
        1 where the country has no JRC data and the coefficients were borrowed.

    Population is folded into ``P``/``B`` here, so the apply side needs no population data.
    *smooth_years* must match the value used at apply time, otherwise the anchor no longer
    reproduces the benchmark exactly. Pass a dict as *diagnostics* to receive the curvature
    provenance and the per-sector fits.
    """
    names = list(national_cdd.columns) if countries is None else list(countries)
    fec = {}
    for name in names:
        series = load_jrc_fec_series(jrc_dir, name, edition)
        if series is None and fallback is not None:
            series = load_jrc_fec_series(fallback[0], name, fallback[1])
        if series is not None:
            fec[name] = series
    if not fec:
        return pd.DataFrame()

    years = [y for y in COOL_FIT_YEARS if y in national_cdd.index]
    pop_m = {c: float(population.loc[c]) / 1e6 for c in names if c in population.index}
    fitted = sorted(c for c in fec if c in national_cdd.columns and c in pop_m)

    cdd_annual = national_cdd.loc[years, fitted]
    cool_twh = pd.Series(
        {c: sum(fec[c][v].reindex(years).mean() for v in _COOL_SECTORS.values()) / 1e3
         for c in fitted}
    )
    groups = cooling_country_groups(cdd_annual, cool_twh)

    # Two stages, as validated in the notebook: the curvature is chosen by the identification rule
    # (which may favour a warm subset, where the bend is actually visible), then the intensity and
    # the per-country baseloads are fitted on ALL countries with fixed effects at that curvature,
    # so every country gets its own baseload and the shared intensity spans the full domain.
    fits, chosen = {}, {}
    for sector, col in _COOL_SECTORS.items():
        cool_pc = pd.DataFrame(
            {c: fec[c][col].reindex(years) / pop_m[c] for c in fitted}, index=years
        )
        chosen[sector] = select_cooling_curvature(cool_pc, cdd_annual, groups)
        fits[sector] = fit_pooled_cooling(cool_pc, cdd_annual, theta=chosen[sector]["theta"])
    if diagnostics is not None:
        diagnostics.update({"chosen": chosen, "fits": fits, "groups": groups})

    # Baseloads for countries JRC lacks: donor mean of the PER-CAPITA value. Taken per capita, the
    # fill needs no population ratio — multiplying by the country's own population below reproduces
    # `parameter_fill_ninja`'s `mean × N/N_ref` exactly.
    baseload = {s: fits[s]["b"].copy() for s in _COOL_SECTORS}
    gap_filled = []
    for country in names:
        if country in fitted or country not in pop_m:
            continue
        donors = [d for d in _PROXY_GROUPS.get(country, []) if d in fitted]
        if not donors:
            continue
        for sector in _COOL_SECTORS:
            baseload[sector][country] = float(baseload[sector][donors].mean())
        gap_filled.append(country)

    # The adoption share is evaluated on a trailing mean at apply time, so the anchor must use the
    # same smoothing — otherwise `s` differs between derivation and application and the anchored
    # model no longer reproduces the benchmark exactly.
    smoothed = national_cdd.astype(float).rolling(smooth_years, min_periods=1).mean()

    def model_window_pc(country: str, window: list[int]) -> float:
        """Per-capita modelled cooling [GWh/yr per Mcap] averaged over the anchor window."""
        total = 0.0
        for sector in _COOL_SECTORS:
            share = 1.0 - np.exp(-fits[sector]["theta"] * smoothed.loc[window, country])
            weather = (national_cdd.loc[window, country] * share).mean()
            total += float(baseload[sector].get(country, np.nan)) + fits[sector]["p0"] * weather
        return float(total)

    rows = {}
    for country in names:
        if country not in pop_m or country not in national_cdd.columns:
            continue
        if not all(np.isfinite(baseload[s].get(country, np.nan)) for s in _COOL_SECTORS):
            continue
        first, last = cooling_anchor_window(country)
        window = [y for y in range(first, last + 1) if y in national_cdd.index]
        cdd_a = float(smoothed.loc[window, country].mean())
        row = {"cool_anchor_cdd": cdd_a,
               "cool_gap_filled": float(country in gap_filled)}
        for sector in _COOL_SECTORS:
            row[f"P_cool_{sector}"] = pop_m[country] * fits[sector]["p0"] / 24.0
            row[f"B_cool_{sector}"] = pop_m[country] * float(baseload[sector][country]) / 8760.0
            row[f"theta_cool_{sector}"] = fits[sector]["theta"]
        if country in fec:
            observed = sum(fec[country][c].reindex(window).mean() for c in _COOL_SECTORS.values())
            modelled = model_window_pc(country, window) * pop_m[country]
            row["corrective_ratio_cool"] = (float(observed) / modelled if modelled > 0 else np.nan)
        else:
            row["corrective_ratio_cool"] = np.nan
        rows[country] = row

    out = pd.DataFrame(rows).T
    # Countries without a benchmark cannot have their own anchor: borrow the donor group's mean.
    ratio = out["corrective_ratio_cool"]
    for country in gap_filled:
        donors = [d for d in _PROXY_GROUPS.get(country, []) if d in ratio.index]
        finite = [d for d in donors if np.isfinite(ratio.get(d, np.nan))]
        if finite:
            out.loc[country, "corrective_ratio_cool"] = float(ratio[finite].mean())
    out["corrective_ratio_cool"] = out["corrective_ratio_cool"].fillna(1.0)
    return out.sort_index()


# ---------------------------------------------------------------------------
# Gap-fill missing countries (Switzerland, Norway, non-EU)
# ---------------------------------------------------------------------------

_POWER_COLS = ["P_heat_thermal", "P_heat_elec", "P_cool"]

#: World-Bank population spellings that differ from the canonical country names.
WB_POP_ALIAS = {
    "Czech Republic": "Czechia",
    "Slovakia": "Slovak Republic",
}

#: Donor countries for every state JRC-IDEES does not cover, shared by all gap-fills.
#: Switzerland/Norway follow ``_DEFAULT_NON_EU_PROXIES`` in ``cordex2energy_analysis.power_system``
#: (alpine / Nordic analogues);
#: the SE-European group and Ukraine follow ``parameter_fill_ninja``'s neighbour means.
_PROXY_GROUPS: dict[str, list[str]] = {
    "Switzerland": ["Austria"],
    "Norway": ["Sweden"],
    "Ukraine": ["Poland", "Slovakia", "Hungary", "Romania"],
    **{c: ["Italy", "Austria", "Hungary", "Romania", "Bulgaria", "Greece"]
       for c in ["Albania", "Bosnia and Herzegovina", "Kosovo", "Montenegro",
                 "North Macedonia", "Serbia"]},
}


def world_bank_population(world_bank_pop_path: Path, year: str = "2020") -> pd.Series:
    """World-Bank population indexed by **canonical** country name (aliases resolved)."""
    pop = (
        pd.read_csv(world_bank_pop_path, skiprows=3)[["Country Name", year]]
        .set_index("Country Name")[year]
    )
    inverse = {v: k for k, v in WB_POP_ALIAS.items()}
    return pop.rename(index=lambda n: inverse.get(n, n))


def fill_missing_coefficients(df: pd.DataFrame, world_bank_pop_path: Path) -> pd.DataFrame:
    """Add rows for countries JRC lacks, mirroring ``fill_missing_country_params``.

    Neighbour-mean / proxy for the rate-like columns; the power columns are scaled by
    the World-Bank 2020 population ratio. Donors come from :data:`_PROXY_GROUPS`
    (Switzerland ← Austria, Norway ← Sweden, the SE-European small states and Ukraine).
    """
    df = df.copy()
    pop = world_bank_population(world_bank_pop_path)

    def proxy(country: str, refs: list[str]):
        present = [r for r in refs if r in df.index]
        if not present:
            return
        df.loc[country] = df.loc[present].mean()
        power = [c for c in _POWER_COLS if c in df.columns]
        df.loc[country, power] *= float(pop.loc[country]) / float(pop.loc[present].mean())

    proxy("Switzerland", _PROXY_GROUPS["Switzerland"])
    proxy("Norway", _PROXY_GROUPS["Norway"])
    for country, refs in _PROXY_GROUPS.items():
        if country not in ("Switzerland", "Norway") and country not in df.index:
            proxy(country, refs)
    return df.sort_index()
