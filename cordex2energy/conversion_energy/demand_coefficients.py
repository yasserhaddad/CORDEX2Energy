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
    residential + tertiary), ``cool_gwh`` (space cooling, ≈ all-electric), indexed by
    year. Returns ``None`` if the country's files are absent (e.g. UK under 2023,
    Switzerland, Norway, non-EU).
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
    cool = _row(res, "Space cooling") + _row(ser, "Space cooling")

    out = pd.DataFrame({
        "heat_thermal_gwh": heat_thermal,
        "heat_electric_gwh": heat_electric,
        "cool_gwh": cool,
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
    ``P_gw_per_degc`` (primary), ``r2_origin``, ``slope_trend``, ``beta``,
    ``r2_trend``, ``n``.
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
           "beta": np.nan, "r2_trend": np.nan, "n": int(x.size)}
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
            "r2_trend": float(r2_trend), "n": int(x.size)}


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
# Gap-fill missing countries (Switzerland, Norway, non-EU)
# ---------------------------------------------------------------------------

_POWER_COLS = ["P_heat_thermal", "P_heat_elec", "P_cool"]


def fill_missing_coefficients(df: pd.DataFrame, world_bank_pop_path: Path) -> pd.DataFrame:
    """Add rows for countries JRC lacks, mirroring ``fill_missing_country_params``.

    Neighbour-mean / proxy for the rate-like columns; the power columns are scaled by
    the World-Bank 2020 population ratio. Switzerland ← Austria, Norway ← Sweden (as
    the reference ``compute_share_df`` does), plus the SE-European small states,
    Luxembourg, Ukraine, Cyprus and Malta if absent.
    """
    df = df.copy()
    pop = (
        pd.read_csv(world_bank_pop_path, skiprows=3)[["Country Name", "2020"]]
        .set_index("Country Name")["2020"]
    )
    wb_alias = {"North Macedonia": "North Macedonia", "Slovakia": "Slovak Republic"}

    def npop(country: str) -> float:
        return float(pop.loc[wb_alias.get(country, country)])

    def proxy(country: str, refs: list[str]):
        present = [r for r in refs if r in df.index]
        if not present:
            return
        df.loc[country] = df.loc[present].mean()
        ref_pop = float(pop.loc[[wb_alias.get(r, r) for r in present]].mean())
        df.loc[country, _POWER_COLS] *= npop(country) / ref_pop

    proxy("Switzerland", ["Austria"])
    proxy("Norway", ["Sweden"])
    neighbours = ["Italy", "Austria", "Hungary", "Romania", "Bulgaria", "Greece"]
    for country in ["Albania", "Bosnia and Herzegovina", "Kosovo", "Montenegro",
                    "North Macedonia", "Serbia"]:
        if country not in df.index:
            proxy(country, neighbours)
    if "Ukraine" not in df.index:
        proxy("Ukraine", ["Poland", "Slovakia", "Hungary", "Romania"])
    return df.sort_index()
