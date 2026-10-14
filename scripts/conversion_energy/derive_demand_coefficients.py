"""Derive per-country demand coefficients (one-time) and write demand_coefficients.csv.

Regresses **JRC-IDEES-2023** annual space-heating/cooling final energy on population-weighted
**CERRA** degree days to produce the coefficients used by ``compute_demand.py``:

* **Heating** ``P_heat_elec`` = through-origin (least-squares) coefficient over the **full**
  window × the latest-year electrified share (weather-stationary; the full window is most
  robust and electrification recency is carried by the share).
* **Cooling** is produced in three forms, newest first. ``compute_demand.py`` prefers the
  **sector-split saturating** one: residential and service fitted separately in per-capita space
  with a *shared* intensity and a saturating adoption share ``s(CDD)=1−exp(−θ·CDD)`` inside the
  level equation, plus ``corrective_ratio_cool`` pinning present-day demand to observed JRC.
  Kept alongside for older consumers: the intercept form (``B_cool`` + ``P_cool_weather``) and the
  legacy through-origin ratio ``P_cool`` = ``ΣE/(24·ΣDD)`` over the **recent window 2011–2020**
  (CERRA 2021 is a partial year and is dropped), with ``cool_LOYO`` as its uncertainty flag. No
  analog-fill (validated to distort tiny-cooling states).

Countries JRC lacks (Switzerland, Norway, non-EU) are gap-filled by neighbour/population proxy.
UK (absent from the 2023 edition) falls back to the 2015 edition.

Population weighting uses the **GPW v4** raster (matching the aggregation), not the WorldPop
raster the degree-days script uses.

Usage:
    python derive_demand_coefficients.py <paths_file>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from cordex2energy.conversion_energy import demand_coefficients as dc
from cordex2energy.conversion_energy import demand_model as dm
from cordex2energy.utils.constants import DEGREE_DAYS_DIRNAME

#: Heating uses the full window; cooling the recent window (CERRA 2021 dropped).
COOL_WINDOW = set(range(2011, 2021))
#: Hot-year window (incl. the 2003 heatwave) for countries on the 2015 fallback edition.
#: UK is absent from JRC-2023, and its recent window (2011–2015) was anomalously cool, so
#: the near-zero-CDD ratio blows up (P_cool 13.9 → end-century cooling above FR/DE). A
#: 2003-inclusive window gives a representative CDD signal (P_cool ~6.5).
COOL_WINDOW_FALLBACK = set(range(2003, 2008))
#: Window for the INTERCEPT fit. The full record is used rather than a short recent window: the
#: intercept + trend form needs enough interannual variation to separate the weather slope from the
#: adoption trend, and a 10-year window leaves them collinear (corr(year, CDD) ~0.8 for FR/NL/BE/DE
#: over 2011-2020, vs ~0.35 over 2000-2020). CERRA degree days end 2021-06-30, so 2021 is dropped.
COOL_WINDOW_INTERCEPT = set(range(2000, 2021))
DD_SLICE = slice("2000", "2020")
POP_RASTER = "gpw_v4/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"
OUTPUT_COLS = ["P_heat_elec", "P_cool", "B_cool", "P_cool_weather", "beta_cool", "r2_cool_trend",
               "cool_LOYO", "electrified_share", "P_heat_thermal", "n",
               # sector-split cooling (the form compute_demand.py prefers); the legacy columns
               # above are kept so older coefficient consumers keep working.
               "P_cool_res", "B_cool_res", "theta_cool_res",
               "P_cool_srv", "B_cool_srv", "theta_cool_srv",
               "cool_anchor_cdd", "corrective_ratio_cool", "cool_gap_filled"]


def parse_arguments():
    p = argparse.ArgumentParser(description="Derive JRC-based demand coefficients.")
    p.add_argument("paths_file", type=str, help="Path to the JSON file containing paths.")
    return p.parse_args()


def load_population_on_grid(pop_path: Path, ref: xr.Dataset) -> xr.DataArray:
    """GPW v4 raster clipped to *ref* and coarsened onto its grid."""
    da = rxr.open_rasterio(pop_path, masked=True).rio.clip_box(
        minx=float(ref.lon.min()), miny=float(ref.lat.min()),
        maxx=float(ref.lon.max()), maxy=float(ref.lat.max()),
    )
    return dm.population_on_grid(da, ref)


def main():
    args = parse_arguments()
    paths = json.load(open(args.paths_file))
    root = Path(paths["path_data_processed"])
    energy_inputs = root / "energy" / "inputs"
    jrc23, jrc15 = energy_inputs / "JRC_2023", energy_inputs / "JRC_2015"
    pop_path = root / "pop" / POP_RASTER
    dd_zarr = root / "energy" / "europe" / DEGREE_DAYS_DIRNAME / "cerra.zarr"
    out_csv = energy_inputs / "demand_coefficients.csv"

    print("Loading CERRA degree days and GPW v4 population …")
    ds = xr.open_zarr(dd_zarr).squeeze(drop=True).sel(time=DD_SLICE)
    pop = load_population_on_grid(pop_path, ds)
    weights, names = dm.build_country_weights(ds["HDD"].isel(time=0), pop)
    nat_hdd = dc.national_annual_degree_days(ds, "HDD", weights, names)
    nat_cdd = dc.national_annual_degree_days(ds, "CDD", weights, names)

    print("Fitting heating (full window, LS) + electrified share …")
    coefs = dc.derive_coefficients(jrc23, nat_hdd, nat_cdd, edition="2023", fallback=(jrc15, "2015"))

    print("Fitting cooling: ratio (legacy) + intercept form B + P·CDD + β·year …")
    for name in coefs.index:
        fec, win = dc.load_jrc_fec_series(jrc23, name, "2023"), COOL_WINDOW
        if fec is None:  # fell back to 2015 (e.g. UK): recent years absent/anomalously cool
            fec, win = dc.load_jrc_fec_series(jrc15, name, "2015"), COOL_WINDOW_FALLBACK
        if fec is None or name not in nat_cdd.columns:
            continue
        st = dc.annual_ratio_stats(nat_cdd[name], fec["cool_gwh"], years=win)
        coefs.loc[name, "P_cool"] = st["P_ratio"]
        coefs.loc[name, "cool_LOYO"] = st["loyo_rmse"]

        # The intercept form. `P_cool` alone cannot work under the Eurostat degree-day definition:
        # it is a through-origin ratio, so a country with near-zero CDD (Ireland ~0.04 °C·d/yr) gets
        # an unbounded slope. Splitting off the weather-INsensitive part (commercial refrigeration
        # and ventilation, which do not follow degree days) is what makes the remaining slope a
        # genuine climate response. Both power terms are clamped at >=0: a negative baseload or a
        # negative cooling response is the fit failing to identify a term, not a physical result.
        fit = dc.regress_coefficient(nat_cdd[name], fec["cool_gwh"],
                                     years=COOL_WINDOW_INTERCEPT, method="ls")
        coefs.loc[name, "B_cool"] = max(fit["intercept"], 0.0) / 8760.0        # GWh/yr -> GW
        coefs.loc[name, "P_cool_weather"] = max(fit["slope_trend"], 0.0)       # GW per °C
        coefs.loc[name, "beta_cool"] = fit["beta"] / 8760.0                    # GW per year
        coefs.loc[name, "r2_cool_trend"] = fit["r2_trend"]

    print("Gap-filling non-JRC countries (CH, NO, non-EU) …")
    coefs = dc.fill_missing_coefficients(coefs, energy_inputs / "World_bank_population.csv")

    # ---- Sector-split cooling (the recommended form) -------------------------------------
    # Residential and service cooling are fitted separately and pooled across countries: the
    # service sector dominates the energy but is largely weather-insensitive and near saturation,
    # while the residential part carries the emerging climate signal. Pooling a shared intensity is
    # what makes the saturation curvature identifiable, since within one country CDD barely moved
    # over 2000-2020. Population is folded into the coefficients here, so `compute_demand.py`
    # needs no population data. Population is taken for 2015 — mid fit window and the basis of the
    # 2015-2020 anchor.
    print("Fitting sector-split cooling (pooled saturating form) …")
    diag: dict = {}
    population = dc.world_bank_population(energy_inputs / "World_bank_population.csv", "2015")
    cooling = dc.derive_cooling_coefficients(
        jrc23, nat_cdd, population, edition="2023", fallback=(jrc15, "2015"),
        countries=list(coefs.index), diagnostics=diag,
    )
    for column in cooling.columns:
        coefs[column] = cooling[column]

    for sector in ("res", "srv"):
        pick, fit = diag["chosen"][sector], diag["fits"][sector]
        print(f"  {sector}: θ = {pick['theta']:.5f} (HRE {dc.HRE_THETA[sector]}) — chosen from "
              f"group '{pick['group']}' spec {pick['spec']} (identification width "
              f"{pick['width']:.0f}x, R²={pick['r2']:.3f}); shared p0 = {fit['p0']:.4f} "
              f"GWh/Mcap per (CDD·s), 90 % saturation at CDD ≈ {-np.log(0.1) / pick['theta']:.0f}")
    benchmarked = coefs.loc[coefs["cool_gap_filled"] == 0, "corrective_ratio_cool"].dropna()
    print(f"  corrective ratio over the {len(benchmarked)} benchmarked countries: median "
          f"{benchmarked.median():.2f}, IQR {benchmarked.quantile(.25):.2f}–"
          f"{benchmarked.quantile(.75):.2f}, >1 in {int((benchmarked > 1).sum())} — the "
          f"un-anchored form has a systematic shortfall, so the anchor corrects a bias, not noise")
    print(f"  gap-filled (no JRC benchmark): {sorted(coefs.index[coefs['cool_gap_filled'] > 0])}")

    cols = [c for c in OUTPUT_COLS if c in coefs.columns]
    coefs[cols].to_csv(out_csv)
    print(f"Wrote {out_csv} ({len(coefs)} countries).")
    print(coefs[["P_heat_elec", "P_cool_res", "B_cool_res", "P_cool_srv", "B_cool_srv",
                 "corrective_ratio_cool"]].round(4).to_string())

    # The weather-insensitive share is the diagnostic that says whether a country's cooling can be
    # modelled from degree days at all. Near 100 % means the observed energy carries no degree-day
    # signal, so the weather slope there is an assumption, not an estimate.
    mean_cool = {}
    for name in coefs.index:
        # `a or b` would raise on DataFrame truthiness — test for None explicitly.
        fec = dc.load_jrc_fec_series(jrc23, name, "2023")
        if fec is None:
            fec = dc.load_jrc_fec_series(jrc15, name, "2015")
        if fec is not None:
            mean_cool[name] = fec["cool_gwh"].mean() / 8760.0
    share = (coefs["B_cool"] / pd.Series(mean_cool)).dropna().sort_values(ascending=False) * 100
    print("\nWeather-insensitive share of cooling energy (%), highest first:")
    print(share.round(0).head(12).to_string())


if __name__ == "__main__":
    main()
