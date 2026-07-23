"""Derive per-country demand coefficients (one-time) and write demand_coefficients.csv.

Regresses **JRC-IDEES-2023** annual space-heating/cooling final energy on population-weighted
**CERRA** degree days to produce the coefficients used by ``compute_demand.py``:

* **Heating** ``P_heat_elec`` = through-origin (least-squares) coefficient over the **full**
  window × the latest-year electrified share (weather-stationary; the full window is most
  robust and electrification recency is carried by the share).
* **Cooling** ``P_cool`` = total-preserving ratio ``ΣE/(24·ΣDD)`` over the **recent window
  2011–2020** (CERRA 2021 is a partial year and is dropped). ``cool_LOYO`` (leave-one-year-out
  RMSE) is shipped as an uncertainty flag. No analog-fill (validated to distort tiny-cooling
  states); own pooled coefficients are kept for all countries.

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

#: Heating uses the full window; cooling the recent window (CERRA 2021 dropped).
COOL_WINDOW = set(range(2011, 2021))
#: Hot-year window (incl. the 2003 heatwave) for countries on the 2015 fallback edition.
#: UK is absent from JRC-2023, and its recent window (2011–2015) was anomalously cool, so
#: the near-zero-CDD ratio blows up (P_cool 13.9 → end-century cooling above FR/DE). A
#: 2003-inclusive window gives a representative CDD signal (P_cool ~6.5).
COOL_WINDOW_FALLBACK = set(range(2003, 2008))
DD_SLICE = slice("2000", "2020")
POP_RASTER = "gpw_v4/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"
OUTPUT_COLS = ["P_heat_elec", "P_cool", "cool_LOYO", "electrified_share", "P_heat_thermal", "n"]


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
    dd_zarr = root / "energy" / "europe" / "degree_days" / "cerra.zarr"
    out_csv = energy_inputs / "demand_coefficients.csv"

    print("Loading CERRA degree days and GPW v4 population …")
    ds = xr.open_zarr(dd_zarr).squeeze(drop=True).sel(time=DD_SLICE)
    pop = load_population_on_grid(pop_path, ds)
    weights, names = dm.build_country_weights(ds["HDD"].isel(time=0), pop)
    nat_hdd = dc.national_annual_degree_days(ds, "HDD", weights, names)
    nat_cdd = dc.national_annual_degree_days(ds, "CDD", weights, names)

    print("Fitting heating (full window, LS) + electrified share …")
    coefs = dc.derive_coefficients(jrc23, nat_hdd, nat_cdd, edition="2023", fallback=(jrc15, "2015"))

    print("Fitting cooling (2011–2020 total-preserving ratio; 2003–2007 for 2015-fallback) + LOYO …")
    for name in coefs.index:
        fec, win = dc.load_jrc_fec_series(jrc23, name, "2023"), COOL_WINDOW
        if fec is None:  # fell back to 2015 (e.g. UK): recent years absent/anomalously cool
            fec, win = dc.load_jrc_fec_series(jrc15, name, "2015"), COOL_WINDOW_FALLBACK
        if fec is None or name not in nat_cdd.columns:
            continue
        st = dc.annual_ratio_stats(nat_cdd[name], fec["cool_gwh"], years=win)
        coefs.loc[name, "P_cool"] = st["P_ratio"]
        coefs.loc[name, "cool_LOYO"] = st["loyo_rmse"]

    print("Gap-filling non-JRC countries (CH, NO, non-EU) …")
    coefs = dc.fill_missing_coefficients(coefs, energy_inputs / "World_bank_population.csv")

    cols = [c for c in OUTPUT_COLS if c in coefs.columns]
    coefs[cols].to_csv(out_csv)
    print(f"Wrote {out_csv} ({len(coefs)} countries).")
    print(coefs[["P_heat_elec", "P_cool", "cool_LOYO"]].round(3).to_string())


if __name__ == "__main__":
    main()
