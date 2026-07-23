"""Utilities for loading and mapping TYNDP 2024 bidding zones to NUTS regions,
and for extracting scenario-consistent demand and baseload profiles.

Main functions
--------------
load_nuts_geodata             Download (or load from cache) NUTS1, NUTS2 and UK NUTS1 geodata.
build_tyndp_nuts_mapping      Map TYNDP bidding-zone codes to NUTS2 (or UK NUTS1) region codes.
build_tyndp_zone_geometries   Dissolve NUTS geometries into per-zone polygons.
drop_overlapping_zones        Remove or repair zones that cause regionmask overlap warnings.
load_tyndp_demand_profiles    Load a TYNDP demand-profile Excel file into an xarray Dataset.
compute_estimated_generation  Multiply capacity factors by TYNDP installed-capacity projections.
preprocess_thermal_demand     Upsample, smooth, correct and rescale CERRA thermal demand.
build_thermal_targets         Build per-country thermal TWh targets from TYNDP scenario data.
compute_tyndp_baseload        Subtract preprocessed thermal demand from TYNDP market profiles.
expand_typical_year           Tile an 8760-step typical-year profile across calendar years.
compute_residual_load         Residual load = total demand − (PV + wind + ROR hydro).
preprocess_hydro_ror          Normalise country names and distribute daily ROR to sub-daily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.ops import snap as _snap_geom

from cordex2energy.utils.constants import REFERENCE_PERIOD

_GISCO_BASE = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"

# Greece uses GR in TYNDP codes but EL in NUTS/Eurostat
_TYNDP_TO_NUTS_COUNTRY: dict[str, str] = {"GR": "EL"}

REALTIME_BALANCING_ZONES = {
    "Nordic":         ['NO', 'SE', 'FI', 'DK'],
    "Central_Western":   ['FR', 'DE', 'LU', 'BE', 'NL', 'CH', 'AT'],
    "Iberian_Peninsula":    ['ES', 'PT'],
    "Central_Eastern":   ['PL', 'CZ', 'SK', 'HU', 'RO', 'BG', 'EE', 'LV', 'LT'],
    "South_East":  ['IT', 'GR', 'SI', 'HR', 'BA', 'RS', 'ME', 'MK', 'AL', 'MT', 'CY'],
    "British_Isles":        ['UK', 'IE']
}


# ---------------------------------------------------------------------------
# NUTS data loading
# ---------------------------------------------------------------------------


def load_nuts_geodata(
    cache_dir: Union[str, Path],
    verbose: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download NUTS1, NUTS2 (2024) and UK NUTS1 (2016) geodata from Eurostat GISCO.

    Files are cached as GeoJSON in *cache_dir* so subsequent calls are instant.
    UK NUTS data is fetched from the 2016 vintage because the UK was removed from
    NUTS after Brexit and is absent from 2021/2024 datasets.

    Parameters
    ----------
    cache_dir:
        Directory where cached GeoJSON files are stored.
    verbose:
        Print a message indicating whether data was downloaded or loaded from cache.

    Returns
    -------
    gdf_nuts1, gdf_nuts2, gdf_uk_nuts1
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    nuts1_path = cache_dir / "nuts1_2024_4326.geojson"
    nuts2_path = cache_dir / "nuts2_2024_4326.geojson"
    uk_nuts1_path = cache_dir / "uk_nuts1_2016_4326.geojson"

    def _load_or_download(path: Path, url: str, label: str) -> gpd.GeoDataFrame:
        if path.exists():
            if verbose:
                print(f"Loaded {label} from cache")
            return gpd.read_file(path)
        gdf = gpd.read_file(url)
        gdf.to_file(path, driver="GeoJSON")
        if verbose:
            print(f"Downloaded and cached {label}")
        return gdf

    gdf_nuts1 = _load_or_download(
        nuts1_path,
        f"{_GISCO_BASE}/NUTS_RG_10M_2024_4326_LEVL_1.geojson",
        "NUTS1 (2024)",
    )
    gdf_nuts2 = _load_or_download(
        nuts2_path,
        f"{_GISCO_BASE}/NUTS_RG_10M_2024_4326_LEVL_2.geojson",
        "NUTS2 (2024)",
    )

    # UK: download full NUTS1 2016, keep only UK regions
    if uk_nuts1_path.exists():
        gdf_uk_nuts1 = gpd.read_file(uk_nuts1_path)
        if verbose:
            print("Loaded UK NUTS1 (2016) from cache")
    else:
        gdf_uk_nuts1 = gpd.read_file(
            f"{_GISCO_BASE}/NUTS_RG_10M_2016_4326_LEVL_1.geojson"
        )
        gdf_uk_nuts1 = gdf_uk_nuts1[gdf_uk_nuts1["CNTR_CODE"] == "UK"].copy()
        gdf_uk_nuts1.to_file(uk_nuts1_path, driver="GeoJSON")
        if verbose:
            print("Downloaded and cached UK NUTS1 (2016)")

    if verbose:
        print(
            f"\nNUTS1: {len(gdf_nuts1)} | NUTS2: {len(gdf_nuts2)} | UK NUTS1: {len(gdf_uk_nuts1)}"
        )

    return gdf_nuts1, gdf_nuts2, gdf_uk_nuts1


# ---------------------------------------------------------------------------
# Zone → NUTS region mapping
# ---------------------------------------------------------------------------


def _build_explicit_nuts2(
    all_regions: set[str],
    uk_nuts1_codes: set[str],
) -> dict[str, list[str]]:
    """Return the hard-coded zone → NUTS2 region lists for sub-national and special zones."""
    return {
        # --- Italy ---
        "ITN1": [c for c in all_regions if c.startswith("ITC") or c.startswith("ITH")],
        "ITCN": ["ITI1", "ITI2"],
        "ITCS": ["ITI3", "ITI4", "ITF1"],
        "ITS1": ["ITF2", "ITF3", "ITF4", "ITF5"],
        "ITCA": ["ITF6"],
        "ITSI": ["ITG1"],
        "ITSA": ["ITG2"],
        # --- Sweden ---
        "SE01": ["SE33"],
        "SE02": ["SE31", "SE32"],
        "SE03": ["SE11", "SE12", "SE21", "SE23"],
        "SE04": ["SE22"],
        # --- Norway (Eurostat statistical regions, NUTS-equivalent) ---
        "NON1": ["NO07"],
        "NOM1": ["NO06"],
        "NOS0": ["NO08", "NO02", "NO09", "NO0A"],
        # --- Denmark ---
        "DKW1": ["DK03", "DK04", "DK05"],
        "DKE1": ["DK01", "DK02"],
        # --- Greece ---
        "GR00": [c for c in all_regions if c.startswith("EL") and c != "EL43"],
        "GR03": ["EL43"],
        # --- UK: use NUTS1 from 2016 boundaries ---
        "UK00": [c for c in uk_nuts1_codes if not c.startswith("UKN")],  # Great Britain
        "UKNI": ["UKN"],                                                   # Northern Ireland
        # --- Virtual/copper-plate nodes ---
        "DEKF": [c for c in all_regions if c.startswith("DE")],
        "DKKF": [c for c in all_regions if c.startswith("DK")],
        # --- Luxembourg internal nodes (single NUTS2 region LU00) ---
        "LUB1": ["LU00"],
        "LUF1": ["LU00"],
        "LUG1": ["LU00"],
        "LUV1": ["LU00"],
    }


def build_tyndp_nuts_mapping(
    tyndp_zones: list[str],
    gdf_nuts2: gpd.GeoDataFrame,
    gdf_uk_nuts1: gpd.GeoDataFrame,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Map TYNDP 2024 bidding-zone codes to NUTS2 (or UK NUTS1) region codes.

    For zones that cover an entire country (``XX00`` codes) the mapping is built
    automatically from the NUTS2 ``CNTR_CODE`` column.  Sub-national zones and
    special virtual nodes use a hard-coded expert mapping.

    Parameters
    ----------
    tyndp_zones:
        Sorted list of 4-character TYNDP bidding-zone codes (e.g. ``["AT00", "BE00", ...]``).
    gdf_nuts2:
        NUTS2 GeoDataFrame (2024 vintage, EU only).
    gdf_uk_nuts1:
        UK NUTS1 GeoDataFrame (2016 vintage).

    Returns
    -------
    tyndp_nuts2_mapping : dict
        ``{zone_code: [nuts_region_code, ...]}``
    df_tyndp_nuts_mapping : pd.DataFrame
        Summary table with columns ``TYNDP Zone``, ``NUTS1``, ``NUTS2``,
        ``# regions``, ``In NUTS data``.
    """
    # Combined lookup: EU NUTS2 + UK NUTS1
    gdf_regions = pd.concat(
        [gdf_nuts2, gdf_uk_nuts1[["NUTS_ID", "CNTR_CODE", "geometry"]]],
        ignore_index=True,
    )
    all_regions: set[str] = set(gdf_regions["NUTS_ID"])
    uk_nuts1_codes: set[str] = set(gdf_uk_nuts1["NUTS_ID"])

    explicit_nuts2 = _build_explicit_nuts2(all_regions, uk_nuts1_codes)

    def _regions_for_country(iso2: str) -> list[str]:
        nuts_cc = _TYNDP_TO_NUTS_COUNTRY.get(iso2, iso2)
        return gdf_regions.loc[gdf_regions["CNTR_CODE"] == nuts_cc, "NUTS_ID"].tolist()

    # Build the full mapping
    mapping: dict[str, list[str]] = {}
    for zone in tyndp_zones:
        if zone in explicit_nuts2:
            mapping[zone] = [c for c in explicit_nuts2[zone] if c in all_regions]
        elif zone.endswith("00"):
            mapping[zone] = _regions_for_country(zone[:2])
        else:
            mapping[zone] = []  # non-EU / non-NUTS country

    # Build summary DataFrame
    rows = []
    for zone, region_codes in mapping.items():
        nuts1_codes = sorted({c[:3] for c in region_codes})
        rows.append(
            {
                "TYNDP Zone": zone,
                "NUTS1": nuts1_codes,
                "NUTS2": region_codes,
                "# regions": len(region_codes),
                "In NUTS data": len(region_codes) > 0,
            }
        )

    return mapping, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Zone geometry construction
# ---------------------------------------------------------------------------


def build_tyndp_zone_geometries(
    tyndp_nuts2_mapping: dict[str, list[str]],
    df_tyndp_nuts_mapping: pd.DataFrame,
    gdf_nuts2: gpd.GeoDataFrame,
    gdf_uk_nuts1: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Dissolve NUTS2 (and UK NUTS1) polygons into per-TYNDP-zone geometries.

    Zones with an empty region list (non-EU / non-NUTS countries) are silently
    skipped and will be absent from the returned GeoDataFrame.

    Parameters
    ----------
    tyndp_nuts2_mapping:
        Output of :func:`build_tyndp_nuts_mapping` — ``{zone: [nuts_codes]}``.
    df_tyndp_nuts_mapping:
        Summary DataFrame from :func:`build_tyndp_nuts_mapping`.
    gdf_nuts2:
        NUTS2 GeoDataFrame (2024, EU only) — used for the CRS.
    gdf_uk_nuts1:
        UK NUTS1 GeoDataFrame (2016 vintage).

    Returns
    -------
    gdf_tyndp_zones : gpd.GeoDataFrame
        One row per zone with columns ``tyndp_zone``, ``geometry``,
        ``NUTS1``, ``NUTS2``, ``# regions``.
    """
    gdf_regions = pd.concat(
        [gdf_nuts2, gdf_uk_nuts1[["NUTS_ID", "CNTR_CODE", "geometry"]]],
        ignore_index=True,
    )

    zone_geoms = []
    for zone, nuts_codes in tyndp_nuts2_mapping.items():
        if not nuts_codes:
            continue
        dissolved = (
            gdf_regions[gdf_regions["NUTS_ID"].isin(nuts_codes)]
            .dissolve()
            .geometry
            .values[0]
        )
        zone_geoms.append({"tyndp_zone": zone, "geometry": dissolved})

    gdf = gpd.GeoDataFrame(zone_geoms, geometry="geometry", crs=gdf_nuts2.crs)
    gdf = gdf.merge(
        df_tyndp_nuts_mapping[["TYNDP Zone", "NUTS1", "NUTS2", "# regions"]].rename(
            columns={"TYNDP Zone": "tyndp_zone"}
        ),
        on="tyndp_zone",
    )
    return gdf


# ---------------------------------------------------------------------------
# Overlap handling
# ---------------------------------------------------------------------------

#: TYNDP virtual / copper-plate nodes that cover an entire country and therefore
#: geometrically overlap with the corresponding country-level zone.
VIRTUAL_ZONES: frozenset = frozenset({"DEKF", "DKKF"})


def drop_overlapping_zones(
    gdf: gpd.GeoDataFrame,
    zone_col: str = "tyndp_zone",
) -> gpd.GeoDataFrame:
    """Remove or repair zones that would cause regionmask overlap warnings.

    Three sources of overlap are handled in order:

    1. **Virtual nodes** listed in :data:`VIRTUAL_ZONES` (e.g. ``DEKF``,
       ``DKKF``) — country-wide copper-plate zones whose geometry duplicates
       the corresponding ``XX00`` zone.  These are dropped entirely.
    2. **NUTS superset / duplicate zones** — a zone whose NUTS region set is a
       strict superset of another zone's set (e.g. ``DK00`` containing all of
       ``DKW1`` + ``DKE1``) is dropped.  Exact duplicates (e.g. all four
       Luxembourg sub-zones resolving to ``LU00``) keep only the first.
    3. **Residual border slivers** — small geometric overlaps caused by
       boundary misalignment between different NUTS vintages (e.g. ``IE00``
       vs ``UKNI``).  The intersection sliver is subtracted from the
       smaller zone so both zones are preserved.

    Parameters
    ----------
    gdf:
        GeoDataFrame returned by :func:`build_tyndp_zone_geometries`.
    zone_col:
        Column that holds zone codes.

    Returns
    -------
    gpd.GeoDataFrame
    """
    # 1. Drop known virtual / copper-plate nodes
    gdf = gdf[~gdf[zone_col].isin(VIRTUAL_ZONES)].copy()

    if "NUTS2" not in gdf.columns:
        return gdf

    # 2. Drop zones whose NUTS region set is a superset of another zone's set,
    #    or an exact duplicate (keep the first occurrence).
    zone_sets: dict[str, frozenset] = {
        row[zone_col]: frozenset(row["NUTS2"]) for _, row in gdf.iterrows()
    }
    zones = list(zone_sets.keys())
    to_drop: set[str] = set()

    for i, zone_a in enumerate(zones):
        if zone_a in to_drop:
            continue
        set_a = zone_sets[zone_a]
        for j, zone_b in enumerate(zones):
            if i == j or zone_b in to_drop:
                continue
            set_b = zone_sets[zone_b]
            if not set_b:
                continue
            # Strict superset → zone_a fully contains zone_b → drop zone_a
            if set_a > set_b:
                to_drop.add(zone_a)
                break
            # Exact duplicate → drop the later occurrence
            if set_a == set_b and i > j:
                to_drop.add(zone_a)
                break

    gdf = gdf[~gdf[zone_col].isin(to_drop)].reset_index(drop=True).copy()

    # 3. Clip residual border slivers: for any remaining pair that still
    #    intersects geometrically, subtract the sliver from the smaller zone.
    #    Vertices are snapped to the partner boundary first so that floating-
    #    point misalignment (e.g. IE00 vs UKNI from different NUTS vintages)
    #    is resolved before the difference is taken.
    areas_m2 = gdf.to_crs("EPSG:3035").geometry.area.values
    for i in range(len(gdf)):
        for j in range(i + 1, len(gdf)):
            inter = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j])
            if inter.is_empty:
                continue
            smaller = i if areas_m2[i] <= areas_m2[j] else j
            larger  = j if smaller == i else i
            geom_s = gdf.geometry.iloc[smaller]
            geom_l = gdf.geometry.iloc[larger]
            # Snap boundary vertices together before differencing to avoid
            # floating-point residuals (tolerance ~1 cm in geographic degrees)
            snapped = _snap_geom(geom_s, geom_l, tolerance=1e-7)
            fixed   = snapped.difference(geom_l).buffer(0)
            gdf.at[gdf.index[smaller], "geometry"] = fixed

    return gdf


# ---------------------------------------------------------------------------
# Demand profile loading
# ---------------------------------------------------------------------------


def load_tyndp_demand_profiles(path, min_hours=8000):
    """Load a TYNDP demand-profile Excel file into an xarray Dataset.

    Each sheet in the file represents one bidding zone.  The column headers
    (years) are at Excel row 12 (pandas ``header=11``).  Each data row is
    one hour of the year.  Year columns are detected automatically: any
    integer column header in [1900, 2100] that also contains at least
    ``min_hours`` non-NaN values is treated as a valid climate year.  This
    guards against stray non-empty cells in otherwise-empty columns (which
    would survive a plain ``dropna(axis=1, how='all')`` and produce
    spurious years in the output).

    Bidding zones are summed to country level (first two characters of the
    zone code) so the returned dataset has a ``country`` dimension.

    Known data errors corrected at load time
    -----------------------------------------
    * Sheet **UK00**: all three climate years are labelled one year too early
      (1994/2007/2008 instead of 1995/2008/2009).  Every year label is
      incremented by 1 when this pattern is detected.

    A warning is printed whenever the final set of detected years differs
    from the expected {1995, 2008, 2009}.

    Parameters
    ----------
    path : str or Path
        Path to the Excel file.
    min_hours : int, optional
        Minimum number of non-NaN rows required for a column to be treated
        as a full climate year.  Defaults to 8000 (safely below the 8760
        hours of a non-leap year).

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(country, time)``.
        Variable name: ``"demand_profile"``.
    """
    _EXPECTED_YEARS = {1995, 2008, 2009}

    xl = pd.ExcelFile(path)
    zone_series: dict[str, pd.Series] = {}

    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=11)
        # Drop fully-empty columns and rows
        df = df.dropna(axis=1, how="all").dropna(how="all")

        # Detect year columns: integer header in [1900, 2100] AND at least
        # min_hours non-NaN values (filters out stray-value columns whose
        # header happens to look like a year, e.g. a partial reference year).
        year_cols: dict = {}
        for c in df.columns:
            try:
                y = int(float(str(c).strip()))
                if 1900 <= y <= 2100 and df[c].notna().sum() >= min_hours:
                    year_cols[c] = y
            except (ValueError, TypeError):
                pass

        # ── Fix known labelling errors ────────────────────────────────────────
        detected_years = set(year_cols.values())

        if sheet == "UK00" and detected_years == {1994, 2007, 2008}:
            # All three climate years are off by one in the UK00 sheet.
            year_cols = {c: y + 1 for c, y in year_cols.items()}

        if set(year_cols.values()) != _EXPECTED_YEARS:
            print(
                f"Warning: sheet '{sheet}' has unexpected climate years "
                f"{set(year_cols.values())} (expected {_EXPECTED_YEARS})."
            )
        # ─────────────────────────────────────────────────────────────────────

        segments = []
        for col, year in sorted(year_cols.items(), key=lambda kv: kv[1]):
            vals = df[col].dropna().values.astype(float)
            # date_range handles leap years (8784 h) automatically
            timestamps = pd.date_range(f"{year}-01-01", periods=len(vals), freq="h")
            segments.append(pd.Series(vals, index=timestamps))

        if segments:
            zone_series[sheet] = pd.concat(segments)

    if not zone_series:
        raise ValueError(f"No demand-profile data found in {path}")

    # Build a common time axis (union across zones)
    all_times = pd.DatetimeIndex(
        sorted(set().union(*[s.index.tolist() for s in zone_series.values()]))
    )
    zones = list(zone_series.keys())
    data = np.stack(
        [zone_series[z].reindex(all_times).values for z in zones], axis=0
    )

    ds = xr.DataArray(
        data,
        dims=["tyndp_zone", "time"],
        coords={"tyndp_zone": zones, "time": all_times},
        name="demand_profile",
    ).to_dataset()

    # ── Aggregate bidding zones → country (first 2 chars of zone code) ───────
    ds = (
        ds.assign_coords(country=("tyndp_zone", [z[:2] for z in zones]))
        .groupby("country")
        .sum()
    )

    return ds


# ---------------------------------------------------------------------------
# Generation estimation
# ---------------------------------------------------------------------------


def compute_estimated_generation(
    ds_pv,
    ds_wind_onshore,
    ds_wind_offshore,
    df_tyndp_solar,
    df_tyndp_wind_onshore,
    df_tyndp_wind_offshore,
    scenario,
    year,
    zone_dim="tyndp_zone",
    country_dim="country",
):
    """Compute estimated energy generation given TYNDP capacity projections.

    Capacity factors (ds_pv, ds_wind_onshore at bidding-zone level;
    ds_wind_offshore at country level) are multiplied by the installed
    capacity extracted from the TYNDP supply input DataFrames for the
    requested scenario/year combination.

    Zones removed by drop_overlapping_zones (e.g. DK00 superseded by
    DKW1+DKE1) are absent from the CF datasets.  When such a zone carries
    non-zero capacity in the TYNDP file its capacity is added to the first
    country peer present in the CF dataset — this is exact because the
    dropped zone shares the same capacity factors as its children.

    After the zone-level multiplication results are summed to country level
    (first two characters of the zone code).  For offshore wind the TYNDP
    projections are aggregated to country level before the multiplication,
    since ds_wind_offshore is already country-aggregated.

    Parameters
    ----------
    ds_pv, ds_wind_onshore : xr.Dataset
        Capacity-factor datasets with a `zone_dim` dimension (TYNDP
        bidding-zone codes, e.g. "FR00", "DKW1").
    ds_wind_offshore : xr.Dataset
        Capacity-factor dataset with a `country_dim` dimension.
    df_tyndp_solar, df_tyndp_wind_onshore, df_tyndp_wind_offshore : pd.DataFrame
        TYNDP supply DataFrames with a "Country" column (bidding-zone codes)
        and numeric capacity columns named ``{year}_{scenario}``
        (e.g. "2040_GA", "2040_DE", "2040_HIGH").
    scenario : str
        Scenario suffix as it appears in column names (e.g. "GA", "DE", "HIGH").
    year : int
        Target year (2030, 2040, or 2050).
    zone_dim : str
        Name of the bidding-zone dimension in ds_pv / ds_wind_onshore.
    country_dim : str
        Name of the country dimension in ds_wind_offshore.

    Returns
    -------
    ds_gen_pv : xr.Dataset
        PV generation at country level   [capacity_unit × CF].
    ds_gen_wind_onshore : xr.Dataset
        Onshore wind generation at country level.
    ds_gen_wind_offshore : xr.Dataset
        Offshore wind generation at country level.
    """
    col = f"{year}_{scenario}"

    # ── 1. Extract capacity series ────────────────────────────────────────────
    def _get_capacity(df):
        return (
            df[["Country", col]]
            .dropna(subset=["Country"])
            .set_index("Country")[col]
            .fillna(0.0)
            .astype(float)
        )

    cap_solar    = _get_capacity(df_tyndp_solar)
    cap_wind_on  = _get_capacity(df_tyndp_wind_onshore)
    cap_wind_off = _get_capacity(df_tyndp_wind_offshore)

    # ── 2. Reintegrate capacity of dropped / overlapping zones ───────────────
    # Dropped zones share the same capacity factors as their country peers, so
    # adding the full capacity to one peer gives the correct country total.
    available_zones = list(ds_pv[zone_dim].values)

    def _reintegrate(cap_series, available_zones):
        cap = cap_series.reindex(
            cap_series.index.union(available_zones), fill_value=0.0
        ).copy()
        for zone in cap_series.index:
            if zone not in available_zones and cap_series.get(zone, 0.0) > 0:
                country = zone[:2]
                children = [z for z in available_zones if z.startswith(country)]
                if children:
                    cap[children[0]] += cap_series[zone]
        return cap.reindex(available_zones, fill_value=0.0)

    cap_solar_z   = _reintegrate(cap_solar,   available_zones)
    cap_wind_on_z = _reintegrate(cap_wind_on, available_zones)

    # ── 3. Helper: build a capacity DataArray ─────────────────────────────────
    def _capacity_da(cap_series, dim):
        return xr.DataArray(
            cap_series.values,
            dims=[dim],
            coords={dim: cap_series.index.values},
        )

    # ── 4. Multiply CF × capacity at zone level ───────────────────────────────
    ds_gen_pv_z   = ds_pv           * _capacity_da(cap_solar_z,   zone_dim)
    ds_gen_wind_z = ds_wind_onshore * _capacity_da(cap_wind_on_z, zone_dim)

    # ── 5. Sum bidding zones → country ────────────────────────────────────────
    zone_to_country = pd.Series({z: z[:2] for z in available_zones})

    def _aggregate_to_country(ds, zone_to_country, zone_dim):
        slices = []
        for country, zones in sorted(zone_to_country.groupby(zone_to_country).groups.items()):
            ds_c = ds.sel({zone_dim: list(zones)}).sum(zone_dim)
            ds_c = ds_c.expand_dims({"country": [country]})
            slices.append(ds_c)
        return xr.concat(slices, dim="country")

    ds_gen_pv      = _aggregate_to_country(ds_gen_pv_z,  zone_to_country, zone_dim)
    ds_gen_wind_on = _aggregate_to_country(ds_gen_wind_z, zone_to_country, zone_dim)

    # ── 6. Offshore: aggregate TYNDP to country level, then multiply ──────────
    cap_wind_off_country = (
        cap_wind_off
        .rename(index=lambda z: z[:2])
        .groupby(level=0)
        .sum()
    )
    available_countries_off = list(ds_wind_offshore[country_dim].values)
    cap_wind_off_aligned = cap_wind_off_country.reindex(
        available_countries_off, fill_value=0.0
    )
    ds_gen_wind_off = ds_wind_offshore * _capacity_da(cap_wind_off_aligned, country_dim)

    # ── 7. Add attributes for traceability
    for ds in [ds_gen_pv, ds_gen_wind_on, ds_gen_wind_off]:
        ds.attrs["source"] = f"TYNDP {year} {scenario}"
        for var in ds.data_vars:
            ds[var].attrs["unit"] = "MWh"

    return ds_gen_pv, ds_gen_wind_on, ds_gen_wind_off


# ---------------------------------------------------------------------------
# Thermal demand pre-processing
# ---------------------------------------------------------------------------


def preprocess_thermal_demand(
    ds_thermal,
    target_twh_per_country=None,
    loss_factor=1.176,
    rolling_window=4,
    upsample_to_1h=True,
    reference_period=REFERENCE_PERIOD,
):
    """Pre-process a thermal demand series for subtraction from / addition to
    TYNDP hourly market demand.

    Applies up to three sequential post-processing steps that bridge the
    structural gap between raw climate-simulation output and ENTSO-E's
    internally derived demand profiles.  **Every step operates independently
    on each data variable of ``ds_thermal``** — so heating and cooling can be
    carried as two separate variables (``heating_demand`` / ``cooling_demand``)
    and each is upsampled, smoothed and rescaled to *its own* target with no
    cross-variable mixing.  A single lumped variable (``demand_profile``) is
    just the one-variable case and reproduces the historical behaviour.

    **Step 1 — Temporal upsampling + thermal inertia smoothing (3 h → 1 h)**

    3-hourly demand.ninja/CERRA output is blocky; real grids operate hourly and
    real buildings have thermal mass.  When ``upsample_to_1h`` is True the data
    is first upsampled to 1 h via cubic spline interpolation, then a centered
    ``rolling_window``-hour rolling mean is applied.  For an already-hourly
    product (the JRC ``cordex2energy_demand`` NetCDFs) pass
    ``upsample_to_1h=False`` to skip the interpolation; the rolling
    thermal-inertia mean is still applied (a model-independent physical
    smoothing).

    **Step 2 — Real-world COP / loss correction**

    A flat ``loss_factor`` multiplier (default 1.176 = 1/0.85, the ENTSO-E DFT
    COP correction).  Note: whenever Step 3 rescales to a target this factor
    cancels exactly (it is absorbed by ``k``); it only affects countries that
    have *no* target (``k = 1``).  Pass ``loss_factor=1.0`` for an
    already-electric product such as ``cordex2energy_demand``.

    **Step 3 — Volumetric rescaling to TYNDP annual TWh target (per variable)**

    When ``target_twh_per_country`` is provided, each *variable* of each
    country is scaled by ``k = E_target / E_simulated`` where ``E_simulated``
    is the mean annual sum over ``reference_period`` only.  Anchoring the
    scaling to the reference period (rather than the whole 1991–2100 series)
    keeps the future heating/cooling trend intact instead of diluting the
    target with it.

    Parameters
    ----------
    ds_thermal : xr.Dataset
        Thermal electricity demand in GW, dims ``(country, time)``.  May carry
        one variable (lumped ``demand_profile``) or two (``heating_demand`` +
        ``cooling_demand``).
    target_twh_per_country : dict or pd.Series, optional
        Annual electrical energy target.  Two accepted forms:

        * **per-variable (nested)** ``{var_name: {country: TWh}}`` — each
          variable is rescaled to its own target;
        * **flat** ``{country: TWh}`` — applied to every variable (the legacy
          single-variable case).

        ``None`` (default) skips rescaling.
    loss_factor : float, optional
        Flat COP/loss multiplier.  Default 1.176 (= 1/0.85).  Set to 1.0 to
        disable.
    rolling_window : int, optional
        Width of the centered rolling mean in hours.  Default 4.
    upsample_to_1h : bool, optional
        Whether to cubic-spline upsample 3 h → 1 h before smoothing.  Default
        True; set False for an already-hourly input.
    reference_period : sequence of two str, optional
        ``(start, end)`` years over which ``E_simulated`` is computed for the
        Step-3 rescaling.  Default :data:`REFERENCE_PERIOD` (1991–2020).  Pass
        ``None`` to use the whole input series (legacy behaviour).

    Returns
    -------
    xr.Dataset
        1-hourly thermal demand with the same variable names as the input, in
        GW.  The variables are **not** summed here — the caller combines them.
    """
    # ── Step 1a: Upsample 3 h → 1 h (cubic spline interpolation), if needed ───
    ds_1h = ds_thermal.resample(time="1h").interpolate("cubic") if upsample_to_1h else ds_thermal

    # ── Step 1b: Thermal inertia smoothing (centered rolling mean) ────────────
    ds_smooth = ds_1h.rolling(
        time=rolling_window, center=True, min_periods=1
    ).mean()

    # ── Step 2: Real-world COP / loss correction ──────────────────────────────
    ds_scaled = ds_smooth * loss_factor

    # ── Step 3: Per-variable volumetric rescaling to TYNDP annual TWh target ──
    if target_twh_per_country is not None:
        # Resolve the target dict for a given variable: nested {var: {country:}}
        # picks the per-variable dict; a flat {country: twh} applies to every
        # variable (the legacy single-variable / lumped case).
        _is_nested = isinstance(next(iter(target_twh_per_country.values()), None), dict)

        def _target_for(var):
            return target_twh_per_country.get(var) if _is_nested else target_twh_per_country

        for v in ds_scaled.data_vars:
            tv = _target_for(v)
            if tv is None:
                continue
            scale_values = []
            for country in ds_scaled[v].country.values:
                c = str(country)
                if c in tv:
                    sim = ds_scaled[v].sel(country=country)
                    if reference_period is not None:
                        sim = sim.sel(time=slice(*reference_period))
                    # Annual GWh = Σ(GW values) × 1 h; /1000 → TWh
                    simulated_twh = float(sim.resample(time="YS").sum().mean(dim="time").values) * 1e-3
                    k = float(tv[c]) / simulated_twh if simulated_twh > 0 else 1.0
                else:
                    k = 1.0
                scale_values.append(k)
            scale_da = xr.DataArray(
                scale_values, dims=["country"],
                coords={"country": ds_scaled[v].country.values},
            )
            ds_scaled[v] = ds_scaled[v] * scale_da

    return ds_scaled


# ---------------------------------------------------------------------------
# Thermal targets
# ---------------------------------------------------------------------------

#: Default proxy pairings for non-EU countries absent from the TYNDP
#: demand spreadsheet (EU-27 only).  Override via ``non_eu_proxy_map``.
_DEFAULT_NON_EU_PROXIES: dict[str, str] = {
    # ── Meteorological + building-stock rationale ─────────────────────────────
    "UK": "FR",   # maritime, comparable building stock & heat-pump rollout
    "CH": "AT",   # alpine, shared border, similar renovation trajectory
    "NO": "SE",   # Nordic, highest heat-pump penetration in Europe
    # ── Western Balkans: intentionally omitted (Physical-Pure k=1 fallback) ──
    # AL, BA, ME, MK, RS are coal-heavy, minimally electrified economies.
    # No EU-27 country has a comparable building-stock topology or
    # electrification trajectory.  The floor cap in compute_tyndp_baseload
    # provides safety for these nodes.
}


def build_thermal_targets(
    df_tyndp_demand,
    scenario,
    year=2050,
    non_eu_proxy_map=None,
    heating_subsectors=("Space heating", "Space heating & hot water"),
    cooling_subsectors=("Cooling",),
    ds_market=None,
):
    """Build a per-country thermal electricity TWh target dict for use in
    :func:`preprocess_thermal_demand`.

    For EU countries whose thermal electricity breakdown appears in the TYNDP
    demand spreadsheet, the target is read directly.  For non-EU countries
    (UK, CH, NO, Balkans …) their thermal target is synthesised by multiplying
    the country's total electricity demand by the *thermal share* (α) of a
    meteorologically and economically similar EU proxy country.

    Non-EU ``E_total`` source
    -------------------------
    ``df_tyndp_demand`` is an EU-27-only export.  When ``ds_market`` is
    provided, ``E_total`` for any non-EU country absent from the spreadsheet
    is derived from the hourly market profiles (mean annual sum across all
    climate years present in the dataset).

    Parameters
    ----------
    df_tyndp_demand : pd.DataFrame
        The ``3_DEMAND_OUTPUT`` sheet with multi-level columns already
        flattened to ``{year}_{scenario}`` format (e.g. ``"2050_GA"``).
    scenario : str
        Scenario identifier (``"GA"``, ``"DE"``, ``"HIGH"``, …).
    year : int, optional
        Target year.  Default 2050.
    non_eu_proxy_map : dict, optional
        Override or extend default proxy pairings.
    heating_subsectors : tuple of str, optional
        ``SUBSECTOR`` values treated as electric heating consumption.
    cooling_subsectors : tuple of str, optional
        ``SUBSECTOR`` values treated as electric cooling consumption.
    ds_market : xr.Dataset, optional
        TYNDP market demand profiles, dims ``(country, time)``,
        variable ``"demand_profile"`` in GW.  Used as ``E_total`` source
        for non-EU countries absent from ``df_tyndp_demand``.

    Returns
    -------
    dict
        ``{"heating": {country: twh}, "cooling": {country: twh},
        "total": {country: twh}}``.  ``"total"`` is the spreadsheet
        ``SUBSECTOR=="Total"`` electricity demand (EU-27 only, never backfilled
        from ``ds_market``) used by the baseload modulation.
    """
    col = f"{year}_{scenario}"
    proxy_map = {**_DEFAULT_NON_EU_PROXIES, **(non_eu_proxy_map or {})}

    def _by_country(subsectors):
        mask = df_tyndp_demand["ENERGY_CARRIER"].eq("Electricity") & \
               df_tyndp_demand["SUBSECTOR"].isin(subsectors)
        return (
            df_tyndp_demand.loc[mask, ["COUNTRY", col]]
            .dropna(subset=[col])
            .groupby("COUNTRY")[col]
            .sum()
        )

    # Heating / cooling electricity by country (EU-27, direct from spreadsheet).
    heating_twh = _by_country(heating_subsectors)
    cooling_twh = _by_country(cooling_subsectors)

    # Total electricity by country (EU-27 only in this spreadsheet).
    total_twh = _by_country(["Total"])

    # Pre-compute market annual totals from ds_market (MWh -> TWh); used ONLY as
    # the non-EU proxy E_total source, never for the returned "total".
    market_twh: dict[str, float] = {}
    if ds_market is not None:
        n_years = max(1, len(np.unique(ds_market.time.dt.year.values)))
        for c in ds_market.country.values:
            v = float(ds_market["demand_profile"].sel(country=c).values.sum()) * 1e-6 / n_years
            if v > 0:
                market_twh[str(c)] = v

    # Per-type target dicts (EU-27 direct entries).
    targets_heat: dict[str, float] = {c: float(v) for c, v in heating_twh.items() if float(v) > 0}
    targets_cool: dict[str, float] = {c: float(v) for c, v in cooling_twh.items() if float(v) > 0}
    total: dict[str, float] = {c: float(v) for c, v in total_twh.items() if float(v) > 0}

    # Non-EU countries: synthesise heating/cooling via the proxy thermal-share
    # method, splitting the proxy share into a heating share and a cooling share.
    for country, proxy in proxy_map.items():
        if country in targets_heat or country in targets_cool:
            continue                              # already has a direct entry
        if proxy not in total_twh or float(total_twh[proxy]) <= 0:
            print(f"Warning: proxy '{proxy}' for '{country}' not found in "
                  f"TYNDP demand data - skipping.")
            continue

        alpha_heat = float(heating_twh.get(proxy, 0.0)) / float(total_twh[proxy])
        alpha_cool = float(cooling_twh.get(proxy, 0.0)) / float(total_twh[proxy])

        # Prefer spreadsheet total; fall back to ds_market annual sum.
        if country in total_twh and float(total_twh[country]) > 0:
            e_total = float(total_twh[country])
            source = "spreadsheet"
        elif country in market_twh:
            e_total = market_twh[country]
            source = "ds_market"
        else:
            print(f"Warning: no total electricity demand found for '{country}' "
                  f"in TYNDP demand data or ds_market - cannot synthesise "
                  f"thermal target (falling back to k=1).")
            continue

        targets_heat[country] = alpha_heat * e_total
        targets_cool[country] = alpha_cool * e_total
        print(f"  {country}: proxy='{proxy}', a_heat={alpha_heat:.3f}, "
              f"a_cool={alpha_cool:.3f}, E_total={e_total:.1f} TWh [{source}] -> "
              f"E_heat={targets_heat[country]:.1f}, E_cool={targets_cool[country]:.1f} TWh")

    return {"heating": targets_heat, "cooling": targets_cool, "total": total}


# ---------------------------------------------------------------------------
# Baseload extraction
# ---------------------------------------------------------------------------


def compute_tyndp_baseload(ds_market, ds_prosumer, floor_fraction=0.15):
    """Extract the non-thermally sensitive base load from TYNDP demand profiles.

    Computes ``base_load = market - prosumer`` with a per-country floor cap
    that prevents physically impossible negative demand values.

    Missing countries in ``ds_prosumer`` are automatically filled with zero
    before subtraction (correct for offshore hubs, LUX sub-nodes, and
    non-EU aggregated zones whose prosumer share is implicitly zero).

    Parameters
    ----------
    ds_market : xr.Dataset
        TYNDP electricity-market demand profiles, dims ``(country, time)``.
    ds_prosumer : xr.Dataset
        TYNDP prosumer demand profiles, dims ``(country, time)``.  May
        contain fewer countries than ``ds_market``.
    floor_fraction : float, optional
        Minimum residual base load as a fraction of each country's mean
        market demand.  Defaults to 0.15 (15 %).

    Returns
    -------
    xr.Dataset
        Non-thermally sensitive base load, dims ``(country, time)``,
        variable name ``"baseload"``.
    """
    # ── Align time axes (intersection) ───────────────────────────────────────
    common_time = np.intersect1d(ds_market.time.values, ds_prosumer.time.values)
    ds_market   = ds_market.sel(time=common_time)
    ds_prosumer = ds_prosumer.sel(time=common_time)
    print(f"Aligned time axis: {len(common_time)} hours from "
          f"{pd.to_datetime(common_time[0])} to {pd.to_datetime(common_time[-1])}.")

    # ── Fill missing countries in prosumer with 0 ─────────────────────────────
    print(f"Filling missing countries in ds_prosumer with 0: "
          f"{set(ds_market.country.values) - set(ds_prosumer.country.values)}")
    ds_prosumer = ds_prosumer.reindex(country=ds_market.country, fill_value=0.0)

    # ── Subtract prosumer from total market load ──────────────────────────────
    ds_baseload = ds_market - ds_prosumer

    # ── Floor cap: prevent negative baseload demand ───────────────────────────
    floor = floor_fraction * ds_market.mean(dim="time")
    ds_baseload = ds_baseload.where(ds_baseload >= floor, other=floor)

    return ds_baseload.rename({"demand_profile": "baseload"})


# ---------------------------------------------------------------------------
# Typical-year expansion
# ---------------------------------------------------------------------------


def expand_typical_year(ds_typical, years, typical_dim="hour_of_year"):
    """Tile an 8760-step typical-year profile across explicit calendar years.

    ``ds_typical`` has a dimension ``typical_dim`` with integer coordinate
    values 0–8759, where 0 = 00:00 Jan 1 and 8759 = 23:00 Dec 31 of a
    non-leap year.  The typical year is usually derived as the mean over
    several climate years (see the ``hour_of_year`` groupby pattern in the
    notebook).

    For leap years, Feb 29 is **skipped**: each calendar year always
    contributes exactly 8760 hourly steps, all mapping cleanly onto the
    8760-step typical pattern.  Dates after Feb 28 in a leap year have their
    ``day_of_year`` shifted back by 1 before the lookup so that Mar 1 of a
    leap year maps to the same typical-year slot as Mar 1 of a non-leap year.

    Parameters
    ----------
    ds_typical : xr.Dataset
        Typical-year dataset.  Must have dimension ``typical_dim`` of
        length 8760 with coordinate values 0–8759.
    years : array-like of int
        Calendar years to tile across (e.g. ``[1995, 2008, 2009]``).
    typical_dim : str, optional
        Name of the hour-of-year dimension.  Default ``"hour_of_year"``.

    Returns
    -------
    xr.Dataset
        Hourly time-indexed dataset spanning ``len(years) × 8760`` steps.
        All other dimensions (e.g. ``country``) are preserved unchanged.
    """
    import calendar as _calendar

    all_times = []
    all_hoy = []

    for year in sorted(years):
        yr_times = pd.date_range(
            start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="1h"
        )
        if _calendar.isleap(year):
            # Drop Feb 29 so the year contributes exactly 8760 steps.
            yr_times = yr_times[~((yr_times.month == 2) & (yr_times.day == 29))]

        doy = yr_times.day_of_year.values.copy()
        if _calendar.isleap(year):
            # Shift doy back by 1 for all dates after Feb 28 so that Mar 1
            # maps to the same typical-year slot as in a non-leap year.
            doy[yr_times.month.values > 2] -= 1

        all_times.append(yr_times)
        all_hoy.append((doy - 1) * 24 + yr_times.hour.values)

    time_index = pd.DatetimeIndex(
        np.concatenate([t.values for t in all_times])
    )
    hoy_array = np.concatenate(all_hoy)

    # Advanced xarray indexing: pick the typical-year slice for every step.
    hoy_da = xr.DataArray(hoy_array, dims=["time"], coords={"time": time_index})
    ds_expanded = ds_typical.sel({typical_dim: hoy_da}).drop_vars(typical_dim)

    return ds_expanded


# ---------------------------------------------------------------------------
# Residual load
# ---------------------------------------------------------------------------


def compute_residual_load(
    ds_gen_pv,
    ds_gen_wind_on,
    ds_gen_wind_off,
    ds_hydro_ror,
    ds_baseload,
    ds_heating,
    ds_cooling,
    target_twh_per_country=None,
    loss_factor=1.176,
    rolling_window_thermal_demand=4,
    rolling_window_residual_load=8,
    upsample_to_1h=True,
    reference_period=REFERENCE_PERIOD,
    country_dim="country",
    return_generation_demand_components=False,
):
    """Compute residual load = total demand − (PV + onshore wind + offshore wind + ROR hydro).

    The demand side is assembled from two components:

    * **Baseload** (``ds_baseload``) — the non-thermally-sensitive part of
      electricity demand, typically derived from TYNDP market profiles after
      subtracting preprocessed thermal demand.  Must be at 1-hourly resolution.
    * **Thermal demand** (``ds_heating`` + ``ds_cooling``) — climate-driven
      heating and cooling electricity demand at the native 3-hourly C2E/CERRA
      resolution.  The two are summed and then fed through
      :func:`preprocess_thermal_demand` (upsample → smooth → rescale) to
      produce a 1-hourly series.

    Both demand components are then summed at 1-hourly resolution and
    resampled to 3-hourly sums so that they are directly comparable to the
    generation datasets, which carry 3-hourly values.

    Country alignment is performed by taking the intersection of the country
    coordinates present across all input datasets.  Missing generation
    technologies for a country (e.g. a landlocked country with no offshore
    wind) are filled with zero rather than dropped.

    All datasets are expected to be in GWh.  The output residual load is also in GWh, with positive
    values indicating unmet demand and negative values indicating generation surplus.

    Parameters
    ----------
    ds_gen_pv : xr.Dataset
        PV capacity-factor × capacity, 3-hourly, ``(country, time)`` dims.
    ds_gen_wind_on : xr.Dataset
        Onshore wind generation, 3-hourly, ``(country, time)`` dims.
    ds_gen_wind_off : xr.Dataset
        Offshore wind generation, 3-hourly, ``(country, time)`` dims.
    ds_hydro_ror : xr.Dataset
        Run-of-river hydro generation, 3-hourly, ``(country, time)`` dims.
        Variable name should be ``"ror"`` or any single data variable.
    ds_baseload : xr.Dataset
        Non-thermally-sensitive baseload demand.  Accepts two forms:

        * **Time-indexed** (``time`` dim): standard 1-hourly dataset.
        * **Typical-year** (``hour_of_year`` dim, length 8760): output of
          :func:`expand_typical_year` / the ``groupby("hour_of_year").mean``
          pattern.  The function auto-expands it across the years present in
          ``ds_gen_pv``, skipping Feb 29 in leap years.

        Variable name: ``"baseload"``.
    ds_heating : xr.Dataset
        Heating electricity demand, 3-hourly (C2E/CERRA resolution).
        Variable name: ``"heating_demand"``.
    ds_cooling : xr.Dataset
        Cooling electricity demand, 3-hourly (C2E/CERRA resolution).
        Variable name: ``"cooling_demand"``.
    target_twh_per_country : dict or pd.Series, optional
        Per-country annual thermal electricity target in TWh, passed directly
        to :func:`preprocess_thermal_demand`.  ``None`` skips rescaling.
    loss_factor : float, optional
        COP correction multiplier passed to :func:`preprocess_thermal_demand`.
        Default 1.176 (= 1/0.85, per ENTSO-E DFT).
    rolling_window_thermal_demand : int, optional
        Thermal inertia smoothing window in hours.  Default 4.
    rolling_window_residual_load : int, optional
        Residual load smoothing window in timesteps.  Default 8.
    country_dim : str, optional
        Name of the country dimension across all datasets.  Default ``"country"``.

    Returns
    -------
    xr.Dataset
        Residual load with a single variable ``"residual_load"``,
        dims ``(country, time)`` at 3-hourly resolution.
        Positive values indicate unmet demand; negative values indicate
        generation surplus.
    """
    # ── Helper: sum all data variables in a Dataset into one DataArray ────────
    def _sum_vars(ds):
        arrays = [ds[v] for v in ds.data_vars]
        return arrays[0] if len(arrays) == 1 else sum(arrays[1:], start=arrays[0])

    # ── Auto-expand typical-year baseload if needed ───────────────────────────
    # If ds_baseload was built with groupby("hour_of_year").mean() it has an
    # hour_of_year dim rather than a time axis.  Expand it here using the
    # calendar years present in the generation datasets.
    if "hour_of_year" in ds_baseload.dims:
        _years = np.unique(ds_gen_pv.time.dt.year.values)
        ds_baseload = expand_typical_year(ds_baseload, years=_years)

    # ── Step 1: Build thermal demand (per-variable, or lumped for legacy) ──────
    # Nested target {var: {country: twh}} -> keep heating & cooling as separate
    # variables so each is rescaled to its own reference-period target. A flat
    # {country: twh} target (or None) reproduces the legacy lumped single series.
    common_time_th = np.intersect1d(ds_heating.time.values, ds_cooling.time.values)
    _nested = (
        isinstance(target_twh_per_country, dict)
        and isinstance(next(iter(target_twh_per_country.values()), None), dict)
    )
    if _nested:
        ds_thermal = xr.Dataset({
            "heating_demand": ds_heating["heating_demand"].sel(time=common_time_th),
            "cooling_demand": ds_cooling["cooling_demand"].sel(time=common_time_th),
        })
    else:
        ds_thermal = xr.Dataset({
            "demand_profile": (
                ds_heating["heating_demand"].sel(time=common_time_th)
                + ds_cooling["cooling_demand"].sel(time=common_time_th)
            )
        })

    # ── Step 2: Preprocess thermal demand → 1-hourly GW (per variable) ────────
    ds_thermal_1h = preprocess_thermal_demand(
        ds_thermal,
        target_twh_per_country=target_twh_per_country,
        loss_factor=loss_factor,
        rolling_window=rolling_window_thermal_demand,
        upsample_to_1h=upsample_to_1h,
        reference_period=reference_period,
    )

    # ── Step 3: Merge baseload + thermal → total 1-hourly demand ──────────────
    # Sum the (1 or 2) thermal variables back together before adding to baseload.
    thermal_total_1h = _sum_vars(ds_thermal_1h)
    common_countries_demand = np.intersect1d(
        ds_baseload[country_dim].values,
        thermal_total_1h[country_dim].values,
    )
    common_time_1h = np.intersect1d(
        ds_baseload.time.values,
        thermal_total_1h.time.values,
    )
    total_demand_1h = (
        ds_baseload["baseload"].sel(
            {country_dim: common_countries_demand, "time": common_time_1h}
        )
        + thermal_total_1h.sel(
            {country_dim: common_countries_demand, "time": common_time_1h}
        )
    )

    # ── Step 4: Resample demand to 3-hourly sums ───────────────────────────────
    # Generation datasets carry 3-hourly values; summing three 1-hour demand
    # steps gives comparable GWh-per-3h totals.
    total_demand_3h = total_demand_1h.resample(time="3h").sum()

    # ── Step 5: Country set = all demand countries; time = intersection ───────
    # Countries present in demand but absent from a generation dataset are
    # treated as having zero generation for that technology (e.g. landlocked
    # countries with no offshore wind, countries without hydro).
    # Time is still intersected: generation data may not span the full demand
    # period and we must not extrapolate.
    all_countries = common_countries_demand

    common_time = total_demand_3h.time.values
    for ds in [ds_gen_pv, ds_gen_wind_on, ds_gen_wind_off, ds_hydro_ror]:
        common_time = np.intersect1d(common_time, ds.time.values)

    # ── Step 6: Align and sum generation, filling missing countries with 0 ────
    def _gen(ds):
        return (
            _sum_vars(ds)
            .reindex({country_dim: all_countries}, fill_value=0.0)
            .sel(time=common_time)
        )

    components = xr.Dataset(
        {
            "gen_pv": _gen(ds_gen_pv),
            "gen_wind_on": _gen(ds_gen_wind_on),
            "gen_wind_off": _gen(ds_gen_wind_off),
            "gen_hydro_ror": _gen(ds_hydro_ror),
        }
    )
    for component in components.data_vars:
        components[component].attrs["units"] = "GWh"
    gen_total = components["gen_pv"] + components["gen_wind_on"] + components["gen_wind_off"] + components["gen_hydro_ror"]

    # ── Step 7: Residual = demand − generation ────────────────────────────────
    demand_aligned = total_demand_3h.sel(
        {country_dim: all_countries, "time": common_time}
    )
    residual = demand_aligned - gen_total

    if rolling_window_residual_load > 1:
        residual = residual.rolling(time=rolling_window_residual_load, center=False, min_periods=1).mean()

    residual = residual.to_dataset(name="residual_load")
    residual.attrs["description"] = "Residual load = total demand − (PV + onshore wind + offshore wind + ROR hydro)"
    residual["residual_load"].attrs["units"] = "GWh"
    residual["residual_load"].attrs["long_name"] = "Residual Load"

    # --- Step 8: Combine datasets for traceability (optional) ---------------------------------------------
    if return_generation_demand_components:
        components["total_demand"] = demand_aligned
        components["total_demand"].attrs["units"] = "GWh"
        return residual, components
    else:
        return residual


# ---------------------------------------------------------------------------
# Hydro run-of-river pre-processing
# ---------------------------------------------------------------------------


def preprocess_hydro_ror(
    df_ror,
    name_to_iso2,
    freq="3h",
    var_name="ror",
):
    """Convert ROR hydro country names to ISO2 codes and distribute daily
    values to sub-daily resolution.

    C2E/CERRA run-of-river output CSVs carry full English country names as
    the row index and daily timestamps as columns.  This function:

    1. Maps each country label through ``name_to_iso2`` (unrecognised names
       are kept as-is).
    2. Constructs an xarray Dataset at daily resolution.
    3. Distributes each daily value evenly across ``n = 24h / freq``
       sub-daily time-steps by forward-filling and then dividing by ``n``.
       The result therefore carries units of ``[original_unit / n]``
       (e.g. GWh/day → GWh/3h when ``freq="3h"``).

    Parameters
    ----------
    df_ror : pd.DataFrame
        Daily ROR generation with shape ``(country, time)``.  The row index
        holds country labels (full names or codes); columns are
        ``pd.DatetimeIndex``-parseable strings.
    name_to_iso2 : dict
        Mapping from full country name to ISO2 code
        (e.g. ``{"France": "FR", ...}``).  Entries absent from the dict are
        passed through unchanged.
    freq : str, optional
        Target sub-daily frequency understood by pandas / xarray
        (e.g. ``"3h"``, ``"1h"``).  Default ``"3h"``.
    var_name : str, optional
        Name of the output data variable.  Default ``"ror"``.

    Returns
    -------
    xr.Dataset
        Sub-daily ROR dataset, dims ``(country, time)``.
    """
    # ── 1. Map country names → ISO2 codes ─────────────────────────────────────
    iso2_index = [
        name_to_iso2.get(str(c).strip(), str(c).strip())
        for c in df_ror.index
    ]

    # ── 2. Build daily xarray Dataset ─────────────────────────────────────────
    ds_daily = xr.DataArray(
        df_ror.values,
        dims=["country", "time"],
        coords={
            "country": iso2_index,
            "time": pd.to_datetime(df_ror.columns),
        },
        name=var_name,
    ).to_dataset()

    # ── 3. Distribute daily → sub-daily ───────────────────────────────────────
    # Number of sub-daily steps per day determines the divisor so that the
    # sum over a day's steps recovers the original daily total.
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))

    # forward-fill repeats each day's value across all intra-day steps, then
    # dividing by steps_per_day gives the correct per-step fraction.
    ds_subdaily = ds_daily.resample(time=freq).ffill() / steps_per_day

    return ds_subdaily


def aggregate_by_balancing_zones(ds, country_dim="country"):
    """
    Groups and sums an xarray Dataset or DataArray along the country dimension
    into the 6 real-time European balancing zones.

    Parameters:
    -----------
    ds : xr.Dataset or xr.DataArray
         The input dataset containing a country dimension.
    country_dim : str
         The exact name of the country dimension coordinate (e.g., 'country' or 'zone').

    Returns:
    --------
    xr.Dataset or xr.DataArray
         The aggregated dataset with the 'country' dimension replaced by 'zone'.
    """
    # 2. Invert the dictionary to create a flat country-to-zone lookup map
    # e.g., {'NO': 'Nordics', 'SE': 'Nordics', 'FR': 'Central_Western', ...}
    country_to_zone = {
        country: zone
        for zone, countries in REALTIME_BALANCING_ZONES.items()
        for country in countries
    }

    # 3. Handle data safety: Filter out countries in the dataset that aren't in our map
    available_countries = [c for c in ds[country_dim].values if c in country_to_zone]
    ds_filtered = ds.sel({country_dim: available_countries})

    # 4. Generate the new zone coordinate labels corresponding to the filtered countries
    zone_labels = [country_to_zone[c] for c in available_countries]

    # 5. Assign the zones as a non-index coordinate aligned with the country dimension
    ds_with_zones = ds_filtered.assign_coords(zone=(country_dim, zone_labels))

    # 6. Group by the new 'zone' coordinate and sum across the country dimension
    # This keeps 'time' and any other dimensions perfectly intact
    ds_zonal = ds_with_zones.groupby("zone").sum(dim=country_dim)

    return ds_zonal


# ---------------------------------------------------------------------------
# Supply capacity loading
# ---------------------------------------------------------------------------

def _parse_supply_sheet(path_excel: Path, sheet_name: str) -> pd.DataFrame:
    """Load and normalise a single TYNDP supply capacity Excel sheet.

    Flattens the two-row header to ``{year}_{scenario}`` column names (e.g.
    ``"2050_HIGH"``), slices from the ``"Country"`` column to the last
    ``"2050_HIGH"`` column, and coerces all numeric columns to float.
    """
    df = pd.read_excel(path_excel, sheet_name=sheet_name, header=[2, 3])
    df.columns = [
        l2 if str(l2) == "Country" else f"{l2}_{l1}"
        for l1, l2 in df.columns
    ]
    df = df.loc[:, "Country":"2050_HIGH"]
    numeric_cols = [c for c in df.columns if c != "Country"]
    df[numeric_cols] = df[numeric_cols].apply(
        lambda s: pd.to_numeric(
            s.astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
    )
    return df


def load_tyndp_supply_capacities(
    path_excel: Union[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TYNDP 2024 supply capacity projections from the standard Excel file.

    Reads sheets ``1.1`` (solar PV), ``1.2`` (onshore wind) and ``1.3``
    (offshore wind), normalises column names to ``{year}_{scenario}`` format
    (e.g. ``"2050_HIGH"``), and coerces all capacity values to float.

    Parameters
    ----------
    path_excel:
        Path to
        ``20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx``
        (or equivalent file).

    Returns
    -------
    df_solar, df_wind_onshore, df_wind_offshore : pd.DataFrame
        Each DataFrame has a ``"Country"`` column (TYNDP bidding-zone codes)
        and numeric capacity columns named ``{year}_{scenario}``.
    """
    path_excel = Path(path_excel)
    df_solar = _parse_supply_sheet(path_excel, "1.1.")
    df_wind_onshore = _parse_supply_sheet(path_excel, "1.2.")
    df_wind_offshore = _parse_supply_sheet(path_excel, "1.3.")
    return df_solar, df_wind_onshore, df_wind_offshore


def load_tyndp_battery_capacities(
    path_excel: Union[str, Path],
    scenario: str = "HIGH",
    year: int = 2050,
    return_separate: bool = False,
) -> "pd.Series | tuple[pd.Series, pd.Series]":
    """Load TYNDP 2024 battery energy storage capacity projections.

    Reads sheet ``1.5.`` (prosumer batteries) and ``1.6.`` (utility batteries).
    By default sums both and returns **GWh per ISO2 country code**.

    Parameters
    ----------
    path_excel : str or Path
        Path to the TYNDP supply Excel file.
    scenario : str
        Scenario name as it appears in the column header (e.g. ``"HIGH"``).
    year : int
        Projection year (e.g. ``2050``).
    return_separate : bool
        When ``True`` return ``(s_prosumer_gwh, s_utility_gwh)`` as two
        separate Series instead of their sum.  Useful when prosumer and
        utility batteries are dispatched with different strategies.

    Returns
    -------
    pd.Series or tuple[pd.Series, pd.Series]
        Total (or prosumer + utility separately) battery energy capacity in
        **GWh**, indexed by two-character ISO2 country code.
    """
    path_excel = Path(path_excel)
    col = f"{year}_{scenario}"

    _aggregate_rows = {"TOTAL", "EU-TOTAL"}

    def _to_iso2_series(df: pd.DataFrame) -> pd.Series:
        mask = df["Country"].apply(
            lambda x: isinstance(x, str) and x not in _aggregate_rows
        )
        d = df.loc[mask, ["Country", col]].copy()
        d["iso2"] = d["Country"].str[:2]
        return d.groupby("iso2")[col].sum().fillna(0.0)

    s_prosumer = _to_iso2_series(_parse_supply_sheet(path_excel, "1.5.")) / 1_000.0
    s_utility  = _to_iso2_series(_parse_supply_sheet(path_excel, "1.6.")) / 1_000.0
    s_prosumer.name = f"prosumer_battery_capacity_gwh_{year}_{scenario}"
    s_utility.name  = f"utility_battery_capacity_gwh_{year}_{scenario}"

    if return_separate:
        return s_prosumer, s_utility

    total_gwh = s_prosumer.add(s_utility, fill_value=0.0)
    total_gwh.name = f"battery_capacity_gwh_{year}_{scenario}"
    return total_gwh


def aggregate_battery_capacities_by_zone(
    s_battery_country: pd.Series,
) -> pd.Series:
    """Aggregate country-level battery capacities to TYNDP balancing zones.

    Uses the same ``REALTIME_BALANCING_ZONES`` mapping as
    :func:`aggregate_by_balancing_zones`.  Countries absent from the input
    Series contribute zero.

    Parameters
    ----------
    s_battery_country : pd.Series
        Battery energy capacity in GWh, indexed by TYNDP country codes.

    Returns
    -------
    pd.Series
        Total battery energy capacity in GWh per balancing zone.
    """
    result = {}
    for zone, countries in REALTIME_BALANCING_ZONES.items():
        result[zone] = sum(float(s_battery_country.get(c, 0.0)) for c in countries)
    return pd.Series(result, name=s_battery_country.name)


# ---------------------------------------------------------------------------
# Country code mapping
# ---------------------------------------------------------------------------

#: Non-EU countries that appear in TYNDP bidding zones but are absent from
#: the EU-27-only demand spreadsheet.
_NON_EU_COUNTRY_CODES: dict[str, str] = {
    "AL": "Albania",
    "BA": "Bosnia and Herzegovina",
    "CH": "Switzerland",
    "HR": "Croatia",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "NO": "Norway",
    "RS": "Serbia",
    "SI": "Slovenia",
}

#: C2E/CERRA/CORDEX CSVs use slightly different names than TYNDP in a few
#: cases.  These are merged into the name→ISO2 lookup by default.
_DEFAULT_C2E_EXTRA_MAPPINGS: dict[str, str] = {
    "Macedonia": "MK",   # short form; TYNDP uses 'North Macedonia'
    "Poland": "PL",      # short form; TYNDP uses 'Poland (PL)'
    "Luxembourg": "LU",
    "Kosovo": "XK",      # not in TYNDP EU demand model
    "Ukraine": "UA",     # not in TYNDP EU demand model
}


def load_tyndp_country_codes(path_demand_excel: Union[str, Path]) -> dict[str, str]:
    """Load the ISO2 → country-name mapping from the TYNDP demand workbook.

    Reads the ``1_ETM_SCENARIO_LINKS`` sheet and extends the EU-27 list with
    common non-EU countries that appear in TYNDP bidding zones but are absent
    from the standard demand spreadsheet.

    Parameters
    ----------
    path_demand_excel:
        Path to
        ``Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb``
        (or equivalent).

    Returns
    -------
    dict
        ``{ISO2_code: country_name}`` mapping (whitespace-stripped).
    """
    df = pd.read_excel(
        path_demand_excel, sheet_name="1_ETM_SCENARIO_LINKS", header=0
    )
    codes = (
        df[["COUNTRY_ID", "COUNTRY_NAME"]]
        .drop_duplicates()
        .set_index("COUNTRY_ID")["COUNTRY_NAME"]
        .to_dict()
    )
    codes.update(_NON_EU_COUNTRY_CODES)
    return {k.strip(): v.strip() for k, v in codes.items()}


def build_c2e_name_to_iso2(
    tyndp_country_codes: dict[str, str],
    extra_mappings: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a full-name → ISO2 lookup for C2E/CERRA/CORDEX CSV country labels.

    Inverts ``tyndp_country_codes`` (``{ISO2: name}``) and merges the default
    C2E name overrides.  Additional project-specific overrides can be supplied
    via ``extra_mappings``.

    Parameters
    ----------
    tyndp_country_codes:
        ``{ISO2: full_name}`` mapping, e.g. from
        :func:`load_tyndp_country_codes`.
    extra_mappings:
        Additional ``{csv_label: ISO2}`` pairs.  Applied after the defaults.

    Returns
    -------
    dict
        ``{country_label: ISO2}`` mapping.
    """
    name_to_iso2 = {v: k for k, v in tyndp_country_codes.items()}
    name_to_iso2.update(_DEFAULT_C2E_EXTRA_MAPPINGS)
    if extra_mappings:
        name_to_iso2.update(extra_mappings)
    return name_to_iso2


def load_c2e_demand_nc(
    path_demand_nc: Union[str, Path],
    name_to_iso2: dict,
    time_slice=None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load the JRC-coefficient (``cordex2energy_demand``) hourly demand NetCDF.

    Returns ``(ds_heating_demand, ds_cooling_demand)`` in exactly the shape the
    demand.ninja CSV loaders yield — two single-variable datasets
    (``heating_demand`` / ``cooling_demand``, dims ``(country, time)``, GW) with
    the ``country`` coordinate remapped to ISO2 via *name_to_iso2* — so the rest
    of the pipeline is agnostic to which demand product was used.  The ``.nc`` is
    **hourly** (vs the 3-hourly CSVs); callers pass ``upsample_to_1h=False`` to
    :func:`preprocess_thermal_demand` accordingly.

    Parameters
    ----------
    path_demand_nc:
        Path to ``energy/europe/demand/{model_scenario}.nc`` (or ``cerra.nc``),
        the output of ``compute_demand.py`` (vars ``heating_demand`` /
        ``cooling_demand`` [GW], dims ``time``/``country`` with 37 canonical
        country names).
    name_to_iso2:
        Full-name → ISO2 mapping (from :func:`build_c2e_name_to_iso2`).  The
        ``.nc`` uses the same canonical names as the demand.ninja CSVs, so this
        yields the same ISO2 country set.
    time_slice:
        Optional ``(start, end)`` passed to ``.sel(time=slice(...))`` (e.g. the
        reference period for CERRA).

    Returns
    -------
    (xr.Dataset, xr.Dataset)
        ``ds_heating_demand`` (var ``heating_demand``) and ``ds_cooling_demand``
        (var ``cooling_demand``).
    """
    ds = xr.open_dataset(path_demand_nc)
    iso2 = [name_to_iso2.get(str(c).strip(), str(c).strip()) for c in ds["country"].values]

    # Diagnostic: flag any country that failed to map to a 2-letter ISO2 code —
    # such a country would silently drop out of the ISO2-based intersections
    # downstream instead of raising.
    unmapped = sorted({str(c) for c, code in zip(ds["country"].values, iso2)
                       if not (isinstance(code, str) and len(code) == 2)})
    if unmapped:
        print(f"  [WARN] load_c2e_demand_nc: {len(unmapped)} country name(s) did not "
              f"map to ISO2 (will not intersect downstream): {unmapped}")

    ds = ds.assign_coords(country=iso2)
    if time_slice is not None:
        ds = ds.sel(time=slice(*time_slice))

    ds_heating = ds[["heating_demand"]]
    ds_cooling = ds[["cooling_demand"]]
    return ds_heating, ds_cooling


# ---------------------------------------------------------------------------
# Demand profile loading helpers
# ---------------------------------------------------------------------------

def load_tyndp_market_profiles(
    path_demand_profiles_dir: Union[str, Path],
    demand_scenario: str,
    demand_year: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load TYNDP market and prosumer hourly demand profiles for a scenario/year.

    Expects the standard ENTSO-E demand-profile directory layout::

        path_demand_profiles_dir/
            {demand_scenario}/
                {demand_year}/
                    ELECTRICITY_MARKET {demand_scenario} {demand_year}.xlsx
                    ELECTRICITY_PROSUMER {demand_scenario} {demand_year}.xlsx

    Parameters
    ----------
    path_demand_profiles_dir:
        Root of the demand-profiles directory.
    demand_scenario:
        Scenario identifier (e.g. ``"DE"``, ``"GA"``, ``"REF"``).
    demand_year:
        Target year (e.g. 2040, 2050).

    Returns
    -------
    ds_market, ds_prosumer : xr.Dataset
        Hourly demand-profile datasets with dimensions ``(country, time)``
        and variable ``"demand_profile"`` in MWh.
    """
    base = Path(path_demand_profiles_dir) / demand_scenario / str(demand_year)
    path_market = base / f"ELECTRICITY_MARKET {demand_scenario} {demand_year}.xlsx"
    path_prosumer = base / f"ELECTRICITY_PROSUMER {demand_scenario} {demand_year}.xlsx"
    ds_market = load_tyndp_demand_profiles(path_market)
    ds_prosumer = load_tyndp_demand_profiles(path_prosumer)
    return ds_market, ds_prosumer


# ---------------------------------------------------------------------------
# Typical-year profile
# ---------------------------------------------------------------------------

def compute_typical_year_profile(ds: xr.Dataset) -> xr.Dataset:
    """Average a time-indexed dataset into an 8760-step typical-year profile.

    Groups all time steps by ``(dayofyear − 1) × 24 + hour`` and takes the
    mean, yielding a dataset with an ``hour_of_year`` coordinate (0–8759).
    Leap-day entries (Feb 29) are excluded before grouping so the output
    always has exactly 8760 values.

    The resulting dataset can be passed directly to
    :func:`compute_residual_load` as ``ds_baseload``; the function detects
    the ``hour_of_year`` dimension and expands it automatically via
    :func:`expand_typical_year`.

    Parameters
    ----------
    ds:
        Time-indexed dataset with a ``time`` dimension.

    Returns
    -------
    xr.Dataset
        Same variables as ``ds`` but with the ``time`` dimension replaced by
        ``hour_of_year`` (length 8760).
    """
    # Exclude Feb 29 to guarantee exactly 8760 unique hour-of-year values.
    mask_leap = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
    ds_no_leap = ds.sel(time=~mask_leap)
    hoy = (ds_no_leap.time.dt.dayofyear - 1) * 24 + ds_no_leap.time.dt.hour
    return (
        ds_no_leap
        .assign_coords(hour_of_year=("time", hoy.values))
        .groupby("hour_of_year")
        .mean("time")
    )


def compute_or_load_baseload(
    path_c2e_cerra: Path,
    path_baseload_cache: Path,
    path_tyndp: Path,
    demand_scenario: str,
    demand_year: int,
    df_tyndp_demand: pd.DataFrame,
    C2E_NAME_TO_ISO2: dict,
    floor_fraction: float = 0.15,
    demand_model: str = "demand.ninja",
    path_energy_demand: Path = None,
    modulate_baseload: bool = True,
    reference_period=REFERENCE_PERIOD,
    legacy_thermal: bool = False,
) -> xr.Dataset:
    """Return the typical-year TYNDP baseload, loading from cache when available.

    The baseload depends on the TYNDP demand scenario/year **and the demand
    model** (the CERRA heating/cooling that is subtracted must match the model
    added back in :func:`compute_residual_load`), so it is computed once from
    CERRA reference-period demand + TYNDP profiles and cached per
    ``(scenario, year, demand_model, legacy)``.

    Extra parameters
    ----------------
    demand_model : str
        ``"demand.ninja"`` (3-hourly CERRA CSVs) or ``"cordex2energy_demand"``
        (hourly ``cerra.nc`` under ``path_energy_demand``).
    path_energy_demand : Path
        ``energy/europe/demand`` directory (required for
        ``cordex2energy_demand`` — holds ``cerra.nc``).
    modulate_baseload : bool
        If True (default) scale each **TYNDP-scenario** country's baseload so its
        annual total equals ``total − heating − cooling``.  Non-scenario
        countries keep their unmodulated baseload.  Ignored under
        ``legacy_thermal``.
    reference_period : sequence of two str
        Years for the CERRA slice and the Step-3 rescale anchor.  Default
        :data:`REFERENCE_PERIOD`.
    legacy_thermal : bool
        Reproduce the historical behaviour (lumped heating+cooling target,
        whole-series ``k``, no modulation).

    Parameters
    ----------
    path_c2e_cerra:
        Root directory of C2E/CERRA output CSVs (contains ``output_variables/`` subdirectory).
    path_baseload_cache:
        Directory where ``baseload_{demand_scenario}_{demand_year}.nc`` is stored.
    path_tyndp:
        Root TYNDP data directory (contains ``Demand Profiles/`` subdirectory).
    demand_scenario:
        TYNDP demand scenario identifier (e.g. "DE").
    demand_year:
        Target year (e.g. 2050).
    df_tyndp_demand:
        Flattened TYNDP ``3_DEMAND_OUTPUT`` DataFrame.
    C2E_NAME_TO_ISO2:
        Full-name → ISO2 country code mapping.
    floor_fraction:
        Floor cap for baseload as fraction of mean market demand.

    Returns
    -------
    xr.Dataset
        Typical-year baseload with ``hour_of_year`` dimension and
        ``"baseload"`` variable.
    """
    dm_tag = {"demand.ninja": "dninja", "cordex2energy_demand": "c2edem"}.get(
        demand_model, demand_model
    )
    cache_path = path_baseload_cache / (
        f"baseload_{demand_scenario}_{demand_year}_{dm_tag}"
        f"{'_legacy' if legacy_thermal else ''}.nc"
    )

    if cache_path.exists():
        print(f"  Found cached baseload -> {cache_path}")
        return xr.open_dataset(cache_path)

    print(f"  Baseload cache not found - computing for {demand_scenario} {demand_year} "
          f"[{demand_model}{', legacy' if legacy_thermal else ''}] ...")

    # Load TYNDP demand profiles for this scenario/year.
    path_demand_profiles_dir = path_tyndp / "Demand Profiles"
    ds_market, ds_prosumer = load_tyndp_market_profiles(
        path_demand_profiles_dir, demand_scenario, demand_year
    )

    # Build per-type (heating/cooling) + total electricity targets.
    targets = build_thermal_targets(
        df_tyndp_demand,
        scenario=demand_scenario,
        year=demand_year,
        ds_market=ds_market,
    )

    # Load reference-period CERRA heating/cooling demand from the SELECTED demand
    # model, so the thermal subtracted here matches what compute_residual_load
    # adds back (subtract-then-add symmetry). Two single-var datasets.
    if demand_model == "cordex2energy_demand":
        if path_energy_demand is None:
            raise ValueError(
                "path_energy_demand is required for demand_model='cordex2energy_demand'"
            )
        ds_heating_cerra, ds_cooling_cerra = load_c2e_demand_nc(
            Path(path_energy_demand) / "cerra.nc",
            C2E_NAME_TO_ISO2,
            time_slice=reference_period,
        )
        upsample, loss = False, 1.0          # already hourly; already electric
    else:
        path_c2e_cerra_dir = path_c2e_cerra / "output_variables"
        df_heating_cerra = pd.read_csv(
            path_c2e_cerra_dir / "heating-demand_CERRA.csv", index_col=0
        )
        df_cooling_cerra = pd.read_csv(
            path_c2e_cerra_dir / "cooling-demand_CERRA.csv", index_col=0
        )
        ds_heating_cerra = (
            xr.DataArray(
                df_heating_cerra.values, dims=["country", "time"],
                coords={"country": [C2E_NAME_TO_ISO2.get(c, c) for c in df_heating_cerra.index],
                        "time": pd.to_datetime(df_heating_cerra.columns)},
                name="heating_demand",
            ).to_dataset().sel(time=slice(*reference_period))
        )
        ds_cooling_cerra = (
            xr.DataArray(
                df_cooling_cerra.values, dims=["country", "time"],
                coords={"country": [C2E_NAME_TO_ISO2.get(c, c) for c in df_cooling_cerra.index],
                        "time": pd.to_datetime(df_cooling_cerra.columns)},
                name="cooling_demand",
            ).to_dataset().sel(time=slice(*reference_period))
        )
        upsample, loss = True, 1 / 0.85      # 3-hourly; ENTSO-E COP correction

    # Two-variable thermal dataset over the common time axis.
    _common_time = np.intersect1d(
        ds_heating_cerra.time.values, ds_cooling_cerra.time.values
    )
    ds_thermal_cerra = xr.Dataset({
        "heating_demand": ds_heating_cerra["heating_demand"].sel(time=_common_time),
        "cooling_demand": ds_cooling_cerra["cooling_demand"].sel(time=_common_time),
    })

    # Preprocess, then sum to the single `demand_profile` that compute_tyndp_baseload
    # subtracts. Heating/cooling are rescaled to SEPARATE targets ONLY for
    # cordex2energy_demand; demand.ninja uses a single LUMPED target (its
    # reference-period cooling is ~0 in cool countries, so a split k_cool =
    # cool_target / ~0 explodes — the lumped, heating-dominated denominator is
    # safe). Legacy additionally drops the reference-period anchor (whole series).
    split_targets = (demand_model == "cordex2energy_demand") and not legacy_thermal
    if split_targets:
        ds_pp = preprocess_thermal_demand(
            ds_thermal_cerra,
            target_twh_per_country={"heating_demand": targets["heating"],
                                    "cooling_demand": targets["cooling"]},
            loss_factor=loss, rolling_window=4, upsample_to_1h=upsample,
            reference_period=reference_period,
        )
        thermal_sum = ds_pp["heating_demand"] + ds_pp["cooling_demand"]
    else:
        ds_thermal_in = xr.Dataset({
            "demand_profile": ds_thermal_cerra["heating_demand"] + ds_thermal_cerra["cooling_demand"]
        })
        lump = {c: targets["heating"].get(c, 0.0) + targets["cooling"].get(c, 0.0)
                for c in set(targets["heating"]) | set(targets["cooling"])}
        ds_pp = preprocess_thermal_demand(
            ds_thermal_in, target_twh_per_country=lump, loss_factor=loss,
            rolling_window=4, upsample_to_1h=upsample,
            reference_period=None if legacy_thermal else reference_period,
        )
        thermal_sum = ds_pp["demand_profile"]
    ds_thermal_preprocessed = xr.Dataset({"demand_profile": thermal_sum})

    # Baseload = (market + prosumer) - thermal, floored. market/prosumer MWh -> GWh.
    ds_total_market = (ds_market + ds_prosumer.reindex(
        country=ds_market.country, fill_value=0.0
    )) * 1e-3
    ds_baseload = compute_tyndp_baseload(
        ds_total_market,
        ds_thermal_preprocessed,
        floor_fraction=floor_fraction,
    )

    # Optional baseload modulation: scale each country's baseload so its annual
    # total matches (TYNDP total electricity - heating - cooling). Restricted to
    # countries present in the TYNDP demand scenario (the EU-27 "total" dict);
    # proxy/ds_market-synthesised countries keep their unmodulated baseload.
    if modulate_baseload and not legacy_thermal:
        total = targets["total"]
        n_years = max(1, len(np.unique(ds_baseload.time.dt.year.values)))
        m_vals = []
        for c in ds_baseload.country.values:
            cc = str(c)
            base_twh = float(ds_baseload["baseload"].sel(country=c).sum().values) * 1e-3 / n_years
            tgt = (total.get(cc, 0.0)
                   - targets["heating"].get(cc, 0.0)
                   - targets["cooling"].get(cc, 0.0))
            m_vals.append(tgt / base_twh if (cc in total and base_twh > 0 and tgt > 0) else 1.0)
        m_da = xr.DataArray(m_vals, dims=["country"],
                            coords={"country": ds_baseload.country.values})
        ds_baseload = ds_baseload * m_da
        # Re-clamp to the floor (modulation can push a country below it).
        floor = (floor_fraction * ds_total_market["demand_profile"].mean(dim="time")).to_dataset(name="baseload")
        ds_baseload = ds_baseload.where(ds_baseload >= floor, other=floor)
        n_mod = int(sum(1 for c in ds_baseload.country.values if str(c) in total))
        print(f"  Modulated baseload for {n_mod} TYNDP-scenario countries "
              f"(total - heating - cooling).")

    # Build a typical-year (hour_of_year) profile by averaging TYNDP climate years.
    ds_baseload_typical = compute_typical_year_profile(ds_baseload)

    # Cache to disk.
    ds_baseload_typical.to_netcdf(cache_path)
    print(f"  Saved baseload cache -> {cache_path}")

    return ds_baseload_typical


def _optimize_battery_gurobi(
    residual_load: np.ndarray,
    E_max: float,
    P_max: float,
    one_way_eff: float,
    min_soc_fraction: float | None = None,
    initial_soc: float | None = None,
    env=None,
    max_signfix_passes: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Solve a battery dispatch QP for one horizon using Gurobi.

    Mirrors :func:`_optimize_reservoir_gurobi` but models a **bidirectional**
    store with round-trip losses.  Minimises ``Σ net_load[t]²`` where
    ``net_load = residual_load + charge − discharge`` subject to a
    state-of-charge balance, charge/discharge power bounds, storage bounds, and
    a boundary condition on storage (see below).

    Complementarity
    ---------------
    The objective depends only on ``charge − discharge``, so the two variables are
    individually degenerate: inflating both lowers the SOC
    (``η·charge − discharge/η``) **without changing the net load**.  That is a free
    energy-disposal valve — the battery could absorb surplus, discard it, and absorb
    again, never filling up.  Round-trip losses do *not* prevent this here, because
    ``Σ net_load²`` penalises surplus (negative residual load) and therefore *rewards*
    disposal.  (The familiar "losses make simultaneous charge/discharge dominated"
    result applies to cost/arbitrage objectives that are monotone in energy consumed.)

    Left unconstrained the relaxation is strongly optimistic: it will flatten a
    pure-surplus year to zero using a battery of arbitrarily small energy capacity.
    We therefore enforce ``charge[t] · discharge[t] == 0`` by **sign-fixing**: solve
    the relaxation, clamp each step to charge-only or discharge-only according to the
    sign of ``charge − discharge``, and re-solve until the sign pattern is stable.
    Each pass is a plain convex QP (no binaries).  The clamped problem is strictly
    more constrained than the relaxation, so the result is always *conservative*
    (it can never overstate the battery's benefit) and is always feasible (the
    do-nothing solution satisfies every clamp).

    Parameters
    ----------
    residual_load :
        Residual load over the horizon (GWh per 3h step).  Must be finite.
    E_max :
        Battery energy capacity (GWh).
    P_max :
        Maximum charge/discharge per 3h step (GWh = installed GW × 3).
    one_way_eff :
        One-way efficiency (``sqrt(roundtrip_efficiency)``); applied on charge
        and on discharge.
    min_soc_fraction :
        If set, SOC cannot drop below ``min_soc_fraction × E_max`` at any step.
    initial_soc :
        Storage boundary condition.  If ``None`` (default), a cyclicity constraint
        ``E[0] == E[T]`` is imposed (perfect foresight over the horizon).  If set,
        ``E[0]`` is fixed to it and **no** cyclicity applies (open horizon; terminal
        SOC is free) — used by the rolling day-ahead scheduler, which seeds each
        window with the SOC carried from the previous window.
    env :
        Optional pre-started Gurobi ``Env`` to reuse across many solves (avoids
        per-call licence/startup cost).  If ``None``, a local env is created and
        disposed within the call.
    max_signfix_passes :
        Maximum number of sign-fixing re-solves used to enforce
        ``charge · discharge == 0`` (see *Complementarity* above).  The pattern is
        typically stable after 2-3 passes.

    Returns
    -------
    (dispatch, net_load, soc) arrays of length T, or (None, None, None) if Gurobi
    does not find an optimal solution.  ``dispatch = discharge − charge``
    (positive = discharging, reduces residual load).  ``soc[k]`` is the storage
    level entering step ``k`` (i.e. ``E[:-1]``).
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = len(residual_load)
    E_min = E_max * min_soc_fraction if min_soc_fraction is not None else 0.0

    _owns_env = env is None
    if _owns_env:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogFile", "")
        env.start()

    try:
        m = gp.Model("Battery_Dispatch", env=env)

        E = m.addMVar(shape=T + 1, lb=E_min, ub=E_max, name="E")
        c = m.addMVar(shape=T, lb=0.0, ub=P_max, name="charge")
        d = m.addMVar(shape=T, lb=0.0, ub=P_max, name="discharge")
        net_load = m.addMVar(shape=T, lb=-GRB.INFINITY, name="net_load")

        m.addConstr(E[1:] == E[:-1] + one_way_eff * c - d / one_way_eff, name="SOC_Balance")
        m.addConstr(net_load == residual_load + c - d, name="Net_Load_Def")

        if initial_soc is not None:
            # Open-horizon window: fixed start SOC, free terminal SOC (no cyclicity).
            m.addConstr(E[0] == initial_soc, name="Initial_SOC")
        else:
            m.addConstr(E[0] == E[T], name="Cyclicity")

        m.setObjective(net_load @ net_load, GRB.MINIMIZE)
        m.optimize()
        # Accept a feasible incumbent even if the barrier could not certify it to
        # tolerance (SUBOPTIMAL/NUMERIC); only bail when no solution exists at all.
        if m.SolCount == 0:
            return None, None, None

        # ── Sign-fixing: enforce charge[t] * discharge[t] == 0 ────────────────
        # The relaxation above is degenerate in (c, d) and exploits it: inflating
        # both drains the SOC without changing net load, letting the battery absorb
        # surplus it never returns.  Clamp each step to the direction the relaxation
        # chose and re-solve; after the first clamped solve a step's charge and
        # discharge bounds are never both > 0, so complementarity holds by
        # construction.  Further passes only refine which steps may act.
        _TOL = 1e-9
        _full = np.full(T, P_max, dtype=float)
        cur_cub, cur_dub = _full, _full
        for _ in range(max_signfix_passes):
            net = c.X - d.X
            new_cub = np.where(net > _TOL, P_max, 0.0)
            new_dub = np.where(net < -_TOL, P_max, 0.0)
            if np.array_equal(new_cub, cur_cub) and np.array_equal(new_dub, cur_dub):
                break  # sign pattern stable
            cur_cub, cur_dub = new_cub, new_dub
            c.UB = cur_cub
            d.UB = cur_dub
            m.optimize()
            if m.SolCount == 0:
                return None, None, None

        return d.X - c.X, net_load.X, E.X[:-1]
    finally:
        if _owns_env:
            env.dispose()


def apply_battery_dispatch(
    ds_residual_zonal: xr.Dataset,
    da_capacity_gwh: xr.DataArray | pd.Series,
    battery_duration_h: float = 4.0,
    roundtrip_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.50,
    dispatch_lookahead_h: float = 0.0,
    zone_dim: str = "zone",
    rl_var: str = "residual_load",
    method: str = "heuristic",
    min_soc_fraction: float | None = None,
    commit_h: float = 24.0,
    verbose: bool = True,
) -> xr.Dataset:
    """Apply battery dispatch zone-by-zone to reduce residual load extremes.

    Charging is always greedy: absorb surplus whenever residual load is negative.
    Discharge strategy depends on ``dispatch_lookahead_h``:

    - ``0`` (default): greedy — discharge as much as needed immediately.
    - ``> 0``: lookahead rationing — count how many positive-RL steps lie within
      the next ``dispatch_lookahead_h`` hours and spread the stored energy evenly
      across them, preventing over-discharge on the first peak.

    One-way efficiency is ``sqrt(roundtrip_efficiency)``; losses are applied
    symmetrically on charge and on discharge.

    Parameters
    ----------
    ds_residual_zonal : xr.Dataset
        Must contain ``rl_var`` with dimensions (``zone_dim``, ``"time"``),
        values in **GWh per 3-hour step**.
    da_capacity_gwh : xr.DataArray or pd.Series
        Battery energy capacity in GWh indexed by zone name.  Zones absent
        from this object get zero capacity (no dispatch).
    battery_duration_h : float
        Assumed duration in hours; sets max power per 3h step as
        ``capacity_gwh / duration_h * 3`` GWh/step.
    roundtrip_efficiency : float
        Round-trip efficiency (fraction, 0–1).
    initial_soc_fraction : float
        Starting state-of-charge as a fraction of total capacity (0–1).
    dispatch_lookahead_h : float
        Lookahead window in hours for discharge rationing.  ``0`` = greedy
        (current behaviour, backward-compatible default).
    zone_dim : str
        Name of the zone coordinate in ``ds_residual_zonal``.
    rl_var : str
        Name of the residual-load variable.
    method : str
        One of ``"heuristic"``, ``"optimal"``, ``"annual"``; anything else raises
        ``ValueError``.

        - ``"heuristic"`` (default): the greedy charge / lookahead-rationed
          discharge loop described above.
        - ``"optimal"``: a **rolling day-ahead** QP scheduler (via Gurobi).  For each
          ``commit_h`` window it solves a QP over ``commit_h + dispatch_lookahead_h``
          hours minimising Σ(net residual load)² under an SOC balance and
          power/energy bounds, commits only the first ``commit_h`` and carries the
          resulting SOC forward.  Models merchant day-ahead scheduling with limited
          foresight.
        - ``"annual"``: a **perfect-foresight annual-cyclical** QP, one solve per
          zone and calendar year with ``E[0] == E[T]`` — the same objective and
          horizon as :func:`apply_reservoir_dispatch`.  Use this to dispatch storage
          for net-load-event mitigation under a scheme consistent with the reservoir.
          ``initial_soc_fraction``, ``dispatch_lookahead_h`` and ``commit_h`` are
          unused in this mode.
    min_soc_fraction : float or None
        Only used when ``method`` is ``"optimal"`` or ``"annual"``.  Lower bound on
        SOC as a fraction of capacity at every step.  ``None`` = no floor.
    commit_h : float
        Only used when ``method="optimal"``.  Length in hours of the committed
        (kept) portion of each rolling solve, and the amount the window slides each
        iteration.  ``dispatch_lookahead_h`` sets the extra foresight beyond it that
        is solved but discarded.
    verbose : bool
        If ``True``, print progress messages.

    Returns
    -------
    xr.Dataset
        Copy of the input with three additional variables:

        - ``residual_load_with_batteries`` – dispatch-adjusted RL (GWh/3h)
        - ``soc_gwh`` – battery state-of-charge trace (GWh)
        - ``battery_dispatch_gwh`` – net battery power per step (GWh/3h);
          positive = discharging (reduces RL), negative = charging (absorbs surplus)
    """
    one_way_eff = np.sqrt(roundtrip_efficiency)
    # Number of 3-hour steps to look ahead when rationing discharge
    lookahead_steps = max(1, round(dispatch_lookahead_h / 3.0)) if dispatch_lookahead_h > 0.0 else 0

    zones = list(ds_residual_zonal[zone_dim].values)
    if isinstance(da_capacity_gwh, pd.Series):
        cap_by_zone = np.array([da_capacity_gwh.get(z, 0.0) for z in zones], dtype=float)
    else:
        z_vals = list(da_capacity_gwh[zone_dim].values)
        cap_by_zone = np.array(
            [float(da_capacity_gwh.sel({zone_dim: z}).values) if z in z_vals else 0.0
             for z in zones],
            dtype=float,
        )

    _VALID_METHODS = ("heuristic", "optimal", "annual")
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown battery dispatch method {method!r}; expected one of {_VALID_METHODS}."
        )

    # Transpose to (zone, time) for the inner loop
    rl_da = ds_residual_zonal[rl_var].transpose(zone_dim, "time")
    rl_arr = rl_da.values.copy()
    n_zones, n_time = rl_arr.shape
    rl_new = rl_arr.copy()
    soc_arr = np.zeros_like(rl_arr)

    if method == "annual":
        # Annual-cyclical perfect-foresight QP, one solve per zone and calendar year.
        # Identical objective/horizon to apply_reservoir_dispatch: minimise the sum of
        # squared net residual load with E[0] == E[T] (leaving initial_soc=None selects
        # that cyclicity), so `initial_soc_fraction` is unused here — the starting SOC
        # is free and determined by the cyclicity constraint.
        import gurobipy as gp

        time_index = pd.DatetimeIndex(rl_da["time"].values)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogFile", "")
        env.start()
        try:
            for iz in range(n_zones):
                cap = cap_by_zone[iz]
                if cap <= 0.0:
                    continue
                max_pwr = cap / battery_duration_h * 3.0  # GWh per 3h step
                for year in np.unique(time_index.year):
                    year_mask = time_index.year == year
                    rl_year = rl_arr[iz, year_mask]
                    if np.all(np.isnan(rl_year)):
                        continue
                    # Gurobi needs finite inputs; treat NaN steps as zero residual load,
                    # then restore NaNs in the adjusted series.
                    nan_mask = np.isnan(rl_year)
                    rl_solve = np.where(nan_mask, 0.0, rl_year)
                    _, net_load, soc = _optimize_battery_gurobi(
                        rl_solve, cap, max_pwr, one_way_eff,
                        min_soc_fraction=min_soc_fraction,
                        env=env,
                    )
                    if net_load is None:
                        if verbose:
                            print(f"    {zones[iz]} {year}: Gurobi infeasible, skipping")
                        continue
                    idx = np.where(year_mask)[0]
                    rl_new[iz, idx] = np.where(nan_mask, np.nan, net_load)
                    soc_arr[iz, idx] = soc
                if verbose:
                    print(f"  Zone {zones[iz]}: dispatched {cap:.2f} GWh battery (annual QP)")
        finally:
            env.dispose()
    elif method == "optimal":
        # Rolling day-ahead QP scheduler.  Each solve covers commit + lookahead
        # hours, only the first `commit_h` is committed, then the window slides
        # forward by `commit_h` carrying the resulting SOC.  Committing before the
        # window end pushes the free-terminal "dump" into the discarded lookahead
        # tail.  The SOC runs continuously across the whole series (no annual reset).
        import gurobipy as gp

        commit_steps = max(1, round(commit_h / 3.0))
        lookahead_steps = max(0, round(dispatch_lookahead_h / 3.0))
        window_steps = commit_steps + lookahead_steps

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogFile", "")
        env.start()
        try:
            for iz in range(n_zones):
                cap = cap_by_zone[iz]
                if cap <= 0.0:
                    continue
                max_pwr = cap / battery_duration_h * 3.0  # GWh per 3h step
                soc0 = cap * initial_soc_fraction
                for start in range(0, n_time, commit_steps):
                    rl_win = rl_arr[iz, start : start + window_steps]
                    if np.all(np.isnan(rl_win)):
                        continue  # nothing to commit; carry SOC unchanged
                    # Gurobi needs finite inputs; treat NaN steps as zero residual
                    # load, then restore NaNs in the committed adjusted series.
                    nan_mask = np.isnan(rl_win)
                    rl_solve = np.where(nan_mask, 0.0, rl_win)
                    _, net_load, soc = _optimize_battery_gurobi(
                        rl_solve, cap, max_pwr, one_way_eff,
                        min_soc_fraction=min_soc_fraction,
                        initial_soc=soc0,
                        env=env,
                    )
                    if net_load is None:
                        continue
                    commit = min(commit_steps, len(rl_win))
                    idx = slice(start, start + commit)
                    rl_new[iz, idx] = np.where(nan_mask[:commit], np.nan, net_load[:commit])
                    soc_arr[iz, idx] = soc[:commit]
                    # soc[k] = storage entering step k; soc[commit] is the level at
                    # the end of the committed window → seed the next window.
                    if commit < len(soc):
                        soc0 = soc[commit]
                if verbose:
                    print(f"  Zone {zones[iz]}: dispatched {cap:.2f} GWh battery")
        finally:
            env.dispose()
    else:
        for iz in range(n_zones):
            cap = cap_by_zone[iz]
            if cap <= 0.0:
                continue

            max_pwr = cap / battery_duration_h * 3.0  # GWh per 3h step
            soc = cap * initial_soc_fraction

            for t in range(n_time):
                rl_t = rl_arr[iz, t]
                if np.isnan(rl_t):
                    soc_arr[iz, t] = soc
                    continue
                if rl_t < 0.0:  # surplus → charge greedily
                    charge = min(-rl_t, max_pwr, (cap - soc) / one_way_eff)
                    soc += charge * one_way_eff
                    rl_new[iz, t] = rl_t + charge
                elif rl_t > 0.0:  # deficit → discharge
                    if lookahead_steps > 0:
                        future = rl_arr[iz, t : t + lookahead_steps]
                        remaining = 1 + int(np.sum(future > 0.0))
                        target = soc * one_way_eff / remaining
                    else:
                        target = soc * one_way_eff
                    discharge = min(rl_t, max_pwr, target)
                    soc -= discharge / one_way_eff
                    rl_new[iz, t] = rl_t - discharge
                soc_arr[iz, t] = soc

            if verbose:
                print(f"  Zone {zones[iz]}: dispatched {cap:.2f} GWh battery")

    # Re-attach xarray coordinates; transpose back to original dim order
    orig_dims = ds_residual_zonal[rl_var].dims
    ds_out = ds_residual_zonal.copy()
    ds_out["residual_load_with_batteries"] = (
        xr.DataArray(
            rl_new,
            coords=rl_da.coords,
            dims=rl_da.dims,
            attrs={"long_name": "Residual load after battery dispatch", "units": "GWh per 3h step"},
        ).transpose(*orig_dims)
    )
    ds_out["soc_gwh"] = (
        xr.DataArray(
            soc_arr,
            coords=rl_da.coords,
            dims=rl_da.dims,
            attrs={"long_name": "Battery state-of-charge", "units": "GWh"},
        ).transpose(*orig_dims)
    )
    # Positive = discharge (reduces RL); negative = charge (absorbs surplus)
    ds_out["battery_dispatch_gwh"] = (
        xr.DataArray(
            rl_arr - rl_new,
            coords=rl_da.coords,
            dims=rl_da.dims,
            attrs={
                "long_name": "Battery net dispatch per 3h step",
                "units": "GWh per 3h step",
                "sign_convention": "positive = discharging (reduces residual load)",
            },
        ).transpose(*orig_dims)
    )
    return ds_out


def apply_prosumer_battery_dispatch(
    ds_residual_zonal: xr.Dataset,
    da_pv_generation: xr.DataArray,
    da_capacity_gwh: xr.DataArray | pd.Series,
    battery_duration_h: float = 4.0,
    roundtrip_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.50,
    zone_dim: str = "zone",
    rl_var: str = "residual_load",
) -> xr.Dataset:
    """Apply PV-aware prosumer battery dispatch to reduce residual load extremes.

    Prosumer batteries are co-located with rooftop PV.  Dispatch strategy:

    - **Charge** only during solar hours (``da_pv_generation > 0``) and only
      when there is a generation surplus (``residual_load < 0``).
    - **Discharge** only when the sun is absent (``da_pv_generation == 0``) and
      there is a grid deficit (``residual_load > 0``).
    - **Daily SOC reset** at midnight: prosumers cannot plan across days, so the
      SOC is reset to ``initial_soc_fraction × capacity`` at each day boundary.

    Parameters
    ----------
    ds_residual_zonal : xr.Dataset
        Must contain ``rl_var`` with dimensions (``zone_dim``, ``"time"``),
        values in **GWh per 3-hour step**.
    da_pv_generation : xr.DataArray
        PV generation signal (any non-negative units) with the same
        ``zone_dim`` and ``"time"`` dimensions as ``ds_residual_zonal``.
        Only the sign (> 0 vs == 0) is used to detect solar vs. non-solar
        timesteps.
    da_capacity_gwh : xr.DataArray or pd.Series
        Prosumer battery energy capacity in GWh indexed by zone name.
    battery_duration_h : float
        Energy-to-power ratio in hours (sets max power per 3h step).
    roundtrip_efficiency : float
        Round-trip efficiency (fraction, 0–1).
    initial_soc_fraction : float
        Starting (and daily reset) state-of-charge as a fraction of capacity.
    zone_dim : str
        Name of the zone coordinate.
    rl_var : str
        Name of the residual-load variable.
    Returns
    -------
    xr.Dataset
        Same structure as :func:`apply_battery_dispatch`: adds
        ``residual_load_with_batteries``, ``soc_gwh``, ``battery_dispatch_gwh``.

    Notes
    -----
    Prosumer batteries are always dispatched with this greedy rule: real home
    batteries run a simple self-consumption controller, and the pipeline carries no
    rooftop-PV or residential-load signal that would justify optimising them.  The
    Gurobi QP (:func:`apply_battery_dispatch` with ``method="optimal"`` or
    ``"annual"``) applies to utility / lumped storage.
    """
    one_way_eff = np.sqrt(roundtrip_efficiency)

    zones = list(ds_residual_zonal[zone_dim].values)
    if isinstance(da_capacity_gwh, pd.Series):
        cap_by_zone = np.array([da_capacity_gwh.get(z, 0.0) for z in zones], dtype=float)
    else:
        z_vals = list(da_capacity_gwh[zone_dim].values)
        cap_by_zone = np.array(
            [float(da_capacity_gwh.sel({zone_dim: z}).values) if z in z_vals else 0.0
             for z in zones],
            dtype=float,
        )

    rl_da = ds_residual_zonal[rl_var].transpose(zone_dim, "time")
    rl_arr = rl_da.values.copy()
    n_zones, n_time = rl_arr.shape

    # Align PV to (zone_dim, time) — use zero if a zone is absent
    pv_da = da_pv_generation.transpose(zone_dim, "time")
    pv_zones = list(pv_da[zone_dim].values)
    if pv_zones == zones:
        pv_arr = pv_da.values
    else:
        pv_arr = np.zeros_like(rl_arr)
        for iz, z in enumerate(zones):
            if z in pv_zones:
                pv_arr[iz] = pv_da.sel({zone_dim: z}).values

    time_index = pd.DatetimeIndex(rl_da["time"].values)
    time_hours = time_index.hour

    rl_new = rl_arr.copy()
    soc_arr = np.zeros_like(rl_arr)

    for iz in range(n_zones):
        cap = cap_by_zone[iz]
        if cap <= 0.0:
            continue

        max_pwr = cap / battery_duration_h * 3.0
        soc = cap * initial_soc_fraction

        for t in range(n_time):
            # Daily SOC reset at midnight (day boundary)
            if t > 0 and time_hours[t] == 0:
                soc = cap * initial_soc_fraction

            rl_t = rl_arr[iz, t]
            if np.isnan(rl_t):
                soc_arr[iz, t] = soc
                continue

            solar_on = pv_arr[iz, t] > 0.0

            if solar_on and rl_t < 0.0:
                # PV surplus → charge prosumer battery
                charge = min(-rl_t, max_pwr, (cap - soc) / one_way_eff)
                soc += charge * one_way_eff
                rl_new[iz, t] = rl_t + charge
            elif (not solar_on) and rl_t > 0.0:
                # No sun, grid deficit → discharge prosumer battery
                discharge = min(rl_t, max_pwr, soc * one_way_eff)
                soc -= discharge / one_way_eff
                rl_new[iz, t] = rl_t - discharge

            soc_arr[iz, t] = soc

    orig_dims = ds_residual_zonal[rl_var].dims
    ds_out = ds_residual_zonal.copy()
    ds_out["residual_load_with_batteries"] = (
        xr.DataArray(
            rl_new,
            coords=rl_da.coords,
            dims=rl_da.dims,
            attrs={"long_name": "Residual load after prosumer battery dispatch", "units": "GWh per 3h step"},
        ).transpose(*orig_dims)
    )
    ds_out["soc_gwh"] = (
        xr.DataArray(
            soc_arr,
            coords=rl_da.coords,
            dims=rl_da.dims,
            attrs={"long_name": "Prosumer battery state-of-charge", "units": "GWh"},
        ).transpose(*orig_dims)
    )
    ds_out["battery_dispatch_gwh"] = (
        xr.DataArray(
            rl_arr - rl_new,
            coords=rl_da.coords,
            dims=rl_da.dims,
            attrs={
                "long_name": "Prosumer battery net dispatch per 3h step",
                "units": "GWh per 3h step",
                "sign_convention": "positive = discharging (reduces residual load)",
            },
        ).transpose(*orig_dims)
    )
    return ds_out


# ---------------------------------------------------------------------------
# Reservoir hydro dispatch
# ---------------------------------------------------------------------------


def _optimize_reservoir_gurobi(
    residual_load: np.ndarray,
    inflow: np.ndarray,
    E_max: float,
    P_max: float,
    min_storage_level_fraction: float | None = None,
    efficiency: float = 0.9,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Solve the annual reservoir dispatch QP using Gurobi.

    Minimises ``Σ (residual_load[t] − dispatch[t])²`` subject to a water-balance
    equality, storage bounds, non-negative spillage, and a cyclical constraint
    ``storage[T] = storage[0]``.

    Parameters
    ----------
    residual_load :
        3-hourly residual load for one calendar year (GWh per 3h step).
    inflow :
        3-hourly inflow for the same year (GWh per 3h step), same length.
    E_max :
        Maximum reservoir storage capacity (GWh).
    P_max :
        Maximum turbine discharge per 3h step (GWh = installed GW × 3).
    min_storage_level_fraction :
        If set, storage cannot drop below ``min_storage_level_fraction × E_max``.
    efficiency :
        Turbine (water-to-wire) conversion efficiency in ``(0, 1]``.  Generating
        ``G`` electrical GWh draws ``G / efficiency`` from storage, so the stored
        potential is only partially recovered as electricity — analogous to the
        battery discharge loss.  Default ``0.9``.

    Returns
    -------
    (dispatch, net_load, storage) arrays of length T, or (None, None, None)
    if Gurobi does not find an optimal solution.  ``dispatch`` is the electrical
    generation ``G`` (what offsets the residual load).
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = len(residual_load)
    E_min = E_max * min_storage_level_fraction if min_storage_level_fraction is not None else 0.0

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogFile", "")
    env.start()
    m = gp.Model("Reservoir_Dispatch", env=env)

    E = m.addMVar(shape=T + 1, lb=E_min, ub=E_max, name="E")
    G = m.addMVar(shape=T, lb=0.0, ub=P_max, name="G")
    S = m.addMVar(shape=T, lb=0.0, name="S")
    net_load = m.addMVar(shape=T, lb=-GRB.INFINITY, name="net_load")

    # Generating G electrical GWh consumes G / efficiency of stored energy.
    m.addConstr(E[1:] == E[:-1] + inflow - G / efficiency - S, name="Storage_Balance")
    m.addConstr(E[0] == E[T], name="Cyclicity")
    m.addConstr(net_load == residual_load - G, name="Net_Load_Def")
    m.setObjective(net_load @ net_load, GRB.MINIMIZE)
    # A single year's residual load can be ill-conditioned (large dynamic range),
    # so push the barrier for accuracy before falling back on the incumbent.
    m.Params.NumericFocus = 3
    m.optimize()

    # Accept the certified optimum, OR any feasible incumbent the barrier could not
    # certify to tolerance (status SUBOPTIMAL/NUMERIC with SolCount > 0). Both
    # satisfy the water-balance constraints exactly — only the optimality *proof* is
    # missing — and using the near-optimal dispatch is far better than skipping the
    # whole country-year (which would leave its residual load undispatched).
    if m.Status == GRB.OPTIMAL or m.SolCount > 0:
        return G.X, net_load.X, E.X[:-1], m.Status
    return None, None, None, m.Status


def apply_reservoir_dispatch(
    ds_residual_load: xr.Dataset,
    ds_hydro_inflow: xr.Dataset,
    df_hydropower_capacities: pd.DataFrame,
    power_col: str = "p_nom_discharge[GW]",
    energy_col: str = "E_store[TWh]",
    min_storage_level_fraction: float | None = None,
    efficiency: float = 0.9,
    rl_var: str = "residual_load",
    country_dim: str = "country",
    verbose: bool = True,
) -> xr.Dataset:
    """Apply annual reservoir hydro dispatch to reduce country-level residual load.

    For each country with installed reservoir capacity, and for each calendar year,
    solves a QP (via Gurobi) that minimises the sum of squared net residual load
    subject to a water-balance equation, storage bounds, and a cyclical storage
    constraint ``s[end] = s[start]``.  This represents perfect-foresight optimal
    operation within each year without inter-annual carry-over.

    The weekly inflow in *ds_hydro_inflow* is linearly interpolated to a 3-hourly
    grid and divided by 56 (7 days × 8 three-hour steps) to obtain GWh per 3h step.

    Parameters
    ----------
    ds_residual_load :
        Country-level residual load, dims ``(country, time)``, 3-hourly GWh.
    ds_hydro_inflow :
        Weekly reservoir inflow, dims ``(country, time)``, in GWh/week.
        Country coordinate must use ISO2 codes.
    df_hydropower_capacities :
        DataFrame containing hydropower capacities, indexed by ISO2 country code.
    power_col :
        Column name in *df_hydropower_capacities* to use.
    energy_col :
        Column name in *df_hydropower_capacities* to use.
    min_storage_level_fraction :
        If set, storage cannot drop below this fraction of E_max at any step.
        Useful to model operational buffers.  Default ``None`` (no floor).
    efficiency :
        Turbine (water-to-wire) conversion efficiency in ``(0, 1]``; generating
        ``G`` electrical GWh draws ``G / efficiency`` from storage.  Default ``0.9``.
    rl_var :
        Name of the residual-load variable in *ds_residual_load*.
    country_dim :
        Name of the country dimension.
    verbose :
        If ``True``, print per-country progress.

    Returns
    -------
    xr.Dataset
        Copy of *ds_residual_load* with *rl_var* replaced by the
        dispatch-adjusted series and two additional variables:
        ``"reservoir_dispatch_gwh"`` and ``"reservoir_storage_gwh"``.
    """
    orig_dims = ds_residual_load[rl_var].dims
    rl_da = ds_residual_load[rl_var].transpose(country_dim, "time")
    rl_arr = rl_da.values.copy()
    time_index = pd.DatetimeIndex(ds_residual_load.time.values)
    all_countries = list(ds_residual_load[country_dim].values)
    inflow_countries = set(ds_hydro_inflow[country_dim].values)

    dispatch_arr = np.zeros_like(rl_arr)
    storage_arr = np.zeros_like(rl_arr)

    reservoir_countries = df_hydropower_capacities.index

    for country in reservoir_countries:
        if country not in inflow_countries or country not in all_countries:
            continue

        # Use the position in all_countries so the arrays are always indexed correctly.
        ic = all_countries.index(country)

        E_max = float(df_hydropower_capacities.loc[country, energy_col]) * 1000 # Max reservoir energy capacity (TWh → GWh)
        P_max = float(df_hydropower_capacities.loc[country, power_col]) * 3.0  # Max turbine power capacity (GW → GWh/3h)

        if E_max <= 0 or P_max <= 0:
            continue

        # Resample weekly inflow to 3h once for this country (interpolate then normalise)
        inflow_weekly = ds_hydro_inflow["inflow"].sel({country_dim: country}).to_series()
        inflow_3h = inflow_weekly.resample("3h").interpolate(method="linear") / (7 * 8)

        if verbose:
            print(f"  Reservoir dispatch: {country}  E_max={E_max:.0f} GWh  P_max={P_max / 3:.1f} GW")

        for year in np.unique(time_index.year):
            year_mask = time_index.year == year
            time_year = time_index[year_mask]
            rl_year = rl_arr[ic, year_mask]

            inflow_year = (
                inflow_3h
                .reindex(time_year, method="nearest")
                .bfill()
                .ffill()
                .fillna(0.0)
                .values
            )

            T = min(len(rl_year), len(inflow_year))
            dispatch, net_load, storage, status = _optimize_reservoir_gurobi(
                rl_year[:T], inflow_year[:T], E_max, P_max, min_storage_level_fraction,
                efficiency=efficiency,
            )

            if dispatch is None:
                if verbose:
                    print(f"    {year}: Gurobi infeasible (status={status}), skipping")
                continue

            yr_idx = np.where(year_mask)[0][:T]
            rl_arr[ic, yr_idx] = net_load
            dispatch_arr[ic, yr_idx] = dispatch
            storage_arr[ic, yr_idx] = storage

    def _rebuild(arr, long_name, units):
        return xr.DataArray(
            arr,
            dims=[country_dim, "time"],
            coords=rl_da.coords,
            attrs={"long_name": long_name, "units": units},
        ).transpose(*orig_dims)

    ds_out = ds_residual_load.copy()
    ds_out[rl_var] = _rebuild(
        rl_arr,
        "Residual load after reservoir hydro dispatch",
        ds_residual_load[rl_var].attrs.get("units", "GWh"),
    )
    ds_out["reservoir_dispatch_gwh"] = _rebuild(
        dispatch_arr, "Reservoir hydro dispatch", "GWh per 3h step"
    )
    ds_out["reservoir_storage_gwh"] = _rebuild(
        storage_arr, "Reservoir storage level", "GWh"
    )
    return ds_out

