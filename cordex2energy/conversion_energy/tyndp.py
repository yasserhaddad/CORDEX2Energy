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
):
    """Pre-process a 3-hourly CERRA-based thermal demand series for
    subtraction from TYNDP hourly market demand.

    Applies three sequential post-processing steps that bridge the
    structural gap between raw climate-simulation output and ENTSO-E's
    internally derived demand profiles:

    **Step 1 — Temporal upsampling + thermal inertia smoothing (3 h → 1 h)**

    CERRA yields blocky 3-hourly steps; real grids operate hourly and
    real buildings have thermal mass.  The data is first upsampled to 1 h
    via cubic spline interpolation (fills intermediate hours smoothly),
    then a centered ``rolling_window``-hour rolling mean is applied.

    **Step 2 — Real-world COP / loss correction**

    The ENTSO-E Demand Forecasting Tool (DFT) mandates a 0.85 COP correction
    factor for defrost cycles, pipe losses, and field efficiency degradation.
    Lowering the COP by 15 % raises the electrical draw by ×1.176 (= 1/0.85).

    **Step 3 — Volumetric rescaling to TYNDP annual TWh target**

    When ``target_twh_per_country`` is provided, each country profile is
    scaled by ``k = E_target / E_simulated``.

    Parameters
    ----------
    ds_thermal : xr.Dataset
        Raw thermal electricity demand in GW at native resolution
        (typically 3-hourly for CERRA/C2E outputs).
        Dimensions: ``(country, time)``.
    target_twh_per_country : dict or pd.Series, optional
        Annual electrical energy target per country ``{country_code: TWh}``.
        Pass ``None`` (default) to skip rescaling.
    loss_factor : float, optional
        Flat multiplier for real-world heat-pump COP degradation.
        Default 1.176 (= 1/0.85, per ENTSO-E DFT).  Set to 1.0 to disable.
    rolling_window : int, optional
        Width of the centered rolling mean in hours applied after
        upsampling.  Default 4.

    Returns
    -------
    xr.Dataset
        1-hourly thermal demand with the same variable names as the input,
        in GW.
    """
    # ── Step 1a: Upsample 3 h → 1 h (cubic spline interpolation) ─────────────
    ds_1h = ds_thermal.resample(time="1h").interpolate("cubic")

    # ── Step 1b: Thermal inertia smoothing (centered rolling mean) ────────────
    ds_smooth = ds_1h.rolling(
        time=rolling_window, center=True, min_periods=1
    ).mean()

    # ── Step 2: Real-world COP / loss correction ──────────────────────────────
    ds_scaled = ds_smooth * loss_factor

    # ── Step 3: Volumetric rescaling to TYNDP annual TWh target ──────────────
    if target_twh_per_country is not None:
        scale_values = []
        for country in ds_scaled.country.values:
            c = str(country)
            if c in target_twh_per_country:
                # Annual GWh = Σ(GW values) × 1 h; /1000 → TWh
                simulated_twh = sum(
                    float(ds_scaled[v].sel(country=country).resample(time="YS").sum().mean(dim="time").values)
                    for v in ds_scaled.data_vars
                ) * 1e-3
                k = (
                    float(target_twh_per_country[c]) / simulated_twh
                    if simulated_twh > 0
                    else 1.0
                )
            else:
                k = 1.0
            scale_values.append(k)

        scale_da = xr.DataArray(
            scale_values,
            dims=["country"],
            coords={"country": ds_scaled.country.values},
        )
        ds_scaled = ds_scaled * scale_da

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
    thermal_subsectors=("Space heating", "Space heating & hot water", "Cooling"),
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
    thermal_subsectors : tuple of str, optional
        ``SUBSECTOR`` values treated as thermal electricity consumption.
    ds_market : xr.Dataset, optional
        TYNDP market demand profiles, dims ``(country, time)``,
        variable ``"demand_profile"`` in GW.  Used as ``E_total`` source
        for non-EU countries absent from ``df_tyndp_demand``.

    Returns
    -------
    dict
        ``{country_code: thermal_twh_target}``.
    """
    col = f"{year}_{scenario}"
    proxy_map = {**_DEFAULT_NON_EU_PROXIES, **(non_eu_proxy_map or {})}

    # ── Thermal electricity by country ────────────────────────────────────────
    mask_thermal = df_tyndp_demand["ENERGY_CARRIER"].eq("Electricity") & \
                   df_tyndp_demand["SUBSECTOR"].isin(thermal_subsectors)
    thermal_twh = (
        df_tyndp_demand.loc[mask_thermal, ["COUNTRY", col]]
        .dropna(subset=[col])
        .groupby("COUNTRY")[col]
        .sum()
    )

    # ── Total electricity by country (EU-27 only in this spreadsheet) ─────────
    mask_total = df_tyndp_demand["ENERGY_CARRIER"].eq("Electricity") & \
                 df_tyndp_demand["SUBSECTOR"].eq("Total")
    total_twh = (
        df_tyndp_demand.loc[mask_total, ["COUNTRY", col]]
        .dropna(subset=[col])
        .groupby("COUNTRY")[col]
        .sum()
    )

    # ── Pre-compute market annual totals from ds_market (GW·h → TWh) ─────────
    # ds_market typically covers multiple climate years (1995, 2008, 2009);
    # divide by the number of unique years to obtain a mean annual total.
    market_twh: dict[str, float] = {}
    if ds_market is not None:
        n_years = max(1, len(np.unique(ds_market.time.dt.year.values)))
        for c in ds_market.country.values:
            # convert from MWh to TWh and average across climate years
            v = float(ds_market["demand_profile"].sel(country=c).values.sum()) * 1e-6 / n_years
            if v > 0:
                market_twh[str(c)] = v

    # ── Build target dict ─────────────────────────────────────────────────────
    targets: dict[str, float] = {}

    # EU countries: use direct thermal target from the spreadsheet
    for country, v in thermal_twh.items():
        if float(v) > 0:
            targets[country] = float(v)

    # Non-EU countries: synthesise via proxy thermal-share method
    for country, proxy in proxy_map.items():
        if country in targets:
            continue                              # already has a direct entry
        if proxy not in thermal_twh or proxy not in total_twh:
            print(f"Warning: proxy '{proxy}' for '{country}' not found in "
                  f"TYNDP demand data — skipping.")
            continue
        if float(total_twh[proxy]) <= 0:
            continue

        alpha = float(thermal_twh[proxy]) / float(total_twh[proxy])

        # Prefer spreadsheet total; fall back to ds_market annual sum
        if country in total_twh and float(total_twh[country]) > 0:
            e_total = float(total_twh[country])
            source = "spreadsheet"
        elif country in market_twh:
            e_total = market_twh[country]
            source = "ds_market"
        else:
            print(f"Warning: no total electricity demand found for '{country}' "
                  f"in TYNDP demand data or ds_market — cannot synthesise "
                  f"thermal target (falling back to k=1).")
            continue

        targets[country] = alpha * e_total
        print(f"  {country}: proxy='{proxy}', \u03b1={alpha:.3f}, "
              f"E_total={e_total:.1f} TWh [{source}] \u2192 "
              f"E_thermal={targets[country]:.1f} TWh")

    return targets


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

    # ── Step 1: Combine heating + cooling → thermal demand (3-hourly) ─────────
    common_time_th = np.intersect1d(ds_heating.time.values, ds_cooling.time.values)
    ds_thermal = xr.Dataset(
        {
            "demand_profile": (
                ds_heating["heating_demand"].sel(time=common_time_th)
                + ds_cooling["cooling_demand"].sel(time=common_time_th)
            )
        }
    )

    # ── Step 2: Preprocess thermal demand → 1-hourly GW ───────────────────────
    ds_thermal_1h = preprocess_thermal_demand(
        ds_thermal,
        target_twh_per_country=target_twh_per_country,
        loss_factor=loss_factor,
        rolling_window=rolling_window_thermal_demand,
    )

    # ── Step 3: Merge baseload + thermal → total 1-hourly demand ──────────────
    common_countries_demand = np.intersect1d(
        ds_baseload[country_dim].values,
        ds_thermal_1h[country_dim].values,
    )
    common_time_1h = np.intersect1d(
        ds_baseload.time.values,
        ds_thermal_1h.time.values,
    )
    total_demand_1h = (
        ds_baseload["baseload"].sel(
            {country_dim: common_countries_demand, "time": common_time_1h}
        )
        + ds_thermal_1h["demand_profile"].sel(
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
) -> pd.Series:
    """Load TYNDP 2024 battery energy storage capacity projections.

    Reads sheet ``1.5.`` (prosumer batteries) and ``1.6.`` (utility batteries),
    sums both, and returns **GWh per ISO2 country code**.

    The raw sheets store values in **MWh** and use TYNDP bidding-zone codes
    (e.g. ``"DE00"``, ``"ITSI"``).  This function strips the trailing suffix to
    obtain the two-character ISO2 code, aggregates all bidding zones for the
    same country, and converts MWh → GWh.  Aggregate rows (``"TOTAL"``,
    ``"EU-TOTAL"``) are excluded before summing.

    Parameters
    ----------
    path_excel : str or Path
        Path to the TYNDP supply Excel file.
    scenario : str
        Scenario name as it appears in the column header (e.g. ``"HIGH"``).
    year : int
        Projection year (e.g. ``2050``).

    Returns
    -------
    pd.Series
        Total battery energy capacity in **GWh**, indexed by two-character
        ISO2 country code.
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

    s_prosumer = _to_iso2_series(_parse_supply_sheet(path_excel, "1.5."))
    s_utility = _to_iso2_series(_parse_supply_sheet(path_excel, "1.6."))

    # Raw values are in MWh — convert to GWh
    total_gwh = s_prosumer.add(s_utility, fill_value=0.0) / 1_000.0
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
) -> xr.Dataset:
    """Return the typical-year TYNDP baseload, loading from cache when available.

    The baseload depends only on the TYNDP demand scenario and year (not on the
    climate model), so it is computed once from CERRA reference data + TYNDP
    demand profiles and then cached.

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
    cache_path = path_baseload_cache / f"baseload_{demand_scenario}_{demand_year}.nc"

    if cache_path.exists():
        print(f"  Found cached baseload → {cache_path}")
        return xr.open_dataset(cache_path)

    print(f"  Baseload cache not found — computing for {demand_scenario} {demand_year} …")

    # Load TYNDP demand profiles for this scenario/year.
    path_demand_profiles_dir = path_tyndp / "Demand Profiles"
    ds_market, ds_prosumer = load_tyndp_market_profiles(
        path_demand_profiles_dir, demand_scenario, demand_year
    )

    # Build thermal electricity targets from the TYNDP demand scenario table.
    thermal_targets = build_thermal_targets(
        df_tyndp_demand,
        scenario=demand_scenario,
        year=demand_year,
        ds_market=ds_market,
    )

    # Load CERRA heating and cooling demand (reference-period reanalysis data).
    path_c2e_cerra = path_c2e_cerra / "output_variables"
    df_heating_cerra = pd.read_csv(
        path_c2e_cerra / "heating-demand_CERRA.csv", index_col=0
    )
    df_cooling_cerra = pd.read_csv(
        path_c2e_cerra / "cooling-demand_CERRA.csv", index_col=0
    )
    ds_heating_cerra = (
        xr.DataArray(
            df_heating_cerra.values,
            dims=["country", "time"],
            coords={
                "country": [C2E_NAME_TO_ISO2.get(c, c) for c in df_heating_cerra.index],
                "time": pd.to_datetime(df_heating_cerra.columns),
            },
            name="heating_demand",
        )
        .to_dataset()
        .sel(time=slice("1991", "2020"))
    )
    ds_cooling_cerra = (
        xr.DataArray(
            df_cooling_cerra.values,
            dims=["country", "time"],
            coords={
                "country": [C2E_NAME_TO_ISO2.get(c, c) for c in df_cooling_cerra.index],
                "time": pd.to_datetime(df_cooling_cerra.columns),
            },
            name="cooling_demand",
        )
        .to_dataset()
        .sel(time=slice("1991", "2020"))
    )

    # Combine heating + cooling into a single thermal demand profile over the
    # intersection of their time axes.
    _common_time = np.intersect1d(
        ds_heating_cerra.time.values, ds_cooling_cerra.time.values
    )
    ds_thermal_cerra = xr.Dataset(
        {
            "demand_profile": (
                ds_heating_cerra["heating_demand"].sel(time=_common_time)
                + ds_cooling_cerra["cooling_demand"].sel(time=_common_time)
            )
        }
    )

    # Upsample, smooth (thermal inertia), COP-correct, and rescale to TYNDP targets.
    ds_thermal_preprocessed = preprocess_thermal_demand(
        ds_thermal_cerra,
        loss_factor=(1 / 0.85),
        rolling_window=4,
        target_twh_per_country=thermal_targets,
    )

    # Subtract thermal demand from total TYNDP market demand → non-thermal baseload.
    # market + prosumer are in MWh; convert to GWh with × 1e-3.
    ds_total_market = (ds_market + ds_prosumer.reindex(
        country=ds_market.country, fill_value=0.0
    )) * 1e-3
    ds_baseload = compute_tyndp_baseload(
        ds_total_market,
        ds_thermal_preprocessed,
        floor_fraction=floor_fraction,
    )

    # Build a typical-year (hour_of_year) profile by averaging TYNDP climate years.
    ds_baseload_typical = compute_typical_year_profile(ds_baseload)

    # Cache to disk.
    ds_baseload_typical.to_netcdf(cache_path)
    print(f"  Saved baseload cache → {cache_path}")

    return ds_baseload_typical


def apply_battery_dispatch(
    ds_residual_zonal: xr.Dataset,
    da_capacity_gwh: xr.DataArray | pd.Series,
    battery_duration_h: float = 4.0,
    roundtrip_efficiency: float = 0.90,
    initial_soc_fraction: float = 0.50,
    zone_dim: str = "zone",
    rl_var: str = "residual_load",
) -> xr.Dataset:
    """Apply greedy battery dispatch zone-by-zone to reduce residual load extremes.

    Simulates a single aggregated battery store per zone.  Strategy: charge
    whenever residual load is negative (surplus), discharge whenever it is
    positive (deficit), respecting power limits and energy capacity.

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
    zone_dim : str
        Name of the zone coordinate in ``ds_residual_zonal``.
    rl_var : str
        Name of the residual-load variable.

    Returns
    -------
    xr.Dataset
        Copy of the input with two additional variables:

        - ``residual_load_with_batteries`` – dispatch-adjusted RL (GWh/3h)
        - ``soc_gwh`` – battery state-of-charge trace (GWh)
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

    # Transpose to (zone, time) for the inner loop
    rl_da = ds_residual_zonal[rl_var].transpose(zone_dim, "time")
    rl_arr = rl_da.values.copy()
    n_zones, n_time = rl_arr.shape
    rl_new = rl_arr.copy()
    soc_arr = np.zeros_like(rl_arr)

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
            if rl_t < 0.0:  # surplus → charge
                charge = min(-rl_t, max_pwr, (cap - soc) / one_way_eff)
                soc += charge * one_way_eff
                rl_new[iz, t] = rl_t + charge
            elif rl_t > 0.0:  # deficit → discharge
                discharge = min(rl_t, max_pwr, soc * one_way_eff)
                soc -= discharge / one_way_eff
                rl_new[iz, t] = rl_t - discharge
            soc_arr[iz, t] = soc

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
    return ds_out

