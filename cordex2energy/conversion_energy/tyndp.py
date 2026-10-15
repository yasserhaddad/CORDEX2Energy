"""Map TYNDP 2024 bidding zones to NUTS regions and build their polygons.

The power-system part of this module (residual load, TYNDP demand and capacity loaders, battery
and reservoir dispatch) moved to ``cordex2energy_analysis.power_system`` in the
CORDEX2Energy-analysis repository, its only user.

Main functions
--------------
load_nuts_geodata             Download (or load from cache) NUTS1, NUTS2 and UK NUTS1 geodata.
build_tyndp_nuts_mapping      Map TYNDP bidding-zone codes to NUTS2 (or UK NUTS1) region codes.
build_tyndp_zone_geometries   Dissolve NUTS geometries into per-zone polygons.
drop_overlapping_zones        Remove or repair zones that cause regionmask overlap warnings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.ops import snap as _snap_geom

_GISCO_BASE = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"

# Greece uses GR in TYNDP codes but EL in NUTS/Eurostat
_TYNDP_TO_NUTS_COUNTRY: dict[str, str] = {"GR": "EL"}



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
