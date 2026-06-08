"""
land_eligibility.py
-------------------
Derives land eligibility masks and eligible fractions for rooftop PV and
onshore wind from CORINE Land Cover 2018 (100m GeoTIFF), aggregated to a
target 10km grid.

For each 10km grid cell the output is a continuous fraction in [0, 1]
representing the share of land that is eligible for each technology —
NOT a binary mask. This allows partial cells (e.g. a cell that is 60%
cropland and 40% forest) to contribute proportionally.

Inputs:
  - CLC 2018 GeoTIFF   : U2018_CLC2018_V2020_20u1.tif  (100m, EPSG:3035)
      Download: https://land.copernicus.eu/pan-european/corine-land-cover/clc2018
  - target grid        : any xr.DataArray with (lat, lon) dims in EPSG:4326
                         (e.g. your CF DataArray)

Outputs:
  - eligible_fraction_pv   : DataArray (lat, lon) in [0, 1]
  - eligible_fraction_wind : DataArray (lat, lon) in [0, 1]
  - built_up_fraction      : DataArray (lat, lon) in [0, 1]  ← for PV power density

CLC class reference:
  https://www.eea.europa.eu/data-and-maps/data/corine-land-cover-2

Dependencies:
    pip install rioxarray rasterio xarray numpy
"""

import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path
from typing import Union

# ── 0.  CLC CLASS DEFINITIONS  ───────────────────────────────────────────────
#
# CORINE Land Cover uses 3-level nomenclature. The GeoTIFF stores the
# 3-digit Level-3 code as the pixel value (e.g. 211 = Non-irrigated arable land).
# Codes 1–11 are also used in older versions (Level-1 aggregated); the 100m
# GeoTIFF uses the full 3-digit codes below. NoData = 0 or 255.
#
# Full code list:
# https://www.eea.europa.eu/data-and-maps/data/corine-land-cover-2/corine-land-cover-classes-and

# --- Artificial surfaces (Level 1: code 1xx) ---------------------------------
URBAN_FABRIC            = [111, 112]          # continuous / discontinuous urban
INDUSTRIAL_COMMERCIAL   = [121]
TRANSPORT               = [122, 123, 124]
MINE_DUMP_CONSTRUCTION  = [131, 132, 133]
URBAN_GREEN             = [141, 142]          # sport/leisure

ARTIFICIAL_ALL = (
    URBAN_FABRIC + INDUSTRIAL_COMMERCIAL + TRANSPORT
    + MINE_DUMP_CONSTRUCTION + URBAN_GREEN
)

# --- Agricultural land (Level 1: code 2xx) -----------------------------------
ARABLE_LAND             = [211, 212, 213]
PERMANENT_CROPS         = [221, 222, 223]
PASTURES                = [231]
HETEROGENEOUS_AG        = [241, 242, 243, 244]

AGRICULTURAL_ALL = ARABLE_LAND + PERMANENT_CROPS + PASTURES + HETEROGENEOUS_AG

# --- Forest and semi-natural (Level 1: code 3xx) -----------------------------
FORESTS                 = [311, 312, 313]
SCRUB_HERBACEOUS        = [321, 322, 323, 324]
OPEN_SPACES             = [331, 332, 333, 334, 335]   # bare rock, sparse veg, glaciers

NATURAL_ALL = FORESTS + SCRUB_HERBACEOUS + OPEN_SPACES

# --- Wetlands (Level 1: code 4xx) --------------------------------------------
WETLANDS                = [411, 412, 421, 422, 423]

# --- Water bodies (Level 1: code 5xx) ----------------------------------------
WATER                   = [511, 512, 521, 522, 523]

# ── ELIGIBILITY RULES  ───────────────────────────────────────────────────
#
# Based on:
#   - Ryberg et al. (2019) "The future of European onshore wind energy potential"
#   - Bódis et al. (2019) "A high-resolution geospatial assessment of the rooftop
#     solar photovoltaic potential in the European Union"
#   - EU JRC ENSPRESO methodology
#
# Rules are conservative defaults — adjust to your study assumptions.

# Rooftop PV: eligible land = built-up area (rooftops)
# Note: eligibility here means "land that HAS rooftops" — the actual
# power density function (pv_power_density) handles the roof utilisation fraction.
ELIGIBLE_PV_CODES = (
    URBAN_FABRIC            # residential rooftops — primary target
    + INDUSTRIAL_COMMERCIAL # large flat roofs — high potential
    + URBAN_GREEN           # leisure/sport facilities — lower priority
)

# Open-field (ground-mounted) PV: eligible land = open, low-conflict land.
# Applied at a reduced fraction (PARTIAL_ELIGIBILITY_PV_OPENFIELD) to reflect
# planning constraints and co-existence with agriculture.
# Based on EU JRC ENSPRESO and Ryberg et al. (2019) methodology.
#
# NOTE on methodology consistency with wind:
#   Wind uses fraction = 1.0 implicitly — all ELIGIBLE_WIND_CODES land is
#   counted at full weight, and turbine spacing is captured by pd_wind via the
#   rotor-diameter layout formula.  For open-field PV the equivalent physical
#   coverage is captured by `ground_coverage_ratio` (GCR) in pv_power_density.
#   Therefore `fraction` here should represent ONLY additional planning /
#   governance constraints beyond what is in the eligible codes themselves
#   (e.g. setbacks, landscape protection, grid access).
#
#   Recommended values:
#     1.0  — technical potential (fully symmetric with wind treatment)
#     0.30 — moderate constraint (some agri-PV, planning-friendly scenario)
#     0.10 — conservative constraint (strict national planning rules)
#     0.03 — very conservative (minimal open-field, strong land-use restrictions)
ELIGIBLE_PV_OPENFIELD_CODES = (
    MINE_DUMP_CONSTRUCTION  # brownfield — minimal conflict, highest priority
    + ARABLE_LAND           # agri-PV / single-axis tracking, co-existence possible
    + PERMANENT_CROPS       # also used for agri-PV in warm climates
    + PASTURES              # low-conflict, common in sunny regions
    + HETEROGENEOUS_AG      # mixed farming — same as wind eligibility
    + SCRUB_HERBACEOUS      # open shrubland, moorland
    + OPEN_SPACES           # sparse vegetation, bare rock, highland plateau
)

# Default: 0.30 (moderate-constraint scenario).
# Matches the implicit assumption for wind (= fraction of eligible land
# available for energy development given planning constraints).
# The physical panel coverage within a PV farm is handled separately by
# ground_coverage_ratio in pv_power_density (default GCR = 0.30).
PARTIAL_ELIGIBILITY_PV_OPENFIELD = {
    "codes":    ELIGIBLE_PV_OPENFIELD_CODES,
    "fraction": 0.30,
}

# Onshore wind: eligible land = open land excluding settlements and water
# Forests included at reduced eligibility (see PARTIAL_ELIGIBILITY_WIND below)
ELIGIBLE_WIND_CODES = (
    ARABLE_LAND             # dominant wind host in NW Europe
    + PERMANENT_CROPS       # often allowed with setback from farmsteads
    + PASTURES              # common in Ireland, UK, Scandinavia
    + HETEROGENEOUS_AG      # mixed landscape
    + SCRUB_HERBACEOUS      # open shrubland, moorland
    + OPEN_SPACES           # sparse vegetation, bare rock at high elevation
    + MINE_DUMP_CONSTRUCTION# brownfield — often preferred
)

# Partially eligible for wind (typically at 50% due to wake losses in forests,
# or country-specific rules — set to 0.0 to fully exclude)
PARTIAL_ELIGIBILITY_WIND = {
    "codes":    FORESTS,
    "fraction": 0.0,    # set to 0.5 to allow wind-in-forest (e.g. Finland/Sweden)
}

# Always excluded regardless of technology
EXCLUDED_CODES = WETLANDS + WATER + TRANSPORT

# ── CLC INDEX ↔ CODE MAPPING  ────────────────────────────────────────────────
#
# The Copernicus CLC 100m GeoTIFF (U2018_CLC2018_V2020_20u1.tif) stores pixel
# values as class **indices** 1–44 (uint8), NOT 3-digit Level-3 codes.
# load_clc() reads this mapping directly from the embedded Raster Attribute
# Table (RAT) via GDAL — the same source QGIS uses when showing the code on
# click.  The hardcoded table below is a fallback for environments where the
# GDAL Python bindings are unavailable or the RAT is missing.

CLC_INDEX_TO_CODE: dict[int, int] = {
    1: 111, 2: 112,                                     # Urban fabric
    3: 121, 4: 122, 5: 123, 6: 124,                    # Industrial / transport
    7: 131, 8: 132, 9: 133,                             # Mine / dump / construction
    10: 141, 11: 142,                                   # Urban green
    12: 211, 13: 212, 14: 213,                          # Arable land
    15: 221, 16: 222, 17: 223,                          # Permanent crops
    18: 231,                                            # Pastures
    19: 241, 20: 242, 21: 243, 22: 244,                # Heterogeneous agricultural
    23: 311, 24: 312, 25: 313,                          # Forests
    26: 321, 27: 322, 28: 323, 29: 324,                # Scrub / herbaceous
    30: 331, 31: 332, 32: 333, 33: 334, 34: 335,       # Open spaces
    35: 411, 36: 412,                                   # Inland wetlands
    37: 421, 38: 422, 39: 423,                          # Coastal wetlands
    40: 511, 41: 512,                                   # Inland waters
    42: 521, 43: 522, 44: 523,                          # Marine waters
}

# ── 1.  PROTECTED AREAS  ─────────────────────────────────────────────────────

# IUCN categories excluded for both PV and wind
IUCN_EXCLUDE_ALL = [
    "Ia", "Ib", "II", "III", "IV",
    "Not Reported", "Not Assigned", "Not Applicable",
]
# IUCN categories excluded for wind only (restricted landscapes may allow PV)
IUCN_EXCLUDE_WIND_ONLY = ["V"]
# Partial exclusion (sustainable use reserves)
IUCN_PARTIAL = {
    "categories":        ["VI"],
    "fraction_excluded": 0.5,
}


def load_natura2000(
    natura_path: Union[str, Path],
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Load Natura 2000 protected area polygons from the EEA GeoPackage.
    All sites are treated as fully excluded (Natura 2000 has no IUCN categories).
    """
    import fiona
    layers = fiona.listlayers(str(natura_path))
    print(f"  Natura 2000 layers: {layers}")
    poly_layer = [lyr for lyr in layers if "epsg" in lyr.lower() or "site" in lyr.lower()]
    layer = poly_layer[0] if poly_layer else layers[0]
    print(f"  Using layer: {layer}")
    gdf = gpd.read_file(natura_path, layer=layer)
    gdf = gdf[gdf.geometry.notnull()].to_crs(target_crs)
    print(f"  Natura 2000: {len(gdf):,} polygons loaded")
    return gdf


def load_wdpa(
    wdpa_path: Union[str, Path],
    target_crs: str = "EPSG:4326",
    status_filter: list = ["Designated", "Inscribed", "Established"],
) -> gpd.GeoDataFrame:
    """
    Load WDPA protected area polygons, filtered to designated sites.
    Points are excluded (no area geometry).
    """
    import fiona
    layers = fiona.listlayers(str(wdpa_path))
    print(f"  WDPA layers: {layers}")
    poly_layer = [lyr for lyr in layers if "poly" in lyr.lower()]
    layer = poly_layer[0] if poly_layer else layers[0]
    print(f"  Using layer: {layer}")
    gdf = gpd.read_file(wdpa_path, layer=layer)
    gdf = gdf[gdf.geometry.notnull()]
    if status_filter:
        gdf = gdf[gdf["STATUS"].isin(status_filter)]
    gdf = gdf.to_crs(target_crs)
    print(
        f"  WDPA: {len(gdf):,} polygons  |  "
        f"IUCN categories: {sorted(gdf['IUCN_CAT'].unique())}"
    )
    return gdf


def _rasterize_to_array(
    geometries,
    burn_value: float,
    shape: tuple,
    transform: rasterio.Affine,
) -> np.ndarray:
    """Rasterise a geometry iterable to a float32 array."""
    pairs = [(geom, burn_value) for geom in geometries if geom is not None]
    if not pairs:
        return np.zeros(shape, dtype=np.float32)
    return rasterize(
        pairs,
        out_shape=shape,
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )

def build_protected_exclusion_mask(
    protected_gdf: gpd.GeoDataFrame,
    target: xr.DataArray,
    source: str = "wdpa",
    iucn_exclude_all:       list = IUCN_EXCLUDE_ALL,
    iucn_exclude_wind_only: list = IUCN_EXCLUDE_WIND_ONLY,
    iucn_partial:           dict = IUCN_PARTIAL,
    intermediate_res_deg:   float = 0.001,
) -> dict:
    """
    Rasterise protected area polygons to the target grid as exclusion
    fractions [0, 1] for PV and wind separately.

    Uses a two-step approach:
      1. Rasterise vector polygons to an intermediate ~100m grid (avoids
         small-polygon aliasing at 10km).
      2. Average-resample to the target 10km grid → continuous fraction.

    Parameters
    ----------
    protected_gdf         : GeoDataFrame in EPSG:4326 (Natura 2000 or WDPA)
    target                : reference DataArray (lat, lon, EPSG:4326)
    source                : "wdpa" applies IUCN category rules;
                            "natura2000" excludes everything fully
    intermediate_res_deg  : intermediate raster resolution (degrees)
                            default 0.001° ≈ 100m

    Returns
    -------
    dict:
        "exclusion_pv"   : DataArray (lat, lon) exclusion fraction for PV
        "exclusion_wind" : DataArray (lat, lon) exclusion fraction for wind
    """
    lon_min = float(target.lon.min())
    lon_max = float(target.lon.max())
    lat_min = float(target.lat.min())
    lat_max = float(target.lat.max())

    width     = int((lon_max - lon_min) / intermediate_res_deg)
    height    = int((lat_max - lat_min) / intermediate_res_deg)
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    shape     = (height, width)
    print(f"  Intermediate raster: {width} × {height} px at {intermediate_res_deg}°")

    if source == "natura2000" or "IUCN_CAT" not in protected_gdf.columns:
        print("  Natura 2000 mode: full exclusion for all sites")
        excl = _rasterize_to_array(protected_gdf.geometry, 1.0, shape, transform)
        excl_pv_arr   = excl
        excl_wind_arr = excl

    else:
        # WDPA: layer IUCN categories with different exclusion strengths
        cat_col = "IUCN_CAT"

        excl_all = _rasterize_to_array(
            protected_gdf[protected_gdf[cat_col].isin(iucn_exclude_all)].geometry,
            1.0, shape, transform,
        )
        excl_wind_only = _rasterize_to_array(
            protected_gdf[protected_gdf[cat_col].isin(iucn_exclude_wind_only)].geometry,
            1.0, shape, transform,
        )
        excl_part = _rasterize_to_array(
            protected_gdf[protected_gdf[cat_col].isin(iucn_partial["categories"])].geometry,
            iucn_partial["fraction_excluded"], shape, transform,
        )

        excl_pv_arr   = np.clip(excl_all + excl_part, 0.0, 1.0)
        excl_wind_arr = np.clip(excl_all + excl_wind_only + excl_part, 0.0, 1.0)

    # Wrap in DataArrays and resample to target grid
    coords = {
        "lat": np.linspace(lat_max, lat_min, height),
        "lon": np.linspace(lon_min, lon_max, width),
    }
    # Register lat/lon as spatial dims so rioxarray can use target as a match template.
    target_rio = target.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=False)

    def _to_da(arr, name):
        da = xr.DataArray(arr, dims=["lat", "lon"], coords=coords)
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.reproject_match(target_rio, resampling=Resampling.average)
        da = da.clip(0.0, 1.0)
        da.attrs = {
            "long_name": f"Protected area exclusion fraction ({name})",
            "units": "1",
            "source": source,
        }
        return da.rename(f"exclusion_{name}")

    print("  Resampling exclusion masks to target grid ...")
    excl_pv_da   = _to_da(excl_pv_arr,   "pv")
    excl_wind_da = _to_da(excl_wind_arr, "wind")

    print(
        f"  Mean exclusion fraction:  PV={float(excl_pv_da.mean()):.3f}  "
        f"Wind={float(excl_wind_da.mean()):.3f}"
    )
    return {
        "exclusion_pv":   excl_pv_da,
        "exclusion_wind": excl_wind_da,
    }


# ── 2.  CLC ELIGIBILITY  ─────────────────────────────────────────────────────

def _read_clc_rat(clc_path: Union[str, Path]) -> dict[int, int] | None:
    """
    Read the index → 3-digit-code mapping from the Raster Attribute Table (RAT)
    embedded in the CLC GeoTIFF, using GDAL.

    The RAT is what QGIS uses to display the 3-digit code on pixel click.
    Returns None if GDAL bindings are unavailable or the RAT lacks a code column.
    """
    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except ImportError:
        return None

    ds = gdal.Open(str(clc_path))
    if ds is None:
        return None
    band = ds.GetRasterBand(1)
    rat  = band.GetDefaultRAT()
    if rat is None:
        return None

    n_cols = rat.GetColumnCount()
    n_rows = rat.GetRowCount()

    # Locate the Value column and any column whose name contains "CODE"
    val_col = code_col = None
    for col in range(n_cols):
        name = rat.GetNameOfCol(col).upper()
        if name == "VALUE":
            val_col = col
        elif "CODE" in name:
            code_col = col

    if val_col is None or code_col is None:
        return None

    mapping = {
        rat.GetValueAsInt(row, val_col): rat.GetValueAsInt(row, code_col)
        for row in range(n_rows)
        if rat.GetValueAsInt(row, val_col) > 0
    }
    return mapping or None


def load_clc(clc_path: Union[str, Path]) -> xr.DataArray:
    """
    Load the CLC 2018 GeoTIFF as a DataArray and normalise to 3-digit codes.

    The Copernicus 100m GeoTIFF stores pixel values as class indices 1–44
    (uint8).  The index → 3-digit-code mapping is read directly from the
    embedded Raster Attribute Table (RAT) via GDAL. If GDAL is
    unavailable or the RAT is absent, falls back to CLC_INDEX_TO_CODE.

    The native CRS is EPSG:3035 (LAEA Europe).
    """
    da = rioxarray.open_rasterio(clc_path, masked=True).squeeze("band", drop=True)
    da = da.fillna(0).astype(np.uint16)
    # Value 48 is the CLC "outside area / NODATA" sentinel (not a valid class index).
    # Zero it out so it is excluded from all eligibility masks.
    da = da.where(da != 48, other=0)

    max_val = int(da.max())
    if max_val <= 48:
        code_map = _read_clc_rat(clc_path)
        if code_map is not None:
            print(
                f"  CLC encoding: read {len(code_map)} index→code entries "
                "from embedded RAT."
            )
        else:
            print(
                "  CLC encoding: no RAT found — using built-in CLC_INDEX_TO_CODE."
            )
            code_map = CLC_INDEX_TO_CODE

        # Vectorised remap via a lookup array
        max_idx = max(code_map)
        lut = np.zeros(max_idx + 1, dtype=np.uint16)   # index 0 stays 0 (NoData)
        for idx, code in code_map.items():
            lut[idx] = code
        da = xr.apply_ufunc(
            lambda arr: lut[arr],
            da,
            dask="parallelized",
            output_dtypes=[np.uint16],
        )
    else:
        print(
            f"  CLC encoding: 3-digit codes already (max={max_val}) — "
            "no remapping needed."
        )

    print(
        f"CLC loaded: {da.rio.width} × {da.rio.height} px  |  "
        f"CRS: {da.rio.crs}  |  res: {da.rio.resolution()} m"
    )
    return da


def _make_binary_mask(clc: xr.DataArray, codes: list[int]) -> xr.DataArray:
    """Return a boolean DataArray: True where CLC code is in `codes`."""
    mask = xr.zeros_like(clc, dtype=bool)
    for code in codes:
        mask = mask | (clc == code)
    return mask


def _aggregate_to_target(
    binary_100m: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray:
    """
    Aggregate a 100m binary mask to the target lat/lon grid as a fraction [0, 1].

    Strategy:
      1. Reproject CLC (EPSG:3035) to EPSG:4326 at an intermediate resolution
         (~0.001° ≈ 100m) to avoid information loss.
      2. Use average resampling to the target 10km grid — this gives the
         fraction of 100m pixels that are eligible within each 10km cell.
    """
    # Step 1: reproject to EPSG:4326 at ~100m (keep native detail)
    mask_4326 = binary_100m.astype(np.float32).rio.reproject(
        "EPSG:4326",
        resampling=Resampling.average,   # average of binary = fraction
    )

    # Step 2: resample to target grid (average = eligible fraction per cell)
    # Register lat/lon as spatial dims so rioxarray can use target as a match template.
    target_rio = target.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=False)
    fraction = mask_4326.rio.reproject_match(
        target_rio,
        resampling=Resampling.average,
    )
    return fraction.clip(0.0, 1.0)

def compute_clc_eligibility(
    clc_path: Union[str, Path],
    target: xr.DataArray,
    eligible_pv_codes:      list[int] = ELIGIBLE_PV_CODES,
    eligible_wind_codes:    list[int] = ELIGIBLE_WIND_CODES,
    partial_wind:           dict      = PARTIAL_ELIGIBILITY_WIND,
    partial_pv_openfield:   dict      = PARTIAL_ELIGIBILITY_PV_OPENFIELD,
) -> dict[str, xr.DataArray]:
    """
    Compute land eligibility fractions for rooftop PV and onshore wind,
    plus built-up fraction (for use in pv_power_density), all at the
    resolution and extent of `target`.

    Parameters
    ----------
    clc_path : str or Path
        Path to U2018_CLC2018_V2020_20u1.tif
    target : xr.DataArray
        Reference grid DataArray with (lat, lon) dims in EPSG:4326.
        Only its coordinates/CRS are used — values are ignored.
    eligible_pv_codes : list of int
        CLC codes counted as fully eligible for rooftop PV.
    eligible_wind_codes : list of int
        CLC codes counted as fully eligible for onshore wind.
    partial_wind : dict
        {"codes": [...], "fraction": float}
        CLC codes with partial wind eligibility and their weight.
    partial_pv_openfield : dict
        {"codes": [...], "fraction": float}
        CLC codes eligible for open-field (ground-mounted) PV at a reduced
        fraction.  Combined additively with rooftop eligibility and clipped
        to [0, 1].  Set fraction=0.0 to disable open-field PV.

    Returns
    -------
    dict with DataArrays (lat, lon), all in [0, 1]:
        "clc_eligible_pv_rooftop"   : fraction of cell eligible for rooftop PV
        "clc_eligible_pv_openfield" : fraction of cell eligible for open-field PV
                                      (already weighted by ``partial_pv_openfield["fraction"]``)
        "clc_eligible_wind"         : fraction of cell eligible for onshore wind
        "built_up_fraction"         : fraction of cell that is built-up area
                                      (subset of rooftop PV; use in pv_power_density)
    """
    print("Loading CLC 2018 ...")
    clc = load_clc(clc_path)

    print("Building binary eligibility masks at 100m ...")
    mask_pv           = _make_binary_mask(clc, eligible_pv_codes)
    mask_pv_openfield = _make_binary_mask(clc, partial_pv_openfield["codes"])
    mask_wind_full    = _make_binary_mask(clc, eligible_wind_codes)
    mask_wind_part    = _make_binary_mask(clc, partial_wind["codes"])
    mask_built_up     = _make_binary_mask(clc, URBAN_FABRIC + INDUSTRIAL_COMMERCIAL)

    print("Aggregating to target grid (this may take a few minutes) ...")
    frac_pv           = _aggregate_to_target(mask_pv,           target)
    frac_pv_openfield = _aggregate_to_target(mask_pv_openfield, target)
    frac_wind_full    = _aggregate_to_target(mask_wind_full,    target)
    frac_wind_part    = _aggregate_to_target(mask_wind_part,    target)
    frac_built_up     = _aggregate_to_target(mask_built_up,     target)

    # Apply partial eligibility weight to open-field fraction (kept separate)
    frac_pv_openfield_weighted = (frac_pv_openfield * partial_pv_openfield["fraction"]).clip(0.0, 1.0)

    # Combine full + partial wind eligibility
    frac_wind = frac_wind_full + frac_wind_part * partial_wind["fraction"]
    frac_wind = frac_wind.clip(0.0, 1.0)

    # Attach metadata
    frac_pv.attrs.update({
        "long_name": "Land eligibility fraction for rooftop PV",
        "units":     "1",
        "clc_codes": str(eligible_pv_codes),
        "source":    "CORINE Land Cover 2018",
    })
    frac_pv_openfield_weighted.attrs.update({
        "long_name": "Land eligibility fraction for open-field PV",
        "units":     "1",
        "clc_codes":          str(partial_pv_openfield["codes"]),
        "openfield_fraction": partial_pv_openfield["fraction"],
        "source":             "CORINE Land Cover 2018",
    })
    frac_wind.attrs.update({
        "long_name": "Land eligibility fraction for onshore wind",
        "units":     "1",
        "clc_codes": str(eligible_wind_codes),
        "partial_codes":    str(partial_wind["codes"]),
        "partial_fraction": partial_wind["fraction"],
        "source":    "CORINE Land Cover 2018",
    })
    frac_built_up.attrs.update({
        "long_name": "Built-up area fraction (urban fabric + industrial)",
        "units":     "1",
        "source":    "CORINE Land Cover 2018",
    })

    print(
        f"\nEligibility summary (area-weighted European mean):\n"
        f"  PV rooftop   : {float(frac_pv.mean()):.3f}\n"
        f"  PV open-field: {float(frac_pv_openfield_weighted.mean()):.3f} "
        f"(fraction={partial_pv_openfield['fraction']})\n"
        f"  Wind     : {float(frac_wind.mean()):.3f}\n"
        f"  Built-up : {float(frac_built_up.mean()):.3f}"
    )

    return {
        "clc_eligible_pv_rooftop":   frac_pv.rename("clc_eligible_pv_rooftop"),
        "clc_eligible_pv_openfield": frac_pv_openfield_weighted.rename("clc_eligible_pv_openfield"),
        "clc_eligible_wind":         frac_wind.rename("clc_eligible_wind"),
        "built_up_fraction":         frac_built_up.rename("built_up_fraction"),
    }

# ── 3.  COMBINE CLC + PROTECTED AREAS  ───────────────────────────────────────

def compute_eligibility(
    clc_path:    Union[str, Path],
    target:      xr.DataArray,
    natura_path: Union[str, Path, None] = None,
    wdpa_path:   Union[str, Path, None] = None,
    eligible_pv_codes:    list[int] = ELIGIBLE_PV_CODES,
    eligible_wind_codes:  list[int] = ELIGIBLE_WIND_CODES,
    partial_wind:         dict      = PARTIAL_ELIGIBILITY_WIND,
    partial_pv_openfield: dict      = PARTIAL_ELIGIBILITY_PV_OPENFIELD,
) -> dict:
    """
    Compute final land eligibility fractions combining CLC land cover
    and protected area exclusions.

    Protected areas are applied multiplicatively:
        eligible_final = clc_eligible × (1 - exclusion_fraction)

    A cell that is 80% CLC-eligible but 50% protected → 40% final eligibility.
    WDPA takes priority over Natura 2000 if both paths are provided.

    Parameters
    ----------
    clc_path    : path to CLC 2018 GeoTIFF
    target      : reference DataArray (lat, lon, EPSG:4326)
    natura_path : path to Natura 2000 GPKG (optional)
    wdpa_path   : path to WDPA GDB or SHP (optional, takes priority)

    Returns
    -------
    dict of DataArrays (lat, lon) in [0, 1]:
        "eligible_fraction_pv_rooftop"   : final rooftop PV eligibility
        "eligible_fraction_pv_openfield" : final open-field PV eligibility
        "eligible_fraction_wind"         : final wind eligibility
        "built_up_fraction"              : built-up fraction (input to pv_power_density)
        "clc_eligible_pv_rooftop"        : CLC-only rooftop PV eligibility (diagnostic)
        "clc_eligible_pv_openfield"      : CLC-only open-field PV eligibility (diagnostic)
        "clc_eligible_wind"              : CLC-only wind eligibility (diagnostic)
        "exclusion_pv"                   : protected area exclusion fraction, PV
        "exclusion_wind"                 : protected area exclusion fraction, wind
    """
    # Step 1: CLC eligibility
    print("\n=== CLC land cover eligibility ===")
    clc = compute_clc_eligibility(
        clc_path, target,
        eligible_pv_codes=eligible_pv_codes,
        eligible_wind_codes=eligible_wind_codes,
        partial_wind=partial_wind,
        partial_pv_openfield=partial_pv_openfield,
    )

    # Step 2: Protected areas
    zero = xr.zeros_like(clc["clc_eligible_pv_rooftop"])
    exclusion = {"exclusion_pv": zero, "exclusion_wind": zero}

    if wdpa_path is not None:
        print("\n=== WDPA protected areas ===")
        gdf = load_wdpa(wdpa_path)
        exclusion = build_protected_exclusion_mask(gdf, target, source="wdpa")
    elif natura_path is not None:
        print("\n=== Natura 2000 protected areas ===")
        gdf = load_natura2000(natura_path)
        exclusion = build_protected_exclusion_mask(gdf, target, source="natura2000")
    else:
        print("\n[!] No protected area dataset provided — CLC eligibility only.")

    # Step 3: Combine — eligible_final = CLC_eligible × (1 − exclusion)
    # Both PV components share the same exclusion mask (PV is excluded from PAs
    # regardless of whether it is rooftop or ground-mounted).
    eligible_pv_rooftop = (
        clc["clc_eligible_pv_rooftop"] * (1 - exclusion["exclusion_pv"])
    ).clip(0.0, 1.0)

    eligible_pv_openfield = (
        clc["clc_eligible_pv_openfield"] * (1 - exclusion["exclusion_pv"])
    ).clip(0.0, 1.0)

    eligible_wind = (
        clc["clc_eligible_wind"] * (1 - exclusion["exclusion_wind"])
    ).clip(0.0, 1.0)

    eligible_pv_rooftop.attrs.update({
        "long_name": "Final land eligibility fraction for rooftop PV",
        "units": "1", "source": "CLC 2018 + protected areas",
    })
    eligible_pv_openfield.attrs.update({
        "long_name": "Final land eligibility fraction for open-field PV",
        "units": "1", "source": "CLC 2018 + protected areas",
    })
    eligible_wind.attrs.update({
        "long_name": "Final land eligibility fraction for onshore wind",
        "units": "1", "source": "CLC 2018 + protected areas",
    })

    print(
        f"\n=== Eligibility summary ===\n"
        f"  PV rooftop  : {float(clc['clc_eligible_pv_rooftop'].mean()):.3f} (CLC)  →  "
        f"{float(eligible_pv_rooftop.mean()):.3f} (after PA exclusion)\n"
        f"  PV openfield: {float(clc['clc_eligible_pv_openfield'].mean()):.3f} (CLC)  →  "
        f"{float(eligible_pv_openfield.mean()):.3f} (after PA exclusion)\n"
        f"  Wind : {float(clc['clc_eligible_wind'].mean()):.3f} (CLC)  →  "
        f"{float(eligible_wind.mean()):.3f} (after PA exclusion)"
    )

    return {
        "eligible_fraction_pv_rooftop":   eligible_pv_rooftop.rename("eligible_fraction_pv_rooftop"),
        "eligible_fraction_pv_openfield": eligible_pv_openfield.rename("eligible_fraction_pv_openfield"),
        "eligible_fraction_wind":         eligible_wind.rename("eligible_fraction_wind"),
        "built_up_fraction":              clc["built_up_fraction"],
        "clc_eligible_pv_rooftop":        clc["clc_eligible_pv_rooftop"],
        "clc_eligible_pv_openfield":      clc["clc_eligible_pv_openfield"],
        "clc_eligible_wind":              clc["clc_eligible_wind"],
        "exclusion_pv":                   exclusion["exclusion_pv"],
        "exclusion_wind":                 exclusion["exclusion_wind"],
    }



# ── 4.  SAVE  ─────────────────────────────────────────────────────────────────

def save_eligibility(eligibility: dict[str, xr.DataArray], out_path: Union[str, Path]):
    """Save all eligibility fractions to a single NetCDF file."""
    ds = xr.Dataset(eligibility)
    ds.attrs = {
        "title":  "Land eligibility fractions from CORINE Land Cover 2018",
        "source": "CLC 2018 v2020_20u1, aggregated to 10km lat/lon grid",
    }
    ds.to_netcdf(out_path)
    print(f"Saved → {out_path}")
