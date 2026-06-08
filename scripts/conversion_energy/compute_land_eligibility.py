"""
compute_land_eligibility.py
---------------------------
Compute land eligibility fractions for rooftop PV and onshore wind from
CORINE Land Cover 2018 (CLC) and, optionally, protected area datasets
(Natura 2000 or WDPA).

Output is a NetCDF file saved under:
    <path_data_processed>/maps/land_eligibility/land_eligibility.nc

The target grid is read from an existing climate model output zarr store so
that the eligibility maps are aligned with the energy simulation grids.

Usage
-----
    python compute_land_eligibility.py <paths_file> \\
        [--clc PATH] [--natura PATH] [--wdpa PATH] \\
        [--scope europe|ch]

Arguments
---------
paths_file
    JSON file with path configuration (e.g. paths.json or paths_so4.json).
--clc
    Path to U2018_CLC2018_V2020_20u1.tif. Defaults to
    <path_data_processed>/../datasets/land_cover/corine_2018/DATA/U2018_CLC2018_V2020_20u1.tif
--natura
    (optional) Path to Natura 2000 GeoPackage (.gpkg).
--wdpa
    (optional) Path to WDPA GDB or SHP. Takes priority over --natura if both given.
--scope
    'europe' (default) or 'ch'. Determines which reference zarr is used to
    define the target grid.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import xarray as xr

from cordex2energy.conversion_energy.land_eligibility import (
    ELIGIBLE_PV_CODES,
    ELIGIBLE_WIND_CODES,
    PARTIAL_ELIGIBILITY_WIND,
    PARTIAL_ELIGIBILITY_PV_OPENFIELD,
    compute_eligibility,
    save_eligibility,
)


def _load_target_grid(paths: dict, scope: str) -> xr.DataArray:
    """
    Load a single (lat, lon) DataArray slice from an existing zarr store to
    use as the target grid template for rioxarray reprojection.
    """
    path_processed = Path(paths["path_data_processed"])

    # Find any zarr store under energy/<scope>/pv to get the grid
    search_root = path_processed / "energy" / scope / "pv"
    zarr_stores = sorted(search_root.glob("**/*.zarr"))
    if not zarr_stores:
        # Fall back to energy/europe/pv if scope-specific not found
        zarr_stores = sorted((path_processed / "energy" / "europe" / "pv").glob("**/*.zarr"))
    if not zarr_stores:
        raise FileNotFoundError(
            f"No zarr store found under {search_root} to derive the target grid. "
            "Please provide a zarr store or adapt _load_target_grid()."
        )

    print(f"  Using target grid from: {zarr_stores[0]}")
    ds = xr.open_zarr(zarr_stores[0])
    # Pick first data variable, drop all dims except lat/lon
    var = list(ds.data_vars)[0]
    da = ds[var]
    # Squeeze extra dims (e.g. time, model) to get a (lat, lon) slice
    extra_dims = [d for d in da.dims if d not in ("lat", "lon")]
    for dim in extra_dims:
        da = da.isel({dim: 0}, drop=True)
    return da.load()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CLC land eligibility maps for PV and wind."
    )
    parser.add_argument(
        "paths_file",
        type=str,
        help="Path to the JSON file containing path configuration.",
    )
    parser.add_argument(
        "--clc",
        type=str,
        default=None,
        help="Path to CLC 2018 GeoTIFF (U2018_CLC2018_V2020_20u1.tif).",
    )
    parser.add_argument(
        "--natura",
        type=str,
        default=None,
        help="Path to Natura 2000 GeoPackage (.gpkg), optional.",
    )
    parser.add_argument(
        "--wdpa",
        type=str,
        default=None,
        help="Path to WDPA geodatabase or shapefile, optional.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="europe",
        choices=["europe", "ch"],
        help="Target grid scope: 'europe' (default) or 'ch'.",
    )
    parser.add_argument(
        "--pv-openfield-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            f"Fraction of open-field CLC-eligible land usable for ground-mounted PV "
            f"(default: {PARTIAL_ELIGIBILITY_PV_OPENFIELD['fraction']:.3f}). "
            "Set to 0 to disable open-field PV."
        ),
    )
    parser.add_argument(
        "--wind-forest-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            f"Fraction of forest CLC land usable for onshore wind turbines "
            f"(default: {PARTIAL_ELIGIBILITY_WIND['fraction']:.3f}). "
            "Set to 0 to exclude forests entirely."
        ),
    )
    args = parser.parse_args()

    if args.pv_openfield_fraction is not None and not 0.0 <= args.pv_openfield_fraction <= 1.0:
        parser.error("--pv-openfield-fraction must be between 0 and 1.")
    if args.wind_forest_fraction is not None and not 0.0 <= args.wind_forest_fraction <= 1.0:
        parser.error("--wind-forest-fraction must be between 0 and 1.")

    # ── Load paths ────────────────────────────────────────────────────────────
    paths_file = Path(args.paths_file)
    if not paths_file.is_file():
        print(f"[ERROR] Paths file not found: {paths_file}", file=sys.stderr)
        sys.exit(1)

    with paths_file.open() as fh:
        paths = json.load(fh)

    path_processed = Path(paths["path_data_processed"])

    # ── Resolve CLC path ──────────────────────────────────────────────────────
    if args.clc is not None:
        clc_path = Path(args.clc)
    else:
        clc_path = (
            path_processed.parent
            / "datasets"
            / "land_cover"
            / "corine_2018"
            / "DATA"
            / "U2018_CLC2018_V2020_20u1.tif"
        )

    if not clc_path.is_file():
        print(
            f"[ERROR] CLC GeoTIFF not found: {clc_path}\n"
            "  Download from https://land.copernicus.eu/pan-european/corine-land-cover/clc2018\n"
            "  or pass the correct path with --clc.",
            file=sys.stderr,
        )
        sys.exit(1)

    natura_path = Path(args.natura) if args.natura else None
    wdpa_path   = Path(args.wdpa)   if args.wdpa   else None

    # ── Output path ───────────────────────────────────────────────────────────
    out_dir = path_processed / "maps" / "land_eligibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "land_eligibility.nc"

    print(f"\nOutput → {out_file}")
    print(f"CLC    → {clc_path}")
    if natura_path:
        print(f"Natura → {natura_path}")
        out_file = out_dir / f"{out_file.stem}_with_natura.nc"
    if wdpa_path:
        print(f"WDPA   → {wdpa_path}")
        out_file = out_dir / f"{out_file.stem}_with_wdpa.nc"

    # ── Target grid ───────────────────────────────────────────────────────────
    print(f"\nLoading target grid (scope={args.scope}) ...")
    target = _load_target_grid(paths, scope=args.scope)
    print(f"  Target grid: {dict(target.sizes)}")

    # ── Override partial eligibility fractions if requested ──────────────────
    partial_pv_openfield = dict(PARTIAL_ELIGIBILITY_PV_OPENFIELD)
    if args.pv_openfield_fraction is not None:
        partial_pv_openfield["fraction"] = args.pv_openfield_fraction
        print(f"  open-field PV fraction overridden → {args.pv_openfield_fraction}")

    partial_wind = dict(PARTIAL_ELIGIBILITY_WIND)
    if args.wind_forest_fraction is not None:
        partial_wind["fraction"] = args.wind_forest_fraction
        print(f"  wind-in-forest fraction overridden  → {args.wind_forest_fraction}")

    # ── Compute eligibility ───────────────────────────────────────────────────
    t0 = time.time()
    eligibility = compute_eligibility(
        clc_path=clc_path,
        target=target,
        natura_path=natura_path,
        wdpa_path=wdpa_path,
        eligible_pv_codes=ELIGIBLE_PV_CODES,
        eligible_wind_codes=ELIGIBLE_WIND_CODES,
        partial_wind=partial_wind,
        partial_pv_openfield=partial_pv_openfield,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_eligibility(eligibility, out_file)


if __name__ == "__main__":
    main()
