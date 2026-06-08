"""Aggregate pre-computed PV and wind power outputs over TYNDP 2024 bidding zones.

Usage
-----
python compute_tyndp_aggregates.py <paths_file> <model_chain> <scenario> [options]

Reads
-----
  energy/europe/pv/<model_tag>/pv_<model_tag>.zarr
  energy/europe/wind_power/<model_tag>/wind_power_<model_tag>_<turbine>.zarr

Writes
------
  energy/europe/pv/<model_tag>/pv_tyndp_agg_<model_tag>.nc
  energy/europe/wind_power/<model_tag>/wind_power_tyndp_agg_<model_tag>_<turbine>.nc
"""

import argparse
import json
import os
import time
from pathlib import Path

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.conversion_energy.aggregate_data import aggregate_data_per_bidding_zone
from cordex2energy.conversion_energy.tyndp import (
    build_tyndp_nuts_mapping,
    build_tyndp_zone_geometries,
    drop_overlapping_zones,
    load_nuts_geodata,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_DEFAULT_TYNDP_FILE = (
    "20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tyndp_zones(path_data_processed: Path, tyndp_filename: str):
    """Load the TYNDP bidding zone list and build overlap-free zone geometries.

    Parameters
    ----------
    path_data_processed:
        Root of the processed data tree.
    tyndp_filename:
        Filename of the TYNDP supply Excel workbook, expected at
        ``<path_data_processed>/extremes/tyndp/<tyndp_filename>``.

    Returns
    -------
    gdf_tyndp_zones : gpd.GeoDataFrame
    """
    path_tyndp = path_data_processed / "extremes" / "tyndp" / tyndp_filename
    if not path_tyndp.exists():
        raise FileNotFoundError(f"TYNDP supply file not found: {path_tyndp}")

    # Read bidding zone codes from the solar sheet (column "Country" = zone code)
    df = pd.read_excel(path_tyndp, sheet_name="1.1.", header=[2, 3])
    df.columns = [
        l2 if str(l2) == "Country" else f"{l2}_{l1}"
        for l1, l2 in df.columns
    ]
    df = df.loc[:, "Country":"2050_HIGH"]
    tyndp_zones = sorted(df["Country"].dropna().iloc[:-2].unique())

    # Build geometries from NUTS
    path_nuts = path_data_processed / "maps" / "nuts"
    gdf_nuts1, gdf_nuts2, gdf_uk_nuts1 = load_nuts_geodata(path_nuts, verbose=False)

    tyndp_nuts2_mapping, df_tyndp_nuts_mapping = build_tyndp_nuts_mapping(
        tyndp_zones, gdf_nuts2, gdf_uk_nuts1
    )
    gdf = build_tyndp_zone_geometries(
        tyndp_nuts2_mapping, df_tyndp_nuts_mapping, gdf_nuts2, gdf_uk_nuts1
    )
    gdf = drop_overlapping_zones(gdf)
    return gdf


def _aggregate_and_save(
    ds: xr.Dataset,
    gdf_zones,
    output_path: Path,
    method: str,
    extra_attrs: dict | None = None,
) -> None:
    """Aggregate *ds* over TYNDP zones and write a NetCDF file."""
    with ProgressBar(dt=30):
        ds_agg = aggregate_data_per_bidding_zone(ds, gdf_zones, method=method).compute()

    if extra_attrs:
        for var in ds_agg.data_vars:
            ds_agg[var].attrs.update(extra_attrs)

    if output_path.exists():
        output_path.unlink()
    ds_agg.to_netcdf(output_path, mode="w")
    print(f"    → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate pre-computed PV and wind power over TYNDP 2024 bidding zones."
    )
    parser.add_argument(
        "paths_file",
        help="Path to the JSON file containing data paths.",
    )
    parser.add_argument(
        "climate_model_chain",
        help="Climate model chain identifier (e.g. CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1).",
    )
    parser.add_argument(
        "climate_scenario",
        help="Climate scenario (e.g. rcp85).",
    )
    parser.add_argument(
        "--method",
        default="median_and_better",
        choices=["mean", "sum", "median_and_better"],
        help="Spatial aggregation method (default: median_and_better).",
    )
    parser.add_argument(
        "--tyndp_file",
        default=_DEFAULT_TYNDP_FILE,
        help="Filename of the TYNDP supply Excel workbook inside "
             "<path_data_processed>/extremes/tyndp/.",
    )
    parser.add_argument(
        "--skip_pv",
        action="store_true",
        help="Skip PV aggregation.",
    )
    parser.add_argument(
        "--skip_wind",
        action="store_true",
        help="Skip wind power aggregation.",
    )
    args = parser.parse_args()

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])
    path_energy_europe = path_data_processed / "energy" / "europe"
    model_tag = f"{args.climate_model_chain}_{args.climate_scenario}"

    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,
            "distributed.worker.memory.spill": 0.8,
            "distributed.worker.memory.pause": 0.9,
            "array.chunk-size": "512MiB",
            "logging.distributed": "error",
        }
    )

    time_start = time.time()

    # ------------------------------------------------------------------
    # Build TYNDP zone geometries (cached NUTS files reused automatically)
    # ------------------------------------------------------------------
    print("Building TYNDP 2024 bidding zone geometries ...")
    gdf_tyndp_zones = build_tyndp_zones(path_data_processed, args.tyndp_file)
    print(f"  {len(gdf_tyndp_zones)} zones after overlap removal.")

    common_attrs = {
        "aggregation_method": args.method,
        "spatial_domain": "TYNDP 2024 bidding zones",
        "climate_model_chain": args.climate_model_chain,
        "climate_scenario": args.climate_scenario,
    }

    # ------------------------------------------------------------------
    # PV
    # ------------------------------------------------------------------
    if not args.skip_pv:
        pv_dir = path_energy_europe / "pv" / model_tag
        pv_zarr = pv_dir / f"pv_{model_tag}.zarr"

        if not pv_zarr.exists():
            print(f"[WARN] PV zarr not found, skipping: {pv_zarr}")
        else:
            print("\nAggregating PV over TYNDP zones ...")
            ds_pv = xr.open_zarr(pv_zarr, chunks="auto")

            extra = {
                **common_attrs,
                "source_zarr": str(pv_zarr),
            }
            _aggregate_and_save(
                ds_pv,
                gdf_tyndp_zones,
                pv_dir / f"pv_tyndp_agg_{model_tag}.nc",
                method=args.method,
                extra_attrs=extra,
            )
            del ds_pv

    # ------------------------------------------------------------------
    # Wind power (one file per turbine type)
    # ------------------------------------------------------------------
    if not args.skip_wind:
        wind_dir = path_energy_europe / "wind_power" / model_tag

        if not wind_dir.exists():
            print(f"[WARN] Wind power directory not found, skipping: {wind_dir}")
        else:
            turbine_zarrs = sorted(
                wind_dir.glob(f"wind_power_{model_tag}_*.zarr")
            )
            if not turbine_zarrs:
                print(f"[WARN] No wind power zarr files found in {wind_dir}")

            for zarr_path in turbine_zarrs:
                # Extract the turbine tag from the filename
                turbine_tag = zarr_path.stem.replace(f"wind_power_{model_tag}_", "")
                print(f"\nAggregating wind power ({turbine_tag}) over TYNDP zones ...")

                ds_wind = xr.open_zarr(zarr_path, chunks="auto")
                extra = {
                    **common_attrs,
                    "turbine_type": turbine_tag.replace("_", "/"),
                    "source_zarr": str(zarr_path),
                }
                _aggregate_and_save(
                    ds_wind,
                    gdf_tyndp_zones,
                    wind_dir / f"wind_power_tyndp_agg_{model_tag}_{turbine_tag}.nc",
                    method=args.method,
                    extra_attrs=extra,
                )
                del ds_wind

    print(
        f"\nTotal time: {(time.time() - time_start) / 60:.1f} minutes."
    )


if __name__ == "__main__":
    main()
