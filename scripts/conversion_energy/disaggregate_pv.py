"""Disaggregate 3-hourly PV output to hourly resolution.

Three methods are supported:

  clearsky    — pvlib Ineichen clear-sky profile weighting.
                Fully Dask-parallel; no rsds input needed.

  catmullrom  — Radiation-weighted, local Catmull-Rom cubic interpolation.
                Fully Dask-parallel; rsds loaded with BC_SUBDAILY_CHUNKS.

  scipy       — Radiation-weighted, global scipy CubicSpline interpolation.
                Requires the full time axis in a single chunk; input data are
                automatically rechunked to REGRIDDED_CHUNKS before processing.

Usage
-----
python disaggregate_pv.py paths_so4.json \\
    CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1 rcp85 \\
    --scope ch --method catmullrom
"""

import argparse
import json
import time
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.conversion_energy.disaggregate import (
    disaggregate_dataset_clearsky,
    disaggregate_dataset_radiation,
)
from cordex2energy.utils.constants import (
    BC_SUBDAILY_CHUNKS,
    COMPRESSED_ZARR_ENCODING,
    EXTENT_CH,
    EXTENT_BC,
    REGRIDDED_CHUNKS,
)
from cordex2energy.utils.utils_xarray import sel_extent

# Direction → azimuth degrees
AZIMUTHS = {"N": 0, "NE": 45, "E": 90, "SE": 135,
            "S": 180, "SW": 225, "W": 270, "NW": 315}

# Hourly output chunks: keep lat/lon from BC_SUBDAILY_CHUNKS, expand time
# to one calendar year (8760 h) so each chunk covers a full year.
HOURLY_CHUNKS = {**BC_SUBDAILY_CHUNKS, "time": 8760}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disaggregate 3-hourly PV to hourly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths_file",
        help="Path to the JSON file containing data paths (e.g. paths_so4.json).",
    )
    parser.add_argument("climate_model_chain", help="Climate model chain identifier.")
    parser.add_argument("climate_scenario",    help="RCP scenario (e.g. rcp85).")
    parser.add_argument(
        "--scope",
        choices=["ch", "europe"],
        default="ch",
        help="Spatial scope: 'ch' (Switzerland) or 'europe'. Default: ch.",
    )
    parser.add_argument(
        "--method",
        choices=["clearsky", "catmullrom", "scipy"],
        default="catmullrom",
        help=(
            "Disaggregation method. "
            "'catmullrom' (default): radiation-weighted, Dask-parallel. "
            "'scipy': radiation-weighted, global spline (slower, auto-rechunked). "
            "'clearsky': pvlib Ineichen profiles, no rsds needed."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="rsds night mask threshold [W/m²]. Only used for radiation methods.",
    )

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

    args = parser.parse_args()

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    extent = EXTENT_CH if args.scope == "ch" else EXTENT_BC

    # ------------------------------------------------------------------
    # Input paths
    # ------------------------------------------------------------------
    pv_zarr = (
        path_data_processed
        / "energy"
        / args.scope
        / "pv"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / f"pv_{args.climate_model_chain}_{args.climate_scenario}.zarr"
    )
    rsds_zarr = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "rsds.zarr"
    )

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    output_zarr = (
        path_data_processed
        / "energy"
        / args.scope
        / "pv_hourly"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / f"pv_hourly_{args.climate_model_chain}_{args.climate_scenario}.zarr"
    )
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load 3-hourly PV (always loaded with BC_SUBDAILY_CHUNKS)
    # ------------------------------------------------------------------
    print(f"Loading PV data from {pv_zarr}")
    ds_pv = xr.open_zarr(pv_zarr, chunks=BC_SUBDAILY_CHUNKS)

    # ------------------------------------------------------------------
    # Disaggregate
    # ------------------------------------------------------------------
    t0 = time.time()

    if args.method == "clearsky":
        print("Method: clear-sky profiles (pvlib Ineichen)")
        ds_pv_hourly = disaggregate_dataset_clearsky(ds_pv, AZIMUTHS)

    else:
        # Both radiation methods need rsds
        print(f"Loading rsds from {rsds_zarr}")
        rsds_3h = sel_extent(
            xr.open_zarr(rsds_zarr, chunks=BC_SUBDAILY_CHUNKS)["rsds"],
            extent,
        )
        # Align spatial coordinates to PV grid
        rsds_3h = rsds_3h.sel(
            lat=ds_pv[list(ds_pv.data_vars)[0]].lat,
            lon=ds_pv[list(ds_pv.data_vars)[0]].lon,
            method="nearest",
            tolerance=0.05,
        )

        if args.method == "scipy":
            # scipy CubicSpline needs the full time axis in one Dask chunk.
            # Rechunk to REGRIDDED_CHUNKS: small spatial tiles, time=-1.
            print(
                "Method: scipy cubic spline — rechunking to REGRIDDED_CHUNKS "
                f"{REGRIDDED_CHUNKS} (time=-1, small spatial tiles)"
            )
            ds_pv   = ds_pv.chunk(REGRIDDED_CHUNKS)
            rsds_3h = rsds_3h.chunk(REGRIDDED_CHUNKS)
            ds_pv_hourly = disaggregate_dataset_radiation(
                ds_pv, rsds_3h, method="scipy", threshold=args.threshold
            )

        else:  # catmullrom
            print("Method: Catmull-Rom (local, Dask-parallel)")
            ds_pv_hourly = disaggregate_dataset_radiation(
                ds_pv, rsds_3h, method="catmullrom", threshold=args.threshold
            )

    # ------------------------------------------------------------------
    # Save to Zarr
    # ------------------------------------------------------------------
    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(HOURLY_CHUNKS.values()),
        }
        for var in ds_pv_hourly.data_vars
    }

    print(f"\nWriting hourly PV to {output_zarr}")
    with ProgressBar(dt=60):
        ds_pv_hourly.chunk(HOURLY_CHUNKS).to_zarr(
            output_zarr, mode="w", encoding=zarr_encoding
        )

    print(f"Done in {(time.time() - t0) / 60:.1f} min — saved to {output_zarr}")


if __name__ == "__main__":
    main()
