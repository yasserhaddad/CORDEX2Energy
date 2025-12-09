import argparse
import json
import os
import sys
import time
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.conversion_energy.wind import (
    CH_TURBINES,
    compute_smooth_power_curve,
    compute_wind_power,
)
from src.utils.constants import COMPRESSED_ZARR_ENCODING, BC_SUBDAILY_CHUNKS


def main():
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()

    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])

    # Threading environment variables to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
    os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")

    dask.config.set({"logging.distributed": "error"})
    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set({"array.chunk-size": "400MiB"})

    turbines = CH_TURBINES

    time_start = time.time()
    print("Computing smoothed power curves...")
    power_curves_smoothed = {
        turbine: compute_smooth_power_curve(turbine) for turbine in turbines
    }

    print("Loading wind data...")

    path_wind_100m = path_data_processed / "wind_downscaled"

    output_path_root = path_wind_100m / "energy"
    output_path_root.mkdir(parents=True, exist_ok=True)

    ds_wind = xr.open_mfdataset(
        path_wind_100m.glob(f"*_{args.climate_scenario}*_final.nc"),
        combine="by_coords",
        parallel=True,
    ).chunk(BC_SUBDAILY_CHUNKS)
    wind_var = list(ds_wind.data_vars.keys())[0]

    print("Computing all wind power outputs...")
    turbine_type = "E-92/2350"

    with ProgressBar(dt=60):
        ds_wind_power = compute_wind_power(
            ds_wind[wind_var],
            power_curves_smoothed[turbine_type][turbine_type.replace("/", "_")],
            turbine_type.replace("/", "_"),
            step=0.01,
            parallelized=True,
        ).to_dataset()

        # write aggregated zarr for CH
        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
            }
            for var in ds_wind_power.data_vars
        }
        output_path = output_path_root / f"wind_power_{args.climate_scenario}.zarr"
        (output_path.parent).mkdir(parents=True, exist_ok=True)
        ds_wind_power.to_zarr(output_path, mode="w", encoding=zarr_encoding)

    print(
        "Time to compute wind power for all turbines: ",
        (time.time() - time_start) / 60,
        " minutes.",
    )


if __name__ == "__main__":
    main()
