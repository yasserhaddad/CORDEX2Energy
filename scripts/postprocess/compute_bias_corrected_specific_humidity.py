import argparse
import sys

sys.path.append("../..")
import json
import time
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar
from src.constants import COMPRESSED_ZARR_ENCODING
from src.convert_climate_vars import (
    calculate_specific_humidity,
)

dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})
dask.config.set(num_workers=10, scheduler="processes")
dask.config.set({"array.chunk-size": "500MiB"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])

    # Load the data paths
    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / args.temp_res
    )


    t_start = time.time()
    print("Loading data")
    # Load the data
    ds_tas = xr.open_zarr(bias_corrected_data_dir / "tas.zarr")
    ds_psl = xr.open_zarr(bias_corrected_data_dir / "psl.zarr")
    ds_hurs = xr.open_zarr(bias_corrected_data_dir / "hurs.zarr")

    if ds_tas["tas"].attrs["units"] == "K":
            ds_tas["tas"] -= 273.15

    if ds_psl["psl"].attrs["units"] == "Pa":
        ds_psl["psl"] /= 100

    print("Calculating specific humidity")
    # Calculate specific humidity

    with ProgressBar(dt=60):
        specific_humidity = calculate_specific_humidity(
            ds_tas["tas"], ds_hurs["hurs"], ds_psl["psl"], dask="allowed"
        ).compute()

    ds_huss = xr.Dataset(
        {
            "huss": specific_humidity
        },
        coords=ds_tas.coords,
    )
    ds_huss["huss"].attrs = {
        "long_name": "Near-Surface Specific Humidity",
        "units": "kg kg-1",
        "standard_name": "specific humidity",
    }

    # Save the specific humidity data
    target_chunks = {
        "time": 365 if args.temp_res == "day" else 2920,
        "lat": 256 if args.temp_res == "day"  else 128,
        "lon": 256 if args.temp_res == "day"  else 128,
    }

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(target_chunks.values()),
        }
        for var in ds_huss.data_vars
    }
    ds_huss.to_zarr(bias_corrected_data_dir / "huss_derived.zarr",
                    encoding=zarr_encoding,
                    mode="w")

    print(f"\tProcessing time: {(time.time() - t_start) / 60:.2f} minutes.")