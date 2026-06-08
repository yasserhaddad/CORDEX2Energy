import argparse
import json
import time
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.data_processing.convert_climate_vars import (
    calculate_specific_humidity,
)
from cordex2energy.utils.constants import (
    BC_DAILY_CHUNKS,
    BC_SUBDAILY_CHUNKS,
    COMPRESSED_ZARR_ENCODING,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate climate data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("temp_res", type=str, help="Temporal resolution.")

    args = parser.parse_args()

    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set({
        "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
        "distributed.worker.memory.spill": 0.8,   # Start spilling at 80%
        "distributed.worker.memory.pause": 0.9,   # Pause worker at 90%
        "array.chunk-size": "512MiB",
        "logging.distributed": "error",           # Less verbose logs
    })

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
    ds_tas = xr.open_zarr(bias_corrected_data_dir / "tas.zarr", chunks="auto")
    ds_ps = xr.open_zarr(bias_corrected_data_dir / "ps.zarr", chunks="auto")
    ds_hurs = xr.open_zarr(bias_corrected_data_dir / "hurs.zarr", chunks="auto")

    if args.temp_res == "day":
        ds_tas = ds_tas.assign_coords(time=ds_tas.time.dt.floor('D'))
        ds_ps = ds_ps.assign_coords(time=ds_ps.time.dt.floor('D'))
        ds_hurs = ds_hurs.assign_coords(time=ds_hurs.time.dt.floor('D'))

    ds_tas = ds_tas.sel(time=ds_ps.time.values)
    ds_hurs = ds_hurs.sel(time=ds_ps.time.values)

    if ds_tas["tas"].attrs["units"] == "K":
            ds_tas["tas"] -= 273.15
            ds_tas["tas"].attrs["units"] = "°C"

    if ds_ps["ps"].attrs["units"] == "Pa":
        ds_ps["ps"] /= 100
        ds_ps["ps"].attrs["units"] = "hPa"

    print("Calculating specific humidity")
    # Calculate specific humidity

    with ProgressBar(dt=60):
        specific_humidity = calculate_specific_humidity(
            ds_tas["tas"], ds_hurs["hurs"], ds_ps["ps"], dask="allowed"
        )

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
        print("Specific humidity calculated. Saving data...")
        print("ds_huss:", ds_huss.time.values[0], "to", ds_huss.time.values[-1])

        # Save the specific humidity data
        target_chunks = BC_DAILY_CHUNKS if args.temp_res == "day" else BC_SUBDAILY_CHUNKS

        zarr_encoding = {
            var: {
                "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
                "chunks": tuple(target_chunks.values()),
            }
            for var in ds_huss.data_vars
        }
        ds_huss.to_zarr(bias_corrected_data_dir / "huss.zarr",
                        encoding=zarr_encoding,
                        mode="w")

        print(f"\tProcessing time: {(time.time() - t_start) / 60:.2f} minutes.")