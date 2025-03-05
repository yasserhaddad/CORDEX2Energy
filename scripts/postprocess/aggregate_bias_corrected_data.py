import argparse
import json
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import xarray as xr

dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})
dask.config.set(num_workers=20, scheduler="processes")
dask.config.set({"array.chunk-size": "500MiB"})

def aggregate_to_daily(ds: xr.Dataset) -> xr.Dataset:
    return ds.resample(time='1D').mean()

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

    bias_corrected_data_dir = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
    )

    for file in (bias_corrected_data_dir / args.temp_res).glob("*.zarr"):
        with ProgressBar(dt=60):
            ds = aggregate_to_daily(
                xr.open_zarr(file)
            ).compute()
            ds.to_zarr(bias_corrected_data_dir / "day" / file.name, mode="w")
