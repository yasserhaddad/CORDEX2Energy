import argparse
import json
import sys
import dask
from pathlib import Path



from cordex2energy.data_processing.load_data import ClimateModelData
from cordex2energy.conversion_energy.aggregate_data import aggregate_data_per_country
from dask.diagnostics import ProgressBar
from cordex2energy.utils.constants import EXTENT_CH, BC_SUBDAILY_CHUNKS
import pandas as pd

vars = ["tas", "rsds", "sfcWind", "huss", "psl"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wind power.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    path_data_processed = Path(paths["path_data_processed"])
    path_data_temporary = Path(paths["path_data_temporary"])

    output_path = (
        path_data_temporary
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "bias_corrected"
        / "3hr"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
            "distributed.worker.memory.spill": 0.8,  # Start spilling at 80%
            "distributed.worker.memory.pause": 0.9,  # Pause worker at 90%
            "array.chunk-size": "512MiB",
            "logging.distributed": "error",  # Less verbose logs
        }
    )
    path_climate_data = path_data_processed / "cordex_processed" / f"{args.climate_model_chain}_{args.climate_scenario}"
    for var in vars:
        print(f"Loading {var} data")
        with ProgressBar(dt=10):
            ds = ClimateModelData(
                path_climate_data
                / ("bias_corrected" if var != "psl" else "regridded")
                / ("3hr" if var != "psl" else "day")
                / f"{var}.zarr",
                extent=EXTENT_CH,
                chunks="auto",
            )

            ds.ds_data = ds.ds_data.sel(
                time=slice("2018", "2052")
            )

            ds_agg = aggregate_data_per_country(
                ds.ds_data, method="mean", offshore=False
            )

            if var == "psl":
                # create 3-hourly index from the first day's 00:00 to the last day's 21:00
                orig_time = pd.to_datetime(ds.ds_data["time"].values)
                start = orig_time[0].replace(hour=0)
                end = orig_time[-1].replace(hour=0) + pd.Timedelta(days=1) - pd.Timedelta(hours=3)
                new_time = pd.date_range(start=start, end=end, freq="3H")

                # replicate daily values into each 3-hour slot by forward-filling
                ds.ds_data = ds.ds_data.reindex({"time": new_time}, method="ffill")

            # Compute before saving to avoid multiprocessing lock error
            ds_data_computed = ds.ds_data.compute()
            output_var_path = output_path / f"{var}_ch.nc"
            if output_var_path.exists():
                output_var_path.unlink()
            ds_data_computed.to_netcdf(output_var_path, mode="w")

            ds_agg_ch = ds_agg.sel(country="Switzerland").compute()
            output_var_agg_path = output_path / f"{var}_agg_ch.nc"
            if output_var_agg_path.exists():
                output_var_agg_path.unlink()
            ds_agg_ch.to_netcdf(
                output_var_agg_path, mode="w"
            )
