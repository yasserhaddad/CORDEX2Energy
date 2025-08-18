import argparse
import json
import sys
import dask
from pathlib import Path

sys.path.append("../..")

from src.data_processing.load_data import ClimateModelData
from src.conversion_energy.aggregate_data import aggregate_data_per_country
from dask.diagnostics import ProgressBar
from src.utils.constants import EXTENT_CH, BC_SUBDAILY_CHUNKS

vars = ["tas", "rsds", "sfcWind", "huss"]

dask.config.set(num_workers=15, scheduler="processes")
dask.config.set({"array.chunk-size": "500MiB"})

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

    for var in vars:
        print(f"Loading {var} data")
        with ProgressBar(dt=10):
            ds = ClimateModelData(
                path_data_processed
                / "cordex_processed"
                / f"{args.climate_model_chain}_{args.climate_scenario}"
                / "bias_corrected"
                / "3hr"
                / f"{var}.zarr",
                extent=EXTENT_CH,
                chunks=BC_SUBDAILY_CHUNKS,
            )

            ds.ds_data = ds.ds_data.sel(
                time=slice("2018", "2052")
            )

            ds_agg = aggregate_data_per_country(
                ds.ds_data, method="mean", offshore=False
            )

            # Compute before saving to avoid multiprocessing lock error
            ds_data_computed = ds.ds_data.compute()
            ds_data_computed.to_netcdf(output_path / f"{var}_ch.nc", mode="w")

            ds_agg_ch = ds_agg.sel(country="Switzerland").compute()
            ds_agg_ch.to_netcdf(
                output_path / f"{var}_agg_ch.nc", mode="w"
            )
