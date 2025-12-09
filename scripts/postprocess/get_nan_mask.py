import argparse
import json
from pathlib import Path
import xarray as xr
import sys
sys.path.append("../..")
from src.utils.utils_xarray import get_nan_mask

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

    path_tas = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "regridded"
        / "3hr"
        / "tas.zarr"
    )
    ds_tas = xr.open_zarr(path_tas, chunks="auto")
    ds_nan_mask = get_nan_mask(
        ds_tas.isel(time=0), var_name="tas", climate_model_chain=args.climate_model_chain
    ).drop_vars("height")

    output_path = (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
        / "mask.nc"
    )
    ds_nan_mask.to_netcdf(output_path)
    print(f"NaN mask saved to {output_path}")
