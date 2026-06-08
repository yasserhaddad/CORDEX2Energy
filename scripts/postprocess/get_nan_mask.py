import argparse
import json
from pathlib import Path
import shutil
import xarray as xr
import sys

from cordex2energy.utils.utils_xarray import get_nan_mask
import dask
from dask.diagnostics import ProgressBar

# Configure Dask
dask.config.set(num_workers=16, threads_per_worker=6)
dask.config.set({"array.chunk-size": "400MiB"})
dask.config.set({"distributed.worker.memory.target": 0.75})
dask.config.set({"distributed.worker.memory.spill": 0.8})
dask.config.set({"distributed.worker.memory.pause": 0.9})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute NaN mask and optionally apply it to bias-corrected data.")
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths."
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument(
        "--apply-mask", action="store_true",
        help="Apply the mask to all bias-corrected variables."
    )

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

    # Apply mask to bias-corrected data if requested
    if args.apply_mask:
        print("\nApplying mask to bias-corrected data...")

        # Define variables by temporal resolution
        VARIABLES_3HR = ["tas", "hurs", "ps", "sfcWind", "rsds", "wind100m", "wind150m"]
        VARIABLES_DAY = ["pr", "mrro", "clt", "psl"]

        base_path = (
            path_data_processed
            / "cordex_processed"
            / f"{args.climate_model_chain}_{args.climate_scenario}"
            / "bias_corrected"
        )

        # Process 3-hourly variables
        for var in VARIABLES_3HR:
            var_path = base_path / "3hr" / f"{var}.zarr"
            if var_path.exists():
                print(f"  Processing {var} (3hr)...")
                ds = xr.open_zarr(var_path, chunks="auto")

                # Apply mask (set NaN values where mask is True)
                ds_masked = ds.where(~ds_nan_mask["mask"])

                # Write to temporary file, then replace original
                temp_path = base_path / "3hr" / f"{var}_masked.zarr"
                with ProgressBar(dt=10):
                    ds_masked.to_zarr(temp_path, mode="w")
                shutil.rmtree(var_path)
                temp_path.rename(var_path)
                print(f"    ✓ {var} masked and saved")
            else:
                print(f"    ⚠ {var} not found at {var_path}")

        # Process daily variables
        for var in VARIABLES_DAY:
            var_path = base_path / "day" / f"{var}.zarr"
            if var_path.exists():
                print(f"  Processing {var} (day)...")
                ds = xr.open_zarr(var_path, chunks="auto")

                # Apply mask (set NaN values where mask is True)
                ds_masked = ds.where(~ds_nan_mask["mask"])

                # Write to temporary file, then replace original
                temp_path = base_path / "day" / f"{var}_masked.zarr"
                with ProgressBar(dt=10):
                    ds_masked.to_zarr(temp_path, mode="w")
                shutil.rmtree(var_path)
                temp_path.rename(var_path)
                print(f"    ✓ {var} masked and saved")
            else:
                print(f"    ⚠ {var} not found at {var_path}")

        print("\n✓ Mask application complete!")