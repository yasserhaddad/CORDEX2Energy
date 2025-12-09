#!/usr/bin/env python3
"""
Convert NetCDF files from 360-day calendar to Gregorian calendar.

This script processes all NetCDF files in a specified directory, converting any files
with a 360-day calendar to Gregorian calendar. For sub-daily data, it fills NaN values
(from added days) using values from the previous day. For daily or coarser data, it uses
forward-fill.

The original files are replaced with the converted versions.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


def get_time_resolution(ds: xr.Dataset) -> pd.Timedelta:
    """
    Get the time resolution of a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with time dimension

    Returns
    -------
    pd.Timedelta
        Time resolution
    """
    if "time" not in ds.dims or len(ds.time) < 2:
        return pd.Timedelta(days=1)

    time_diff = ds.time.diff(dim="time")
    return pd.Timedelta(time_diff.median().values)


def convert_360day_to_gregorian(nc_file: Path, dry_run: bool = False) -> bool:
    """
    Convert a NetCDF file from 360-day calendar to Gregorian.

    Parameters
    ----------
    nc_file : Path
        Path to NetCDF file
    dry_run : bool
        If True, only report what would be done without modifying files

    Returns
    -------
    bool
        True if file was converted, False if no conversion needed
    """
    print(f"\nProcessing: {nc_file.name}")

    try:
        # Open without chunking first to check calendar
        ds = xr.open_dataset(nc_file).chunk({"time": 50})

        # Check if calendar conversion is needed
        if not hasattr(ds.time, "encoding"):
            print("  → No time encoding found, skipping")
            ds.close()
            return False

        calendar = ds.time.encoding.get("calendar", None)

        if calendar not in ["360_day"]:
            print(f"  → Calendar is '{calendar}', no conversion needed")
            ds.close()
            return False

        if dry_run:
            print("  → Would convert from 360_day to gregorian")
            ds.close()
            return True

        print("  → Converting from 360_day to gregorian...")

        units = ds.time.encoding.get("units", "days since 1850-01-01")
        dtype = ds.time.encoding.get("dtype", np.dtype('float64'))

        # Keep only data variables (exclude coordinates and bounds)
        data_vars_to_keep = [var for var in ds.data_vars if ds[var].dtype != object]
        if data_vars_to_keep:
            ds = ds[data_vars_to_keep]

        # Convert calendar
        ds = ds.convert_calendar(
            "gregorian",
            dim="time",
            missing=np.nan,
            align_on="year",
        )

        # Detect time resolution
        time_res = get_time_resolution(ds)
        print(f"  → Time resolution: {time_res}")

        # Fill NaN values from calendar conversion
        if time_res < pd.Timedelta(days=1):
            # Sub-daily: fill with previous day's values
            periods_per_day = int(round(pd.Timedelta(days=1) / time_res))

            if periods_per_day >= 1:
                print(f"  → Filling NaNs with shifted values ({periods_per_day} periods per day)...")

                # # Detect NaN timesteps by checking the first data variable
                # first_var = list(ds.data_vars)[0]
                # # Check which timesteps have all-NaN spatial data
                # nan_mask = ds[first_var].isnull().all(dim=[d for d in ds[first_var].dims if d != 'time']).compute()
                # nan_times = ds.time.where(nan_mask, drop=True)

                # if len(nan_times) > 0:
                #     print(f"  → Found {len(nan_times)} NaN timesteps to fill")
                #     # For each NaN timestep, get the value from one day earlier
                #     for var in list(ds.data_vars):
                #         # Calculate the shifted times (one day earlier)
                #         shifted_times = nan_times.values - pd.Timedelta(days=1)

                #         # Get values from shifted times (lazy operation)
                #         shifted_data = ds[var].sel(time=shifted_times)

                #         # Assign nan_times as the time coordinate for shifted_data
                #         shifted_data['time'] = nan_times

                #         # Fill NaN values at those specific times (lazy operation)
                #         ds[var] = ds[var].combine_first(shifted_data)

                #         del shifted_data
                # else:
                #     print("  → No NaN timesteps found")
                # shift data by one day (in periods) so value at t comes from t - 1 day
                ds_prev_day = ds.shift(time=periods_per_day)
                # fill NaNs with previous-day same time-of-day values
                ds = ds.where(
                    ~ds.isnull(), ds_prev_day
                )
                del ds_prev_day

            # Fallback: forward-fill any remaining NaNs along time
            ds = ds.ffill(dim="time")
        else:
            # Daily or coarser: simple forward fill
            print("  → Using forward-fill for daily or coarser data")
            ds = ds.ffill(dim="time")

        # Save to temporary file
        temp_file = nc_file.parent / f"{nc_file.stem}_temp.nc"
        print("  → Saving to temporary file...")
        if temp_file.exists():
            temp_file.unlink()

        # Prepare encoding to preserve compression
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 1,
                'shuffle': True,
            }
            # Preserve original dtype if possible
            if var in ds.data_vars:
                encoding[var]['dtype'] = ds[var].dtype

        # Set time encoding to gregorian
        encoding['time'] = {
            'dtype': dtype,
            'calendar': 'gregorian',
            'units': units,
        }

        with ProgressBar(dt=5):
            ds.to_netcdf(temp_file, encoding=encoding)

        ds.close()

        # Replace original file with converted file
        print("  → Replacing original file...")
        nc_file.unlink()
        temp_file.rename(nc_file)

        print(f"  ✓ Successfully converted {nc_file.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {nc_file.name}: {str(e)}")
        # Clean up temp file if it exists
        temp_file = nc_file.parent / f"{nc_file.stem}_temp.nc"
        if temp_file.exists():
            temp_file.unlink()
        return False


def main():
    """Main function to process all NetCDF files in a directory."""
    parser = argparse.ArgumentParser(
        description="Convert NetCDF files from 360-day calendar to Gregorian calendar."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing NetCDF files to convert"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually converting files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.nc",
        help="File pattern to match (default: *.nc)"
    )

    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    # Find all NetCDF files
    if args.recursive:
        nc_files = list(directory.rglob(args.pattern))
    else:
        nc_files = list(directory.glob(args.pattern))

    if not nc_files:
        print(f"No NetCDF files found matching pattern '{args.pattern}' in {directory}")
        sys.exit(0)

    print(f"Found {len(nc_files)} NetCDF file(s) in {directory}")

    if args.dry_run:
        print("\n--- DRY RUN MODE: No files will be modified ---\n")

    # Process each file
    converted_count = 0
    skipped_count = 0
    error_count = 0

    for nc_file in nc_files:
        try:
            if convert_360day_to_gregorian(nc_file, dry_run=args.dry_run):
                converted_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"\n✗ Unexpected error processing {nc_file.name}: {str(e)}")
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(nc_files)}")
    print(f"Converted: {converted_count}")
    print(f"Skipped (no conversion needed): {skipped_count}")
    print(f"Errors: {error_count}")

    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")


if __name__ == "__main__":
    main()
