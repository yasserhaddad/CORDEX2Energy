from typing import List
import xarray as xr
import numpy as np
import pandas as pd


def xr_reorder_coords(ds: xr.Dataset, new_order: List[str]) -> xr.Dataset:
    """
    Reorder the given coordinates in an xarray.Dataset while keeping the others unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    new_order : list of str
        The desired order of some coordinates.

    Returns
    -------
    xr.Dataset
        The dataset with reordered coordinates.

    Raises
    ------
    ValueError
        If some coordinates in new_order are not in the dataset dimensions.
    """
    # Get the current coordinate order
    current_order = list(ds.dims)

    # Ensure all new_order elements exist in current_order
    if not set(new_order).issubset(set(current_order)):
        raise ValueError(
            "Some coordinates in new_order are not in the dataset dimensions"
        )

    # Get the coordinates that should remain unchanged
    unchanged_coords = [dim for dim in current_order if dim not in new_order]

    # Combine new order with unchanged coordinates
    final_order = new_order + unchanged_coords

    # Reorder dimensions
    return ds.transpose(*final_order)


def get_time_resolution(ds, time_dim="time"):
    """
    Determines the time resolution of an xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The input dataset or data array.
    time_dim : str, optional
        The name of the time dimension (default is "time").

    Returns
    -------
    pandas.Timedelta
        The most common time difference between consecutive timesteps.

    Raises
    ------
    ValueError
        If the dataset does not contain the specified time dimension or if there are not enough time points to determine the resolution.
    """
    if time_dim not in ds.coords:
        raise ValueError(f"Dataset does not contain the time dimension '{time_dim}'")

    time_values = ds[time_dim].values  # Extract time values as numpy datetime array
    if len(time_values) < 2:
        raise ValueError("Not enough time points to determine resolution")

    time_diffs = pd.Series(
        time_values[1:] - time_values[:-1]
    )  # Compute time differences
    return time_diffs.mode()[0]  # Return the most frequent time resolution


def get_valid_hours(time_resolution, start_hour=0):
    """
    Get all possible hours corresponding to a given time resolution, starting from a specific hour.

    Parameters
    ----------
    time_resolution : pandas.Timedelta
        The time step resolution (e.g., from get_time_resolution function).
    start_hour : int, optional
        The starting hour (default is 0).

    Returns
    -------
    list
        A list of valid hours in a day.

    Raises
    ------
    ValueError
        If time_resolution is not a pandas.Timedelta object or if the resolution is too fine (<1 hour).
    """
    # Ensure resolution is a pandas Timedelta
    if not isinstance(time_resolution, pd.Timedelta):
        raise ValueError("time_resolution must be a pandas Timedelta object")

    # Convert resolution to hours
    hours_step = time_resolution.total_seconds() / 3600  # Convert to hours
    if hours_step < 1:
        raise ValueError("Resolution is too fine (<1 hour), which is unusual for this method.")

    # Generate hours starting from start_hour up to 24
    valid_hours = np.arange(start_hour, 24, hours_step, dtype=int)

    return list(valid_hours)


def get_specific_hourly_data(ds: xr.Dataset, hour: int) -> xr.Dataset:
    """
    Select data from an xarray Dataset for a specific hour of the day.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset containing time series data.
    hour : int
        The specific hour of the day (0-23) to filter the data.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing data only for the specified hour.
    """

    return ds.sel(time=ds.time.dt.hour == hour)


def match_time_steps(ds_data: xr.Dataset, ds_reference: xr.Dataset, time_res: str, skip_leap_days: bool = True) -> xr.Dataset:
    """
    Adjust the time steps of the reference dataset to match the time steps of the data dataset.

    Parameters
    ----------
    ds_data : xr.Dataset
        The dataset containing the desired time steps.
    ds_reference : xr.Dataset
        The reference dataset to be adjusted.
    time_res : str
        The time resolution to use for adjusting the time steps.

    Returns
    -------
    xr.Dataset
        The reference dataset with adjusted time steps.
    """
    if not np.array_equal(np.unique(ds_data.time.dt.hour), np.unique(ds_reference.time.dt.hour)):
        print("Adjusting time steps of the reference dataset")
        original_length = len(ds_reference.time)
        first_time_step = pd.to_datetime(ds_data.time.values[0])
        start_date = pd.to_datetime(ds_reference.time.values[0]).replace(
            hour=first_time_step.hour, minute=first_time_step.minute
        )
        if skip_leap_days:
            new_time = []
            current_date = start_date
            while len(new_time) < original_length:
                if not (current_date.month == 2 and current_date.day == 29):
                    new_time.append(current_date)
                current_date += pd.to_timedelta(time_res)
        else:
            new_time = pd.date_range(start=start_date, periods=original_length, freq=time_res)
        ds_reference = ds_reference.assign_coords(time=new_time)

    return ds_reference


def get_min_max(ds: xr.Dataset, time_resolution: str) -> xr.Dataset:
    """
    Calculate the minimum and maximum values of an xarray Dataset over a given time resolution.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    time_resolution : str
        The time resolution to calculate the min and max values over.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing the minimum and maximum values over the specified time resolution.

    Raises
    ------
    ValueError
        If time_resolution is not in the format expected by the resample function of xarray.
    """
    # Check if time_resolution is in the correct format
    try:
        pd.to_timedelta(time_resolution)
    except ValueError:
        raise ValueError("time_resolution must be a string in a format recognized by pandas.to_timedelta")

    return ds.resample(time=time_resolution).min(), ds.resample(time=time_resolution).max()