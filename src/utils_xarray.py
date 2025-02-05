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
    if time_dim not in ds:
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
