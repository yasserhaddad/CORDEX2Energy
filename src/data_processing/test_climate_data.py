import xarray as xr
import numpy as np

def has_duplicate_coords(ds: xr.Dataset) -> dict:
    """
    Check for duplicate values in all coordinates of an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset whose coordinates will be checked for duplicates.

    Returns
    -------
    dict
        A dictionary where the keys are coordinate names and the values are booleans
        indicating whether duplicates exist (True) or not (False) for each coordinate.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.Dataset({'a': ('x', [1, 2, 3])}, coords={'x': [0, 1, 1]})
    >>> has_duplicate_coords(ds)
    {'x': True}
    """
    result = {}
    for dim in ds.dims:
        values = ds[dim].values
        # For datetime64, convert to list for hashing
        if hasattr(values, 'tolist'):
            values = values.tolist()
        result[dim] = len(set(values)) != len(values)
    return result


def is_time_sorted(ds: xr.Dataset) -> bool:
    """
    Check if the 'time' coordinate in an xarray Dataset is sorted in ascending order.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to check for a sorted 'time' coordinate.

    Returns
    -------
    bool
        True if the 'time' coordinate is sorted in ascending order or not present; False otherwise.
    """
    if 'time' not in ds.coords:
        return True
    time_values = ds['time'].values
    return all(time_values[i] <= time_values[i+1] for i in range(len(time_values)-1))


def has_identical_random_timesteps(ds: xr.Dataset) -> bool:
    """
    Randomly selects two different timesteps from the dataset and checks if they are identical.

    This function compares all variable values for two randomly chosen timesteps in the input
    xarray Dataset. If all values are equal for both timesteps across all variables and dimensions,
    the function returns True; otherwise, it returns False.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset containing a 'time' dimension.

    Returns
    -------
    bool
        True if the two randomly selected timesteps are identical across all variables and dimensions,
        False otherwise.
    """
    if 'time' not in ds.dims or ds.sizes['time'] < 2:
        return False
    time_len = ds.sizes['time']
    idx = np.random.choice(time_len, size=2, replace=False)
    for var in ds.data_vars:
        arr0 = ds[var].isel(time=idx[0]).values
        arr1 = ds[var].isel(time=idx[1]).values
        if not np.array_equal(arr0, arr1):
            return False
    return True
