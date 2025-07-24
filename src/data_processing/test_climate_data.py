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
        if hasattr(values, "tolist"):
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
    if "time" not in ds.coords:
        return True
    time_values = ds["time"].values
    return all(
        time_values[i] <= time_values[i + 1] for i in range(len(time_values) - 1)
    )


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
    if "time" not in ds.dims or ds.sizes["time"] < 2:
        return False
    time_len = ds.sizes["time"]
    idx = np.random.choice(time_len, size=2, replace=False)
    for var in ds.data_vars:
        arr0 = ds[var].isel(time=idx[0]).values
        arr1 = ds[var].isel(time=idx[1]).values
        if not np.array_equal(arr0, arr1):
            return False
    return True

def parse_cdo_grid_file(grid_file_path: str) -> dict:
    """
    Parse a CDO grid description file to extract expected latitude and longitude ranges.

    Supports CDO ASCII grid files with 'xvals' and 'yvals' or 'xfirst', 'xlast', 'yfirst', 'ylast'.

    Parameters
    ----------
    grid_file_path : str
        Path to the CDO grid description file.

    Returns
    -------
    dict
        Dictionary with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max' and float values (or None if not found).
    """
    lat_min = None
    lat_max = None
    lon_min = None
    lon_max = None
    with open(grid_file_path, "r") as f:
        for line in f:
            # if xfirst/xlast/ylast/yfirst are present, extract min/max values
            if "xfirst" in line:
                lon_min = float(line.split("=")[1].strip().replace(",", ""))
            if "xlast" in line:
                lon_max = float(line.split("=")[1].strip().replace(",", ""))
            if "yfirst" in line:
                lat_min = float(line.split("=")[1].strip().replace(",", ""))
            if "ylast" in line:
                lat_max = float(line.split("=")[1].strip().replace(",", ""))
            # Alternative: parse xvals/yvals if present
            if "xvals" in line:
                vals = [float(x) for x in line.split("=")[1].strip().split()]
                lon_min, lon_max = min(vals), max(vals)
            if "yvals" in line:
                vals = [float(x) for x in line.split("=")[1].strip().split()]
                lat_min, lat_max = min(vals), max(vals)
    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def check_coordinate_ranges(ds: xr.Dataset, grid_file_path: str) -> bool:
    """
    Check if the latitude and longitude coordinates of an xarray Dataset are within the expected range from a CDO grid file.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing latitude ('lat') and longitude ('lon') coordinates to check.
    grid_file_path : str
        Path to the CDO grid file specifying the expected coordinate ranges.

    Returns
    -------
    bool
        True if both latitude and longitude ranges are within the expected bounds, False otherwise.

    Warns
    -----
    Prints a warning message if the latitude or longitude range is outside the expected range specified in the grid file.
    """
    grid_info = parse_cdo_grid_file(grid_file_path)
    lat = ds.lat.values
    lon = ds.lon.values
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    in_lat_range = (
        grid_info["lat_min"] is None or lat_min >= grid_info["lat_min"]
    ) and (grid_info["lat_max"] is None or lat_max <= grid_info["lat_max"])
    in_lon_range = (
        grid_info["lon_min"] is None or lon_min >= grid_info["lon_min"]
    ) and (grid_info["lon_max"] is None or lon_max <= grid_info["lon_max"])
    if not in_lat_range:
        print(
            f"Warning: Latitude range [{lat_min}, {lat_max}] is outside expected [{grid_info['lat_min']}, {grid_info['lat_max']}]"
        )
    if not in_lon_range:
        print(
            f"Warning: Longitude range [{lon_min}, {lon_max}] is outside expected [{grid_info['lon_min']}, {grid_info['lon_max']}]"
        )
    return in_lat_range and in_lon_range


def check_monotonic_latlon(ds: xr.Dataset) -> bool:
    """
    Check if latitude and longitude coordinates in an xarray Dataset are monotonic.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing 'lat' and 'lon' coordinates to check.

    Returns
    -------
    bool
        True if both latitude and longitude coordinates are monotonic (either increasing or decreasing), False otherwise.

    Warns
    -----
    Prints a warning message if either latitude or longitude coordinates are not monotonic.
    """
    lat = ds.lat.values
    lon = ds.lon.values
    lat_mono = np.all(np.diff(lat) >= 0) or np.all(np.diff(lat) <= 0)
    lon_mono = np.all(np.diff(lon) >= 0) or np.all(np.diff(lon) <= 0)
    if not lat_mono:
        print("Warning: Latitude coordinate is not monotonic.")
    if not lon_mono:
        print("Warning: Longitude coordinate is not monotonic.")
    return lat_mono and lon_mono
