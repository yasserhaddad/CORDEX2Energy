from ast import literal_eval
import numpy as np
import pandas as pd
import windpowerlib.power_curves as pc
from windpowerlib.data import store_turbine_data_from_oedb
import xarray as xr

EUROPE_TURBINES = ['E-126/7580', 'SWT120/3600', 'SWT142/3150']
CH_TURBINES = ["E-82/2300", "E-92/2350", "V90/2000", "V112/3300"]

EUROPE_HUB_HEIGHTS = {
    "SWT142/3150": 129,  # this turbine has 3 possible hub heights. We choose the one in the middle.
    "SWT120/3600": 90,
    "E-126/7580": 127,
}

CH_HUB_HEIGHTS = {
    "E-82/2300": 98,
    "E-92/2350": 98,
    "V90/2000": 95,
    "V112/3300": 94
}


def cubic_spline_interpol(power: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cubic spline interpolation on a power curve to a resolution of 0.1 m/s.

    Parameters
    ----------
    power : pd.DataFrame
        DataFrame with wind speed as the index and power generation as the value.

    Returns
    -------
    pd.DataFrame
        Interpolated power curve with wind speed index at 0.1 m/s resolution.
    """
    # Define the interpolation range
    start, end = power.first_valid_index(), power.last_valid_index()
    new_index = np.arange(start, end + 0.01, 0.01)

    # Reindex and interpolate
    power = power.reindex(new_index, tolerance=1e-7, method='nearest')
    power = power.interpolate(method='spline', order=3, s=0)

    # Round the index for consistency
    power.index = np.round(power.index, decimals=2)
    return power


def compute_smooth_power_curve(turbine_type: str) -> pd.DataFrame:
    """
    Compute a smoothed power curve for a given turbine type.
    This function retrieves turbine data, extracts the power curve for the specified
    turbine type, normalizes it, and applies smoothing using a turbulence intensity
    method. The smoothed power curve is then interpolated using cubic spline interpolation.

    Parameters
    ----------
    turbine_type : str
        The type of turbine for which the smoothed power curve is to be computed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the smoothed power curve with wind speeds as the index
        and corresponding normalized power values.

    Raises
    ------
    ValueError
        If the power curve for the specified turbine type is not available.

    Notes
    -----
    - The turbulence intensity used for smoothing is based on Equation 3-13 in Knorr (2016).
    - The power curve values are normalized by dividing by the maximum value.

    References
    ----------
    Knorr, K. (2016). "Wind Turbine Power Curve Modeling and Analysis."
    """

    df_turbine = store_turbine_data_from_oedb()
    power_curve = df_turbine.loc[df_turbine['turbine_type'] == turbine_type].iloc[0]
    if not power_curve.has_power_curve:
        raise ValueError(f"Power curve for turbine type {turbine_type} not available.")
    power_curve['power_curve_wind_speeds'] = np.array(literal_eval(power_curve['power_curve_wind_speeds']))
    power_curve['power_curve_values'] = np.array(literal_eval(power_curve['power_curve_values']))

    power_curve.power_curve_values /= power_curve.power_curve_values.max()
    power_curve_smoothed = pc.smooth_power_curve(
        pd.Series(power_curve.power_curve_wind_speeds),
        pd.Series(power_curve.power_curve_values),
        standard_deviation_method="turbulence_intensity",
        turbulence_intensity=0.1246, # turbulence intensity taken from eq. 3-13 in Knorr (2016)
    )

    power_curve_smoothed = power_curve_smoothed.set_index('wind_speed')
    power_curve_smoothed = cubic_spline_interpol(power_curve_smoothed)
    power_curve_smoothed[power_curve_smoothed < 0] = 0

    return power_curve_smoothed.rename(columns={'value': turbine_type.replace("/", "_")})


def compute_power_law_exponent(
    wind_speed_h1: xr.DataArray,
    wind_speed_h2: xr.DataArray,
    height_h1: float,
    height_h2: float,
) -> xr.DataArray:
    """
    Compute the power law exponent for wind speed at two different heights.

    Parameters
    ----------
    wind_speed_h1 : xr.DataArray
        Wind speed at height h1.
    wind_speed_h2 : xr.DataArray
        Wind speed at height h2.
    height_h1 : float
        Height h1 in meters.
    height_h2 : float
        Height h2 in meters.

    Returns
    -------
    xr.DataArray
        Power law exponent for the wind speed at the two heights.
    """
    return np.log(wind_speed_h2 / wind_speed_h1) / np.log(height_h2 / height_h1)


def extrapolate_wind_speed(
    wind_speed: xr.DataArray,
    original_height: float,
    target_height: float,
    exponent: xr.DataArray,
) -> xr.DataArray:
    """
    Extrapolate wind speed from one height to another using the power law.

    Parameters
    ----------
    wind_speed : xr.DataArray
        Wind speed at the original height.
    original_height : float
        Original height in meters.
    target_height : float
        Target height in meters.
    exponent : xr.DataArray
        Power law exponent.

    Returns
    -------
    xr.DataArray
        Extrapolated wind speed at the target height.
    """
    return wind_speed * (target_height / original_height) ** exponent


# def compute_wind_power(
#     wind_speed: xr.DataArray,
#     power_curve: pd.Series,
#     turbine_type: str,
#     step: float = 0.01,
#     parallelized: bool = True,
# ) -> xr.DataArray:
#     """
#     Compute normalized power output using a precomputed power curve and wind speed data.

#     Parameters
#     ----------
#     wind_speed : xr.DataArray
#         Wind speed data at hub height.
#     power_curve : pd.Series
#         Series containing the smoothed power curve with wind speeds as the index
#         and corresponding (normalized) power values.
#     turbine_type : str
#         The type of turbine for which the power is computed.
#     step : float, optional
#         Wind speed resolution in m/s (default is 0.01).
#     parallelized : bool, optional
#         Whether to use Dask for parallelized computation (default is True).

#     Returns
#     -------
#     xr.DataArray
#         (Normalized) Wind power output as an xarray DataArray.
#     """
#     # Extract power table and wind speed range
#     power_table = power_curve.values
#     min_ws = power_curve.index.min()
#     max_ws = power_curve.index.max()

#     def fast_lookup(ws_array):
#         # Initialize power output array
#         power = np.zeros_like(ws_array, dtype=float)

#         # Mask valid wind speeds within the power curve range
#         valid = (ws_array >= min_ws) & (ws_array <= max_ws)

#         # Compute indices for valid wind speeds
#         idx = ((ws_array[valid] - min_ws) / step).astype(int)
#         idx = np.clip(idx, 0, len(power_table) - 1)

#         # Assign power values using fast array indexing
#         power[valid] = power_table[idx]
#         return power

#     # Apply the lookup function to the wind speed DataArray
#     wind_power = xr.apply_ufunc(
#         fast_lookup,
#         wind_speed.round(2),
#         input_core_dims=[["time", "lat", "lon"]],
#         output_core_dims=[["time", "lat", "lon"]],
#         output_dtypes=[float],
#         vectorize=True,
#         dask="parallelized" if parallelized else "forbidden",   # Enable Dask parallelization
#     )

#     wind_power.name = turbine_type

#     return wind_power

def compute_wind_power(
    wind_speed: xr.DataArray,
    power_curve: pd.Series,
    turbine_type: str,
    step: float = 0.01,
    parallelized: bool = True,
) -> xr.DataArray:
    """
    Compute normalized power output using a precomputed power curve and wind speed data.
    Uses xarray.map_blocks for high-performance Dask execution.
    """
    power_table = power_curve.values
    min_ws = power_curve.index.min()
    max_ws = power_curve.index.max()

    def blockwise_lookup(ws_block: np.ndarray) -> np.ndarray:
        """
        Fast lookup applied to each block of wind speeds (NumPy array).
        """
        idx = ((ws_block - min_ws) / step).astype(int)
        idx = np.clip(idx, 0, len(power_table) - 1)
        power = power_table[idx]
        power[(ws_block < min_ws) | (ws_block > max_ws)] = 0.0
        return power

    # Wrap the blockwise function to make it xarray-compatible
    def xr_blockwise_lookup(ws_block: xr.DataArray) -> xr.DataArray:
        result = xr.apply_ufunc(
            blockwise_lookup,
            ws_block,
            dask="allowed" if parallelized else "forbidden",
            output_dtypes=[float],
        )
        return result

    # Apply to all chunks
    wind_power = xr.map_blocks(xr_blockwise_lookup, wind_speed)

    # Assign name and copy metadata
    wind_power.name = turbine_type

    return wind_power