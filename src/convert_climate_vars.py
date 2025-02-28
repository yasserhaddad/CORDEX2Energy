from typing import Union
import numpy as np
import xarray as xr


def convert_wind_components_to_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Convert the wind components to wind speed.

    Parameters
    ----------
    u : xr.DataArray
        The zonal wind component.
    v : xr.DataArray
        The meridional wind component.

    Returns
    -------
    xr.DataArray
        The wind speed.
    """
    return xr.apply_ufunc(
        lambda x, y: np.sqrt(x**2 + y**2),
        u,
        v,
        # dask="parallelized",
        output_dtypes=[np.float32],
    )


def calculate_specific_humidity(
    temperature: xr.DataArray,
    relative_humidity: xr.DataArray,
    surface_pressure: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate specific humidity from temperature, relative humidity, and surface pressure.

    Parameters
    ----------
    temperature : xr.DataArray
        The temperature in degrees Celsius.
    relative_humidity : xr.DataArray
        The relative humidity as a percentage (0 to 100)
    surface_pressure : xr.DataArray
        The surface pressure in hPa.

    Returns
    -------
    xr.DataArray
        The specific humidity.
    """

    def specific_humidity_func(temp, rh, sp):
        # Calculate saturation vapor pressure (e_s) using the Clausius-Clapeyron equation (Bolton's approximation)
        e_s = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
        # Compute actual vapor pressure (e) from relative humidity and saturation vapor pressure
        e = rh * e_s / 100
        # Calculate specific humidity (q) from actual vapor pressure and surface pressure
        q = 0.622 * (e / sp)
        # q = 0.622 * e / (sp - (1 - 0.622) * e)

        return q

    return xr.apply_ufunc(
        specific_humidity_func,
        temperature,
        relative_humidity,
        surface_pressure,
        output_dtypes=[np.float32],
    )