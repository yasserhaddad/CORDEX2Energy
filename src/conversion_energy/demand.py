from typing import Dict, List
import numpy as np
import xarray as xr


def process_population_data(
    da_pop: xr.DataArray, ds_tas: xr.Dataset, coarsen: bool = True
) -> xr.DataArray:
    da_pop = da_pop.where(da_pop >= 0, np.nan)
    # Step 1: Coarsen (if needed to reduce data size)
    da_pop_coarse = da_pop.coarsen(lat=10, lon=10, boundary="trim").sum()

    # Step 2: Align to temperature dataset grid
    da_pop_coarse = da_pop_coarse.interp(
        lat=ds_tas.lat, lon=ds_tas.lon, method="nearest"
    )
    return da_pop_coarse


def normalize_population_data(
    da_pop: xr.DataArray, da_country_mask: xr.DataArray
) -> xr.DataArray:
    list_da_pop_countries = []
    for country in np.unique(da_country_mask.values):
        if not np.isnan(country):
            da_pop_country = da_pop.where(da_country_mask == country, 0)
            da_pop_country = da_pop_country / da_pop_country.sum()
            list_da_pop_countries.append(da_pop_country)

    return sum(list_da_pop_countries)


def heating_degree_days(
    x: np.ndarray, reference_temp: float = 18, base_temp: float = 15
) -> np.ndarray:
    """
    Calculate the heating degree days (HDD) for a given temperature array.

    Heating degree days are a measure of the demand for energy needed to heat a building.
    It is calculated as the difference between a reference temperature and the actual
    temperature, only when the actual temperature is below a specified base temperature.

    Parameters
    ----------
    x : np.ndarray
        Array of temperatures (in degrees Celsius).
    reference_temp : float, optional
        Reference temperature (in degrees Celsius) used for the calculation.
        Default is 18°C.
    base_temp : float, optional
        Base temperature (in degrees Celsius) below which heating is considered necessary.
        Default is 15°C.

    Returns
    -------
    np.ndarray
        Array of heating degree days corresponding to the input temperatures.
        Values are zero where the input temperature is above the base temperature.

    """
    return np.where(x <= base_temp, reference_temp - x, 0)


def cooling_degree_days(
    x: np.ndarray, reference_temp: float = 21, base_temp: float = 24
) -> np.ndarray:
    """
    Calculate cooling degree days (CDD) based on temperature data.
    Cooling degree days are a measure of how much (in degrees) and for how long
    the outside air temperature is above a certain base temperature. This is
    often used to estimate the energy demand for cooling buildings.
    Parameters
    ----------
    x : np.ndarray
        Array of temperature values (in degrees Celsius).
    reference_temp : float, optional
        The reference temperature used for calculating the degree days.
        Default is 21°C.
    base_temp : float, optional
        The base temperature above which cooling is considered necessary.
        Default is 24°C.
    Returns
    -------
    np.ndarray
        An array of cooling degree days corresponding to the input temperature
        values. If a temperature value is below the base temperature, the
        cooling degree day is 0 for that value.
    """
    return np.where(x >= base_temp, x - reference_temp, 0)


def compute_cdd_hdd_xr(
    da_tas: xr.DataArray,
    da_pop_normalized: xr.DataArray,
    parallelized: bool = True,
) -> xr.Dataset:
    metrics_funcs = {
        "HDD": heating_degree_days,
        "CDD": cooling_degree_days,
    }
    list_ds = []
    for name, metric_func in metrics_funcs.items():
        ds_metric = xr.apply_ufunc(
            metric_func,
            da_tas,
            keep_attrs=True,
            dask="allowed" if parallelized else "forbidden",
        )
        list_ds.append(
            (
                (
                    (ds_metric * da_pop_normalized).resample(time="1D").sum("time") / 24
                ).rename(name)
            )
        )

    ds_metric_weighted = xr.merge(list_ds)
    ds_metric_weighted.attrs["units"] = "degree days"
    ds_metric_weighted.attrs["long_name"] = "Degree days"
    ds_metric_weighted.attrs["description"] = (
        "Heating and cooling degree days calculated from temperature data."
    )

    return ds_metric_weighted


def compute_cooling_days(x: np.ndarray, base_temp: float = 24):
    return x >= base_temp


def compute_heating_days(x: np.ndarray, base_temp: float = 15):
    return x <= base_temp


def compute_cdm_hdm_xr(
    da_tas: xr.DataArray,
    da_pop_normalized: xr.DataArray,
    temperature_weighting: bool = False,
    parallelized: bool = True,
) -> xr.Dataset:
    metrics_funcs = {
        "heating_days": compute_heating_days,
        "cooling_days": compute_cooling_days,
    }
    list_ds = []
    for name, metric_func in metrics_funcs.items():
        ds_metric = xr.apply_ufunc(
            metric_func,
            da_tas,
            keep_attrs=True,
            dask="allowed" if parallelized else "forbidden",
        )
        da_metric = ds_metric if temperature_weighting else (ds_metric * da_pop_normalized)
        da_metric = (da_metric.resample(time="ME").sum("time")).rename(name)
        list_ds.append(da_metric)

    ds_metric_weighted = xr.merge(list_ds)
    ds_metric_weighted.attrs["units"] = "days"
    ds_metric_weighted.attrs["long_name"] = "Heating and cooling days per month"
    ds_metric_weighted.attrs["description"] = (
        "Heating and cooling days per month calculated from temperature data."
    )

    return ds_metric_weighted
