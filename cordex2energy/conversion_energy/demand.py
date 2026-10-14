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

    Follows the Eurostat/EC definition, which is stated in terms of ``Tm``, the **mean air
    temperature of the day**: ``if Tm <= 15 then HDD = 18 - Tm else 0``. The rule is non-linear,
    so feeding it sub-daily temperatures and averaging afterwards is a different quantity — see
    :func:`compute_cdd_hdd_xr` and its ``eurostat_daily`` argument.

    Parameters
    ----------
    x : np.ndarray
        Array of **daily mean** temperatures (in degrees Celsius).
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

    Follows the Eurostat/EC definition, which is stated in terms of ``Tm``, the **mean air
    temperature of the day**: ``if Tm >= 24 then CDD = Tm - 21 else 0``. The rule is non-linear and
    the threshold sits in the tail of the European temperature distribution, so applying it to
    sub-daily temperatures and averaging afterwards gives a very different result — 8.6x the annual
    CDD for the UK, 75x for Ireland. See :func:`compute_cdd_hdd_xr` and its ``eurostat_daily``
    argument.

    Parameters
    ----------
    x : np.ndarray
        Array of **daily mean** temperature values (in degrees Celsius).
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
    eurostat_daily: bool = True,
) -> xr.Dataset:
    """Population-weighted heating and cooling degree days from a temperature field.

    ``eurostat_daily`` selects *what temperature the threshold is applied to*, which matters a
    great deal because the degree-day rule is non-linear:

    * ``True`` (default) — resample to the **daily mean** first, then apply the kernel. This is the
      Eurostat/EC definition: "``Tm`` is the mean air temperature of day *i*".
    * ``False`` — apply the kernel at the input (sub-daily) resolution and average afterwards. This
      is the historical behaviour, retained so earlier products remain reproducible.

    The two are not interchangeable. Sub-daily application counts warm afternoons on days whose mean
    never reaches the base temperature, so it yields far more cooling degree days in cool climates —
    on CERRA 1991-2020 the annual CDD ratio (sub-daily / Eurostat) is 1.4x for Spain but 8.6x for
    the UK and 75x for Ireland. Heating is barely affected (<=7 %) because most of the year the
    daily mean is well below the heating base temperature, so the threshold rarely binds.

    *da_pop_normalized* is time-invariant, so it may be applied on either side of the daily mean.
    """
    metrics_funcs = {
        "HDD": heating_degree_days,
        "CDD": cooling_degree_days,
    }
    da_input = da_tas.resample(time="1D").mean("time") if eurostat_daily else da_tas

    list_ds = []
    for name, metric_func in metrics_funcs.items():
        ds_metric = xr.apply_ufunc(
            metric_func,
            da_input,
            keep_attrs=True,
            dask="allowed" if parallelized else "forbidden",
        )
        weighted = ds_metric * da_pop_normalized
        if not eurostat_daily:
            weighted = weighted.resample(time="1D").mean("time")
        list_ds.append(weighted.rename(name))

    ds_metric_weighted = xr.merge(list_ds)
    ds_metric_weighted.attrs["units"] = "degree days"
    ds_metric_weighted.attrs["long_name"] = "Degree days"
    ds_metric_weighted.attrs["description"] = (
        "Heating and cooling degree days calculated from temperature data."
    )
    ds_metric_weighted.attrs["definition"] = (
        "eurostat_daily (threshold on the daily mean temperature)"
        if eurostat_daily
        else "sub-daily (threshold at input resolution, averaged to daily)"
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
