import datetime
from typing import Dict, Union
import xarray as xr
import pandas as pd

from ..utils.utils_xarray import get_time_resolution


def extract_adjusted_timesteps(
    df_climate_models_mapping: pd.DataFrame,
    climate_model_chain: str,
    climate_scenario: str,
) -> Dict[str, Union[pd.DatetimeIndex, int]]:
    """Extract adjusted timesteps from EURO-CORDEX climate models mapping file for a given
    climate model chain and scenario combination. This mapping shrinks the timesteps of the
    original simulation to obtain a faster warming that aligns with the CMIP6 GCM model equivalent
    of the GCM driving the EURO-CORDEX model.

    Parameters
    ----------
    df_climate_models_mapping : pd.DataFrame
        pandas DataFrame containing the timesteps mapping for the EURO-CORDEX climate models.
        It contains the columns "name", "scenarios_mapped", "year_list", "start_year".
    climate_model_chain : str
        The name of the EURO-CORDEX climate model chain.
    climate_scenario : str
        The acronym of the climate scenario. Can either be "rcp26", "rcp45" or "rcp85".

    Returns
    -------
    Dict[str, pd.DatetimeIndex]
        A dictionary containing the original timesteps to select from the simulation
        and the new timesteps for the adjusted timeseries.

    Raises
    ------
    ValueError
        If the climate model chain and scenario combination is not found in the mapping file.
    """
    mapping = df_climate_models_mapping[
        (df_climate_models_mapping["name"].str.contains(climate_model_chain))
        & (df_climate_models_mapping["scenarios_mapped"] == climate_scenario)
    ]
    if mapping.empty:
        raise ValueError(
            f"""Climate model chain {climate_model_chain} for climate scenario
            {climate_scenario} not found in mapping file"""
        )

    year_list = mapping["year_list"].item()
    start_year = mapping["start_year"].item()

    if "mode" in mapping.columns:
        if "original" in mapping["mode"].item():
            if mapping["start_year"].item() > 1991:
                start_year = 1991
                missing_years = list(range(1991, mapping["start_year"].item()))
                year_list = missing_years + year_list
            if mapping["year_list"].item()[0] > 1991:
                missing_years = list(range(1991, mapping["year_list"].item()[0]))
                year_list = missing_years + year_list
            if mapping["year_list"].item()[-1] < 2099:
                missing_years = list(range(mapping["year_list"].item()[-1] + 1, 2100))
                year_list = year_list + missing_years

    timesteps = None
    for y in year_list:
        r = pd.date_range(f"01-01-{y} 00:00", f"12-31-{y} 00:00", freq="D")
        if timesteps is None:
            timesteps = r
        else:
            timesteps = timesteps.union(r)
    adjusted_timesteps = {
        "original_timesteps": timesteps,
        "start_year": start_year,
    }

    return adjusted_timesteps


def get_new_timesteps(
    adjusted_timesteps: Dict[str, Union[pd.DatetimeIndex, int]], ds_data: xr.Dataset
) -> pd.DatetimeIndex:
    """Get the new timesteps for the adjusted timeseries, after extracting the adjusted
    timesteps from the xarray Dataset.

    Parameters
    ----------
    adjusted_timesteps : Dict[str, Union[pd.DatetimeIndex, int]]
        A dictionary containing the original timesteps to select from the simulation
        and the new timesteps for the adjusted timeseries, along with the start year
        of the new timeseries.
    ds_data : xr.Dataset
        xarray Dataset containing the climate data.

    Returns
    -------
    pd.DatetimeIndex
        The new timesteps for the adjusted timeseries.
    """
    start_hour = ds_data.time[0].dt.hour.item()
    start_minute = ds_data.time[0].dt.minute.item()
    time_resolution = timedelta_to_freq(
        pd.to_timedelta(ds_data.time[1].item() - ds_data.time[0].item())
    )
    new_timesteps = pd.date_range(
        f"01-01-{adjusted_timesteps['start_year']} {start_hour}:{start_minute}",
        periods=len(ds_data.time),
        freq=time_resolution,
    )
    return new_timesteps


def timedelta_to_freq(timedelta: Union[pd.Timedelta, datetime.timedelta]) -> str:
    """
    Convert a Pandas Timedelta or a timedelta-like object into a frequency string for pd.date_range.

    Parameters
    ----------
    timedelta : (pd.Timedelta or datetime.timedelta)
        The timedelta to convert.

    Returns
    -------
    str
        Frequency string (e.g., '3H', '30T').
    """
    # Ensure the input is a Pandas Timedelta
    td = pd.to_timedelta(timedelta)

    # Extract components
    days = td.days
    seconds = td.seconds
    microseconds = td.microseconds

    # Convert components to frequency
    if days > 0:
        return f"{days}D"
    elif seconds >= 3600:  # Hours
        hours = seconds // 3600
        return f"{hours}h"
    elif seconds >= 60:  # Minutes
        minutes = seconds // 60
        return f"{minutes}min"
    elif seconds > 0:  # Seconds
        return f"{seconds}s"
    elif microseconds > 0:  # Microseconds
        return f"{microseconds}us"
    else:
        raise ValueError("Cannot determine frequency for zero timedelta.")

def fill_missing_timesteps(ds: xr.Dataset, drop_leap_days: bool = True) -> xr.Dataset:
    """
    Expand the time dimension of the dataset to include all expected timesteps and interpolate missing values.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset containing a 'time' dimension.
    drop_leap_days : bool, optional
        If True, drop February 29 from the expected timeseries before comparison.

    Returns
    -------
    xr.Dataset
        The dataset with filled timesteps.
    """
    times = pd.DatetimeIndex(ds["time"].values)
    # remove timezone awareness for comparison
    if times.tz is not None:
        times = times.tz_convert(None)

    times = times.sort_values().unique()
    freq = pd.infer_freq(times)
    if freq is None:
        # fallback: use the mode of the diffs
        diffs = times.to_series().diff().dropna()
        mode = diffs.mode()
        freq = mode.iloc[0]

    expected = pd.date_range(start=times[0], end=times[-1], freq=freq)

    if drop_leap_days:
        expected = expected[~((expected.month == 2) & (expected.day == 29))]

    # Reindex dataset to include all expected timesteps
    ds_filled = ds.reindex(time=expected)

    # Interpolate missing values
    ds_filled = ds_filled.interpolate_na(dim="time", method="linear", fill_value="extrapolate")

    return ds_filled

def fill_missing_timesteps_ffill_like(ds: xr.Dataset, drop_leap_days: bool = True) -> xr.Dataset:
    """
    Expand the time dimension to include all expected timesteps and fill missing values.
    Daily data uses simple forward-fill; subdaily data forward-fills based on matching time-of-day.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset with a 'time' dimension.
    drop_leap_days : bool, optional
        Whether to drop February 29 when generating expected timestamps.

    Returns
    -------
    xr.Dataset
        Dataset with missing timesteps filled using forward-fill logic.
    """
    # ---- Detect frequency ----
    times = pd.DatetimeIndex(ds["time"].values)
    if times.tz is not None:
        times = times.tz_convert(None)

    times = times.sort_values().unique()
    freq = pd.infer_freq(times)

    if freq is None:
        diffs = times.to_series().diff().dropna()
        freq = diffs.mode().iloc[0]

    expected = pd.date_range(start=times[0], end=times[-1], freq=freq)

    if drop_leap_days:
        expected = expected[~((expected.month == 2) & (expected.day == 29))]

    # ---- Reindex to full timeline ----
    ds = ds.reindex(time=expected)

    # ---- DAILY CASE → simple forward fill ----
    if freq == "D":
        return ds.ffill(dim="time")

    # ---- SUBDAILY CASE → forward fill by time-of-day ----
    time_res = get_time_resolution(ds)
    # # how many periods make up one day
    periods_per_day = int(round(pd.Timedelta(days=1) / time_res))
    if periods_per_day >= 1:
        # shift data by one day (in periods) so value at t comes from t - 1 day
        ds_prev_day = ds.shift(time=periods_per_day)
        # fill NaNs with previous-day same time-of-day values
        ds = ds.where(
            ~ds.isnull(), ds_prev_day
        )
        del ds_prev_day

    # fallback: forward-fill any remaining NaNs along time
    ds = ds.ffill(dim="time")

    return ds