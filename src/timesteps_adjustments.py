import datetime
from typing import Dict, Union
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


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

    timesteps = None
    for y in mapping.year_list.item():
        r = pd.date_range(f"01-01-{y} 00:00", f"12-31-{y} 00:00", freq="D")
        if timesteps is None:
            timesteps = r
        else:
            timesteps = timesteps.union(r)
    adjusted_timesteps = {
        "original_timesteps": timesteps,
        "start_year": mapping.start_year.item(),
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
        return f"{hours}H"
    elif seconds >= 60:  # Minutes
        minutes = seconds // 60
        return f"{minutes}T"
    elif seconds > 0:  # Seconds
        return f"{seconds}S"
    elif microseconds > 0:  # Microseconds
        return f"{microseconds}U"
    else:
        raise ValueError("Cannot determine frequency for zero timedelta.")
