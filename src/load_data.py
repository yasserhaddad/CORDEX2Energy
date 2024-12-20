from pathlib import Path
from typing import Dict, List
import numpy as np
import xarray as xr
import pandas as pd


def extract_adjusted_timesteps(
    df_climate_models_mapping: pd.DataFrame,
    climate_model_chain: str,
    climate_scenario: str,
) -> Dict[str, pd.DatetimeIndex]:
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
            f'''Climate model chain {climate_model_chain} for climate scenario
            {climate_scenario} not found in mapping file'''
        )

    timesteps = None
    for y in mapping.year_list.item():
        r = pd.date_range(f"12-01-{y-1} 00:00", f"11-30-{y} 00:00", freq="D")
        if timesteps is None:
            timesteps = r
        else:
            timesteps = timesteps.union(r)
    adjusted_timesteps = {"original_timesteps": timesteps}
    adjusted_timesteps["new_timesteps"] = pd.date_range(
        f"12-01-{mapping.start_year.item()-1} 00:00", periods=len(timesteps), freq="D"
    )

    return adjusted_timesteps


class ClimateData:
    def __init__(
        self,
        path_climate_data: Path,
        path_climate_data_historical: Path = None,
        adjusted_timesteps: Dict[str, pd.DatetimeIndex] = None,
        chunks: Dict[str, int] = None,
        extent: Dict[str, float] = None,
    ):
        self.ds_data = xr.open_mfdataset(path_climate_data.glob("*")).sortby("time")

        if path_climate_data_historical is not None:
            self.ds_data = xr.concat(
                [
                    xr.open_mfdataset(path_climate_data_historical.glob("*")).sortby(
                        "time"
                    ),
                    self.ds_data,
                ],
                dim="time",
            )

        if adjusted_timesteps is not None:
            self.ds_data = self.ds_data.sel(
                time=adjusted_timesteps["original_timesteps"]
            )
            self.ds_data["time"] = adjusted_timesteps["new_timesteps"]

        if extent is not None:
            self.ds_data = self.ds_data.where(
                (self.ds_data["lat"] >= extent["lat_min"])
                & (self.ds_data["lat"] <= extent["lat_max"])
                & (self.ds_data["lon"] >= extent["lon_min"])
                & (self.ds_data["lon"] <= extent["lon_max"]),
                drop=True,
            )

        if chunks is not None:
            self.ds_data = self.ds_data.chunk(chunks)
