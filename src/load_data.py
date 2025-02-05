from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from ibicus.debias import QuantileMapping, ISIMIP

from .timesteps_adjustments import get_new_timesteps
from .utils_xarray import (
    xr_reorder_coords,
    get_time_resolution,
    get_specific_hourly_data,
    get_valid_hours,
)
import subprocess

ISIMIP_BIAS_CORRECTION_VARS = [
    "hurs",
    "pr",
    "prsnratio",
    "psl",
    "rsds",
    "rlds",
    "sfcwind",
    "tas",
    "tasrange",
    "taskew",
]


REMAP_TYPE_MAP = {
    "conservative": "remapcon",
    "bilinear": "remapbil",
    "bicubic": "remapbic",
    "nearest_neighbor": "remapnn",
    "distance_weighted": "remapdis",
}


def cdo_remap(
    input_file_path: str, output_file_path: str, remap_type: str, target_grid_path: str
) -> None:
    """
    Remap the input file to the target grid using CDO.

    Parameters
    ----------
    input_file_path : str
        The path to the input file.
    output_file_path : str
        The path to the output file.
    remap_type : str
        The type of remapping to perform. Supported types are:
        - 'conservative'
        - 'bilinear'
        - 'bicubic'
        - 'nearest_neighbor'
        - 'distance_weighted'
    target_grid_path : str
        The path to the target grid file.
    """
    # Get the CDO remap command
    cdo_remap_command = REMAP_TYPE_MAP.get(remap_type.lower())
    if cdo_remap_command is None:
        raise ValueError(f"Unsupported remap type: {remap_type}")

    # Construct the CDO command
    cdo_command = [
        "cdo",
        f"{cdo_remap_command},{target_grid_path}",
        input_file_path,
        output_file_path,
    ]

    # Execute the CDO command
    subprocess.run(cdo_command, check=True)


class ClimateModelData:
    def __init__(
        self,
        path_climate_data: Path,
        path_climate_data_historical: Path = None,
        adjusted_timesteps: Dict[str, pd.DatetimeIndex] = None,
        subset_data_vars: List[str] = None,
        chunks: Dict[str, int] = None,
        extent: Dict[str, float] = None,
        reordered_coords: List[str] = None,
        load: bool = True,
    ):
        if path_climate_data.is_dir():
            self.ds_data = xr.open_mfdataset(
                path_climate_data.glob("*"), chunks=chunks
            ).sortby("time")
        else:
            self.ds_data = xr.open_dataset(path_climate_data, chunks=chunks).sortby(
                "time"
            )

        if path_climate_data_historical is not None:
            self.ds_data = xr.concat(
                [
                    xr.open_mfdataset(
                        path_climate_data_historical.glob("*"), chunks=chunks
                    ).sortby("time"),
                    self.ds_data,
                ],
                dim="time",
            )

        if adjusted_timesteps is not None:
            self.ds_data = self.ds_data.sel(
                time=self.ds_data.time.dt.floor("D").isin(
                    adjusted_timesteps["original_timesteps"]
                )
            )
            self.ds_data["time"] = get_new_timesteps(adjusted_timesteps, self.ds_data)

        if subset_data_vars is not None:
            self.ds_data = self.ds_data[subset_data_vars]

        if extent is not None:
            self.ds_data = self.ds_data.where(
                (self.ds_data["lat"] >= extent["lat_min"])
                & (self.ds_data["lat"] <= extent["lat_max"])
                & (self.ds_data["lon"] >= extent["lon_min"])
                & (self.ds_data["lon"] <= extent["lon_max"]),
                drop=True,
            )

        if reordered_coords is not None:
            self.ds_data = xr_reorder_coords(self.ds_data, reordered_coords)

        if load:
            self.ds_data = self.ds_data.compute()

    def plot_annual_mean(
        self,
        data_var: str,
        spatial_mean: bool = False,
        ax: plt.Axes = None,
        title: str = None,
        save: bool = False,
        output_filename: str = None,
        file_format: str = "pdf",
        **kwargs_plot,
    ):
        if ax is None:
            _, ax = plt.subplots(**kwargs_plot)
        ds_mean = self.ds_data[data_var].groupby("time.year").mean(skipna=True)
        if spatial_mean:
            ds_mean = ds_mean.mean(dim=["y", "x"], skipna=True)

        ds_mean.plot(ax=ax)
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()

    def bias_correction(
        self,
        ds_reference: xr.Dataset,
        data_var: str,
        reference_period: Tuple[str, str],
        future_period: Tuple[str, str],
        bias_correction_method: str,
        parametric_existing_settings: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Perform bias correction of the climate model data using the reference data.

        Parameters
        ----------
        reference_data : xr.Dataset
            The reference data used for bias correction. The name of the variable
            should be identical to the one from the climate model data.
        bias_correction_method : str
            The bias correction method to use. Supported methods are:
            - "quantile_mapping"
            - "ISIMIP"
        **kwargs
            Additional keyword arguments to pass to the bias
            correction method function.
        """
        if bias_correction_method == "quantile_mapping":
            bc = QuantileMapping()
        elif bias_correction_method == "ISIMIP":
            bc = ISIMIP()

        if parametric_existing_settings:
            if data_var.startswith("wind"):
                bc = bc.from_variable("sfcWind")
            else:
                bc = bc.from_variable(data_var)

        time_res = get_time_resolution(self.ds_data)
        if time_res < pd.Timedelta(days=1):
            hours = get_valid_hours(self.ds_data, 0)
        else:
            hours = []

        # Select the data for the reference and future periods
        ds_reference_data = ds_reference.sel(time=slice(*reference_period)).rename(
            {"lat": "y", "lon": "x"}
        )[data_var]
        ds_climate_model_data_past = self.ds_data.sel(
            time=slice(*reference_period)
        ).rename({"lat": "y", "lon": "x"})[data_var]
        ds_climate_model_data_future = self.ds_data.sel(
            time=slice(*future_period)
        ).rename({"lat": "y", "lon": "x"})[data_var]

        # Perform bias correction for each hour of the day if sub-daily data, otherwise for the whole day
        if len(hours) < 1:
            ds_climate_model_data_future_debiased = ds_climate_model_data_future.copy(
                data=bc.apply(
                    ds_reference_data.values,
                    ds_climate_model_data_past.values,
                    ds_climate_model_data_future.values,
                    **kwargs
                )
            )
        else:
            list_ds_debiased = []
            for hour in hours:
                list_ds_debiased.append(
                    get_specific_hourly_data(
                        ds_climate_model_data_future, hour
                    ).copy(
                        data=bc.apply(
                            get_specific_hourly_data(ds_reference_data, hour).values,
                            get_specific_hourly_data(
                                ds_climate_model_data_past, hour
                            ).values,
                            get_specific_hourly_data(
                                ds_climate_model_data_future, hour
                            ).values,
                            **kwargs
                        )
                    )
                )

            ds_climate_model_data_future_debiased = xr.concat(
                list_ds_debiased, dim="time"
            ).sortby("time")

        return ds_climate_model_data_future_debiased


class ClimateReanalysisData:
    def __init__(
        self,
        path_reanalysis_data: Path,
        chunks: Dict[str, int] = None,
        subset_data_vars: List[str] = None,
        extent: Dict[str, float] = None,
        reordered_coords: List[str] = None,
        load: bool = True,
    ):
        if path_reanalysis_data.is_dir():
            self.ds_data = xr.open_mfdataset(
                path_reanalysis_data.glob("*"), chunks=chunks
            ).sortby("time")
        else:
            self.ds_data = xr.open_dataset(path_reanalysis_data, chunks=chunks).sortby(
                "time"
            )

        if subset_data_vars is not None:
            self.ds_data = self.ds_data[subset_data_vars]

        if extent is not None:
            self.ds_data = self.ds_data.where(
                (self.ds_data["lat"] >= extent["lat_min"])
                & (self.ds_data["lat"] <= extent["lat_max"])
                & (self.ds_data["lon"] >= extent["lon_min"])
                & (self.ds_data["lon"] <= extent["lon_max"]),
                drop=True,
            )

        if reordered_coords is not None:
            self.ds_data = xr_reorder_coords(self.ds_data, reordered_coords)

        if load:
            self.ds_data = self.ds_data.compute()
