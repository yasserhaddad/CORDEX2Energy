from pathlib import Path
import time
from typing import Dict, List, Tuple
import dask
from dask.diagnostics import ProgressBar
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
    match_time_steps,
)
from .bias_correction_xclim import (
    empirical_quantile_mapping,
    DICT_DEFAULT_XCLIM_PARAMETERS,
)

import subprocess

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
        drop_leap_days: bool = True,
        load: bool = False,
    ):
        if path_climate_data.name.endswith(".zarr"):
            self.ds_data = xr.open_zarr(path_climate_data, chunks=chunks)
        elif path_climate_data.is_dir():
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
            self.ds_data = self.ds_data.sel(
                lat=slice(extent["lat_min"], extent["lat_max"]),
                lon=slice(extent["lon_min"], extent["lon_max"]),
            )

        if reordered_coords is not None:
            self.ds_data = xr_reorder_coords(self.ds_data, reordered_coords)

        if drop_leap_days:
            self.ds_data = self.ds_data.sel(
                time=~(
                    (self.ds_data.time.dt.month == 2) & (self.ds_data.time.dt.day == 29)
                )
            )
            self.drop_leap_days = True
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
        running_window_mode: bool = True,
        running_window_length: int = 91,
        print_time: bool = False,
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
        if print_time:
            time_start = time.time()

        if bias_correction_method == "quantile_mapping":
            bc = QuantileMapping
        elif bias_correction_method == "ISIMIP":
            bc = ISIMIP

        if parametric_existing_settings:
            if data_var.startswith("wind"):
                bc = bc.from_variable(
                    "sfcWind",
                    running_window_mode=running_window_mode,
                    running_window_length=running_window_length,
                )
            elif data_var == "pr":
                bc = bc.for_precipitation(
                    running_window_mode=running_window_mode,
                    running_window_length=running_window_length,
                )
            else:
                bc = bc.from_variable(
                    data_var,
                    running_window_mode=running_window_mode,
                    running_window_length=running_window_length,
                )
        else:
            bc = bc(
                running_window_mode=running_window_mode,
                running_window_length=running_window_length,
            )

        time_res = get_time_resolution(self.ds_data)
        time_res_ref = get_time_resolution(ds_reference)
        if time_res != time_res_ref:
            raise ValueError(
                "The time resolution of the climate model data and the reference data must be the same."
            )

        # Check if the time steps are the same, if not, adjust the time steps of the reference dataset
        ds_reference = match_time_steps(
            self.ds_data, ds_reference, time_res, skip_leap_days=self.drop_leap_days
        )

        first_time_step = pd.to_datetime(self.ds_data.time.values[0])
        if time_res < pd.Timedelta(days=1):
            hours = get_valid_hours(time_res, first_time_step.hour)
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
                    **kwargs,
                )
            ).rename({"y": "lat", "x": "lon"})
            if print_time:
                print(f"\tTime: {(time.time() - time_start) / 60:.2f} min")
        else:
            list_ds_debiased = []
            for hour in hours:
                if print_time:
                    print(f"\tHour: {hour}")
                list_ds_debiased.append(
                    get_specific_hourly_data(ds_climate_model_data_future, hour).copy(
                        data=bc.apply(
                            get_specific_hourly_data(ds_reference_data, hour).values,
                            get_specific_hourly_data(
                                ds_climate_model_data_past, hour
                            ).values,
                            get_specific_hourly_data(
                                ds_climate_model_data_future, hour
                            ).values,
                            **kwargs,
                        )
                    )
                )
                if print_time:
                    print(
                        f"\tHour: {hour}, Time: {(time.time() - time_start) / 60:.2f} min"
                    )

            ds_climate_model_data_future_debiased = (
                xr.concat(list_ds_debiased, dim="time")
                .sortby("time")
                .to_dataset()
                .rename({"y": "lat", "x": "lon"})
            )

        return ds_climate_model_data_future_debiased

    def bias_correction_xclim(
        self,
        ds_reference: xr.Dataset,
        data_var: str,
        reference_period: Tuple[str, str],
        future_period: Tuple[str, str],
        bias_correction_method: str,
        running_window_length: int = 91,
        default_parameters: bool = True,
        load_data: bool = True,
        nr_processes: bool = 10,
        print_time: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        if nr_processes > 1:
            dask.config.set(num_workers=nr_processes, scheduler="processes")

        if print_time:
            time_start = time.time()

        if bias_correction_method == "quantile_mapping":
            bc = empirical_quantile_mapping
        else:
            raise ValueError(
                f"Unsupported bias correction method: {bias_correction_method}"
            )

        time_res = get_time_resolution(self.ds_data)
        time_res_ref = get_time_resolution(ds_reference)
        if time_res != time_res_ref:
            raise ValueError(
                "The time resolution of the climate model data and the reference data must be the same."
            )

        first_time_step = pd.to_datetime(self.ds_data.time.values[0])
        # Check if the time steps are the same, if not, adjust the time steps of the reference dataset
        ds_reference = match_time_steps(
            self.ds_data, ds_reference, time_res, skip_leap_days=self.drop_leap_days
        )

        if time_res < pd.Timedelta(days=1):
            hours = get_valid_hours(time_res, first_time_step.hour)
        else:
            hours = []

        # Select the data for the reference and future periods
        da_reference_data = ds_reference.sel(time=slice(*reference_period))[data_var]
        da_climate_model_data_past = self.ds_data.sel(time=slice(*reference_period))[
            data_var
        ]
        da_climate_model_data_future = self.ds_data.sel(time=slice(*future_period))[
            data_var
        ]

        if load_data:
            da_reference_data = da_reference_data.load()
            da_climate_model_data_past = da_climate_model_data_past.load()
            da_climate_model_data_future = da_climate_model_data_future.load()

        param_var = "sfcWind" if data_var.startswith("wind") else data_var
        params = (
            DICT_DEFAULT_XCLIM_PARAMETERS[param_var] if default_parameters else kwargs
        )
        # Perform bias correction for each hour of the day if sub-daily data, otherwise for the whole day
        if len(hours) < 1:
            with ProgressBar(dt=10):
                ds_climate_model_data_future_debiased = da_climate_model_data_future.copy(
                    data=bc(
                        da_climate_model_data_past,
                        da_climate_model_data_future,
                        da_reference_data,
                        window_length=running_window_length,
                        **params,
                    ).compute().transpose("time", "lat", "lon")
                ).to_dataset()
            if print_time:
                print(f"\tTime: {(time.time() - time_start) / 60:.2f} min")
        else:
            list_ds_debiased = []
            for hour in hours:
                if print_time:
                    print(f"\tHour: {hour}")
                with ProgressBar(dt=10):
                    list_ds_debiased.append(
                        get_specific_hourly_data(da_climate_model_data_future, hour).copy(
                            data=bc(
                                get_specific_hourly_data(da_reference_data, hour),
                                get_specific_hourly_data(da_climate_model_data_past, hour),
                                get_specific_hourly_data(
                                    da_climate_model_data_future, hour
                                ),
                                window_length=running_window_length,
                                **params,
                            ).compute().transpose("time", "lat", "lon")
                        )
                    )
                if print_time:
                    print(
                        f"\tHour: {hour}, Time: {(time.time() - time_start) / 60:.2f} min"
                    )

            ds_climate_model_data_future_debiased = (
                xr.concat(list_ds_debiased, dim="time")
                .sortby("time")
                .to_dataset()
            )

        return ds_climate_model_data_future_debiased


class ClimateReanalysisData:
    def __init__(
        self,
        path_reanalysis_data: Path,
        chunks: Dict[str, int] = None,
        subset_data_vars: List[str] = None,
        extent: Dict[str, float] = None,
        reordered_coords: List[str] = None,
        drop_leap_days: bool = True,
        load: bool = True,
    ):
        if path_reanalysis_data.name.endswith(".zarr"):
            self.ds_data = xr.open_zarr(path_reanalysis_data, chunks=chunks)
        elif path_reanalysis_data.is_dir():
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
            self.ds_data = self.ds_data.sel(
                lat=slice(extent["lat_min"], extent["lat_max"]),
                lon=slice(extent["lon_min"], extent["lon_max"]),
            )

        if reordered_coords is not None:
            self.ds_data = xr_reorder_coords(self.ds_data, reordered_coords)

        if drop_leap_days:
            self.ds_data = self.ds_data.sel(
                time=~(
                    (self.ds_data.time.dt.month == 2) & (self.ds_data.time.dt.day == 29)
                )
            )

        if load:
            self.ds_data = self.ds_data.compute()
