import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from .bias_correction_xclim import (
    DICT_DEFAULT_XCLIM_PARAMETERS,
    empirical_quantile_mapping,
)
from .load_data import ClimateModelData, ClimateReanalysisData
from ..utils.utils_xarray import (
    get_specific_hourly_data,
    get_time_resolution,
    get_valid_hours,
    match_time_steps
)
from ..utils.metrics import compute_trend_statsmodel
from ..utils.constants import BC_DAILY_CHUNKS, BC_SUBDAILY_CHUNKS

sns.set_style("whitegrid", {"grid.color": ".93"})

# Define a color-blind friendly palette
COLORBLIND_PALETTE = sns.color_palette("colorblind")
sns.set_palette(COLORBLIND_PALETTE)


class BiasCorrection:
    def __init__(
        self,
        climate_data: ClimateModelData,
        reanalysis_data: ClimateReanalysisData,
        reference_period: Tuple[str, str],
        future_period: Tuple[str, str],
        bias_correction_method: str,
        client: Client = None,
    ):
        self.climate_data = climate_data
        self.reanalysis_data = reanalysis_data
        self.reference_period = reference_period
        self.future_period = future_period
        self.bias_correction_method = bias_correction_method
        self.client = client

    def load_and_preprocess_data(
        self, data_var: str, reference_data_var: str
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, List[int]]:
        """
        Load and preprocess climate model and reference data.

        This function loads climate model data and reference reanalysis data, ensures they have the same time resolution,
        matches the time steps, and extracts the specified data variable for the reference period and future period.

        Parameters
        ----------
        data_var : str
            The name of the data variable to be extracted from the climate model dataset.
        reference_data_var : str
            The name of the data variable to be extracted from the reanalysis dataset.

        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray, xr.DataArray, List[int]]
            A tuple containing:
            - da_reference_data: xr.DataArray
                The reference reanalysis data for the specified variable and reference period.
            - da_climate_model_data_past: xr.DataArray
                The climate model data for the specified variable and reference period.
            - da_climate_model_data_future: xr.DataArray
                The climate model data for the specified variable and future period.
            - hours: List[int]
                A list of valid hours if the time resolution is less than a day, otherwise an empty list.

        Raises
        ------
        ValueError
            If the time resolution of the climate model data and the reference data are not the same.
        """
        time_res = get_time_resolution(self.climate_data.ds_data)
        time_res_ref = get_time_resolution(self.reanalysis_data.ds_data)
        if time_res != time_res_ref:
            raise ValueError(
                "The time resolution of the climate model data and the reference data must be the same."
            )

        first_time_step = pd.to_datetime(self.climate_data.ds_data.time.values[0])
        if not np.array_equal(
            np.unique(self.climate_data.ds_data.time.dt.hour), np.unique(self.reanalysis_data.ds_data.time.dt.hour)
        ):
            self.reanalysis_data.ds_data = match_time_steps(
                self.climate_data.ds_data,
                self.reanalysis_data.ds_data,
                time_res,
                skip_leap_days=self.climate_data.drop_leap_days,
            )

        if time_res < pd.Timedelta(days=1):
            hours = get_valid_hours(time_res, first_time_step.hour)
            print(f"Valid hours: {hours}")
        else:
            hours = []

        da_reference_data = self.reanalysis_data.ds_data.sel(
            time=slice(*self.reference_period)
        ).rename({reference_data_var: data_var})[data_var]
        da_climate_model_data_past = self.climate_data.ds_data.sel(
            time=slice(*self.reference_period)
        )[data_var]
        da_climate_model_data_future = self.climate_data.ds_data.sel(
            time=slice(*self.future_period)
        )[data_var]

        return (
            da_reference_data,
            da_climate_model_data_past,
            da_climate_model_data_future,
            hours,
        )

    def update_attributes(
        self, ds_bias_corrected: xr.Dataset, data_var: str
    ) -> xr.Dataset:
        """
        Update the attributes of the bias-corrected dataset with additional metadata.

        Parameters
        ----------
        ds_bias_corrected : xr.Dataset
            The bias-corrected xarray Dataset whose attributes need to be updated.
        data_var : str
            The name of the data variable within the dataset whose attributes are used for updating.

        Returns
        -------
        xr.Dataset
            The bias-corrected xarray Dataset with updated attributes.

        Notes
        -----
        The method updates the global attributes, based on the original climata data and
        adds the following ones:
        - `history`: Appends the history attribute from the data variable.
        - `bias_adjustment`: Copies the bias adjustment attribute from the data variable.
        - `reference_period`: Sets the reference period based on the instance's reference period.
        - `future_period`: Sets the future period based on the instance's future period.
        """

        new_attrs = self.climate_data.ds_data.attrs
        if "history" in new_attrs:
            new_attrs["history"] += f" | {ds_bias_corrected[data_var].attrs['history']}"
        else:
            new_attrs["history"] = ds_bias_corrected[data_var].attrs["history"]

        new_attrs["bias_adjustment"] = ds_bias_corrected[data_var].attrs[
            "bias_adjustment"
        ]
        new_attrs["reference_period"] = (
            f"{self.reference_period[0]} - {self.reference_period[1]}"
        )
        new_attrs["future_period"] = (
            f"{self.future_period[0]} - {self.future_period[1]}"
        )
        ds_bias_corrected = ds_bias_corrected.assign_attrs(new_attrs)

        return ds_bias_corrected

    def perform_bias_correction(
        self,
        bc: callable,
        da_reference_data: xr.DataArray,
        da_climate_model_data_past: xr.DataArray,
        da_climate_model_data_future: xr.DataArray,
        load_data: bool,
        running_window_length: int,
        **params,
    ) -> xr.Dataset:
        """
        Perform bias correction on the climate model data.

        Parameters
        ----------
        bc : callable
            The bias correction function to be applied.
        da_reference_data_hourly : xr.DataArray
            The reference reanalysis data for the specified variable (and optionally hour).
        da_climate_model_data_past_hourly : xr.DataArray
            The climate model data for the specified variable (and optionally hour) in the reference period.
        da_climate_model_data_future_hourly : xr.DataArray
            The climate model data for the specified variable (and optionally hour) in the future period.
        load_data : bool
            Whether to load the data into memory.
        running_window_length : int
            The length of the running window for the bias correction.
        **params : dict
            Additional parameters for the bias correction function.

        Returns
        -------
        xr.Dataset
            The debiased climate model data for the specified variable (and optionally hour).
        """
        if load_data:
            da_reference_data = da_reference_data.load()
            da_climate_model_data_past = da_climate_model_data_past.load()
            da_climate_model_data_future = da_climate_model_data_future.load()

        debiased_data = self.update_attributes(
            bc(
                da_climate_model_data_past,
                da_climate_model_data_future,
                da_reference_data,
                adjust_on_past=True,
                window_length=running_window_length,
                **params,
            )
            .compute()
            .transpose("time", "lat", "lon")
            .rename(da_climate_model_data_future.name)
            .to_dataset(),
            da_climate_model_data_future.name,
        )

        return debiased_data

    def save_debiased_data(self, debiased_data, output_path, data_var, hour) -> None:
        """
        Save debiased data to a Zarr file.

        Parameters
        ----------
        debiased_data : xarray.Dataset
            The debiased data to be saved.
        output_path : pathlib.Path
            The directory path where the Zarr file will be saved.
        data_var : str
            The name of the data variable.
        hour : int
            The hour associated with the data.

        Returns
        -------
        None
        """
        if self.client is not None:
            debiased_data = debiased_data.persist()
        debiased_data.to_zarr(output_path / f"{data_var}_hour_{hour}.zarr", mode="w")

    def concatenate_saved_data(self, output_path, data_var, hours):
        """
        Concatenate saved climate model data files into a single dataset.

        Parameters
        ----------
        output_path : pathlib.Path
            The directory path where the data files are stored.
        data_var : str
            The variable name of the climate data.
        hours : list of int
            List of hours corresponding to the data files to be concatenated.

        Returns
        -------
        xarray.Dataset
            The concatenated dataset sorted by time.
        """
        print("Contactenating saved data...")

        dask.config.set(num_workers=16, threads_per_worker=4)
        with ProgressBar(dt=10):
            datasets = [
                xr.open_zarr(output_path / f"{data_var}_hour_{hour}.zarr", chunks="auto").load()
                for hour in hours
            ]
        ds_climate_model_data_future_debiased = xr.concat(datasets, dim="time").sortby("time")

        return ds_climate_model_data_future_debiased

    def bias_correction_xclim(
        self,
        data_var: str,
        reference_data_var: str,
        running_window_length: int = 91,
        default_parameters: bool = True,
        load_data: bool = False,
        print_time: bool = False,
        output_path: Path = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Perform bias correction using xclim.

        Parameters
        ----------
        data_var : str
            The name of the data variable to be bias corrected.
        reference_data_var : str
            The name of the data variable in the reference dataset.
        running_window_length : int, optional
            The length of the running window for the bias correction, by default 91.
        default_parameters : bool, optional
            Whether to use default parameters for the bias correction function, by default True.
        load_data : bool, optional
            Whether to load the data into memory, by default False.
        nr_processes : int, optional
            The number of processes to use for parallel processing, by default 10.
        print_time : bool, optional
            Whether to print the time taken for each step, by default False.
        output_path : Path, optional
            The directory path where the output files will be saved, by default None.
        **kwargs
            Additional parameters for the bias correction function.

        Returns
        -------
        xr.Dataset
            The debiased climate model data for the specified variable.

        Raises
        ------
        ValueError
            If the specified bias correction method is not supported.
        """
        if print_time:
            time_start = time.time()

        if self.bias_correction_method == "quantile_mapping":
            bc = empirical_quantile_mapping
        else:
            raise ValueError(
                f"Unsupported bias correction method: {self.bias_correction_method}"
            )

        (
            da_reference_data,
            da_climate_model_data_past,
            da_climate_model_data_future,
            hours,
        ) = self.load_and_preprocess_data(data_var, reference_data_var)

        param_var = "sfcWind" if data_var.startswith("wind") else data_var
        params = (
            DICT_DEFAULT_XCLIM_PARAMETERS[param_var] if default_parameters else kwargs
        )

        def process_and_save_hour(hour):
            if (output_path / f"{data_var}_hour_{hour}.zarr").exists():
                return

            da_reference_data_hourly = get_specific_hourly_data(da_reference_data, hour)
            da_climate_model_data_past_hourly = get_specific_hourly_data(
                da_climate_model_data_past, hour
            )
            da_climate_model_data_future_hourly = get_specific_hourly_data(
                da_climate_model_data_future, hour
            )

            debiased_data = self.perform_bias_correction(
                bc,
                da_reference_data_hourly,
                da_climate_model_data_past_hourly,
                da_climate_model_data_future_hourly,
                load_data,
                running_window_length,
                **params,
            )

            if print_time:
                print(f"\tTime to perform bias correction: {(time.time() - time_start) / 60:.2f} min")

            self.save_debiased_data(debiased_data, output_path, data_var, hour)
            del debiased_data

        if len(hours) < 1:
            with ProgressBar(dt=60):
                ds_climate_model_data_future_debiased = self.perform_bias_correction(
                    bc,
                    da_reference_data,
                    da_climate_model_data_past,
                    da_climate_model_data_future,
                    load_data,
                    running_window_length,
                    **params,
                )
                ds_climate_model_data_future_debiased = ds_climate_model_data_future_debiased
            if print_time:
                print(f"\tTime: {(time.time() - time_start) / 60:.2f} min")
        else:
            for hour in hours:
                print(f"\tHour: {hour}")
                with ProgressBar(dt=60):
                    process_and_save_hour(hour)
                if print_time:
                    print(
                        f"\tHour: {hour} Time: {(time.time() - time_start) / 60:.2f} min"
                    )

            ds_climate_model_data_future_debiased = self.concatenate_saved_data(
                output_path, data_var, hours
            )

        return ds_climate_model_data_future_debiased