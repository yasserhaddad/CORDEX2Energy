from pathlib import Path
import time
from typing import List, Tuple
import dask
from dask.diagnostics import ProgressBar
import xarray as xr
import pandas as pd

from .load_data import ClimateModelData, ClimateReanalysisData
from .utils_xarray import (
    get_time_resolution,
    get_specific_hourly_data,
    get_valid_hours,
    match_time_steps,
)
from .bias_correction_xclim import (
    empirical_quantile_mapping,
    DICT_DEFAULT_XCLIM_PARAMETERS,
)


class BiasCorrection:
    def __init__(
        self,
        climate_data: ClimateModelData,
        reanalysis_data: ClimateReanalysisData,
        reference_period: Tuple[str, str],
        future_period: Tuple[str, str],
        bias_correction_method: str,
    ):
        self.climate_data = climate_data
        self.reanalysis_data = reanalysis_data
        self.reference_period = reference_period
        self.future_period = future_period
        self.bias_correction_method = bias_correction_method

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
        self.reanalysis_data.ds_data = match_time_steps(
            self.climate_data.ds_data,
            self.reanalysis_data.ds_data,
            time_res,
            skip_leap_days=self.climate_data.drop_leap_days,
        )

        if time_res < pd.Timedelta(days=1):
            hours = get_valid_hours(time_res, first_time_step.hour)
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

        debiased_data = da_climate_model_data_future.copy(
            data=bc(
                da_climate_model_data_past,
                da_climate_model_data_future,
                da_reference_data,
                adjust_on_past=True,
                window_length=running_window_length,
                **params,
            )
            .compute()
            .transpose("time", "lat", "lon")
        ).to_dataset()

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
        ds_climate_model_data_future_debiased = xr.open_mfdataset(
            [output_path / f"{data_var}_hour_{hour}.zarr" for hour in hours],
            concat_dim="time",
            combine="nested",
        ).sortby("time")
        return ds_climate_model_data_future_debiased

    def bias_correction_xclim(
        self,
        data_var: str,
        reference_data_var: str,
        running_window_length: int = 91,
        default_parameters: bool = True,
        load_data: bool = False,
        nr_processes: int = 10,
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
        if nr_processes > 1:
            dask.config.set(num_workers=nr_processes, scheduler="processes")

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
            if print_time:
                print(f"\tHour: {hour}")

            da_reference_data_hourly = get_specific_hourly_data(da_reference_data, hour)
            da_climate_model_data_past_hourly = get_specific_hourly_data(
                da_climate_model_data_past, hour
            )
            da_climate_model_data_future_hourly = get_specific_hourly_data(
                da_climate_model_data_future, hour
            )

            mask_constant_future = (
                da_climate_model_data_future_hourly.std(dim="time") == 0
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

            if "jitter_under_thresh_value" in params and mask_constant_future.any():
                debiased_data = debiased_data.where(
                    ~mask_constant_future,
                    da_climate_model_data_future_hourly.mean("time"),
                )

            self.save_debiased_data(debiased_data, output_path, data_var, hour)

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
