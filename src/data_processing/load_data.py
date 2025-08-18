from pathlib import Path
from typing import Dict, List
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from .timesteps_adjustments import get_new_timesteps
from ..utils.utils_xarray import (
    xr_reorder_coords
)


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
        check_calendar: bool = True,
        drop_leap_days: bool = True,
        load: bool = False,
    ):
        if isinstance(path_climate_data, list):
            engine = "zarr" if path_climate_data[0].name.endswith(".zarr") else "netcdf"
            self.ds_data = xr.open_mfdataset(path_climate_data, chunks=chunks, combine_attrs="override", engine=engine).sortby("time")
        elif path_climate_data.name.endswith(".zarr"):
            self.ds_data = xr.open_zarr(path_climate_data, chunks=chunks)
        elif path_climate_data.is_dir():
            self.ds_data = xr.open_mfdataset(
                sorted(path_climate_data.glob("*")), chunks=chunks, combine_attrs="override"
            ).sortby("time")
        else:
            self.ds_data = xr.open_dataset(path_climate_data, chunks=chunks).sortby(
                "time"
            )

        self.drop_leap_days = drop_leap_days

        if path_climate_data_historical is not None:
            # Load historical data
            ds_hist = xr.open_mfdataset(
                path_climate_data_historical.glob("*"), chunks=chunks
            ).sortby("time")
            # Select only timesteps not present in self.ds_data
            new_times = ds_hist.time.to_index().difference(self.ds_data.time.to_index())
            ds_hist = ds_hist.sel(time=new_times)
            # Concatenate along time
            self.ds_data = xr.concat(
                [ds_hist, self.ds_data],
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

        if check_calendar and hasattr(self.ds_data.time, 'encoding'):
            calendar = self.ds_data.time.encoding.get('calendar', None)
            if calendar not in [None, 'proleptic_gregorian', 'standard', 'gregorian']:
                self.ds_data['time'] = self.ds_data.convert_calendar("gregorian", dim="time", align_on="year" if calendar=="360_day" else None)
        if drop_leap_days:
            self.ds_data = self.ds_data.sel(
                time=~(
                    (self.ds_data.time.dt.month == 2) & (self.ds_data.time.dt.day == 29)
                )
            )

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


class ClimateReanalysisData:
    def __init__(
        self,
        path_reanalysis_data: Path,
        chunks: Dict[str, int] = None,
        subset_data_vars: List[str] = None,
        extent: Dict[str, float] = None,
        reordered_coords: List[str] = None,
        drop_leap_days: bool = True,
        load: bool = False,
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


