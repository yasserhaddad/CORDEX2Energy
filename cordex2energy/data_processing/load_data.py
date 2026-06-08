from pathlib import Path
from typing import Dict, List
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from .timesteps_adjustments import get_new_timesteps
from ..utils.utils_xarray import xr_reorder_coords, get_time_resolution


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
            self.ds_data = xr.open_mfdataset(
                path_climate_data,
                chunks=chunks,
                parallel=True,
                combine_attrs="override",
                engine=engine,
            ).sortby("time")
        elif path_climate_data.name.endswith(".zarr"):
            self.ds_data = xr.open_zarr(path_climate_data, chunks=chunks)
        elif path_climate_data.is_dir():
            self.ds_data = xr.open_mfdataset(
                sorted(path_climate_data.glob("*")),
                parallel=True,
                chunks=chunks,
                combine_attrs="override",
            ).sortby("time")
        else:
            self.ds_data = xr.open_dataset(path_climate_data, chunks=chunks).sortby(
                "time"
            )

        self.drop_leap_days = drop_leap_days

        if path_climate_data_historical is not None:
            # Load historical data
            ds_hist = xr.open_mfdataset(
                path_climate_data_historical.glob("*"), chunks=chunks, parallel=True
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

        if check_calendar and hasattr(self.ds_data.time, "encoding"):
            calendar = self.ds_data.time.encoding.get("calendar", None)
            if calendar not in [None, "proleptic_gregorian", "standard", "gregorian"]:
                self.ds_data = self.ds_data.convert_calendar(
                    "gregorian",
                    dim="time",
                    missing=np.nan,
                    align_on="year" if calendar == "360_day" else None,
                )
                if calendar == "360_day":
                    time_res = get_time_resolution(self.ds_data)
                    if time_res < pd.Timedelta(days=1):
                        # # how many periods make up one day
                        periods_per_day = int(round(pd.Timedelta(days=1) / time_res))
                        if periods_per_day >= 1:
                            # shift data by one day (in periods) so value at t comes from t - 1 day
                            ds_prev_day = self.ds_data.shift(time=periods_per_day)
                            # fill NaNs with previous-day same time-of-day values
                            self.ds_data = self.ds_data.where(
                                ~self.ds_data.isnull(), ds_prev_day
                            )
                            del ds_prev_day

                        # fallback: forward-fill any remaining NaNs along time
                        self.ds_data = self.ds_data.ffill(dim="time")
                    else:
                        # not sub-daily: simple forward fill
                        self.ds_data = self.ds_data.ffill(dim="time")

        if drop_leap_days:
            self.ds_data = self.ds_data.sel(
                time=~(
                    (self.ds_data.time.dt.month == 2) & (self.ds_data.time.dt.day == 29)
                )
            )

        if load:
            self.ds_data = self.ds_data.compute()


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
