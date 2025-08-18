import json
import sys
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.utils.constants import (
    EUROPE_TURBINES,
    CH_TURBINES,
    EXTENT_CH,
    REFERENCE_PERIOD,
    MID_CENTURY_PERIOD,
)
from src.utils.utils_xarray import sel_extent

paths = json.load(open("../../paths_so4.json"))

path_data_processed = Path(paths["path_data_processed"])

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

climate_scenario = "rcp85"
selected_turbine_europe = EUROPE_TURBINES[0]
selected_turbine_ch = CH_TURBINES[0]

path_energy_europe = path_data_processed / "energy" / "europe"
path_energy_ch = path_data_processed / "energy" / "ch"

# Load PV
path_pv = path_energy_europe / "pv"
path_pv_zarr = sorted(path_pv.glob(f"*/*{climate_scenario}*.zarr"))
data_var_pv = list(xr.open_zarr(path_pv_zarr[0], chunks="auto").data_vars.keys())[0]
ds_pv = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")
        for path in path_pv_zarr
    ],
    dim="model"
).rename({data_var_pv: "pv"})
ds_pv["model"] = [path.stem.split("pv_")[1] for path in path_pv_zarr]

# Load Wind
path_wind = path_energy_europe / "wind_power"
path_wind_zarr = sorted(
    path_wind.glob(f"*/*{climate_scenario}*{selected_turbine_europe.replace('/', '_')}*.zarr")
)
data_var_wind = list(xr.open_zarr(path_wind_zarr[0], chunks="auto").data_vars.keys())[0]
ds_wind = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")
        for path in path_wind_zarr
    ], dim="model"
).rename({data_var_wind: "wind"})
ds_wind["model"] = [path.stem.split("wind_power_")[1].split(f"_{selected_turbine_europe.replace('/', '_')}")[0] for path in path_wind_zarr]


path_wind_ch = path_energy_ch / "wind_power"
path_wind_ch_zarr = sorted(
    path_wind_ch.glob(f"*{climate_scenario}*.zarr")
)

ds_wind_ch = xr.concat(
    [xr.open_zarr(path, chunks="auto")[[selected_turbine_ch.replace('/', '_')]] for path in path_wind_ch_zarr], dim="model"
)
ds_wind_ch["model"] = [
    path.stem.split("wind_power_")[1]
    for path in path_wind_ch_zarr
]
data_var_wind_ch = list(ds_wind_ch.data_vars.keys())[0]
ds_wind_ch = ds_wind_ch.rename({data_var_wind_ch: "wind"})

# Load demand
path_demand = path_energy_europe / "demand"
paths_demand_zarr = sorted(path_demand.glob(f"*{climate_scenario}.zarr"))
ds_cdd_hdd = xr.concat(
    [
        xr.open_zarr(path, chunks="auto").isel(band=0).drop_duplicates(dim="time")
        for path in paths_demand_zarr
    ], dim="model"
)
# Compute yearly sums for demand (CDD/HDD)
ds_cdd_hdd["model"] = [path.stem for path in paths_demand_zarr]
ds_cdd_hdd = ds_cdd_hdd.resample(time="YE").sum(dim="time")

# Analysis
print("Computing reference period means")
with ProgressBar(dt=1):
    mean_ref_pv = ds_pv.sel(time=slice(*REFERENCE_PERIOD)).mean(dim="time").compute()
    mean_ref_wind = (
        ds_wind.sel(time=slice(*REFERENCE_PERIOD)).mean(dim="time").compute()
    )
    mean_ref_cdd_hdd = ds_cdd_hdd.sel(time=slice(*REFERENCE_PERIOD)).mean(dim="time").compute()
    mean_ref_wind_ch = ds_wind_ch.sel(time=slice(*REFERENCE_PERIOD)).mean(dim="time").compute()

mean_ref = xr.merge(
    [mean_ref_pv, mean_ref_wind, mean_ref_cdd_hdd]
)
path_mean_ref = path_data_processed / "energy" / "mean_reference_period.nc"
if path_mean_ref.exists():
    path_mean_ref.unlink()
mean_ref.to_netcdf(path_mean_ref)

mean_ref_ch = xr.merge(
    [sel_extent(mean_ref[["pv", "HDD", "CDD"]], extent=EXTENT_CH), mean_ref_wind_ch]
)[["pv", "wind", "HDD", "CDD"]]
path_mean_ref_ch = path_data_processed / "energy" / "mean_reference_period_ch.nc"
if path_mean_ref_ch.exists():
    path_mean_ref_ch.unlink()
mean_ref_ch.to_netcdf(path_mean_ref_ch)


print("Computing mid-century means")
with ProgressBar(dt=1):
    mean_mid_century_pv = (
        ds_pv.sel(time=slice(*MID_CENTURY_PERIOD)).mean(dim="time").compute()
    )
    mean_mid_century_wind = (
        ds_wind.sel(time=slice(*MID_CENTURY_PERIOD)).mean(dim="time").compute()
    )
    mean_mid_century_cdd_hdd = (
        ds_cdd_hdd.sel(time=slice(*MID_CENTURY_PERIOD)).mean(dim="time").compute()
    )
    mean_mid_century_wind_ch = (
        ds_wind_ch.sel(time=slice(*MID_CENTURY_PERIOD)).mean(dim="time").compute()
    )

mean_mid_century = xr.merge(
    [
        mean_mid_century_pv,
        mean_mid_century_wind,
        mean_mid_century_cdd_hdd,
    ]
)
path_mean_mid_century = path_data_processed / "energy" / "mean_mid_century.nc"
if path_mean_mid_century.exists():
    path_mean_mid_century.unlink()
mean_mid_century.to_netcdf(path_mean_mid_century)

mean_mid_century_ch = xr.merge(
    [sel_extent(mean_mid_century[["pv", "HDD", "CDD"]], extent=EXTENT_CH), mean_mid_century_wind_ch]
)[["pv", "wind", "HDD", "CDD"]]
path_mean_mid_century_ch = path_data_processed / "energy" / "mean_mid_century_ch.nc"
if path_mean_mid_century_ch.exists():
    path_mean_mid_century_ch.unlink()
mean_mid_century_ch.to_netcdf(path_mean_mid_century_ch)