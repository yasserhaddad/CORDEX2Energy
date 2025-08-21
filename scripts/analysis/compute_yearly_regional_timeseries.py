import json
import sys
from pathlib import Path

import dask
import xarray as xr
import geopandas as gpd
from dask.diagnostics import ProgressBar

sys.path.append("../..")

from src.utils.constants import (
    CH_TURBINES,
    EXTENT_CH,
)
from src.utils.utils_xarray import sel_extent
from src.conversion_energy.aggregate_data import create_regions_mask
from src.results.analysis import compute_seasonal_statistics, compute_regional_values

paths = json.load(open("../../paths_so4.json"))

path_data_processed = Path(paths["path_data_processed"])

dask.config.set(num_workers=12, threads_per_worker=4)
dask.config.set({"array.chunk-size": "500MiB"})

climate_scenario = "rcp85"
selected_turbine_ch = CH_TURBINES[0]
selected_pv_tilt_azimuth_ch = "tilt_35_azimuth_S"

path_energy_europe = path_data_processed / "energy" / "europe"
path_energy_ch = path_data_processed / "energy" / "ch"

# Load data
path_pv_ch = path_energy_ch / "pv"
path_pv_ch_zarr = sorted(path_pv_ch.glob(f"*{climate_scenario}*.zarr"))

ds_pv_ch = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")[[selected_pv_tilt_azimuth_ch]]
        for path in path_pv_ch_zarr
    ],
    dim="model",
).rename({selected_pv_tilt_azimuth_ch: "pv"})
ds_pv_ch["model"] = [path.stem.split("pv_")[1] for path in path_pv_ch_zarr]
ds_pv_ch = ds_pv_ch

path_wind_ch = path_energy_ch / "wind_power"
path_wind_ch_zarr = sorted(path_wind_ch.glob(f"*{climate_scenario}*.zarr"))

ds_wind_ch = xr.concat(
    [
        xr.open_zarr(path, chunks="auto")[[selected_turbine_ch.replace("/", "_")]]
        for path in path_wind_ch_zarr
    ],
    dim="model",
).rename({selected_turbine_ch.replace("/", "_"): "wind"})
ds_wind_ch["model"] = [path.stem.split("wind_power_")[1] for path in path_wind_ch_zarr]

# Load hydropower
path_hydropower = path_energy_europe / "hydropower"
path_hydropower_zarr = sorted(path_hydropower.glob(f"*/*{climate_scenario}*.zarr"))
ds_hydropower_ch = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto")
            .convert_calendar("standard", align_on="year")
            .sortby("lat", ascending=True)
            for path in path_hydropower_zarr
        ],
        dim="model",
    ),
    extent=EXTENT_CH,
)
ds_hydropower_ch["lat"] = ds_pv_ch["lat"]
ds_hydropower_ch["lon"] = ds_pv_ch["lon"]
ds_hydropower_ch["model"] = [
    path.stem.split("hydropower_")[1] for path in path_hydropower_zarr
]

path_demand = path_energy_europe / "demand"
paths_demand_zarr = sorted(path_demand.glob(f"*{climate_scenario}.zarr"))

ds_cdd_hdd_ch = sel_extent(
    xr.concat(
        [
            xr.open_zarr(path, chunks="auto").isel(band=0).drop_duplicates(dim="time")
            for path in paths_demand_zarr
        ],
        dim="model",
    ),
    extent=EXTENT_CH,
)
ds_cdd_hdd_ch["model"] = [path.stem for path in paths_demand_zarr]

gdf_regions_ch = gpd.read_file(
    path_data_processed / "maps" / "regions_ch" / "PRODREG.shp"
).to_crs("EPSG:4326")
regions_ch, regions_mask = create_regions_mask(
    gdf_regions_ch, names="ProdregN_3", da=ds_wind_ch.wind
)

print("Computing yearly timeseries of PV, Wind, Hydropower and Demand")
with ProgressBar(dt=1):
    ds_pv_ch_mean_yearly = compute_regional_values(
        ds_pv_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_wind_ch_mean_yearly = compute_regional_values(
        ds_wind_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_hydropower_ch_mean_yearly = compute_regional_values(
        ds_hydropower_ch.resample(time="YE").mean(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_cdd_hdd_sum_yearly = compute_regional_values(
        ds_cdd_hdd_ch.resample(time="YE").sum(dim="time"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
        method="sum",
    ).compute()

ds_mean_yearly = xr.merge(
    [ds_pv_ch_mean_yearly, ds_wind_ch_mean_yearly, ds_hydropower_ch_mean_yearly, ds_cdd_hdd_sum_yearly]
)
path_mean_yearly = path_energy_ch / "statistics" / "regional_stats_yearly.nc"
path_mean_yearly.parent.mkdir(parents=True, exist_ok=True)
if path_mean_yearly.exists():
    path_mean_yearly.unlink()
ds_mean_yearly.to_netcdf(path_mean_yearly)

print("Computing seasonal timeseries of PV, Wind, Hydropower and Demand")
with ProgressBar(dt=1):
    ds_pv_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_pv_ch),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_wind_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_wind_ch),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_hydropower_ch_mean_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_hydropower_ch),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()
    ds_cdd_hdd_sum_seasonal = compute_regional_values(
        compute_seasonal_statistics(ds_cdd_hdd_ch, method="sum"),
        regions_mask,
        gdf_regions_ch["ProdregN_3"].values,
    ).compute()

ds_mean_seasonal = xr.merge(
    [ds_pv_ch_mean_seasonal, ds_wind_ch_mean_seasonal, ds_hydropower_ch_mean_seasonal, ds_cdd_hdd_sum_seasonal]
)
path_mean_seasonal = path_energy_ch / "statistics" / "regional_stats_seasonal.nc"
path_mean_seasonal.parent.mkdir(parents=True, exist_ok=True)
if path_mean_seasonal.exists():
    path_mean_seasonal.unlink()
ds_mean_seasonal.to_netcdf(path_mean_seasonal)
