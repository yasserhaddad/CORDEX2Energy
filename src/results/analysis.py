import numpy as np
import xarray as xr

def compute_gmst_anomalies(ds, var, reference_period, time_dim="time") -> tuple[np.ndarray, xr.DataArray]:
    mean = ds[var].mean(dim=["lat", "lon"])
    years = ds[time_dim].values if time_dim in ds else np.arange(len(mean))
    ref_mean = mean.sel(year=reference_period).mean()
    anomalies = mean - ref_mean
    return years, anomalies


def compute_percentage_change(ref: xr.Dataset, target: xr.Dataset, round: int = 2) -> xr.Dataset:
    """
    Compute the percentage change between a reference dataset and a target dataset.

    Parameters
    ----------
    ref : xr.Dataset
        Reference dataset.
    target : xr.Dataset
        Target dataset to compare against the reference.
    round : int, optional
        Number of decimal places to round the result, default is 2.

    Returns
    -------
    xr.DataArray
        Percentage change between target and reference datasets, calculated as
        100 * (target - ref) / ref.

    """
    return (100 * (target - ref) / ref).round(round)

def compute_seasonal_statistics(ds: xr.Dataset | xr.DataArray, time_dim: str = "time", method: str = "mean") -> xr.Dataset | xr.DataArray:
    """
    Compute seasonal statistics for each year in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable of interest.
    time_dim : str, optional
        Name of the time dimension, default is "time".
    method : str, optional
        Statistical method to use, default is "mean".

    Returns
    -------
    xr.DataArray
        Seasonal averages with dimensions ('year', 'season', ...).
    """
    # # Ensure time is a datetime type
    if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
        ds = ds.copy()
        ds[time_dim] = xr.decode_cf(ds[[time_dim]])[time_dim]
    if method not in ["mean", "median", "sum"]:
        raise ValueError(f"Unknown method: {method}")
    ds = ds.resample({time_dim: 'QS-DEC'})
    if method == "sum":
        ds = ds.sum(dim=time_dim)
    elif method == "median":
        ds = ds.median(dim=time_dim)
    else:
        ds = ds.mean(dim=time_dim)
    ds = ds.assign_coords(
        year=(time_dim, ds[time_dim].dt.year.data),
        season=(time_dim, ds[time_dim].dt.season.data),
    )
    return ds

def compute_regional_values(ds: xr.Dataset | xr.DataArray, regions_mask: xr.DataArray, regions_list: list[str], method: str = "mean") -> xr.Dataset | xr.DataArray:
    """
    Compute regional averages for the given dataset using the provided regions mask.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input dataset containing the variable of interest.
    regions_mask : xr.DataArray
        Boolean mask indicating the regions over which to compute the averages.
    regions_list : list[str]
        List of region names corresponding to the regions mask.
    method : str, optional
        Aggregation method to use, default is "mean".

    Returns
    -------
    xr.Dataset | xr.DataArray
        Regional averages with dimensions ('region', ...).
    """
    if method not in ["mean", "median", "sum"]:
        raise ValueError(f"Unknown method: {method}")
    ds_regional = ds.groupby(regions_mask)
    if method == "median":
        ds_regional = ds_regional.median(dim=("lat", "lon"))
    elif method == "sum":
        ds_regional = ds_regional.sum(dim=("lat", "lon"))
    else:
        ds_regional = ds_regional.mean(dim=("lat", "lon"))
    ds_regional["mask"] = regions_list
    return ds_regional

def compute_sign_agreement(ds: xr.Dataset | xr.DataArray, model_dim: str = "model") -> xr.Dataset | xr.DataArray:
    """
    Compute the number of models that agree in sign with the multi-model mean.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Input data array with a model dimension.
    model_dim : str, optional
        Name of the model dimension, default is "model".

    Returns
    -------
    xr.Dataset | xr.DataArray
        Array with the count of models agreeing in sign with the multi-model mean.
    """
    return xr.apply_ufunc(
        lambda x: np.sum(np.sign(x) == np.sign(np.nanmean(x))),
        ds,
        input_core_dims=[[model_dim]],
        vectorize=True,
    )