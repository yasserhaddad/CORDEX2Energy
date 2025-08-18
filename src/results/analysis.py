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

def compute_seasonal_averages(ds: xr.Dataset | xr.DataArray, time_dim: str = "time") -> xr.Dataset | xr.DataArray:
    """
    Compute seasonal averages for each year in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable of interest.
    var : str
        Name of the variable to compute seasonal averages for.
    time_dim : str, optional
        Name of the time dimension, default is "time".

    Returns
    -------
    xr.DataArray
        Seasonal averages with dimensions ('year', 'season', ...).
    """
    # Ensure time is a datetime type
    if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
        ds = ds.copy()
        ds[time_dim] = xr.decode_cf(ds[[time_dim]])[time_dim]

    # For each season, select the timesteps corresponding to the season and take the yearly average
    season_names = ["DJF", "MAM", "JJA", "SON"]
    seasonal_yearly_averages = []

    for season in season_names:
        # Select timesteps for the season
        da_season = ds.sel(time=ds["time"].dt.season == season)
        # Group by year and take the mean
        avg = da_season.groupby("time.year").mean(dim="time")
        # Add a season coordinate for concatenation
        avg = avg.expand_dims(season=[season])
        seasonal_yearly_averages.append(avg)

    # Concatenate along the new 'season' dimension
    return xr.concat(seasonal_yearly_averages, dim="season")


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