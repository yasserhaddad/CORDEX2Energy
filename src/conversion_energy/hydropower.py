import xarray as xr

def simple_hydropower_potential(ds: xr.Dataset, reference_period: list, quantile_max: float, quantile_min: float = 0.05, dim: str = "time") -> xr.Dataset:
    """
    Computes a simple hydropower potential ratio based on a specified quantile computed over a reference period.

    Assumes constant efficiency and head, and calculates the ratio of all values to a specified quantile per location,
    representing the design discharge.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with dimensions including `dim` and location dimensions.
    quantile : float
        Quantile to compute (between 0 and 1) for the design discharge.
    dim : str, optional
        Dimension over which to compute the quantile (default is "time").

    Returns
    -------
    xr.Dataset
        Dataset containing the ratio of original values to the computed quantile per location.
    """
    quant_max = ds.sel(time=slice(*reference_period)).quantile(quantile_max, dim=dim, skipna=True)
    quant_min = ds.sel(time=slice(*reference_period)).quantile(quantile_min, dim=dim, skipna=True)
    results = xr.where(ds < quant_min, 0,
                xr.where(ds > quant_max, 1,
                    ds / quant_max))
    return results