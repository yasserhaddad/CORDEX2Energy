import xarray as xr

GRAVITY = 9.81  # m/s^2
WATER_DENSITY = 1000  # kg/m^3

def mean_discharge_weight(
    ds: xr.Dataset,
    reference_period: list,
    exponent: float = 0.5,
    dim: str = "time",
):
    q_mean = ds.sel(time=slice(*reference_period)).mean(dim)
    weight = (q_mean / q_mean.max()) ** exponent
    return weight

def persistence_weight(
    ds: xr.Dataset,
    reference_period: list,
    dim: str = "time",
):
    ref = ds.sel(time=slice(*reference_period))
    return ref.mean(dim) / ref.std(dim)


def get_streamflow_bounds(
    ds: xr.Dataset,
    reference_period: list,
    quantile_min: float = 0.05,
    quantile_max: float = 0.95,
    dim: str = "time",
):
    quant_min = ds.sel(time=slice(*reference_period)).quantile(
        quantile_min, dim=dim, skipna=True
    )
    quant_max = ds.sel(time=slice(*reference_period)).quantile(
        quantile_max, dim=dim, skipna=True
    )
    return quant_min, quant_max


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
    quant_min, quant_max = get_streamflow_bounds(ds, reference_period, quantile_min, quantile_max, dim)
    results = xr.where(ds < quant_min, 0,
                xr.where(ds > quant_max, 1,
                    ds / quant_max))
    return results


def compute_ror_hydropower_potential(
    ds: xr.Dataset,
    reference_period: list,
    quantile_min: float = 0.05,
    quantile_max: float = 0.95,
    dim: str = "time",
    hydraulic_head: float = 20,
    efficiency: float = 0.8
) -> xr.Dataset:
    """
    Computes a hypothetical run-of-river hydropower potential based on streamflow data.
    """
    quant_min, quant_max = get_streamflow_bounds(ds, reference_period, quantile_min, quantile_max, dim)
    discharge_capped = xr.where(ds < quant_min, 0,
                         xr.where(ds > quant_max, quant_max, ds))
    power = (WATER_DENSITY * GRAVITY * hydraulic_head * efficiency * discharge_capped)
    return power

def compute_hydropower_reservoir_inflows(
    ds: xr.Dataset,
    storage_window: int = 90,
    dim: str = "time"
):
    """
    Computes hydropower reservoir inflows as rolling sums over a specified storage window.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with dimensions including `dim` and location dimensions.
    storage_window : int, optional
        Number of days over which to compute the rolling sum (default is 90).
    dim : str, optional
        Dimension over which to compute the rolling sum (default is "time").

    Returns
    -------
    xr.Dataset
        Dataset containing the rolling sum of inflows over the specified storage window.
    """
    inflows = ds.rolling({dim: storage_window}, center=False).sum()
    return inflows