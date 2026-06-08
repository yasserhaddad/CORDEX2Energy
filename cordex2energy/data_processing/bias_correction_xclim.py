import xarray as xr
from xclim import sdba


def empirical_quantile_mapping(
    da_clim_past: xr.DataArray,
    da_clim_future: xr.DataArray,
    da_ref: xr.DataArray,
    adjust_on_past: bool = True,
    window_length: int = 91,
    nquantiles: int = 99,
    adjustment_kind: str = "+",
    extrapolation: str = "constant",
    interp: str = "nearest",
    adapt_freq_thresh: str = None,
    jitter_under_thresh_value: str = None,
) -> xr.Dataset:
    """
    Apply empirical quantile mapping for bias correction.

    Parameters
    ----------
    da_clim_past : xr.DataArray
        Climate data from the past period.
    da_clim_future : xr.DataArray
        Climate data from the future period to be adjusted.
    da_ref : xr.DataArray
        Reference climate data for the past period.
    adjust_on_past : bool, optional
        If True, adjust the past data based trained quantile mapping functions, by default True.
    window_length : int, optional
        Length of the moving window for grouping data by day of the year, by default 91.
    nquantiles : int, optional
        Number of quantiles to use for the quantile mapping, by default 99.
    adjustment_kind : str, optional
        Type of adjustment to apply ('+' for additive, '*' for multiplicative), by default "+".
    extrapolation : str, optional
        Method for extrapolation beyond the range of quantiles ('constant', 'nan', 'nearest'), by default "constant".
    interp : str, optional
        Interpolation method for quantile mapping ('nearest', 'linear'), by default "nearest".
    adapt_freq_thresh : str, optional
        Threshold for frequency adaptation, by default None, meaning that frequency adaptation is not performed.
    jitter_under_thresh_value : str, optional
        Threshold under which to add uniform random noise to values, a quantity with units.
        By default None, meaning that jitter under thresh is not performed.

    Returns
    -------
    xr.Dataset
        Adjusted climate data for the future period.
    """
    group = sdba.Grouper("time.dayofyear", window=window_length)
    QM = sdba.EmpiricalQuantileMapping.train(
        da_ref,
        da_clim_past,
        nquantiles=nquantiles,
        group=group,
        adapt_freq_thresh=adapt_freq_thresh,
        kind=adjustment_kind,
        jitter_under_thresh_value=jitter_under_thresh_value
    )

    future = QM.adjust(da_clim_future, extrapolation=extrapolation, interp=interp)

    if adjust_on_past:
        past = QM.adjust(da_clim_past, extrapolation=extrapolation, interp=interp)
        future = xr.concat([past, future], dim="time")

    return future


DICT_DEFAULT_XCLIM_PARAMETERS = {
    "pr": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": "0.1 mm d-1",
        "jitter_under_thresh_value": "0.001 mm d-1",
    },
    "tas": {
        "nquantiles": 99,
        "adjustment_kind": "+",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
    "hurs": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
    "huss": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
    "sfcWind": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
    "psl": {
        "nquantiles": 99,
        "adjustment_kind": "+",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
    "ps": {
        "nquantiles": 99,
        "adjustment_kind": "+",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
    "rsds": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
        "jitter_under_thresh_value": "0.01 W m**-2",
    },
    "mrro": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
        "jitter_under_thresh_value": "0.001 kg m**-2",
    },
    "clt": {
        "nquantiles": 99,
        "adjustment_kind": "+",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
}
