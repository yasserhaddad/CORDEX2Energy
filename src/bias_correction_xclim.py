import xarray as xr
from xclim import sdba


def empirical_quantile_mapping(
    da_clim_past: xr.DataArray,
    da_clim_future: xr.DataArray,
    da_ref: xr.DataArray,
    window_length: int = 91,
    nquantiles: int = 99,
    adjustment_kind: str = "+",
    extrapolation: str = "constant",
    interp: str = "nearest",
    adapt_freq_thresh: str = None,
) -> xr.Dataset:
    group = sdba.Grouper("time.dayofyear", window=window_length)
    QM = sdba.EmpiricalQuantileMapping.train(
        da_ref,
        da_clim_past,
        nquantiles=nquantiles,
        group=group,
        adapt_freq_thresh=adapt_freq_thresh,
        kind=adjustment_kind,
    )
    return QM.adjust(da_clim_future, extrapolation=extrapolation, interp=interp)


DICT_DEFAULT_XCLIM_PARAMETERS = {
    "pr": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": "0.1 mm d-1",
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
    "rsds": {
        "nquantiles": 99,
        "adjustment_kind": "*",
        "extrapolation": "constant",
        "interp": "linear",
        "adapt_freq_thresh": None,
    },
}
