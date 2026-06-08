"""PV disaggregation: 3-hourly → hourly.

Three methods are available:

* ``"clearsky"``   — uses pvlib Ineichen clear-sky profiles as weighting
  functions.  Fully Dask-compatible; only requires geometry parameters.

* ``"catmullrom"`` — uses bias-corrected rsds weighted by local Catmull-Rom
  cubic interpolation at t±1 h.  Fully Dask-compatible (only ``.shift()``
  + elementwise ops; no global spline fit; Dask needs just a 2-step time
  overlap).

* ``"scipy"``      — uses bias-corrected rsds weighted by a global
  ``scipy.interpolate.CubicSpline`` via ``xr.interp``.  Requires the
  **entire time axis** to fit in a single chunk; callers *must* rechunk
  input data to ``REGRIDDED_CHUNKS`` (``{"time": -1, "lat": 20, "lon": 20}``)
  beforehand.

Public functions
----------------
disaggregate_pv_clearsky
disaggregate_pv_radiation
disaggregate_dataset_clearsky
disaggregate_dataset_radiation
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from cordex2energy.conversion_energy.pv import (
    disaggregate_pv_to_hourly,
    generate_clearsky_profiles,
)

__all__ = [
    "disaggregate_pv_clearsky",
    "disaggregate_pv_radiation",
    "disaggregate_dataset_clearsky",
    "disaggregate_dataset_radiation",
]

_OFFSETS = np.array([-1, 0, 1], dtype=np.int64)
_ONE_HOUR_NS = np.timedelta64(1, "h").astype("timedelta64[ns]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _catmull_rom_offsets(da: xr.DataArray) -> xr.DataArray:
    """Estimate values at t−1 h, t, t+1 h via local Catmull-Rom interpolation.

    Coefficients come from evaluating the uniform Catmull-Rom formula at the
    sub-interval fractions that correspond to ±1 h offsets within 3-hour
    segments:

    - At u = 2/3 in segment [P₋₁, P₀] → t₀ − 1 h:
        − 1/27·P₋₂ + 1/3·P₋₁ + 7/9·P₀ − 2/27·P₊₁

    - At u = 1/3 in segment [P₀, P₊₁] → t₀ + 1 h:
        − 2/27·P₋₁ + 7/9·P₀  + 1/3·P₊₁ − 1/27·P₊₂

    Only ``.shift()`` + elementwise operations are used, so this is fully
    Dask-compatible with no requirement to load the full time axis.

    Parameters
    ----------
    da : xr.DataArray
        Must have a ``time`` dimension (3-hourly cadence).

    Returns
    -------
    xr.DataArray with dims ``(time, offset, ...)``
        ``offset`` coordinate values are ``[-1, 0, 1]``.
    """
    Pm2 = da.shift(time=2)   # P₋₂ : 2 knots (6 h) earlier
    Pm1 = da.shift(time=1)   # P₋₁ : 1 knot  (3 h) earlier
    P0  = da                  # P₀  : current knot
    Pp1 = da.shift(time=-1)  # P₊₁ : 1 knot  (3 h) ahead
    Pp2 = da.shift(time=-2)  # P₊₂ : 2 knots (6 h) ahead

    val_m1 = -(1 / 27) * Pm2 + (1 / 3) * Pm1 + (7 / 9) * P0 - (2 / 27) * Pp1
    val_p1 = -(2 / 27) * Pm1 + (7 / 9) * P0  + (1 / 3) * Pp1 - (1 / 27) * Pp2

    spatial_dims = [d for d in da.dims if d != "time"]
    return (
        xr.concat(
            [val_m1, P0, val_p1],
            dim=xr.DataArray(_OFFSETS, dims="offset"),
        )
        .clip(min=0.0)
        .transpose("time", "offset", *spatial_dims)
    )


def _rsds_blocks_scipy(rsds_3h: xr.DataArray) -> xr.DataArray:
    """Estimate rsds at t−1 h, t, t+1 h via scipy global cubic spline.

    Internally calls ``xr.interp(method="cubic")`` which fits a
    ``scipy.interpolate.CubicSpline`` over the full time axis.  The entire
    time dimension *must* be in a single Dask chunk; callers should rechunk
    to ``{"time": -1, "lat": 20, "lon": 20}`` (``REGRIDDED_CHUNKS``) before
    calling this function.

    Parameters
    ----------
    rsds_3h : xr.DataArray
        3-hourly rsds with dimension ``time`` and optional spatial dims.

    Returns
    -------
    xr.DataArray with dims ``(time, offset, ...)``
        ``offset`` coordinate values are ``[-1, 0, 1]``.
    """
    times_3h = rsds_3h.time.values
    new_times_2d = times_3h[:, None] + _OFFSETS[None, :] * _ONE_HOUR_NS
    target_times = np.sort(np.unique(new_times_2d.ravel()))

    rsds_1h = rsds_3h.interp(
        time=target_times,
        method="cubic",
        kwargs={"fill_value": 0.0},
    ).clip(min=0.0)

    spatial_dims = [d for d in rsds_3h.dims if d != "time"]
    return (
        xr.concat(
            [
                rsds_1h.sel(time=times_3h + off * _ONE_HOUR_NS)
                       .assign_coords(time=times_3h)
                for off in _OFFSETS
            ],
            dim=xr.DataArray(_OFFSETS, dims="offset"),
        )
        .transpose("time", "offset", *spatial_dims)
    )


def _apply_radiation_weights(
    pv_3h: xr.DataArray,
    rsds_blocks: xr.DataArray,
    threshold: float = 5.0,
) -> xr.DataArray:
    """Distribute *pv_3h* over (t−1 h, t, t+1 h) using *rsds_blocks* weights.

    Energy-conserving: each hourly output = pv_3h_mean × weight × 3.
    Night mask: hours where rsds < *threshold* are set to zero.

    Parameters
    ----------
    pv_3h : xr.DataArray  (time, ...)
    rsds_blocks : xr.DataArray  (time, offset, ...)  — offset ∈ [-1, 0, 1]
    threshold : float  [W/m²]

    Returns
    -------
    xr.DataArray  (time, offset, ...)
    """
    spatial_dims = [d for d in pv_3h.dims if d != "time"]
    block_sum = rsds_blocks.sum("offset").where(lambda x: x > 0, other=1.0)
    weights = rsds_blocks / block_sum
    pv_expanded = pv_3h.expand_dims(offset=_OFFSETS).transpose(
        "time", "offset", *spatial_dims
    )
    return (pv_expanded * weights * 3).where(rsds_blocks >= threshold, 0.0)


def _flatten(pv_hourly_2d: xr.DataArray, times_3h: np.ndarray) -> xr.DataArray:
    """Flatten ``(time, offset)`` → sorted 1-D hourly ``time`` axis."""
    hourly_ts = (times_3h[:, None] + _OFFSETS[None, :] * _ONE_HOUR_NS).ravel()
    return (
        pv_hourly_2d
        .stack(new_time=("time", "offset"))
        .drop_vars(["new_time", "time", "offset"])
        .assign_coords(new_time=hourly_ts)
        .rename(new_time="time")
        .sortby("time")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def disaggregate_pv_clearsky(
    pv_3h: xr.DataArray,
    tilt: int,
    azimuth: int,
    spatial_chunks: dict | None = None,
) -> xr.DataArray:
    """Disaggregate 3-hourly PV to hourly using pvlib clear-sky profiles.

    Calls :func:`~cordex2energy.conversion_energy.pv.generate_clearsky_profiles`
    and :func:`~cordex2energy.conversion_energy.pv.disaggregate_pv_to_hourly`
    from the PV conversion module.  Fully Dask-compatible.

    Parameters
    ----------
    pv_3h : xr.DataArray (time, lat, lon)
        3-hourly PV averages.
    tilt : int
        Panel tilt in degrees.
    azimuth : int
        Panel azimuth in degrees (0 = N, 90 = E, 180 = S, 270 = W).
    spatial_chunks : dict, optional
        Chunks forwarded to ``generate_clearsky_profiles``.
        Defaults to ``{"lat": 20, "lon": 20, "day": 30}``.

    Returns
    -------
    xr.DataArray (time, lat, lon)  — hourly
    """
    if spatial_chunks is None:
        spatial_chunks = {"lat": 20, "lon": 20, "day": 30}

    profiles = generate_clearsky_profiles(
        pv_3h.lat.values,
        pv_3h.lon.values,
        tilt,
        azimuth,
        chunks=spatial_chunks,
    )
    result = disaggregate_pv_to_hourly(pv_3h, profiles)
    result.attrs["method"] = "clear-sky profile weighting"
    return result


def disaggregate_pv_radiation(
    pv_3h: xr.DataArray,
    rsds_3h: xr.DataArray,
    method: str = "catmullrom",
    threshold: float = 5.0,
) -> xr.DataArray:
    """Disaggregate 3-hourly PV to hourly using radiation-weighted interpolation.

    Parameters
    ----------
    pv_3h : xr.DataArray (time, lat, lon)
        3-hourly PV averages.
    rsds_3h : xr.DataArray (time, lat, lon)
        Bias-corrected 3-hourly surface downwelling shortwave radiation [W/m²].
        Must share the same ``time`` coordinate as *pv_3h*.
    method : {"catmullrom", "scipy"}
        ``"catmullrom"`` — local 4-point Catmull-Rom coefficients; fully
        Dask-compatible, no rechunking required.

        ``"scipy"``      — global ``scipy.interpolate.CubicSpline`` via
        ``xr.interp``; requires ``time=-1`` chunks (full time axis in one
        Dask chunk).  Pass data rechunked to ``REGRIDDED_CHUNKS`` before
        calling this function.
    threshold : float
        rsds values below this [W/m²] are treated as night → output = 0.

    Returns
    -------
    xr.DataArray (time, lat, lon)  — hourly
    """
    if method == "catmullrom":
        rsds_blocks = _catmull_rom_offsets(rsds_3h)
    elif method == "scipy":
        rsds_blocks = _rsds_blocks_scipy(rsds_3h)
    else:
        raise ValueError(f"method must be 'catmullrom' or 'scipy', got {method!r}")

    pv_hourly_2d = _apply_radiation_weights(pv_3h, rsds_blocks, threshold)
    result = _flatten(pv_hourly_2d, pv_3h.time.values)
    result.attrs.update({
        "description": "Hourly PV disaggregated from 3-hourly averages",
        "units": pv_3h.attrs.get("units", ""),
        "method": f"radiation interpolation ({method})",
    })
    return result


def disaggregate_dataset_clearsky(
    ds_pv: xr.Dataset,
    azimuths_map: dict[str, int],
    spatial_chunks: dict | None = None,
) -> xr.Dataset:
    """Apply :func:`disaggregate_pv_clearsky` to every variable in *ds_pv*.

    Variable names must follow the pattern ``tilt_{tilt}_azimuth_{dir}``
    where ``{dir}`` is a key in *azimuths_map* (e.g. ``"S"``, ``"SW"``).

    Parameters
    ----------
    ds_pv : xr.Dataset
        3-hourly PV dataset.
    azimuths_map : dict[str, int]
        Mapping from direction string to azimuth degrees.
        Example: ``{"N": 0, "E": 90, "S": 180, "W": 270,
        "NE": 45, "SE": 135, "SW": 225, "NW": 315}``.
    spatial_chunks : dict, optional
        Forwarded to ``generate_clearsky_profiles``.

    Returns
    -------
    xr.Dataset  — hourly, same variable names as *ds_pv*
    """
    hourly_vars: dict[str, xr.DataArray] = {}
    for vname in ds_pv:
        parts   = vname.split("_")
        tilt    = int(parts[1])
        azimuth = azimuths_map[parts[-1]]
        hourly_vars[vname] = disaggregate_pv_clearsky(
            ds_pv[vname], tilt, azimuth, spatial_chunks=spatial_chunks
        )
    return xr.Dataset(hourly_vars)


def disaggregate_dataset_radiation(
    ds_pv: xr.Dataset,
    rsds_3h: xr.DataArray,
    method: str = "catmullrom",
    threshold: float = 5.0,
) -> xr.Dataset:
    """Apply :func:`disaggregate_pv_radiation` to every variable in *ds_pv*.

    The radiation weight blocks (rsds at t±1 h) are computed **once** from
    *rsds_3h* and reused for all PV variables, avoiding redundant computation.

    Parameters
    ----------
    ds_pv : xr.Dataset
        3-hourly PV dataset.
    rsds_3h : xr.DataArray (time, lat, lon)
        Bias-corrected rsds.  When ``method="scipy"``, must be rechunked to
        ``{"time": -1, "lat": 20, "lon": 20}`` (``REGRIDDED_CHUNKS``) before
        calling this function.
    method : {"catmullrom", "scipy"}
    threshold : float  [W/m²]

    Returns
    -------
    xr.Dataset  — hourly, same variable names as *ds_pv*
    """
    if method == "catmullrom":
        rsds_blocks = _catmull_rom_offsets(rsds_3h)
    elif method == "scipy":
        rsds_blocks = _rsds_blocks_scipy(rsds_3h)
    else:
        raise ValueError(f"method must be 'catmullrom' or 'scipy', got {method!r}")

    times_3h = rsds_3h.time.values
    hourly_vars: dict[str, xr.DataArray] = {}
    for vname in ds_pv:
        pv_hourly_2d = _apply_radiation_weights(ds_pv[vname], rsds_blocks, threshold)
        result = _flatten(pv_hourly_2d, times_3h)
        result.attrs.update({
            "description": "Hourly PV disaggregated from 3-hourly averages",
            "units": ds_pv[vname].attrs.get("units", ""),
            "method": f"radiation interpolation ({method})",
        })
        hourly_vars[vname] = result
    return xr.Dataset(hourly_vars)
