import subprocess

REMAP_TYPE_MAP = {
    "conservative": "remapcon",
    "conservative2": "remapcon2",
    "bilinear": "remapbil",
    "bicubic": "remapbic",
    "nearest_neighbor": "remapnn",
    "distance_weighted": "remapdis",
}

def cdo_remap(
    input_file_path: str, output_file_path: str, remap_type: str, target_grid_path: str, input_grid_path: str = None
) -> None:
    """
    Remap the input file to the target grid using CDO.

    Parameters
    ----------
    input_file_path : str
        The path to the input file.
    output_file_path : str
        The path to the output file.
    remap_type : str
        The type of remapping to perform. Supported types are:
        - 'conservative'
        - 'bilinear'
        - 'bicubic'
        - 'nearest_neighbor'
        - 'distance_weighted'
    target_grid_path : str
        The path to the target grid file.
    """
    # Get the CDO remap command
    cdo_remap_command = REMAP_TYPE_MAP.get(remap_type.lower())
    if cdo_remap_command is None:
        raise ValueError(f"Unsupported remap type: {remap_type}")

    # Construct the CDO command
    cdo_command = [
        "cdo",
        f"{cdo_remap_command},{target_grid_path}",
    ]

    if input_grid_path:
        cdo_command.append(f"-setgrid,{input_grid_path}")

    cdo_command += [
        input_file_path,
        output_file_path,
    ]

    # Execute the CDO command
    subprocess.run(cdo_command, check=True)

def cdo_setgrid(input_file_path: str, grid_path: str, output_file_path: str) -> None:
    """
    Set the grid of the input file to the specified grid using CDO.

    Parameters
    ----------
    input_file_path : str
        The path to the input file.
    grid_path : str
        The path to the grid file.
    output_file_path : str
        The path to the output file.
    """
    # Construct the CDO command
    cdo_command = ["cdo", f"setgrid,{grid_path}", input_file_path, output_file_path]

    # Execute the CDO command
    subprocess.run(cdo_command, check=True)


def save_cdo_griddes(input_path: str, output_path: str) -> None:
    """
    Save the grid definition of the input file to the specified output file.

    Parameters
    ----------
    input_path : str
        The path to the input file.
    output_path : str
        The path to the output file.
    """
    try:
        # Run the cdo griddes command
        result = subprocess.run(["cdo", "griddes", input_path], capture_output=True, text=True, check=True)

        # Write the output to the specified file
        with open(output_path, "w") as f:
            f.write(result.stdout)

        print(f"Grid definition saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running CDO: {e}")
    except FileNotFoundError:
        print("Error: CDO is not installed or not found in PATH.")

import numpy as np
import xarray as xr

try:
    import xesmf as xe
except ImportError as e:
    raise ImportError("Install xesmf and esmpy (e.g., conda install -c conda-forge xesmf esmpy)") from e


def _ensure_lat_lon(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    # Normalize coordinate names to lat/lon and sort ascending
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "var")
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if "lat" in ds.coords:
        if np.any(np.diff(ds.lat.values) < 0):
            ds = ds.sortby("lat")
    if "lon" in ds.coords:
        if np.any(np.diff(ds.lon.values) < 0):
            ds = ds.sortby("lon")
    return ds


def regrid_with_xesmf(
    src: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray,
    var_names: list[str] | None = None,
    method: str = "bilinear",          # or "conservative" if both grids have bounds
    weights_path: str | None = None,   # file to cache weights
    reuse_weights: bool = True,
    periodic: bool = False,
    extrap_method: str = "nearest_s2d",
    chunks: dict | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Regrid a source xarray object to a target grid using xESMF, with optional Dask support and weight caching.

    Parameters
    ----------
    src : xr.Dataset or xr.DataArray
        Source data to be regridded. Must contain latitude and longitude coordinates.
    target : xr.Dataset or xr.DataArray
        Target grid definition. Must contain latitude and longitude coordinates.
    var_names : list of str, optional
        List of variable names to regrid if `src` is a Dataset. If None, all data variables are regridded.
    method : str, default "bilinear"
        Regridding method to use. Options include "bilinear", "conservative", etc.
    weights_path : str, optional
        Path to a file for caching and reusing regridding weights. If None, weights are not cached.
    reuse_weights : bool, default True
        Whether to reuse existing weights if `weights_path` is provided and the file exists.
    periodic : bool, default False
        Whether the grid is periodic in longitude (e.g., global grids).
    extrap_method : str, default "nearest_s2d"
        Method for handling points outside the source grid. See xESMF documentation for options.
    chunks : dict, optional
        Dictionary specifying chunk sizes for Dask. If provided, output will be chunked accordingly.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The regridded data, matching the type of the input `src`. If `src` is a Dataset, only the specified variables are regridded.

    Notes
    -----
    - Requires xESMF and xarray.
    - If `src` is a Dataset, only variables in `var_names` are regridded.
    - Weight caching via `weights_path` can significantly speed up repeated regridding operations.
    """
    src_ds = _ensure_lat_lon(src)
    tgt_ds = _ensure_lat_lon(target)

    regridder = xe.Regridder(
        src_ds, tgt_ds, method=method, periodic=periodic,
        extrap_method=extrap_method,
        filename=weights_path if weights_path else None,
        reuse_weights=bool(weights_path and reuse_weights),
    )

    if isinstance(src, xr.DataArray):
        out = regridder(src)
    else:
        if var_names is None:
            var_names = list(src_ds.data_vars)
        data_vars = {v: regridder(src_ds[v]) for v in var_names}
        out = xr.Dataset(data_vars=data_vars, attrs=src_ds.attrs)

    if chunks:
        out = out.chunk(chunks)
    return out