import xarray as xr
import zarr
from pathlib import Path
import shutil
import dask
from dask.diagnostics import ProgressBar

import sys

sys.path.append("../..")
from cordex2energy.utils.constants import (
    COMPRESSED_ZARR_ENCODING,
    BC_SUBDAILY_CHUNKS
)

def gradual_concat_and_save_zarr(
    zarr_paths,
    output_path,
    concat_dim="time",
    chunking=None,
    sort_by="time",
    tmp_store_path=None,
):
    """
    Gradually concatenate multiple Zarr files along a dimension, rechunk, sort, and save to a new Zarr store.

    Args:
        zarr_paths (list of str): List of input Zarr file paths.
        output_path (str): Output Zarr file path.
        concat_dim (str): Dimension to concatenate along.
        chunking (dict): Chunking scheme for output.
        sort_by (str): Variable name to sort by before saving.
        tmp_store_path (str): Optional path for a temporary Zarr store.
    """
    # Use a temporary store if provided, else use output_path directly
    store_path = tmp_store_path or output_path

    # Open the first file and initialize the output store
    ds0 = xr.open_zarr(zarr_paths[0], chunks=chunking)
    ds0 = ds0.sortby(sort_by)
    ds0.to_zarr(store_path, mode="w")

    # Process and append each subsequent file
    for zarr_path in zarr_paths[1:]:
        ds = xr.open_zarr(zarr_path)
        ds = ds.sortby(sort_by)
        # Append along the concat_dim
        ds.to_zarr(store_path, mode="a", append_dim=concat_dim)

    # If using a tmp store, rechunk and save to final output
    if tmp_store_path:
        ds_final = xr.open_zarr(store_path, chunks=chunking)
        ds_final = ds_final.sortby(sort_by)
        ds_final = ds_final.chunk(chunking)
        ds_final.to_zarr(output_path, mode="w")
        # Optionally, remove the tmp store
        shutil.rmtree(tmp_store_path)

if __name__ == "__main__":
    dask.config.set(num_workers=15)
    dask.config.set({"array.chunk-size": "500MiB"})
    # Example usage
    # zarr_files = [
    #     "/path/to/file1.zarr",
    #     "/path/to/file2.zarr",
    #     # ...
    # ]
    # output_zarr = "/path/to/output.zarr"
    # chunking = {"time": 100, "lat": 50, "lon": 50}  # Example chunking

    # gradual_concat_and_save_zarr(
    #     zarr_files,
    #     output_zarr,
    #     concat_dim="time",
    #     chunking=chunking,
    #     sort_by="time",
    #     tmp_store_path="/tmp/tmp_zarr_store"
    # )
    data_var = "hurs"
    temp_storage = Path("/net/exo/landclim_nobackup/yhaddad/clim2energy-ch/cordex_processed/CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp45/temp/bias_corrected/hurs")
    print(list(temp_storage.glob("*.zarr")))
    with ProgressBar(dt=1):
        datasets = [
            xr.open_zarr(path).load()
            for path in temp_storage.glob("*.zarr")
        ]
    ds_bias_corrected = xr.concat(datasets, dim="time").sortby("time")
    ds_bias_corrected = ds_bias_corrected.clip(0, 100)
    print("Values clipped to [0, 100].")

    print("Bias correction completed. Saving data...")
    # Save the bias-corrected data

    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
        }
        for var in ds_bias_corrected.data_vars
    }
    output_bias_corrected_cordex_var_path = Path("/net/exo/landclim/yhaddad/clim2energy-ch/cordex_processed/CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_rcp45/bias_corrected/3hr/hurs")
    if (output_bias_corrected_cordex_var_path / f"{data_var}.zarr").exists():
        shutil.rmtree(output_bias_corrected_cordex_var_path / f"{data_var}.zarr")


    ds_bias_corrected.to_zarr(
        output_bias_corrected_cordex_var_path / f"{data_var}.zarr",
        mode="w",
        encoding=zarr_encoding,
    )
    print(
        f"Data saved to {output_bias_corrected_cordex_var_path / f'{data_var}.zarr'}."
    )
