import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import sys

from cordex2energy.utils.constants import (
    COMPRESSED_ZARR_ENCODING,
    BC_SUBDAILY_CHUNKS
)

dask.config.set(num_workers=15, threads_per_worker=16)
dask.config.set(
    {
        "distributed.worker.memory.target": 0.7,  # Spill to disk at 70%
        "distributed.worker.memory.spill": 0.8,  # Start spilling at 80%
        "distributed.worker.memory.pause": 0.9,  # Pause worker at 90%
        "array.chunk-size": "512MiB",
        "logging.distributed": "error",  # Less verbose logs
    }
)

variable = "psl"
climate_scenario = "rcp85"
path_zarr = [
    Path(f"/landclim/yhaddad/clim2energy-ch/cordex_processed/CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_{climate_scenario}/bias_corrected/3hr/{variable}.zarr"),
    Path(f"/landclim_nobackup/yhaddad/clim2energy-ch/cordex_processed/CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1_{climate_scenario}/temp/bias_corrected/{variable}/{variable}_hour_0.zarr")
]
with ProgressBar(dt=1):
    datasets = [
        xr.open_zarr(path, chunks="auto")
        for path in path_zarr
    ]
    ds_climate_model_data_future_debiased = xr.concat(datasets, dim="time").sortby("time")
    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
        }
        for var in ds_climate_model_data_future_debiased.data_vars
    }
    ds_climate_model_data_future_debiased.chunk(BC_SUBDAILY_CHUNKS).to_zarr(path_zarr[0].parent / f"{variable}_complete.zarr", mode="w", encoding=zarr_encoding)