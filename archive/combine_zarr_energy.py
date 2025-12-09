import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from pathlib import Path

import sys

sys.path.append("../..")
from src.utils.constants import COMPRESSED_ZARR_ENCODING, BC_SUBDAILY_CHUNKS

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

variable = "pv"
climate_model_chain = "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
climate_scenario = "rcp85"
path_zarr = Path(
    f"/landclim_nobackup/yhaddad/clim2energy-ch/energy/europe/{variable}/{climate_model_chain}_{climate_scenario}/"
)
path_output = Path(
    f"/landclim/yhaddad/clim2energy-ch/energy/europe/{variable}/{climate_model_chain}_{climate_scenario}/"
)
with ProgressBar(dt=1):
    ds_var = xr.open_mfdataset(
        path_zarr.glob(f"pv_{climate_model_chain}_{climate_scenario}*.zarr"),
        engine="zarr",
        combine="by_coords",
        parallel=True,
    )
    zarr_encoding = {
        var: {
            "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
            "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
        }
        for var in ds_var.data_vars
    }
    ds_var.chunk(BC_SUBDAILY_CHUNKS).to_zarr(
        path_output / f"{variable}_{climate_model_chain}_{climate_scenario}.zarr",
        mode="w",
        encoding=zarr_encoding,
    )
