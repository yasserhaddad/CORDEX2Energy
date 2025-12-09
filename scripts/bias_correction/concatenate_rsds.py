import sys
sys.path.append("../..")

from pathlib import Path
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from src.utils.constants import COMPRESSED_ZARR_ENCODING, BC_SUBDAILY_CHUNKS

# Configure Dask
dask.config.set(num_workers=16, threads_per_worker=2)
dask.config.set({"array.chunk-size": "400MiB"})
dask.config.set({"distributed.worker.memory.target": 0.75})
dask.config.set({"distributed.worker.memory.spill": 0.8})
dask.config.set({"distributed.worker.memory.pause": 0.9})

var = "hurs"
climate_model_chain = "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
climate_scenario = "rcp85"
temp_res = "3hr"
input_path = Path(f"/net/exo/landclim_nobackup/yhaddad/clim2energy-ch/cordex_processed/{climate_model_chain}_{climate_scenario}/temp/bias_corrected/{var}/{temp_res}")
output_path = Path(f"/net/exo/landclim/yhaddad/clim2energy-ch/cordex_processed/{climate_model_chain}_{climate_scenario}/bias_corrected/{temp_res}/{var}.zarr")

print(f"Reading zarr files from: {input_path}")

# Get all zarr directories
zarr_files = sorted(input_path.glob("*.zarr"))
print(f"Found {len(zarr_files)} zarr files")

if not zarr_files:
    raise ValueError(f"No zarr files found in {input_path}")

# Open all zarr files
print("Opening zarr files...")
with ProgressBar(dt=10):
    datasets = [xr.open_zarr(zarr_file, chunks="auto").load() for zarr_file in zarr_files]

# Concatenate along time dimension
print("Concatenating datasets...")
ds_combined = xr.concat(datasets, dim="time")

# Sort by time
print("Sorting by time...")
ds_combined = ds_combined.sortby("time")

# Remove duplicate times if any
print("Checking for duplicate times...")
if ds_combined.get_index("time").has_duplicates:
    print("  Removing duplicate times...")
    ds_combined = ds_combined.drop_duplicates(dim="time")

# Rechunk for optimal storage
print("Rechunking data...")
ds_combined = ds_combined.chunk(BC_SUBDAILY_CHUNKS)

# Prepare output
output_path.parent.mkdir(parents=True, exist_ok=True)

# Set up encoding
zarr_encoding = {
    var: {
        "compressor": COMPRESSED_ZARR_ENCODING["compressor"],
        "chunks": tuple(BC_SUBDAILY_CHUNKS.values()),
    }
    for var in ds_combined.data_vars
}

# Save to zarr
print(f"Saving to: {output_path}")
ds_combined.to_zarr(output_path, mode="w", encoding=zarr_encoding)

print("✓ Concatenation complete!")
print(f"  Time range: {ds_combined.time.min().values} to {ds_combined.time.max().values}")
print(f"  Total timesteps: {len(ds_combined.time)}")

