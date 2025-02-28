# ClimateScenarios4Energy-CH/src

This repository contains scripts and modules for bias correction and processing of climate model data for energy applications in Switzerland. Below is a brief description of the key files and their functionalities.

## Files and Their Descriptions

### `bias_correction.py`
This script contains the `BiasCorrection` class, which is responsible for performing bias correction on climate model data using various methods. Key functionalities include:
- Loading and preprocessing climate model and reference reanalysis data.
- Performing bias correction using empirical quantile mapping.
- Updating attributes of the bias-corrected dataset.
- Saving and concatenating debiased data.

### `bias_correction_xclim.py`
This script provides functions for bias correction using the `xclim` library. Key functionalities include:
- `empirical_quantile_mapping`: Applies empirical quantile mapping for bias correction.
- `DICT_DEFAULT_XCLIM_PARAMETERS`: A dictionary containing default parameters for different climate variables.

### `constants.py`
This script defines constants used throughout the project. Key functionalities include:
- `REDUCED_NETCDF_ENCODINGS`: A dictionary specifying encoding settings for NetCDF files.
- `COMPRESSED_ZARR_ENCODING`: A dictionary specifying encoding settings for Zarr files using Blosc compression.

### `convert_climate_vars.py`
This script contains functions for converting and calculating climate variables. Key functionalities include:
- `convert_wind_components_to_speed`: Converts zonal and meridional wind components to wind speed.
- `calculate_specific_humidity`: Calculates specific humidity from temperature, relative humidity, and surface pressure.

### `load_data.py`
This script provides classes and functions for loading and processing climate model and reanalysis data. Key functionalities include:
- `ClimateModelData`: A class for loading and processing climate model data.
- `ClimateReanalysisData`: A class for loading and processing reanalysis data.
- `cdo_remap`: A function for remapping climate data to a target grid using CDO.

### `timesteps_adjustments.py`
This script contains functions for adjusting and extracting timesteps from climate model data. Key functionalities include:
- `extract_adjusted_timesteps`: Extracts adjusted timesteps from a mapping file for a given climate model chain and scenario.
- `get_new_timesteps`: Generates new timesteps for adjusted timeseries based on the extracted timesteps.
- `timedelta_to_freq`: Converts a timedelta object into a frequency string for use with `pd.date_range`.

### `utils_load_data.py`
This script provides utility functions for loading climate model data. Key functionalities include:
- `get_components_climate_model_chain`: Extracts components of a climate model chain string.
- `get_raw_cordex_data_paths`: Generates file paths for raw CORDEX data based on input parameters.

### `utils_xarray.py`
This script provides utility functions for working with xarray datasets. Key functionalities include:
- `xr_reorder_coords`: Reorders coordinates in an xarray dataset.
- `get_time_resolution`: Determines the time resolution of an xarray dataset.
- `get_valid_hours`: Generates valid hours for a given time resolution.
- `get_specific_hourly_data`: Selects data for a specific hour of the day from an xarray dataset.
- `match_time_steps`: Adjusts the time steps of a reference dataset to match those of another dataset.
- `get_min_max`: Calculates the minimum and maximum values of an xarray dataset over a specified time resolution.
- `add_values_missing_ocean`: Fills missing ocean data with random values within a specified range.
- `add_noise_to_constant_grid_cells`: Adds noise to constant grid cells in an xarray dataset.