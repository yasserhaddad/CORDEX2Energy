# Data Documentation — CORDEX2Energy

This document describes the full data processing chain, from raw climate model output to final energy-relevant quantities for Switzerland and Europe.

---

## Table of Contents

0. [General Information](#0-general-information)
1. [Climate Model Data](#1-climate-model-data)
   - 1.1 [Raw CORDEX Data](#11-raw-cordex-data)
   - 1.2 [Reference Reanalysis (CERRA)](#12-reference-reanalysis-cerra)
   - 1.3 [Preprocessing](#13-preprocessing)
   - 1.4 [How to Open the Climate Data](#14-how-to-open-the-climate-data)
2. [Bias Correction](#2-bias-correction)
   - 2.1 [Method](#21-method)
   - 2.2 [Variable-by-Variable Notes](#22-variable-by-variable-notes)
3. [Runoff Routing](#3-runoff-routing)
   - 3.1 [Switzerland — PREVAH Hydrological Model](#31-switzerland--prevah-hydrological-model)
   - 3.2 [Europe — mRm Routing Model](#32-europe--mrm-routing-model)
4. [Energy Conversion](#4-energy-conversion)
   - 4.1 [Wind Power](#41-wind-power)
   - 4.2 [Solar Photovoltaics (PV)](#42-solar-photovoltaics-pv)
   - 4.3 [Hydropower](#43-hydropower)
   - 4.4 [Energy Demand (HDD / CDD)](#44-energy-demand-hdd--cdd)
   - 4.5 [How to Open the Energy Data](#45-how-to-open-the-energy-data)
5. [Data Formats and Storage](#5-data-formats-and-storage)
6. [Scenarios and Time Periods](#6-scenarios-and-time-periods)
7. [Spatial Domains](#7-spatial-domains)
8. [References](#8-references)

---

## 0. General Information

### Dataset Title

**CORDEX2Energy** — Bias-corrected EURO-CORDEX data converted into energy variables for energy system analysis in Switzerland and Europe

### Contact

For questions regarding the dataset, processing methods, or data access, please contact:

> **Name**: *Yann Yasser Haddad*
> **Institution**: *ETH Zürich, Switzerland*
> **Email**: *yasser.haddad@env.ethz.ch*

### Funding and Acknowledgements

The authors gratefully acknowledge funding from the Joint Initiative SPEED2ZERO. SPEED2ZERO conducts research, tools, action plans and technologies to support a sustainable transformation in Switzerland. A transformation that meets international and national climate targets guarantees a resilient energy supply, and in which biodiversity can regain its richness. SPEED2ZERO received support from the ETH-Board under the Joint Initiatives scheme.

### License

See [LICENSE](LICENSE) in the repository root.

---

## 1. Climate Model Data

### 1.1 Raw CORDEX Data

The primary climate forcing data comes from the **EURO-CORDEX** ensemble at **0.11° horizontal resolution** (EUR-11). Each model chain combines a Regional Climate Model (RCM) driven by a Global Climate Model (GCM).

| RCM | GCM | Ensemble | Available scenarios |
|-----|-----|----------|---------------------|
| CNRM-ALADIN63 | CNRM-CERFACS-CNRM-CM5 | r1i1p1 | rcp26, rcp45, rcp85 |
| CNRM-ALADIN63 | MOHC-HadGEM2-ES | r1i1p1 | rcp85 |
| CNRM-ALADIN63 | NCC-NorESM1-M | r1i1p1 | rcp85 |
| CNRM-ALADIN63 | MPI-M-MPI-ESM-LR | r1i1p1 | rcp85 |
| MOHC-HadREM3-GA7-05 | MOHC-HadGEM2-ES | r1i1p1 | rcp26, rcp85 |
| MOHC-HadREM3-GA7-05 | CNRM-CERFACS-CNRM-CM5 | r1i1p1 | rcp85 |
| MOHC-HadREM3-GA7-05 | MPI-M-MPI-ESM-LR | r1i1p1 | rcp85 |
| MOHC-HadREM3-GA7-05 | NCC-NorESM1-M | r1i1p1 | rcp85 |
| MOHC-HadREM3-GA7-05 | ICHEC-EC-EARTH | r12i1p1 | rcp26, rcp85 |

The following variables are extracted and regridded onto a common grid prior to further processing:

| CF variable | Description | Temporal res. | Units (raw) |
|-------------|-------------|---------------|-------------|
| `tas` | Near-surface air temperature (2 m) | 3-hourly | K |
| `hurs` | Near-surface relative humidity (2 m) | 3-hourly | % |
| `rsds` | Surface downwelling shortwave radiation | 3-hourly | W m⁻² |
| `sfcWind` | Near-surface wind speed (10 m) | 3-hourly | m s⁻¹ |
| `ps` | Surface pressure | 3-hourly | Pa |
| `psl` | Mean sea level pressure | Daily | Pa |
| `pr` | Precipitation | Daily | kg m⁻² s⁻¹ |
| `mrro` | Surface runoff | Daily | kg m⁻² s⁻¹ |
| `clt` | Total cloud cover fraction | Daily | % |
| `wind100m` | Wind speed at 100 m (optional) | 3-hourly | m s⁻¹ |
| `wind150m` | Wind speed at 150 m (optional) | 3-hourly | m s⁻¹ |

> **Note on derived variables**: `tas_min` and `tas_max` are computed from 3-hourly `tas` as part of postprocessing. `huss` (specific humidity) is derived from `hurs`, `tas`, and `ps`.

### 1.2 Reference Reanalysis (CERRA)

**CERRA** (Copernicus European Regional ReAnalysis) serves as the observational reference for bias correction. CERRA-Land is used for runoff. The mapping between CORDEX and CERRA variable names is:

| CORDEX variable | CERRA variable | Height / notes |
|-----------------|----------------|----------------|
| `tas` | `2t` | height = 2 m |
| `hurs` | `2r` | height = 2 m |
| `sfcWind` | `10si` | height = 10 m |
| `ps` | `sp` | — |
| `psl` | `msl` | — |
| `rsds` | `ssrd` | J m⁻² → W m⁻² conversion applied |
| `pr` | `tp` | kg m⁻² |
| `mrro` | `ro` | sum of `sro` (surface runoff) + `perc` (percolation) from CERRA-Land |
| `clt` | `tcc` | — |
| `wind100m/150m` | `ws` | at respective height |

### 1.3 Preprocessing

Raw CORDEX files are preprocessed into uniform Zarr archives stored under:

```
cordex_processed/{model}_{scenario}/regridded/{temp_res}/{variable}.zarr
```

Steps applied per variable:
- **Regridding**: bilinear for most variables; conservative for `mrro` onto a hydrological grid (`g010_MERIT_2.txt`)
- **Calendar conversion**: 360-day model calendars are expanded to a standard Gregorian calendar
- **Temporal gap filling**: missing timesteps are forward-filled or filled from climatology
- **Duplicate/monotonicity checks**: duplicate coordinates and non-monotonic time axes are corrected

### 1.4 How to Open the Climate Data

All processed climate data is stored as **Zarr** archives and can be opened with [xarray](https://docs.xarray.dev). The path structure follows the convention described in §1.3.

```python
import xarray as xr
from pathlib import Path

path_data_processed = Path("/path/to/processed_data")
model  = "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario = "rcp85"

# --- Regridded (pre-bias-correction) ---
ds_regridded = xr.open_zarr(
    path_data_processed
    / "cordex_processed"
    / f"{model}_{scenario}"
    / "regridded"
    / "3hr"          # or "day" for daily variables
    / "tas.zarr",
    chunks="auto",
)

# --- Bias-corrected ---
ds_bc = xr.open_zarr(
    path_data_processed
    / "cordex_processed"
    / f"{model}_{scenario}"
    / "bias_corrected"
    / "3hr"          # or "day" for daily variables
    / "tas.zarr",
    chunks="auto",
)

# Spatial subset (e.g. Switzerland)
ds_ch = ds_bc.sel(lat=slice(45.8, 47.9), lon=slice(5.9, 10.5))

# Temporal subset
ds_future = ds_bc.sel(time=slice("2021", "2100"))
```

> **Tip**: pass `chunks="auto"` to keep data lazy (Dask-backed). For small subsets call `.load()` to bring the data into memory.

---

## 2. Bias Correction

### 2.1 Method

Bias correction is applied variable by variable using **Empirical Quantile Mapping (EQM)** as implemented in [xclim](https://xclim.readthedocs.io/).

| Parameter | Value |
|-----------|-------|
| Method | Empirical Quantile Mapping (EQM) |
| Training period (reference) | 1991–2020 |
| Projection period (future) | 2021–2100 |
| Running window | 91 days (seasonal detrending) |
| Spatial domain (general) | lat [30, 75], lon [−30, 41] |
| Spatial domain (runoff) | lat [20.8, 71.8], lon [−51.7, 72.7] |

Output path:
```
cordex_processed/{model}_{scenario}/bias_corrected/{temp_res}/{variable}.zarr
```

### 2.2 Variable-by-Variable Notes

All variables use 99 quantiles and `extrapolation = "constant"` unless noted otherwise.

| Variable | Adjustment kind | Interp. | Freq. adaptation threshold | Jitter threshold | Special handling |
|----------|----------------|---------|---------------------------|------------------|-----------------|
| `tas` | additive (`+`) | linear | — | — | No unit transformation; units remain K |
| `hurs` | multiplicative (`*`) | linear | — | — | — |
| `sfcWind` | multiplicative (`*`) | linear | — | — | Reference height set to 10 m; same parameters applied to `wind100m`/`wind150m` |
| `ps` | additive (`+`) | linear | — | — | — |
| `psl` | additive (`+`) | linear | — | — | — |
| `rsds` | multiplicative (`*`) | linear | — | 0.01 W m⁻² | CERRA `ssrd` converted from J m⁻² to W m⁻² by dividing by time step duration; leap days removed from reference; time-offset alignment applied |
| `pr` | multiplicative (`*`) | linear | 0.1 mm d⁻¹ | 0.001 mm d⁻¹ | CORDEX units converted from kg m⁻² s⁻¹ to mm d⁻¹; negative post-correction values clipped to 0 |
| `mrro` | multiplicative (`*`) | linear | — | 0.001 kg m⁻² | CORDEX units converted from kg m⁻² s⁻¹ to kg m⁻²; land/sea NaN mask re-applied; regridded conservatively onto hydrological grid |
| `clt` | additive (`+`) | linear | — | — | Post-correction values clipped to [0, 100] |

**Postprocessing steps** (after bias correction):
- `tas_min` and `tas_max` are derived from the bias-corrected 3-hourly `tas`
- `huss` (specific humidity) is computed from bias-corrected `hurs`, `tas`, and `ps`
- A NaN mask from the pre-correction regridded field is re-applied to `mrro` to restore missing ocean/glacier cells

---

## 3. Runoff Routing

### 3.1 Switzerland — PREVAH Hydrological Model

For Switzerland and the surrounding alpine region, bias-corrected daily climate fields are fed into the **PREVAH** (Precipitation–Runoff–EVApotranspiration HRU model) to produce spatially distributed streamflow. This enables routing of surface runoff through Swiss catchments.

**Spatial extent for PREVAH input**: lat [44.4, 48.6], lon [4.5, 16.5]

**Input variables** prepared for PREVAH (daily, bias-corrected):

| Variable | Description |
|----------|-------------|
| `tas` | Mean daily temperature |
| `tas_min` | Daily minimum temperature |
| `tas_max` | Daily maximum temperature |
| `huss` | Specific humidity |
| `hurs` | Relative humidity |
| `pr` | Precipitation |
| `rsds` | Shortwave radiation |
| `sfcWind` | Wind speed |
| `clt` | Cloud cover |
| `psl` | Sea level pressure |

Files are written to:
```
cordex_processed/{model}_{scenario}/prevah/input_data/
```

PREVAH outputs routed streamflow at gauging stations (RGS files), which are later combined and converted to hydropower capacity factors.

### 3.2 Europe — mRm Routing Model

For Europe-wide hydropower estimates, bias-corrected daily `mrro` is routed through the **mRm** (multiscale Routing Model) to produce gridded river discharge across the European domain.

**Postprocessing of `mrro` as input to mRm**:

Before being passed to the routing model, the bias-corrected runoff is postprocessed as follows:
1. Converted from kg m⁻² (daily accumulation) back to kg m⁻² s⁻¹
2. Re-masked with the original land/ocean NaN pattern (to restore missing ocean/glacier cells that were filled prior to bias correction)
3. Regridded conservatively onto the mRm hydrological grid (`g010_MERIT_2.txt`)

Postprocessed output:
```
cordex_processed/{model}_{scenario}/bias_corrected/day/mrro.zarr  (masked, regridded onto mRm grid)
```

mRm routing output (routed discharge) is stored under:
```
{path_routed_runoff}/{model}_{scenario}/
```

---

## 4. Energy Conversion

### 4.1 Wind Power

**Input variable**: bias-corrected 3-hourly `sfcWind` (10 m wind speed)

**Processing steps**:
1. **Power-law exponent** computed from wind speed at multiple heights (where available) to characterise the vertical wind profile. When model chains don't provide wind speeds at multiple heights, we use grid-cell specific exponents derived from the CERRA reanalysis
2. **Hub-height extrapolation** using the computed exponent
3. **Smoothed power curves** (cubic spline, 0.01 m s⁻¹ resolution) are generated for each turbine type
4. Wind speed → power output using the turbine-specific power curve
5. Aggregated spatially per country using installed-capacity weighting

**Turbine configurations**:

| Scope | Turbines | Hub height (m) |
|-------|----------|----------------|
| Europe | E-126/7580 | 127 |
| Europe | SWT120/3600 | 90 |
| Europe | SWT142/3150 | 129 |
| Switzerland | E-82/2300 | 98 |
| Switzerland | E-92/2350 | 98 |
| Switzerland | V90/2000 | 95 |
| Switzerland | V112/3300 | 94 |

Output path:
```
energy/{scope}/wind/{model}_{scenario}/
```

### 4.2 Solar Photovoltaics (PV)

**Input variables**: bias-corrected 3-hourly `rsds` (incoming surface shortwave radiation), `tas` (temperature) and `sfcWind` (surface wind speed).

**Processing steps**:
1. **PV model**: [pvlib](https://pvlib-python.readthedocs.io/) `ModelChain` with the SAPM temperature model (`close_mount_glass_glass`)
2. Global horizontal irradiance (GHI) is decomposed into direct and diffuse components using the **Erbs** model
3. Power output is computed using a normalised rooftop PV system (`pdc0 = 1000 W`, `γ_pdc = −0.004 %/°C`, inverter efficiency 97%)
4. Multiple panel orientations (tilt / azimuth) are combined using a fixed weighting configuration (`NORMALISED_ROOFTOP_PV_CONFIGURATION`)
5. _**Temporal disaggregation** (3-hourly → hourly): clear-sky irradiance profiles generated with pvlib are used to weight 3-hourly PV estimates within each 3-hour block. Night-time values are masked using a radiation threshold (> 5 W m⁻²)_ (In progress)
6. Aggregated per country

Output path:
```
energy/{scope}/pv/{model}_{scenario}/pv_{model}_{scenario}.zarr
```

Units: kW per kWp

### 4.3 Hydropower

**Input variable**: daily routed streamflow (Switzerland: from PREVAH; Europe: from mRm)

Two approaches are implemented:

**Run-of-River (RoR)**:
- Hydropower potential is expressed as the ratio of daily discharge to the design discharge (defined as a high quantile of the reference-period distribution)
- Optionally weighted by mean discharge or persistence (signal-to-noise ratio) across catchments

**Reservoir inflows**:
- Inflow to reservoir storage estimated from routed discharge

Output path:
```
energy/europe/hydropower/{model}_{scenario}/
```

### 4.4 Energy Demand (HDD / CDD)

**Input variable**: bias-corrected 3-hourly `tas` (near-surface temperature)

**Processing steps**:
1. Temperature is aggregated to daily means
2. **Heating Degree Days (HDD)** and **Cooling Degree Days (CDD)** are computed relative to standard base temperatures
3. Optionally weighted by **gridded population data** to produce nationally representative demand indices
4. Aggregated per country

Output path:
```
energy/europe/demand/{model}_{scenario}/
```

### 4.5 How to Open the Energy Data

All energy output files are Zarr archives (gridded) or CSV files (country/canton aggregates).

#### Gridded energy data (xarray / Zarr)

```python
import xarray as xr
from pathlib import Path

path_energy = Path("/path/to/processed_data") / "energy"
model    = "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario = "rcp85"

# Wind power (Europe)
ds_wind = xr.open_zarr(
    path_energy / "europe" / "wind" / f"{model}_{scenario}" / "wind.zarr",
    chunks="auto",
)

# Solar PV (Switzerland)
ds_pv = xr.open_zarr(
    path_energy / "ch" / "pv"
    / f"{model}_{scenario}"
    / f"pv_{model}_{scenario}.zarr",
    chunks="auto",
)
```

#### Country/canton aggregated data (xarray / NetCDF)

Country-aggregated energy outputs are stored as NetCDF (`.nc`) files and can be opened with `xarray.open_dataset`.

```python
import xarray as xr
from pathlib import Path

path_energy = Path("/path/to/processed_data") / "energy"
model    = "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario = "rcp85"
turbine  = "E-126_7580"   # note: '/' replaced by '_' in filenames

# Wind power — country aggregates (one file per turbine type)
ds_wind_agg = xr.open_dataset(
    path_energy / "europe" / "wind" / f"{model}_{scenario}"
    / f"wind_power_agg_{model}_{scenario}_{turbine}.nc"
)
# Offshore wind
ds_wind_offshore = xr.open_dataset(
    path_energy / "europe" / "wind" / f"{model}_{scenario}"
    / f"wind_power_offshore_{model}_{scenario}_{turbine}.nc"
)

# Solar PV — country aggregates (Europe scope)
ds_pv_agg = xr.open_dataset(
    path_energy / "europe" / "pv" / f"{model}_{scenario}"
    / f"pv_agg_{model}_{scenario}.nc"
)
# Variables are named e.g. pv_<tilt>_az_<direction> (tilt in degrees, direction N/S/E/SE/…)
# Units: fraction (0–1) capacity factor
```

---

## 5. Data Formats and Storage

| Format | Usage | Compression |
|--------|-------|-------------|
| Zarr | All processed and bias-corrected data | Blosc/Zstd level 3, SHUFFLE |
| NetCDF (.nc) | Intermediate regridding blocks and aggregated country/canton time series | zlib level 1 |

**Chunk sizes**:

| Stage | Time | Lat | Lon |
|-------|------|-----|-----|
| Regridded | unlimited | 20 | 20 |
| Bias-corrected (daily) | 365 | 256 | 256 |
| Bias-corrected (3-hourly) | 2920 | 128 | 128 |

---

## 6. Scenarios and Time Periods

| Period label | Years | Description |
|--------------|-------|-------------|
| Reference | 1991–2020 | Used for bias correction training and climatological baselines |
| Near-future | 2021–2050 | First half of projection period |
| Mid-century | 2041–2060 | Overlap period used for mid-century analyses |
| Future | 2021–2100 | Full projection period |

---

## 7. Spatial Domains

| Domain | lat range | lon range | Notes |
|--------|-----------|-----------|-------|
| Europe (BC) | [30, 75] | [−30, 41] | Bias correction and energy variables |
| Runoff (BC) | [20.8, 71.8] | [−51.7, 72.7] | Extended domain for `mrro` |
| Switzerland | [45.8, 47.9] | [5.9, 10.5] | Fine-scale CH analysis |
| PREVAH extent | [44.4, 48.6] | [4.5, 16.5] | Alpine region for hydrological routing |

---

## 8. References

### Climate Models and Reanalysis

- **EURO-CORDEX**: Jacob, D. et al. EURO-CORDEX: new high-resolution climate change projections for European impact research. Reg Environ Change 14, 563–578 (2014).

- **CERRA reanalysis**: Ridal, M. et al. CERRA, the Copernicus European Regional Reanalysis system. Quarterly Journal of the Royal Meteorological Society 150, 3385–3411 (2024).

### Hydrological Models

- **PREVAH**: Viviroli, D., Zappa, M., Gurtz, J. & Weingartner, R. An introduction to the hydrological modelling system PREVAH and its pre- and post-processing-tools. Environmental Modelling & Software 24, 1209–1222 (2009).

- **mRm (mesoscale Routing Model)**: Thober, S. et al. The multiscale routing model mRM v1.0: simple river routing at resolutions from 1 to 50&thinsp;km. Geoscientific Model Development 12, 2501–2521 (2019).


### Software and Tools

- **xclim** (bias correction): Bourgault, P. et al. xclim: xarray-based climate data analytics. Journal of Open Source Software 8, 5415 (2023).


- **pvlib** (PV modelling): Anderson, K. S. et al. pvlib python: 2023 project update. Journal of Open Source Software 8, 5994 (2023).


- **windpowerlib** (wind power modelling): Sabine Haas et al. wind-python/windpowerlib: Update release. Zenodo https://doi.org/10.5281/ZENODO.10685057 (2024).
