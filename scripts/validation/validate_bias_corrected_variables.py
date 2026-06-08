#!/usr/bin/env python3
"""
Comprehensive validation of bias-corrected climate variables.

This script creates validation plots for bias-corrected variables including:
- Sub-daily variables (3hr): tas, hurs, ps, sfcWind, rsds
- Daily variables: pr, mrro, clt, psl

Validation metrics include:
- Temporal statistics (mean, std, quantiles)
- Spatial patterns (maps of means and extremes)
- Threshold exceedances
- Diurnal cycles (for sub-daily data)
- Seasonal cycles
- Probability distributions
"""

import argparse
import json
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from cordex2energy.utils.constants import (
    EXTENT_PREVAH,
    EXTENT_CH,
    EXTENT_BC,
    REFERENCE_PERIOD,
    FUTURE_PERIOD,
)
from cordex2energy.utils.utils_xarray import sel_extent

# Configure Dask
dask.config.set(num_workers=16, threads_per_worker=2)
dask.config.set({"array.chunk-size": "400MiB"})
dask.config.set({"distributed.worker.memory.target": 0.75})
dask.config.set({"distributed.worker.memory.spill": 0.8})
dask.config.set({"distributed.worker.memory.pause": 0.9})

# Variable configurations with physical units and validation thresholds
VARIABLE_CONFIG = {
    # Sub-daily variables (3hr)
    "tas": {
        "name": "Near-Surface Air Temperature",
        "units": "K",
        "convert": lambda x: x - 273.15,  # K to °C
        "units_display": "°C",
        "thresholds": {"cold": -20, "hot": 40},
        "expected_range": (-40, 50),
        "temp_res": "3hr",
    },
    "hurs": {
        "name": "Near-Surface Relative Humidity",
        "units": "%",
        "convert": None,
        "units_display": "%",
        "thresholds": {"low": 10, "high": 100},
        "expected_range": (0, 100),
        "temp_res": "3hr",
    },
    "ps": {
        "name": "Surface Air Pressure",
        "units": "Pa",
        "convert": lambda x: x / 100,  # Pa to hPa
        "units_display": "hPa",
        "thresholds": {"low": 95000, "high": 105000},  # 950-1050 hPa
        "expected_range": (850, 1100),
        "temp_res": "3hr",
    },
    "sfcWind": {
        "name": "Near-Surface Wind Speed",
        "units": "m s-1",
        "convert": None,
        "units_display": "m s-1",
        "thresholds": {"moderate": 10, "strong": 20},
        "expected_range": (0, 50),
        "temp_res": "3hr",
    },
    "rsds": {
        "name": "Surface Downwelling Shortwave Radiation",
        "units": "J m-2",
        "convert": lambda x: x / (3 * 3600),  # J/m² to W/m² for 3hr data
        "units_display": "W m-2",
        "thresholds": {"high": 800},
        "expected_range": (0, 1400),
        "temp_res": "3hr",
    },
    # Daily variables
    "pr": {
        "name": "Precipitation",
        "units": ["kg m-2 s-1", "kg m-2", "mm d-1"],
        "conversions": {
            "kg m-2 s-1": lambda x: x * 86400,  # kg/m²/s to mm/day (rate to total)
            "kg m-2": lambda x: x,  # kg/m² to mm/day (equivalent, no conversion needed)
        },
        "convert": lambda x: x
        * 86400,  # Default conversion (for backward compatibility)
        "units_display": "mm d-1",
        "thresholds": {"low": 0, "high": 150},  # mm/day after conversion
        "expected_range": (0, 300),  # mm/day
        "temp_res": "day",
    },
    "mrro": {
        "name": "Runoff",
        "units": "kg m-2 s-1",
        "convert": lambda x: x * 86400,  # kg/m²/s to mm/day
        "units_display": "kg m-2",
        "thresholds": {"high": 10, "low": 0.5},
        "expected_range": (0, 200),
        "temp_res": "day",
    },
    "clt": {
        "name": "Total Cloud Cover",
        "units": "%",
        "convert": None,
        "units_display": "%",
        "thresholds": {"low": 20, "high": 80},
        "expected_range": (0, 100),
        "temp_res": "day",
    },
    "psl": {
        "name": "Sea Level Pressure",
        "units": "Pa",
        "convert": lambda x: x / 100,  # Pa to hPa
        "units_display": "hPa",
        "thresholds": {"low": 980, "high": 1030},
        "expected_range": (950, 1080),
        "temp_res": "day",
    },
}


def needs_conversion(da, expected_units, display_units, conversions=None):
    """Check if data needs unit conversion and return the appropriate conversion function.

    Args:
        da: DataArray to check
        expected_units: Expected input units (can be string or list) - for backward compatibility
        display_units: Target display units
        conversions: Dict mapping input units to conversion functions (optional)

    Returns:
        conversion function if conversion needed, None otherwise, or True for legacy behavior
    """
    if not hasattr(da, "units"):
        # If no units specified and we have conversions dict, use the first one
        if conversions and isinstance(conversions, dict):
            return list(conversions.values())[0]
        return True  # Legacy behavior: needs conversion

    # Check if already in display units
    current_units = (
        da.units.strip() if hasattr(da.units, "strip") else str(da.units).strip()
    )

    # Normalize unit representations (handle various formats for superscripts)
    def normalize_units(unit_str):
        return (
            unit_str.replace("⁻", "-")
            .replace("⁻¹", "-1")
            .replace("⁻²", "-2")
            .replace("**-1", "-1")
            .replace("**-2", "-2")
            .replace("^-1", "-1")
            .replace("^-2", "-2")
        )

    display_units_normalized = normalize_units(display_units)
    current_units_normalized = normalize_units(current_units)

    if current_units_normalized == display_units_normalized:
        return None  # Already in display units, no conversion needed

    # If conversions dict provided, look up the appropriate conversion
    if conversions and isinstance(conversions, dict):
        for input_unit, conversion_func in conversions.items():
            if normalize_units(input_unit) == current_units_normalized:
                return conversion_func
        # Current units not in conversions dict
        print(
            f"  Warning: Unexpected units '{current_units}' (expected one of {list(conversions.keys())} or '{display_units}')"
        )
        return list(conversions.values())[0]  # Use first conversion as fallback

    # Legacy behavior: check against expected_units
    if isinstance(expected_units, list):
        expected_units_normalized = [normalize_units(u) for u in expected_units]
    else:
        expected_units_normalized = [normalize_units(expected_units)]

    if current_units_normalized in expected_units_normalized:
        return True  # In expected units, needs conversion (legacy)
    else:
        # Unknown units, log warning and assume conversion needed
        print(
            f"  Warning: Unexpected units '{current_units}' (expected {expected_units} or '{display_units}')"
        )
        return True


class BiasCorrectingValidator:
    """Comprehensive validation of bias-corrected climate variables."""

    def __init__(
        self,
        climate_model_chain: str,
        climate_scenario: str,
        path_data_processed: Path,
        output_dir: Path,
        extent: dict = None,
        temp_res_override: str = None,
    ):
        self.climate_model_chain = climate_model_chain
        self.climate_scenario = climate_scenario
        self.path_data_processed = path_data_processed
        self.output_dir = output_dir
        self.extent = extent or EXTENT_PREVAH
        self.temp_res_override = temp_res_override

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directory structure
        self.bias_corrected_dir = (
            path_data_processed
            / "cordex_processed"
            / f"{climate_model_chain}_{climate_scenario}"
            / "bias_corrected"
        )
        self.regridded_dir = (
            path_data_processed
            / "cordex_processed"
            / f"{climate_model_chain}_{climate_scenario}"
            / "regridded"
        )
        self.cerra_dir = path_data_processed / "cerra_processed" / "regridded"

    def load_data(self, variable: str, period: str = "reference"):
        """Load regridded, bias-corrected, and reference data for a variable.

        Args:
            variable: Variable name
            period: Either 'reference' or 'future'
        """
        config = VARIABLE_CONFIG[variable]
        # Use override if specified, otherwise fall back to config
        temp_res = (
            self.temp_res_override if self.temp_res_override else config["temp_res"]
        )

        # Select time period
        if period == "reference":
            time_slice = REFERENCE_PERIOD
        elif period == "future":
            time_slice = FUTURE_PERIOD
        else:
            raise ValueError(
                f"Invalid period: {period}. Must be 'reference' or 'future'"
            )

        print(
            f"  Loading {variable} data for {period} period ({time_slice[0]}-{time_slice[1]})..."
        )

        # Load bias-corrected data
        bc_path = self.bias_corrected_dir / temp_res / f"{variable}.zarr"
        if not bc_path.exists():
            raise FileNotFoundError(f"Bias-corrected data not found: {bc_path}")
        ds_bc = xr.open_zarr(bc_path, chunks="auto")
        da_bc = ds_bc[variable]

        # Load regridded (raw) data
        raw_path = self.regridded_dir / temp_res / f"{variable}.zarr"
        if not raw_path.exists():
            raise FileNotFoundError(f"Regridded data not found: {raw_path}")
        ds_raw = xr.open_zarr(raw_path, chunks="auto")
        da_raw = ds_raw[variable]

        # Remove any extra dimensions beyond lat, lon, time
        expected_dims = {"lat", "lon", "time"}
        for da in [da_raw, da_bc]:
            extra_dims = set(da.dims) - expected_dims
            if extra_dims:
                print(f"  Removing extra dimensions from data: {extra_dims}")
                for dim in extra_dims:
                    da_raw = da_raw.isel({dim: 0}) if dim in da_raw.dims else da_raw
                    da_bc = da_bc.isel({dim: 0}) if dim in da_bc.dims else da_bc

        # Load CERRA reference data (only for reference period)
        da_cerra = None
        if period == "reference":
            cerra_var_map = {
                "tas": "2t",
                "hurs": "2r",
                "ps": "sp",
                "sfcWind": "10si",
                "rsds": "ssrd",
                "pr": "tp",
                "mrro": "ro",
                "clt": "tcc",
                "psl": "msl",
            }
            cerra_var = cerra_var_map.get(variable)
            if cerra_var:
                cerra_path = self.cerra_dir / f"{cerra_var}.zarr"
                if cerra_path.exists():
                    ds_cerra = xr.open_zarr(cerra_path, chunks="auto")
                    da_cerra = ds_cerra[cerra_var]

                    # Remove any extra dimensions from CERRA data
                    extra_dims = set(da_cerra.dims) - expected_dims
                    if extra_dims:
                        print(
                            f"  Removing extra dimensions from CERRA data: {extra_dims}"
                        )
                        for dim in extra_dims:
                            da_cerra = da_cerra.isel({dim: 0})
                else:
                    print(f"  Warning: CERRA data not found: {cerra_path}")

        # Select extent and time period
        da_bc = sel_extent(da_bc.sel(time=slice(*time_slice)), self.extent)
        da_raw = sel_extent(da_raw.sel(time=slice(*time_slice)), self.extent)
        if da_cerra is not None:
            da_cerra = sel_extent(da_cerra.sel(time=slice(*time_slice)), self.extent)

        # Remove leap days (February 29th)
        da_bc = da_bc.sel(
            time=~((da_bc.time.dt.month == 2) & (da_bc.time.dt.day == 29))
        )
        da_raw = da_raw.sel(
            time=~((da_raw.time.dt.month == 2) & (da_raw.time.dt.day == 29))
        )
        if da_cerra is not None:
            da_cerra = da_cerra.sel(
                time=~((da_cerra.time.dt.month == 2) & (da_cerra.time.dt.day == 29))
            )

        # Apply unit conversions for display if needed
        conversions = config.get("conversions", None)

        if conversions is not None or config["convert"] is not None:
            # Check each dataset and convert only if necessary
            convert_func_bc = needs_conversion(
                da_bc, config["units"], config["units_display"], conversions
            )
            if convert_func_bc is not None:
                if callable(convert_func_bc):
                    da_bc_display = convert_func_bc(da_bc)
                elif convert_func_bc is True and config["convert"] is not None:
                    # Legacy behavior
                    da_bc_display = config["convert"](da_bc)
                else:
                    da_bc_display = da_bc
            else:
                da_bc_display = da_bc

            convert_func_raw = needs_conversion(
                da_raw, config["units"], config["units_display"], conversions
            )
            if convert_func_raw is not None:
                if callable(convert_func_raw):
                    da_raw_display = convert_func_raw(da_raw)
                elif convert_func_raw is True and config["convert"] is not None:
                    da_raw_display = config["convert"](da_raw)
                else:
                    da_raw_display = da_raw
            else:
                da_raw_display = da_raw

            if da_cerra is not None:
                convert_func_cerra = needs_conversion(
                    da_cerra, config["units"], config["units_display"], conversions
                )
                if convert_func_cerra is not None:
                    if callable(convert_func_cerra):
                        da_cerra_display = convert_func_cerra(da_cerra)
                    elif convert_func_cerra is True and config["convert"] is not None:
                        da_cerra_display = config["convert"](da_cerra)
                    else:
                        da_cerra_display = da_cerra
                else:
                    da_cerra_display = da_cerra
            else:
                da_cerra_display = None
        else:
            da_bc_display = da_bc
            da_raw_display = da_raw
            da_cerra_display = da_cerra

        return da_raw_display, da_bc_display, da_cerra_display

    def validate_variable(self, variable: str):
        """Run comprehensive validation for a single variable."""
        print(f"\n{'='*80}")
        print(f"Validating {variable} - {VARIABLE_CONFIG[variable]['name']}")
        print(f"{'='*80}")

        config = VARIABLE_CONFIG[variable]
        # Use override if specified, otherwise fall back to config
        temp_res = (
            self.temp_res_override if self.temp_res_override else config["temp_res"]
        )

        # Load data for both periods
        da_raw_ref, da_bc_ref, da_cerra_ref = self.load_data(
            variable, period="reference"
        )
        da_raw_fut, da_bc_fut, _ = self.load_data(variable, period="future")

        # Create figure directory for this variable (with temp_res subdirectory)
        fig_dir = self.output_dir / temp_res / variable
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 1. Null value spatial patterns
        print("  Computing null value patterns...")
        self._plot_null_patterns(
            da_raw_ref,
            da_bc_ref,
            da_cerra_ref,
            da_raw_fut,
            da_bc_fut,
            variable,
            config,
            fig_dir,
        )

        # 2. Spatial mean patterns
        print("  Computing spatial patterns...")
        self._plot_spatial_patterns(
            da_raw_ref,
            da_bc_ref,
            da_cerra_ref,
            da_raw_fut,
            da_bc_fut,
            variable,
            config,
            fig_dir,
        )

        # 3. Temporal statistics
        print("  Computing temporal statistics...")
        self._plot_temporal_statistics(
            da_raw_ref,
            da_bc_ref,
            da_cerra_ref,
            da_raw_fut,
            da_bc_fut,
            variable,
            config,
            fig_dir,
        )

        # 3. Probability distributions
        # print("  Computing probability distributions...")
        # self._plot_distributions(
        #     da_raw_ref, da_bc_ref, da_cerra_ref, da_raw_fut, da_bc_fut,
        #     variable, config, fig_dir
        # )

        # 4. Extreme values analysis
        print("  Computing extreme values...")
        self._plot_extremes(
            da_raw_ref,
            da_bc_ref,
            da_cerra_ref,
            da_raw_fut,
            da_bc_fut,
            variable,
            config,
            fig_dir,
        )

        # 5. Seasonal cycle
        print("  Computing seasonal cycle...")
        self._plot_seasonal_cycle(
            da_raw_ref,
            da_bc_ref,
            da_cerra_ref,
            da_raw_fut,
            da_bc_fut,
            variable,
            config,
            fig_dir,
        )

        # 6. Diurnal cycle (for sub-daily data only)
        if temp_res == "3hr":
            print("  Computing diurnal cycle...")
            self._plot_diurnal_cycle(
                da_raw_ref,
                da_bc_ref,
                da_cerra_ref,
                da_raw_fut,
                da_bc_fut,
                variable,
                config,
                fig_dir,
            )

        # 8. Threshold exceedances
        if config.get("thresholds"):
            print("  Computing threshold exceedances...")
            self._plot_threshold_exceedances(
                da_raw_ref,
                da_bc_ref,
                da_cerra_ref,
                da_raw_fut,
                da_bc_fut,
                variable,
                config,
                fig_dir,
            )

        print(f"  ✓ Validation plots saved to {fig_dir}")

    def _plot_null_patterns(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot spatial patterns of null/NaN values for both periods."""
        with ProgressBar(dt=2):
            # Count NaN values across time dimension (for spatial maps)
            null_raw_ref = da_raw_ref.isnull().sum(dim="time").compute()
            null_bc_ref = da_bc_ref.isnull().sum(dim="time").compute()
            null_raw_fut = da_raw_fut.isnull().sum(dim="time").compute()
            null_bc_fut = da_bc_fut.isnull().sum(dim="time").compute()

            # Count number of timesteps that have at least one null value
            timesteps_with_nulls_raw_ref = (
                da_raw_ref.isnull().any(dim=["lat", "lon"]).sum().compute().values
            )
            timesteps_with_nulls_bc_ref = (
                da_bc_ref.isnull().any(dim=["lat", "lon"]).sum().compute().values
            )
            timesteps_with_nulls_raw_fut = (
                da_raw_fut.isnull().any(dim=["lat", "lon"]).sum().compute().values
            )
            timesteps_with_nulls_bc_fut = (
                da_bc_fut.isnull().any(dim=["lat", "lon"]).sum().compute().values
            )

            # Total timesteps for percentage calculation
            n_time_ref = len(da_raw_ref.time)
            n_time_fut = len(da_raw_fut.time)

            if da_cerra_ref is not None:
                null_cerra_ref = da_cerra_ref.isnull().sum(dim="time").compute()
                timesteps_with_nulls_cerra_ref = (
                    da_cerra_ref.isnull().any(dim=["lat", "lon"]).sum().compute().values
                )

        # Convert to percentage
        null_pct_raw_ref = (null_raw_ref / n_time_ref) * 100
        null_pct_bc_ref = (null_bc_ref / n_time_ref) * 100
        null_pct_raw_fut = (null_raw_fut / n_time_fut) * 100
        null_pct_bc_fut = (null_bc_fut / n_time_fut) * 100

        has_cerra = da_cerra_ref is not None
        n_cols = 3 if has_cerra else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

        # Determine appropriate colorbar range (use max across all datasets, or cap at 100%)
        all_null_pcts = [
            null_pct_raw_ref,
            null_pct_bc_ref,
            null_pct_raw_fut,
            null_pct_bc_fut,
        ]
        if has_cerra:
            null_pct_cerra_ref = (null_cerra_ref / n_time_ref) * 100
            all_null_pcts.append(null_pct_cerra_ref)

        vmax_null = min(max([pct.max().values for pct in all_null_pcts]), 100)
        # If max is very small, set a reasonable minimum for visibility
        if vmax_null < 1:
            vmax_null = max(vmax_null, 1)

        # Reference period (top row)
        im1 = null_pct_raw_ref.plot(
            ax=axes[0, 0], cmap="YlOrRd", vmin=0, vmax=vmax_null, add_colorbar=False
        )
        axes[0, 0].set_title(
            f"Raw - Reference\n({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )
        plt.colorbar(im1, ax=axes[0, 0], label="% Null Values")

        im2 = null_pct_bc_ref.plot(
            ax=axes[0, 1], cmap="YlOrRd", vmin=0, vmax=vmax_null, add_colorbar=False
        )
        axes[0, 1].set_title(
            f"Bias-Corrected - Reference\n({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )
        plt.colorbar(im2, ax=axes[0, 1], label="% Null Values")

        if has_cerra:
            im3 = null_pct_cerra_ref.plot(
                ax=axes[0, 2], cmap="YlOrRd", vmin=0, vmax=vmax_null, add_colorbar=False
            )
            axes[0, 2].set_title(
                f"CERRA Reference\n({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
            )
            plt.colorbar(im3, ax=axes[0, 2], label="% Null Values")

        # Future period (bottom row)
        im4 = null_pct_raw_fut.plot(
            ax=axes[1, 0], cmap="YlOrRd", vmin=0, vmax=vmax_null, add_colorbar=False
        )
        axes[1, 0].set_title(f"Raw - Future\n({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})")
        plt.colorbar(im4, ax=axes[1, 0], label="% Null Values")

        im5 = null_pct_bc_fut.plot(
            ax=axes[1, 1], cmap="YlOrRd", vmin=0, vmax=vmax_null, add_colorbar=False
        )
        axes[1, 1].set_title(
            f"Bias-Corrected - Future\n({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})"
        )
        plt.colorbar(im5, ax=axes[1, 1], label="% Null Values")

        if has_cerra:
            # Show summary statistics for null values
            total_ref = null_pct_raw_ref.size
            null_count_raw_ref = (null_pct_raw_ref > 0).sum().values
            null_count_bc_ref = (null_pct_bc_ref > 0).sum().values
            null_count_raw_fut = (null_pct_raw_fut > 0).sum().values
            null_count_bc_fut = (null_pct_bc_fut > 0).sum().values

            axes[1, 2].axis("off")
            summary_text = (
                f"Null Value Summary\n\n"
                f"Reference Period ({n_time_ref} timesteps):\n"
                f"  Raw: {null_count_raw_ref}/{total_ref} locations\n"
                f"       ({null_count_raw_ref/total_ref*100:.1f}%)\n"
                f"       {timesteps_with_nulls_raw_ref:.0f} timesteps affected\n"
                f"  BC:  {null_count_bc_ref}/{total_ref} locations\n"
                f"       ({null_count_bc_ref/total_ref*100:.1f}%)\n"
                f"       {timesteps_with_nulls_bc_ref:.0f} timesteps affected\n"
                f"  CERRA: {timesteps_with_nulls_cerra_ref:.0f} timesteps affected\n\n"
                f"Future Period ({n_time_fut} timesteps):\n"
                f"  Raw: {null_count_raw_fut}/{total_ref} locations\n"
                f"       ({null_count_raw_fut/total_ref*100:.1f}%)\n"
                f"       {timesteps_with_nulls_raw_fut:.0f} timesteps affected\n"
                f"  BC:  {null_count_bc_fut}/{total_ref} locations\n"
                f"       ({null_count_bc_fut/total_ref*100:.1f}%)\n"
                f"       {timesteps_with_nulls_bc_fut:.0f} timesteps affected"
            )
            axes[1, 2].text(
                0.1,
                0.5,
                summary_text,
                fontsize=10,
                verticalalignment="center",
                family="monospace",
            )

        plt.suptitle(
            f"{config['name']} - Null Value Patterns\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_null_patterns.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_spatial_patterns(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot spatial mean patterns for reference and future periods."""
        with ProgressBar(dt=2):
            mean_raw_ref = da_raw_ref.mean(dim="time").compute()
            mean_bc_ref = da_bc_ref.mean(dim="time").compute()
            mean_raw_fut = da_raw_fut.mean(dim="time").compute()
            mean_bc_fut = da_bc_fut.mean(dim="time").compute()
            if da_cerra_ref is not None:
                mean_cerra_ref = da_cerra_ref.mean(dim="time").compute()

        has_cerra = da_cerra_ref is not None
        n_cols = 3 if has_cerra else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

        # Reference period (top row)
        mean_raw_ref.plot(ax=axes[0, 0], cmap="RdYlBu_r")
        axes[0, 0].set_title(
            f"Raw - Reference\n({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )

        mean_bc_ref.plot(ax=axes[0, 1], cmap="RdYlBu_r")
        axes[0, 1].set_title(
            f"Bias-Corrected - Reference\n({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )

        if has_cerra:
            mean_cerra_ref.plot(ax=axes[0, 2], cmap="RdYlBu_r")
            axes[0, 2].set_title(
                f"CERRA Reference\n({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
            )

        # Future period (bottom row)
        mean_raw_fut.plot(ax=axes[1, 0], cmap="RdYlBu_r")
        axes[1, 0].set_title(f"Raw - Future\n({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})")

        mean_bc_fut.plot(ax=axes[1, 1], cmap="RdYlBu_r")
        axes[1, 1].set_title(
            f"Bias-Corrected - Future\n({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})"
        )

        if has_cerra:
            # Show difference for future period
            diff_fut = mean_bc_fut - mean_bc_ref
            vmax = np.abs(diff_fut).quantile(0.95).values
            diff_fut.plot(ax=axes[1, 2], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[1, 2].set_title(f"Future - Reference\nChange (Bias-Corrected)")

        plt.suptitle(
            f"{config['name']} ({config['units_display']}) - Spatial Mean\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_spatial_mean.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_temporal_statistics(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot temporal statistics: yearly and monthly means as boxplots."""
        print("    Computing yearly and monthly means...")

        with ProgressBar(dt=2):
            # Compute yearly means (spatial average first, then temporal grouping)
            yearly_raw_ref = (
                da_raw_ref.mean(dim=["lat", "lon"])
                .groupby("time.year")
                .mean(dim="time")
                .compute()
            )
            yearly_bc_ref = (
                da_bc_ref.mean(dim=["lat", "lon"])
                .groupby("time.year")
                .mean(dim="time")
                .compute()
            )
            yearly_raw_fut = (
                da_raw_fut.mean(dim=["lat", "lon"])
                .groupby("time.year")
                .mean(dim="time")
                .compute()
            )
            yearly_bc_fut = (
                da_bc_fut.mean(dim=["lat", "lon"])
                .groupby("time.year")
                .mean(dim="time")
                .compute()
            )

            # Compute all monthly values (for each month separately across all years)
            # Get values for each month individually
            monthly_values_raw_ref = {}
            monthly_values_bc_ref = {}
            monthly_values_raw_fut = {}
            monthly_values_bc_fut = {}

            for month in range(1, 13):
                monthly_values_raw_ref[month] = (
                    da_raw_ref.sel(time=da_raw_ref.time.dt.month == month)
                    .mean(dim=["lat", "lon"])
                    .compute()
                    .values.ravel()
                )
                monthly_values_bc_ref[month] = (
                    da_bc_ref.sel(time=da_bc_ref.time.dt.month == month)
                    .mean(dim=["lat", "lon"])
                    .compute()
                    .values.ravel()
                )
                monthly_values_raw_fut[month] = (
                    da_raw_fut.sel(time=da_raw_fut.time.dt.month == month)
                    .mean(dim=["lat", "lon"])
                    .compute()
                    .values.ravel()
                )
                monthly_values_bc_fut[month] = (
                    da_bc_fut.sel(time=da_bc_fut.time.dt.month == month)
                    .mean(dim=["lat", "lon"])
                    .compute()
                    .values.ravel()
                )

            if da_cerra_ref is not None:
                yearly_cerra_ref = (
                    da_cerra_ref.mean(dim=["lat", "lon"])
                    .groupby("time.year")
                    .mean(dim="time")
                    .compute()
                )
                monthly_values_cerra_ref = {}
                for month in range(1, 13):
                    monthly_values_cerra_ref[month] = (
                        da_cerra_ref.sel(time=da_cerra_ref.time.dt.month == month)
                        .mean(dim=["lat", "lon"])
                        .compute()
                        .values.ravel()
                    )

        # Create figure with 2 subplots: yearly and monthly
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))

        # Prepare data for yearly boxplots
        yearly_data = [
            yearly_raw_ref.values.ravel(),
            yearly_bc_ref.values.ravel(),
            yearly_raw_fut.values.ravel(),
            yearly_bc_fut.values.ravel(),
        ]
        yearly_labels = ["Raw\nRef", "BC\nRef", "Raw\nFut", "BC\nFut"]
        yearly_colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]

        if da_cerra_ref is not None:
            yearly_data.insert(2, yearly_cerra_ref.values.ravel())
            yearly_labels.insert(2, "CERRA\nRef")
            yearly_colors.insert(2, "#2ca02c")

        # Plot yearly boxplots
        bp1 = axes[0].boxplot(
            yearly_data,
            labels=yearly_labels,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops=dict(
                marker="o", markerfacecolor="gray", markersize=4, alpha=0.5
            ),
        )

        # Color the boxes
        for patch, color in zip(bp1["boxes"], yearly_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        axes[0].set_ylabel(f"{config['name']} ({config['units_display']})")
        axes[0].set_title(
            f"Yearly Means\n({len(yearly_raw_ref)} years ref, {len(yearly_raw_fut)} years fut)"
        )
        axes[0].grid(True, alpha=0.3, axis="y")

        # Add period separator
        if da_cerra_ref is not None:
            axes[0].axvline(
                x=3.5, color="black", linestyle="--", linewidth=1, alpha=0.5
            )
        else:
            axes[0].axvline(
                x=2.5, color="black", linestyle="--", linewidth=1, alpha=0.5
            )

        # Plot monthly boxplots - one group per month
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        # Prepare positions for each dataset within each month
        n_datasets = 5 if da_cerra_ref is not None else 4
        dataset_width = 0.15
        group_width = n_datasets * dataset_width + 0.2

        positions_all = []
        data_all = []
        colors_all = []

        for month_idx, month in enumerate(range(1, 13)):
            base_pos = month_idx * group_width

            # Raw Ref
            positions_all.append(base_pos)
            data_all.append(monthly_values_raw_ref[month])
            colors_all.append("#1f77b4")

            # BC Ref
            positions_all.append(base_pos + dataset_width)
            data_all.append(monthly_values_bc_ref[month])
            colors_all.append("#ff7f0e")

            if da_cerra_ref is not None:
                # CERRA Ref
                positions_all.append(base_pos + 2 * dataset_width)
                data_all.append(monthly_values_cerra_ref[month])
                colors_all.append("#2ca02c")

                # Raw Fut
                positions_all.append(base_pos + 3 * dataset_width)
                data_all.append(monthly_values_raw_fut[month])
                colors_all.append("#1f77b4")

                # BC Fut
                positions_all.append(base_pos + 4 * dataset_width)
                data_all.append(monthly_values_bc_fut[month])
                colors_all.append("#ff7f0e")
            else:
                # Raw Fut
                positions_all.append(base_pos + 2 * dataset_width)
                data_all.append(monthly_values_raw_fut[month])
                colors_all.append("#1f77b4")

                # BC Fut
                positions_all.append(base_pos + 3 * dataset_width)
                data_all.append(monthly_values_bc_fut[month])
                colors_all.append("#ff7f0e")

        # Plot all monthly boxplots
        bp2 = axes[1].boxplot(
            data_all,
            positions=positions_all,
            patch_artist=True,
            widths=dataset_width * 0.8,
            showfliers=False,
            manage_ticks=False,
        )

        # Color the boxes
        for patch, color in zip(bp2["boxes"], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Set x-axis ticks at the center of each month group
        month_centers = [
            i * group_width + (n_datasets - 1) * dataset_width / 2 for i in range(12)
        ]
        axes[1].set_xticks(month_centers)
        axes[1].set_xticklabels(month_names)

        axes[1].set_ylabel(f"{config['name']} ({config['units_display']})")
        axes[1].set_title("Monthly Means (per month across all years)")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", alpha=0.6, label="Raw Ref"),
            Patch(facecolor="#ff7f0e", alpha=0.6, label="BC Ref"),
        ]
        if da_cerra_ref is not None:
            legend_elements.append(
                Patch(facecolor="#2ca02c", alpha=0.6, label="CERRA Ref")
            )
        legend_elements.extend(
            [
                Patch(facecolor="#1f77b4", alpha=0.6, label="Raw Fut"),
                Patch(facecolor="#ff7f0e", alpha=0.6, label="BC Fut"),
            ]
        )
        axes[1].legend(handles=legend_elements, loc="best", ncol=5)

        plt.suptitle(
            f"{config['name']} - Temporal Statistics\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_temporal_statistics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_distributions(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot probability distributions using computed histograms for both periods."""
        # Compute quantiles to determine bin range
        # Rechunk to single chunk for quantile computation
        print("    Computing distribution ranges...")
        with ProgressBar(dt=2):
            da_raw_ref_rechunk = da_raw_ref.chunk({"time": -1, "lat": -1, "lon": -1})
            da_bc_ref_rechunk = da_bc_ref.chunk({"time": -1, "lat": -1, "lon": -1})
            da_raw_fut_rechunk = da_raw_fut.chunk({"time": -1, "lat": -1, "lon": -1})
            da_bc_fut_rechunk = da_bc_fut.chunk({"time": -1, "lat": -1, "lon": -1})

            q_min = min(
                da_raw_ref_rechunk.quantile(0.01).compute().values,
                da_bc_ref_rechunk.quantile(0.01).compute().values,
                da_raw_fut_rechunk.quantile(0.01).compute().values,
                da_bc_fut_rechunk.quantile(0.01).compute().values,
            )
            q_max = max(
                da_raw_ref_rechunk.quantile(0.99).compute().values,
                da_bc_ref_rechunk.quantile(0.99).compute().values,
                da_raw_fut_rechunk.quantile(0.99).compute().values,
                da_bc_fut_rechunk.quantile(0.99).compute().values,
            )
            if da_cerra_ref is not None:
                da_cerra_ref_rechunk = da_cerra_ref.chunk(
                    {"time": -1, "lat": -1, "lon": -1}
                )
                q_min = min(q_min, da_cerra_ref_rechunk.quantile(0.01).compute().values)
                q_max = max(q_max, da_cerra_ref_rechunk.quantile(0.99).compute().values)

        # Define bins
        bins = np.linspace(q_min, q_max, 101)

        # Compute histograms using xarray.histogram (works with dask)
        print("    Computing histograms...")
        with ProgressBar(dt=2):
            hist_raw_ref = xr.apply_ufunc(
                lambda x: np.histogram(x[~np.isnan(x)], bins=bins, density=True)[0],
                da_raw_ref,
                input_core_dims=[["time", "lat", "lon"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            ).compute()

            hist_bc_ref = xr.apply_ufunc(
                lambda x: np.histogram(x[~np.isnan(x)], bins=bins, density=True)[0],
                da_bc_ref,
                input_core_dims=[["time", "lat", "lon"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            ).compute()

            hist_raw_fut = xr.apply_ufunc(
                lambda x: np.histogram(x[~np.isnan(x)], bins=bins, density=True)[0],
                da_raw_fut,
                input_core_dims=[["time", "lat", "lon"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            ).compute()

            hist_bc_fut = xr.apply_ufunc(
                lambda x: np.histogram(x[~np.isnan(x)], bins=bins, density=True)[0],
                da_bc_fut,
                input_core_dims=[["time", "lat", "lon"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            ).compute()

            if da_cerra_ref is not None:
                hist_cerra_ref = xr.apply_ufunc(
                    lambda x: np.histogram(x[~np.isnan(x)], bins=bins, density=True)[0],
                    da_cerra_ref,
                    input_core_dims=[["time", "lat", "lon"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                ).compute()

        bin_centers = (bins[:-1] + bins[1:]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Reference period
        axes[0].plot(bin_centers, hist_raw_ref, alpha=0.7, label="Raw", linewidth=2)
        axes[0].fill_between(bin_centers, 0, hist_raw_ref, alpha=0.3)
        axes[0].plot(
            bin_centers, hist_bc_ref, alpha=0.7, label="Bias-Corrected", linewidth=2
        )
        axes[0].fill_between(bin_centers, 0, hist_bc_ref, alpha=0.3)
        if da_cerra_ref is not None:
            axes[0].plot(
                bin_centers, hist_cerra_ref, alpha=0.7, label="CERRA", linewidth=2
            )
            axes[0].fill_between(bin_centers, 0, hist_cerra_ref, alpha=0.3)
        axes[0].set_xlabel(f"{config['name']} ({config['units_display']})")
        axes[0].set_ylabel("Probability Density")
        axes[0].set_title(
            f"Reference Period ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Future period
        axes[1].plot(bin_centers, hist_raw_fut, alpha=0.7, label="Raw", linewidth=2)
        axes[1].fill_between(bin_centers, 0, hist_raw_fut, alpha=0.3)
        axes[1].plot(
            bin_centers, hist_bc_fut, alpha=0.7, label="Bias-Corrected", linewidth=2
        )
        axes[1].fill_between(bin_centers, 0, hist_bc_fut, alpha=0.3)
        axes[1].set_xlabel(f"{config['name']} ({config['units_display']})")
        axes[1].set_ylabel("Probability Density")
        axes[1].set_title(f"Future Period ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"{config['name']} - Probability Distribution\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_extremes(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot extreme values (95th and 99th percentiles) for both periods."""
        # Rechunk along time dimension for quantile computation
        with ProgressBar(dt=2):
            da_raw_ref_rechunk = da_raw_ref.chunk({"time": -1})
            da_bc_ref_rechunk = da_bc_ref.chunk({"time": -1})
            da_raw_fut_rechunk = da_raw_fut.chunk({"time": -1})
            da_bc_fut_rechunk = da_bc_fut.chunk({"time": -1})

            p95_raw_ref = da_raw_ref_rechunk.quantile(0.95, dim="time").compute()
            p95_bc_ref = da_bc_ref_rechunk.quantile(0.95, dim="time").compute()
            p99_raw_ref = da_raw_ref_rechunk.quantile(0.99, dim="time").compute()
            p99_bc_ref = da_bc_ref_rechunk.quantile(0.99, dim="time").compute()

            p95_raw_fut = da_raw_fut_rechunk.quantile(0.95, dim="time").compute()
            p95_bc_fut = da_bc_fut_rechunk.quantile(0.95, dim="time").compute()
            p99_raw_fut = da_raw_fut_rechunk.quantile(0.99, dim="time").compute()
            p99_bc_fut = da_bc_fut_rechunk.quantile(0.99, dim="time").compute()

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Reference period - 95th percentile
        p95_raw_ref.plot(ax=axes[0, 0], cmap="RdYlBu_r")
        axes[0, 0].set_title(
            f"Raw - 95th Percentile\nReference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )

        p95_bc_ref.plot(ax=axes[0, 1], cmap="RdYlBu_r")
        axes[0, 1].set_title(
            f"Bias-Corrected - 95th Percentile\nReference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )

        # Reference period - 99th percentile
        p99_raw_ref.plot(ax=axes[0, 2], cmap="RdYlBu_r")
        axes[0, 2].set_title(
            f"Raw - 99th Percentile\nReference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )

        p99_bc_ref.plot(ax=axes[0, 3], cmap="RdYlBu_r")
        axes[0, 3].set_title(
            f"Bias-Corrected - 99th Percentile\nReference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )

        # Future period - 95th percentile
        p95_raw_fut.plot(ax=axes[1, 0], cmap="RdYlBu_r")
        axes[1, 0].set_title(
            f"Raw - 95th Percentile\nFuture ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})"
        )

        p95_bc_fut.plot(ax=axes[1, 1], cmap="RdYlBu_r")
        axes[1, 1].set_title(
            f"Bias-Corrected - 95th Percentile\nFuture ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})"
        )

        # Future period - 99th percentile
        p99_raw_fut.plot(ax=axes[1, 2], cmap="RdYlBu_r")
        axes[1, 2].set_title(
            f"Raw - 99th Percentile\nFuture ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})"
        )

        p99_bc_fut.plot(ax=axes[1, 3], cmap="RdYlBu_r")
        axes[1, 3].set_title(
            f"Bias-Corrected - 99th Percentile\nFuture ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})"
        )

        plt.suptitle(
            f"{config['name']} ({config['units_display']}) - Extreme Values\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(fig_dir / f"{variable}_extremes.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_seasonal_cycle(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot seasonal cycle for both periods."""
        with ProgressBar(dt=2):
            monthly_raw_ref = (
                da_raw_ref.groupby("time.month")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            monthly_bc_ref = (
                da_bc_ref.groupby("time.month")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            monthly_raw_fut = (
                da_raw_fut.groupby("time.month")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            monthly_bc_fut = (
                da_bc_fut.groupby("time.month")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            if da_cerra_ref is not None:
                monthly_cerra_ref = (
                    da_cerra_ref.groupby("time.month")
                    .mean(dim="time")
                    .mean(dim=["lat", "lon"])
                    .compute()
                )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Reference period
        axes[0].plot(
            monthly_raw_ref.month,
            monthly_raw_ref.values,
            marker="o",
            label="Raw",
            linewidth=2,
        )
        axes[0].plot(
            monthly_bc_ref.month,
            monthly_bc_ref.values,
            marker="s",
            label="Bias-Corrected",
            linewidth=2,
        )
        if da_cerra_ref is not None:
            axes[0].plot(
                monthly_cerra_ref.month,
                monthly_cerra_ref.values,
                marker="^",
                label="CERRA",
                linewidth=2,
            )
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel(f"{config['name']} ({config['units_display']})")
        axes[0].set_title(
            f"Reference Period ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )
        axes[0].set_xticks(range(1, 13))
        axes[0].set_xticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Future period
        axes[1].plot(
            monthly_raw_fut.month,
            monthly_raw_fut.values,
            marker="o",
            label="Raw",
            linewidth=2,
        )
        axes[1].plot(
            monthly_bc_fut.month,
            monthly_bc_fut.values,
            marker="s",
            label="Bias-Corrected",
            linewidth=2,
        )
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel(f"{config['name']} ({config['units_display']})")
        axes[1].set_title(f"Future Period ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})")
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(
            ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"{config['name']} - Seasonal Cycle\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_seasonal_cycle.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_diurnal_cycle(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot diurnal cycle (for sub-daily data) for both periods."""
        with ProgressBar(dt=2):
            hourly_raw_ref = (
                da_raw_ref.groupby("time.hour")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            hourly_bc_ref = (
                da_bc_ref.groupby("time.hour")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            hourly_raw_fut = (
                da_raw_fut.groupby("time.hour")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            hourly_bc_fut = (
                da_bc_fut.groupby("time.hour")
                .mean(dim="time")
                .mean(dim=["lat", "lon"])
                .compute()
            )
            if da_cerra_ref is not None:
                hourly_cerra_ref = (
                    da_cerra_ref.groupby("time.hour")
                    .mean(dim="time")
                    .mean(dim=["lat", "lon"])
                    .compute()
                )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Reference period
        axes[0].plot(
            hourly_raw_ref.hour,
            hourly_raw_ref.values,
            marker="o",
            label="Raw",
            linewidth=2,
        )
        axes[0].plot(
            hourly_bc_ref.hour,
            hourly_bc_ref.values,
            marker="s",
            label="Bias-Corrected",
            linewidth=2,
        )
        if da_cerra_ref is not None:
            axes[0].plot(
                hourly_cerra_ref.hour,
                hourly_cerra_ref.values,
                marker="^",
                label="CERRA",
                linewidth=2,
            )
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel(f"{config['name']} ({config['units_display']})")
        axes[0].set_title(
            f"Reference Period ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})"
        )
        axes[0].set_xticks(range(0, 24, 3))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Future period
        axes[1].plot(
            hourly_raw_fut.hour,
            hourly_raw_fut.values,
            marker="o",
            label="Raw",
            linewidth=2,
        )
        axes[1].plot(
            hourly_bc_fut.hour,
            hourly_bc_fut.values,
            marker="s",
            label="Bias-Corrected",
            linewidth=2,
        )
        axes[1].set_xlabel("Hour of Day")
        axes[1].set_ylabel(f"{config['name']} ({config['units_display']})")
        axes[1].set_title(f"Future Period ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})")
        axes[1].set_xticks(range(0, 24, 3))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f"{config['name']} - Diurnal Cycle\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_diurnal_cycle.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_threshold_exceedances(
        self,
        da_raw_ref,
        da_bc_ref,
        da_cerra_ref,
        da_raw_fut,
        da_bc_fut,
        variable,
        config,
        fig_dir,
    ):
        """Plot frequency of threshold exceedances for both periods."""
        thresholds = config.get("thresholds", {})
        if not thresholds:
            return

        # Determine number of rows based on CERRA availability
        n_rows = 3 if da_cerra_ref is not None else 2
        fig, axes = plt.subplots(
            n_rows, len(thresholds) * 2, figsize=(6 * len(thresholds) * 2, 5 * n_rows)
        )
        if len(thresholds) == 1:
            axes = axes.reshape(n_rows, 2)

        for idx, (thr_name, thr_value) in enumerate(thresholds.items()):
            # Determine if this is a "high/hot" threshold (exceedances above) or "low/cold" threshold (counts below)
            is_high_threshold = thr_name in [
                "hot",
                "high",
                "moderate",
                "strong",
                "heavy",
                "extreme",
            ]

            # Thresholds already in display units
            thr_compare = thr_value

            with ProgressBar(dt=2):
                # Compute exceedances/counts based on threshold type
                if is_high_threshold:
                    # Exceedances above threshold
                    exc_raw_ref = (da_raw_ref > thr_compare).sum(dim="time").compute()
                    exc_bc_ref = (da_bc_ref > thr_compare).sum(dim="time").compute()
                    exc_raw_fut = (da_raw_fut > thr_compare).sum(dim="time").compute()
                    exc_bc_fut = (da_bc_fut > thr_compare).sum(dim="time").compute()
                    if da_cerra_ref is not None:
                        exc_cerra_ref = (
                            (da_cerra_ref > thr_compare).sum(dim="time").compute()
                        )
                    comparison_str = ">"
                else:
                    # Counts below threshold
                    exc_raw_ref = (da_raw_ref < thr_compare).sum(dim="time").compute()
                    exc_bc_ref = (da_bc_ref < thr_compare).sum(dim="time").compute()
                    exc_raw_fut = (da_raw_fut < thr_compare).sum(dim="time").compute()
                    exc_bc_fut = (da_bc_fut < thr_compare).sum(dim="time").compute()
                    if da_cerra_ref is not None:
                        exc_cerra_ref = (
                            (da_cerra_ref < thr_compare).sum(dim="time").compute()
                        )
                    comparison_str = "<"

            # Determine common colorbar range across all datasets
            all_values_ref = [exc_raw_ref, exc_bc_ref]
            all_values_fut = [exc_raw_fut, exc_bc_fut]
            if da_cerra_ref is not None:
                all_values_ref.append(exc_cerra_ref)

            vmin = min([v.min().values for v in all_values_ref + all_values_fut])
            vmax = max([v.max().values for v in all_values_ref + all_values_fut])

            # Reference period (left column)
            col_ref = idx * 2

            # Raw
            im = exc_raw_ref.plot(
                ax=axes[0, col_ref],
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            axes[0, col_ref].set_title(
                f"Raw - Reference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})\n"
                f"Count {comparison_str} {thr_compare:.1f} {config['units_display']}"
            )
            plt.colorbar(im, ax=axes[0, col_ref], label="Count")

            # Bias-Corrected
            im = exc_bc_ref.plot(
                ax=axes[1, col_ref],
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            axes[1, col_ref].set_title(
                f"Bias-Corrected - Reference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})\n"
                f"Count {comparison_str} {thr_compare:.1f} {config['units_display']}"
            )
            plt.colorbar(im, ax=axes[1, col_ref], label="Count")

            # CERRA (if available)
            if da_cerra_ref is not None:
                im = exc_cerra_ref.plot(
                    ax=axes[2, col_ref],
                    cmap="YlOrRd",
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False,
                )
                axes[2, col_ref].set_title(
                    f"CERRA - Reference ({REFERENCE_PERIOD[0]}-{REFERENCE_PERIOD[1]})\n"
                    f"Count {comparison_str} {thr_compare:.1f} {config['units_display']}"
                )
                plt.colorbar(im, ax=axes[2, col_ref], label="Count")

            # Future period (right column)
            col_fut = idx * 2 + 1

            # Raw
            im = exc_raw_fut.plot(
                ax=axes[0, col_fut],
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            axes[0, col_fut].set_title(
                f"Raw - Future ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})\n"
                f"Count {comparison_str} {thr_compare:.1f} {config['units_display']}"
            )
            plt.colorbar(im, ax=axes[0, col_fut], label="Count")

            # Bias-Corrected
            im = exc_bc_fut.plot(
                ax=axes[1, col_fut],
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            axes[1, col_fut].set_title(
                f"Bias-Corrected - Future ({FUTURE_PERIOD[0]}-{FUTURE_PERIOD[1]})\n"
                f"Count {comparison_str} {thr_compare:.1f} {config['units_display']}"
            )
            plt.colorbar(im, ax=axes[1, col_fut], label="Count")

            # Empty plot for CERRA future (not available)
            if da_cerra_ref is not None:
                axes[2, col_fut].axis("off")
                axes[2, col_fut].text(
                    0.5,
                    0.5,
                    "CERRA\n(No future data)",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=axes[2, col_fut].transAxes,
                )

        plt.suptitle(
            f"{config['name']} - Threshold Exceedances\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"{variable}_threshold_exceedances.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_summary_statistics(self, variables: list = None):
        """Create a comprehensive summary plot for all variables."""
        if variables is None:
            variables = list(VARIABLE_CONFIG.keys())

        print(f"\n{'='*80}")
        print("CREATING SUMMARY STATISTICS PLOT")
        print(f"Variables: {', '.join(variables)}")
        print(f"{'='*80}")

        # Collect statistics for all variables
        stats_data = {}

        for variable in variables:
            if variable not in VARIABLE_CONFIG:
                print(f"Warning: Unknown variable {variable}, skipping")
                continue

            try:
                print(f"\n  Processing {variable}...")
                config = VARIABLE_CONFIG[variable]
                temp_res = (
                    self.temp_res_override
                    if self.temp_res_override
                    else config["temp_res"]
                )

                # Load data for both periods
                da_raw_ref, da_bc_ref, da_cerra_ref = self.load_data(
                    variable, period="reference"
                )
                da_raw_fut, da_bc_fut, _ = self.load_data(variable, period="future")

                with ProgressBar(dt=2):
                    # Compute statistics
                    # 1. Timesteps with null values
                    null_timesteps_raw_ref = (
                        da_raw_ref.isnull()
                        .any(dim=["lat", "lon"])
                        .sum()
                        .compute()
                        .values
                    )
                    null_timesteps_bc_ref = (
                        da_bc_ref.isnull()
                        .any(dim=["lat", "lon"])
                        .sum()
                        .compute()
                        .values
                    )
                    null_timesteps_raw_fut = (
                        da_raw_fut.isnull()
                        .any(dim=["lat", "lon"])
                        .sum()
                        .compute()
                        .values
                    )
                    null_timesteps_bc_fut = (
                        da_bc_fut.isnull()
                        .any(dim=["lat", "lon"])
                        .sum()
                        .compute()
                        .values
                    )

                    # 2. Min and Max values
                    min_raw_ref = da_raw_ref.min().compute().values
                    max_raw_ref = da_raw_ref.max().compute().values
                    min_bc_ref = da_bc_ref.min().compute().values
                    max_bc_ref = da_bc_ref.max().compute().values
                    min_raw_fut = da_raw_fut.min().compute().values
                    max_raw_fut = da_raw_fut.max().compute().values
                    min_bc_fut = da_bc_fut.min().compute().values
                    max_bc_fut = da_bc_fut.max().compute().values

                    # 3. Threshold exceedances (total count across all grid points and timesteps)
                    threshold_counts = {}
                    if config.get("thresholds"):
                        for thr_name, thr_value in config["thresholds"].items():
                            is_high_threshold = thr_name in [
                                "hot",
                                "high",
                                "moderate",
                                "strong",
                                "heavy",
                                "extreme",
                            ]

                            if is_high_threshold:
                                threshold_counts[thr_name] = {
                                    "raw_ref": (da_raw_ref > thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                    "bc_ref": (da_bc_ref > thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                    "raw_fut": (da_raw_fut > thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                    "bc_fut": (da_bc_fut > thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                }
                            else:
                                threshold_counts[thr_name] = {
                                    "raw_ref": (da_raw_ref < thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                    "bc_ref": (da_bc_ref < thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                    "raw_fut": (da_raw_fut < thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                    "bc_fut": (da_bc_fut < thr_value)
                                    .sum()
                                    .compute()
                                    .values,
                                }

                stats_data[variable] = {
                    "config": config,
                    "null_timesteps": {
                        "raw_ref": null_timesteps_raw_ref,
                        "bc_ref": null_timesteps_bc_ref,
                        "raw_fut": null_timesteps_raw_fut,
                        "bc_fut": null_timesteps_bc_fut,
                    },
                    "min_values": {
                        "raw_ref": min_raw_ref,
                        "bc_ref": min_bc_ref,
                        "raw_fut": min_raw_fut,
                        "bc_fut": min_bc_fut,
                    },
                    "max_values": {
                        "raw_ref": max_raw_ref,
                        "bc_ref": max_bc_ref,
                        "raw_fut": max_raw_fut,
                        "bc_fut": max_bc_fut,
                    },
                    "thresholds": threshold_counts,
                }

            except Exception as e:
                print(f"    ✗ Error processing {variable}: {str(e)}")
                continue

        # Create the summary plot
        n_vars = len(stats_data)
        if n_vars == 0:
            print("  No data to plot!")
            return

        # Determine number of columns (5: null timesteps, min, max, threshold 1, threshold 2)
        n_cols = 5
        fig, axes = plt.subplots(n_vars, n_cols, figsize=(28, 4 * n_vars))

        # Ensure axes is 2D
        if n_vars == 1:
            axes = axes.reshape(1, -1)

        var_names = list(stats_data.keys())

        for idx, variable in enumerate(var_names):
            data = stats_data[variable]
            config = data["config"]

            # Column 0: Null timesteps
            ax = axes[idx, 0]
            x_pos = np.arange(4)
            values = [
                data["null_timesteps"]["raw_ref"],
                data["null_timesteps"]["bc_ref"],
                data["null_timesteps"]["raw_fut"],
                data["null_timesteps"]["bc_fut"],
            ]
            colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                ["Raw\nRef", "BC\nRef", "Raw\nFut", "BC\nFut"], fontsize=9
            )
            ax.set_ylabel("Timesteps with Nulls", fontsize=10)
            ax.set_title(
                f"{variable.upper()}\nNull Timesteps", fontsize=11, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="y")
            # Add values on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(val)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Column 1: Min values
            ax = axes[idx, 1]
            values = [
                data["min_values"]["raw_ref"],
                data["min_values"]["bc_ref"],
                data["min_values"]["raw_fut"],
                data["min_values"]["bc_fut"],
            ]
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                ["Raw\nRef", "BC\nRef", "Raw\nFut", "BC\nFut"], fontsize=9
            )
            ax.set_ylabel(f'Min ({config["units_display"]})', fontsize=10)
            ax.set_title(
                f"{variable.upper()}\nMinimum Value", fontsize=11, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="y")
            # Add values on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                y_pos = (
                    height
                    if height > 0
                    else height - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                )
                va = "bottom" if height > 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos,
                    f"{val:.1f}",
                    ha="center",
                    va=va,
                    fontsize=8,
                )

            # Column 2: Max values
            ax = axes[idx, 2]
            values = [
                data["max_values"]["raw_ref"],
                data["max_values"]["bc_ref"],
                data["max_values"]["raw_fut"],
                data["max_values"]["bc_fut"],
            ]
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                ["Raw\nRef", "BC\nRef", "Raw\nFut", "BC\nFut"], fontsize=9
            )
            ax.set_ylabel(f'Max ({config["units_display"]})', fontsize=10)
            ax.set_title(
                f"{variable.upper()}\nMaximum Value", fontsize=11, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="y")
            # Add values on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Columns 3-4: Threshold exceedances (split across two subplots)
            if data["thresholds"]:
                threshold_items = list(data["thresholds"].items())
                n_thresholds = len(threshold_items)

                # Split thresholds across two columns
                thresholds_per_col = (n_thresholds + 1) // 2  # Ceiling division

                for col_offset in range(2):
                    ax = axes[idx, 3 + col_offset]
                    start_idx = col_offset * thresholds_per_col
                    end_idx = min(start_idx + thresholds_per_col, n_thresholds)

                    if start_idx < n_thresholds:
                        col_thresholds = threshold_items[start_idx:end_idx]
                        n_col_thresholds = len(col_thresholds)

                        if n_col_thresholds > 0:
                            bar_width = 0.18
                            group_spacing = 0.1

                            # Collect threshold names for title
                            threshold_names_for_title = []
                            for thr_name, _ in col_thresholds:
                                thr_value = config["thresholds"][thr_name]
                                is_high = thr_name in [
                                    "hot",
                                    "high",
                                    "moderate",
                                    "strong",
                                    "heavy",
                                    "extreme",
                                ]
                                comp = ">" if is_high else "<"
                                threshold_names_for_title.append(
                                    f"{thr_name} {comp}{thr_value}"
                                )

                            # Plot each threshold as a group of 4 bars
                            all_positions = []
                            all_values = []
                            for thr_idx, (thr_name, thr_data) in enumerate(
                                col_thresholds
                            ):
                                base_pos = thr_idx * (4 * bar_width + group_spacing)
                                positions = [base_pos + i * bar_width for i in range(4)]

                                values = [
                                    thr_data["raw_ref"],
                                    thr_data["bc_ref"],
                                    thr_data["raw_fut"],
                                    thr_data["bc_fut"],
                                ]

                                all_positions.extend(positions)
                                all_values.extend(values)

                            # Plot all bars
                            bars = ax.bar(
                                all_positions,
                                all_values,
                                width=bar_width,
                                color=colors * n_col_thresholds,
                                alpha=0.7,
                            )

                            # Add values on bars
                            for bar, val in zip(bars, all_values):
                                height = bar.get_height()
                                ax.text(
                                    bar.get_x() + bar.get_width() / 2.0,
                                    height,
                                    f"{int(val)}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=7,
                                )

                            # Set x-axis labels - one set of data labels per threshold group
                            all_tick_positions = []
                            all_tick_labels = []
                            data_labels = ["Raw\nRef", "BC\nRef", "Raw\nFut", "BC\nFut"]

                            for thr_idx in range(n_col_thresholds):
                                base_pos = thr_idx * (4 * bar_width + group_spacing)
                                tick_positions = [
                                    base_pos + i * bar_width for i in range(4)
                                ]
                                all_tick_positions.extend(tick_positions)
                                all_tick_labels.extend(data_labels)

                            ax.set_xticks(all_tick_positions)
                            ax.set_xticklabels(all_tick_labels, fontsize=8)
                            ax.set_ylabel("Count", fontsize=10)

                            # Create title with threshold names
                            title_thresholds = ", ".join(threshold_names_for_title)
                            ax.set_title(
                                f"{variable.upper()}\n{title_thresholds} {config['units_display']}",
                                fontsize=11,
                                fontweight="bold",
                            )
                            ax.grid(True, alpha=0.3, axis="y")
                        else:
                            ax.axis("off")
                    else:
                        ax.axis("off")
            else:
                # No thresholds - turn off both threshold columns
                for col_offset in range(2):
                    ax = axes[idx, 3 + col_offset]
                    if col_offset == 0:
                        ax.text(
                            0.5,
                            0.5,
                            "No thresholds\ndefined",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                        )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col_offset == 1:
                        ax.axis("off")

        plt.suptitle(
            f"Summary Statistics - All Variables\n"
            f"{self.climate_model_chain} - {self.climate_scenario}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        # Save to output directory root
        output_file = self.output_dir / "summary_statistics.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n  ✓ Summary plot saved to {output_file}")

    def validate_all_variables(self, variables: list = None):
        """Validate all specified variables."""
        if variables is None:
            variables = list(VARIABLE_CONFIG.keys())

        print(f"\n{'='*80}")
        print("VALIDATING BIAS-CORRECTED VARIABLES")
        print(f"Climate Model Chain: {self.climate_model_chain}")
        print(f"Climate Scenario: {self.climate_scenario}")
        print(f"Variables: {', '.join(variables)}")
        print(f"{'='*80}")

        for variable in variables:
            if variable not in VARIABLE_CONFIG:
                print(f"Warning: Unknown variable {variable}, skipping")
                continue

            try:
                self.validate_variable(variable)
            except Exception as e:
                print(f"  ✗ Error validating {variable}: {str(e)}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*80}")
        print("VALIDATION COMPLETE")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate bias-corrected climate variables"
    )
    parser.add_argument(
        "paths_file", type=str, help="Path to the JSON file containing paths"
    )
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario")
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help="Variables to validate (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: figs/validation/<model>_<scenario>)",
    )
    parser.add_argument(
        "--temp_res",
        type=str,
        choices=["3hr", "day"],
        default=None,
        help="Override temporal resolution (default: use value from variable config)",
    )
    parser.add_argument(
        "--extent",
        type=str,
        choices=["prevah", "switzerland", "europe"],
        default="prevah",
        help="Domain for validation extent (default: prevah)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Create summary statistics plot for all variables",
    )

    args = parser.parse_args()

    # Load paths
    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])

    if args.extent == "prevah":
        extent = EXTENT_PREVAH
    elif args.extent == "switzerland":
        extent = EXTENT_CH
    else:
        extent = EXTENT_BC

    if (
        path_data_processed
        / "cordex_processed"
        / f"{args.climate_model_chain}_{args.climate_scenario}"
    ).is_dir():
        # Set output directory
        if args.output_dir is None:
            output_dir = (
                Path("../../figs/validation")
                / f"{args.climate_model_chain}_{args.climate_scenario}"
            )
        else:
            output_dir = Path(args.output_dir)

        # Create validator
        validator = BiasCorrectingValidator(
            climate_model_chain=args.climate_model_chain,
            climate_scenario=args.climate_scenario,
            path_data_processed=path_data_processed,
            output_dir=output_dir,
            extent=extent,
            temp_res_override=args.temp_res,
        )

        # Run validation
        if args.summary:
            # Create summary plot
            validator._plot_summary_statistics(variables=args.variables)
        else:
            # Run full validation
            validator.validate_all_variables(variables=args.variables)
    else:
        print(
            f"Processed data for {args.climate_model_chain} - {args.climate_scenario} not found in {path_data_processed}/cordex_processed"
        )
