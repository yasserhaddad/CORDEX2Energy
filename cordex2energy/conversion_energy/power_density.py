"""
power_density.py
----------------
Compute technology-specific power densities [MW/km²] from:
  - pvlib-style PV system configuration (rooftop)
  - Wind turbine name + hub height (via windpowerlib turbine database)
"""

import numpy as np
import xarray as xr
from typing import Union

# ── PV ────────────────────────────────────────────────────────────────────────

def pv_power_density(
    pv_config: dict,
    built_up_fraction: Union[xr.DataArray, float],
    eligible_pv_openfield: Union[xr.DataArray, float] = 0.0,
    module_area_m2: float = 1.7,
    usable_roof_fraction: float = 0.40,
    roof_to_footprint_ratio: float = 1.0,
    packing_efficiency: float = 0.90,
    ground_coverage_ratio: float = 0.30,
) -> Union[xr.DataArray, float]:
    """
    Compute combined rooftop + open-field PV power density [MW/km²] from a
    pvlib-style configuration.

    Both components use the same module (same `pdc0`); they differ only in how
    panels are arranged on the land surface.

    Parameters
    ----------
    pv_config : dict
        pvlib-style system configuration dict. Must contain:
            pv_config["module_parameters"]["pdc0"]  : module peak power [W]
    built_up_fraction : DataArray or float
        Fraction of each grid cell classified as built-up / urban [0, 1].
        Drives the rooftop PV contribution.
    eligible_pv_openfield : DataArray or float
        Fraction of each grid cell eligible for open-field (ground-mounted) PV
        [0, 1], as returned by `compute_eligibility` under key
        ``"eligible_fraction_pv_openfield"``.  Already weighted by the
        open-field eligibility fraction from the land eligibility step.
        Default 0.0 (rooftop only, no open-field contribution).
    module_area_m2 : float
        Physical area of one module [m²]. Default 1.7 m² (typical 400W mono-Si,
        ~1.0 × 1.7 m).
    usable_roof_fraction : float
        Fraction of roof area that is usable for PV after excluding chimneys,
        HVAC units, shading obstacles, and edge zones. Default 0.40.
        Literature range: 0.25–0.50 (Mainzer et al. 2014, Bódis et al. 2019).
    roof_to_footprint_ratio : float
        Ratio of actual roof surface area to building footprint area.
        1.0 for flat roofs; 1.05–1.30 for pitched roofs. Default 1.0.
    packing_efficiency : float
        Fraction of usable roof area covered by panels (accounts for
        inter-panel gaps and mounting hardware). Default 0.90 for close-mount.
    ground_coverage_ratio : float
        Fraction of open-field eligible land covered by panel area, accounting
        for inter-row spacing needed to avoid self-shading. Default 0.30.
        Literature range: 0.25–0.40 for fixed-tilt; ~0.25–0.35 for single-axis
        tracking (Deng et al. 2015, Ryberg et al. 2019).

    Returns
    -------
    DataArray or float : total PV power density [MW/km²]
        Rooftop + open-field contributions combined.
        Has the same spatial dimensions as `built_up_fraction`.

    Notes
    -----
    Derivation:
        panel_W_m2      = pdc0 [W] / module_area [m²]       # W/m² of panel

        rooftop:
          density_rt    = panel_W_m2
                          × packing_efficiency               # panels on usable roof
                          × usable_roof_fraction             # usable / total roof
                          × roof_to_footprint_ratio          # roof / footprint
                          × built_up_fraction                # footprint / cell area

        open-field:
          density_of    = panel_W_m2
                          × ground_coverage_ratio            # panel area / eligible ground
                          × eligible_pv_openfield            # eligible ground / cell area

        total           = density_rt + density_of
                          [W/m²_cell] == [MW/km²]            # units cancel exactly
    """
    pdc0_W = pv_config["module_parameters"]["pdc0"]
    panel_W_per_m2 = pdc0_W / module_area_m2

    density_rooftop = (
        panel_W_per_m2
        * packing_efficiency
        * usable_roof_fraction
        * roof_to_footprint_ratio
        * built_up_fraction
    )

    density_openfield = (
        panel_W_per_m2
        * ground_coverage_ratio
        * eligible_pv_openfield
    )

    power_density = density_rooftop + density_openfield

    # Attach metadata if output is a DataArray
    if isinstance(power_density, xr.DataArray):
        power_density.attrs.update({
            "long_name":                "PV power density (rooftop + open-field)",
            "units":                    "MW km-2",
            "module_pdc0_W":            pdc0_W,
            "module_area_m2":           module_area_m2,
            "panel_W_per_m2":           round(panel_W_per_m2, 1),
            "usable_roof_fraction":     usable_roof_fraction,
            "roof_to_footprint":        roof_to_footprint_ratio,
            "packing_efficiency":       packing_efficiency,
            "ground_coverage_ratio":    ground_coverage_ratio,
        })

    return power_density


# ── WIND ──────────────────────────────────────────────────────────────────────

def wind_power_density(
    turbine_name: str,
    hub_height_m: float,
    spacing_crosswind_D: float = 4.0,
    spacing_downwind_D: float = 8.0,
) -> dict:
    """
    Compute onshore wind power density [MW/km²] from turbine specs via
    windpowerlib's turbine database, plus a grid layout assumption.

    Parameters
    ----------
    turbine_name : str
        Turbine name as it appears in the windpowerlib database.
        E.g. "E-126/7580". Run `windpowerlib.get_turbine_types()` to list all.
    hub_height_m : float
        Hub height in metres. Must be available for the chosen turbine in the
        windpowerlib database (determines the power/thrust curve used).
    spacing_crosswind_D : float
        Turbine spacing perpendicular to prevailing wind, in rotor diameters.
        Default 4.0 D. Literature range: 3–5 D.
    spacing_downwind_D : float
        Turbine spacing along prevailing wind direction, in rotor diameters.
        Default 8.0 D. Literature range: 6–10 D.
        IEC recommendation: 4 D × 8 D minimum.

    Returns
    -------
    dict with keys:
        turbine_name          : str
        hub_height_m          : float
        rotor_diameter_m      : float
        rated_power_MW        : float
        area_per_turbine_km2  : float
        power_density_MW_km2  : float   ← main output
        spacing_crosswind_D   : float
        spacing_downwind_D    : float
        sensitivity           : dict    ← power density for ±1D spacing variants

    Notes
    -----
    Requires:  pip install windpowerlib
    Derivation:
        area_per_turbine = (sx_D × D) × (sd_D × D)   [m²]
        power_density    = rated_power [MW] / area_per_turbine [km²]
    """
    try:
        from windpowerlib import WindTurbine
    except ImportError:
        raise ImportError(
            "windpowerlib is required for wind power density calculation.\n"
            "Install with:  pip install windpowerlib"
        )

    turbine = WindTurbine(turbine_type=turbine_name, hub_height=hub_height_m)

    rotor_diameter_m = turbine.rotor_diameter          # metres
    rated_power_MW   = turbine.nominal_power / 1e6     # W → MW

    spacing_cross_m = spacing_crosswind_D * rotor_diameter_m
    spacing_down_m  = spacing_downwind_D  * rotor_diameter_m

    area_per_turbine_km2  = (spacing_cross_m * spacing_down_m) / 1e6
    power_density_MW_km2  = rated_power_MW / area_per_turbine_km2

    # Sensitivity: vary each spacing by ±1D
    sensitivity = {}
    for sx in [spacing_crosswind_D - 1, spacing_crosswind_D, spacing_crosswind_D + 1]:
        for sd in [spacing_downwind_D - 1, spacing_downwind_D, spacing_downwind_D + 1]:
            a = (sx * rotor_diameter_m * sd * rotor_diameter_m) / 1e6
            sensitivity[f"{sx}D×{sd}D"] = round(rated_power_MW / a, 2)

    result = {
        "turbine_name":         turbine_name,
        "hub_height_m":         hub_height_m,
        "rotor_diameter_m":     rotor_diameter_m,
        "rated_power_MW":       rated_power_MW,
        "spacing_crosswind_D":  spacing_crosswind_D,
        "spacing_downwind_D":   spacing_downwind_D,
        "area_per_turbine_km2": round(area_per_turbine_km2, 4),
        "power_density_MW_km2": round(power_density_MW_km2, 2),
        "sensitivity":          sensitivity,
    }

    _print_wind_summary(result)
    return result


def _print_wind_summary(r: dict):
    print(f"\nWind turbine: {r['turbine_name']}  |  hub height: {r['hub_height_m']} m")
    print(f"  Rotor diameter   : {r['rotor_diameter_m']} m")
    print(f"  Rated power      : {r['rated_power_MW']} MW")
    print(f"  Layout           : {r['spacing_crosswind_D']}D × {r['spacing_downwind_D']}D")
    print(f"  Area / turbine   : {r['area_per_turbine_km2']} km²")
    print(f"  Power density    : {r['power_density_MW_km2']} MW/km²")
    print(f"\n  Sensitivity to spacing (MW/km²):")
    for spacing, pd in r["sensitivity"].items():
        marker = " ← selected" if spacing == f"{r['spacing_crosswind_D']}D×{r['spacing_downwind_D']}D" else ""
        print(f"    {spacing:>8}  :  {pd:5.1f}{marker}")