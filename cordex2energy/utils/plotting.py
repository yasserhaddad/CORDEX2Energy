from pathlib import Path
from typing import Optional, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import regionmask
import seaborn as sns
import xarray as xr
import geopandas as gpd
import yaml

sns.set_style("whitegrid", {"grid.color": ".93"})

FONTSIZE_LEGEND = 18
FONTSIZE_TICKS = 16
FONTSIZE_LABELS = 19
FONTSIZE_TITLE = 22

COMPONENTS_SIZE_CONFIG_DEFAULT = {
    "FONTSIZE_LEGEND": 18,
    "FONTSIZE_TICKS": 16,
    "FONTSIZE_LABELS": 19,
    "FONTSIZE_TITLE": 22,
}

COLORBLIND_PALETTE = sns.color_palette("colorblind")
BLUE = COLORBLIND_PALETTE[0]
ORANGE = COLORBLIND_PALETTE[1]
GREEN = COLORBLIND_PALETTE[2]
RED = COLORBLIND_PALETTE[3]
PURPLE = COLORBLIND_PALETTE[4]
BROWN = COLORBLIND_PALETTE[5]
PINK = COLORBLIND_PALETTE[6]
GREY = COLORBLIND_PALETTE[7]
YELLOW = COLORBLIND_PALETTE[8]
SKY_BLUE = COLORBLIND_PALETTE[9]

river_cmap = mcolors.LinearSegmentedColormap.from_list(
    "river", ["#0b1c2d", "#1c4e80", "#4ea3c8", "#cfe9f2", "#ffffff"]
)

river_cmap = mcolors.LinearSegmentedColormap.from_list(
    "river", ["#1c4e80", "#4ea3c8", "#cfe9f2", "#ffffff"]
)

CMAPS = {
    "pv": "plasma",
    "wind": "Greens",
    "hydropower": "PuBu",
    "hydropower_ror": river_cmap,
    "hydropower_reservoir": river_cmap,
    "HDD": "Blues",
    "CDD": "YlOrRd",
}

def load_warming_level_periods(
    yml_path: Union[str, Path],
    warming_levels: list[float] | None = None,
    gcms: Union[list[str], dict[str, str], None] = None,
) -> dict[tuple[str, str], dict[str, dict[float, dict | None]]]:
    """Load warming-level periods from a pre-computed YAML file.

    The YAML format is the one produced by Mathias Hauser's ``warming_levels.py``
    script (e.g. ``cmip5_warming_levels_all_ens_1850_1900.yml``).  Each section
    key encodes the level as an integer with one implicit decimal place:
    ``warming_level_15`` → 1.5 °C, ``warming_level_061`` → 0.61 °C.

    Parameters
    ----------
    yml_path : str | Path
        Path to the YAML file.
    warming_levels : list[float] | None
        Levels to extract (e.g. ``[1.5, 2.0, 3.0]``).  ``None`` returns all
        levels present in the file.
    gcms : list[str] | dict[str, str] | None
        Models to include.

        - ``None`` — return all models.
        - ``list[str]`` — include all ensemble members for the listed models.
        - ``dict[str, str]`` — ``{model: ensemble}`` mapping; keeps only the
          specified ensemble member per model (e.g.
          ``{"CNRM-CM5": "r1i1p1", "EC-EARTH": "r12i1p1"}``).

    Returns
    -------
    dict
        Nested dict keyed by ``(model, ensemble)``::

            {
                (model, ensemble): {
                    scenario: {
                        warming_level: {
                            "start_year": int,
                            "end_year": int,
                            "central_year": int,
                        }
                    }
                }
            }

        Missing combinations (level not reached, model absent from the file)
        are simply not present in the dict; use ``.get()`` to handle them.
    """
    def _suffix_to_float(suffix: str) -> float:
        # "15" → 1.5,  "20" → 2.0,  "061" → 0.61
        return int(suffix) / 10 ** (len(suffix) - 1)

    with open(Path(yml_path)) as fh:
        data = yaml.safe_load(fh)

    # Map float level → YAML section key for every level in the file
    available: dict[float, str] = {}
    for key in data:
        if key.startswith("warming_level_"):
            suffix = key[len("warming_level_"):]
            available[_suffix_to_float(suffix)] = key

    keys_to_read = (
        {wl: available[wl] for wl in warming_levels if wl in available}
        if warming_levels is not None
        else available
    )

    # Normalise gcms argument into (model_set | None, ens_map | None)
    if gcms is None:
        model_set, ens_map = None, None
    elif isinstance(gcms, dict):
        model_set, ens_map = set(gcms.keys()), gcms
    else:
        model_set, ens_map = set(gcms), None

    result: dict[tuple[str, str], dict[str, dict[float, dict]]] = {}
    for wl, key in keys_to_read.items():
        for entry in data.get(key, []):
            model, ens = entry["model"], entry["ensemble"]
            if model_set is not None and model not in model_set:
                continue
            if ens_map is not None and ens_map.get(model) != ens:
                continue
            sy = entry["start_year"]
            me_key = (model, ens)
            result.setdefault(me_key, {}).setdefault(entry["exp"], {})[wl] = {
                "start_year": sy,
                "end_year": entry["end_year"],
                "central_year": sy + 10,
            }

    return result


def compute_gmst_anomalies(
    ds, var, reference_period, time_dim="time"
) -> tuple[np.ndarray, xr.DataArray]:
    mean = ds[var].mean(dim=["lat", "lon"])
    years = ds[time_dim].values if time_dim in ds else np.arange(len(mean))
    ref_mean = mean.sel(year=reference_period).mean()
    anomalies = mean - ref_mean
    return years, anomalies

def cm_to_inch(cm: float) -> float:
    """
    Convert centimeters to inches for matplotlib figsize.

    Parameters
    ----------
    cm : float
        Length in centimeters.

    Returns
    -------
    float
        Length in inches.
    """
    return cm / 2.54


def parse_figsize(
    figsize: tuple[float, float] | tuple[str, str] | tuple[float | str, float | str],
) -> tuple[float, float]:
    """
    Parse figsize input and convert to matplotlib-compatible format.

    Accepts figsize as:
    - Tuple of numbers: (width, height) in inches
    - Tuple of strings with 'cm': ('20cm', '15cm') converts to inches
    - Mixed formats: (20, '15cm') or ('20cm', 15)

    Parameters
    ----------
    figsize : tuple of float or str
        Figure size as (width, height). Values can be numbers (inches) or
        strings with 'cm' suffix (e.g., '20cm').

    Returns
    -------
    tuple of float
        Figure size in inches as (width, height).

    Examples
    --------
    >>> parse_figsize((10, 8))
    (10, 8)
    >>> parse_figsize(('20cm', '15cm'))
    (7.874015748031496, 5.905511811023622)
    >>> parse_figsize((10, '15cm'))
    (10, 5.905511811023622)
    """
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must be a tuple or list of length 2")

    width, height = figsize

    # Parse width
    if isinstance(width, str):
        if width.lower().endswith("cm"):
            width = cm_to_inch(float(width[:-2]))
        else:
            width = float(width)
    elif not isinstance(width, (int, float)):
        raise ValueError(f"Invalid width type: {type(width)}")

    # Parse height
    if isinstance(height, str):
        if height.lower().endswith("cm"):
            height = cm_to_inch(float(height[:-2]))
        else:
            height = float(height)
    elif not isinstance(height, (int, float)):
        raise ValueError(f"Invalid height type: {type(height)}")

    return (float(width), float(height))


def plot_gmst_anomalies(
    scenarios: list[str],
    subset_gcms: list[Path],
    var: str,
    time_dim: str,
    reference_period: slice,
    figsize=(18, 5),
    components_size_config: dict = None,
    path_savefig: Path = None,
):
    """
    Plot annual GMST anomalies for all GCMs in subset per scenario.

    Parameters
    ----------
    scenarios : list of str
        List of scenario names.
    subset_gcms : list of pathlib.Path
        List of file paths to GCM NetCDF files.
    var : str
        Variable name in the NetCDF files to plot.
    time_dim : str
        Name of the time dimension (e.g., "year").
    reference_period : slice
        Reference period (e.g., slice("1871", "1900")) for anomaly calculation.
    color_palette : list
        List of colors for plotting each GCM.
    figsize : tuple, optional
        Figure size.
    path_savefig : pathlib.Path, optional
        Path to directory where the figure will be saved.

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize, sharey=True)
    lines = []
    labels = []

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        j = 0
        for file in subset_gcms:
            if scenario in file.name:
                ds = xr.open_dataset(file).sel(year=slice(reference_period.start, None))
                years, anomalies = compute_gmst_anomalies(
                    ds, var, reference_period, time_dim
                )
                gcm = file.stem.split("_")[2]
                (line,) = ax.plot(
                    years,
                    anomalies.values,
                    label=gcm,
                    alpha=1,
                    color=COLORBLIND_PALETTE[j],
                )
                if gcm not in labels:
                    lines.append(line)
                    labels.append(gcm)
                ds.close()
                j += 1
        ax.set_xlabel("Year", fontsize=FONTSIZE_LABELS)
        if i == 0:
            ax.set_ylabel(
                f"Annual GMST anomaly relative to {reference_period.start}-{reference_period.stop} (K)",
                fontsize=components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS),
            )
        ax.set_title(
            f"{scenario.upper()}",
            fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
        )
        ax.tick_params(
            axis="both",
            labelsize=components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS),
        )

    fig.suptitle(
        "Annual GMST anomalies for all GCMs in subset (per scenario)",
        fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
    )
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND),
    )
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_gmst_anomalies_subset_vs_all(
    scenarios: list[str],
    subset_gcms: list[Path],
    all_other_gcms_data: list[Path],
    var: str,
    time_dim: str,
    reference_period: slice,
    figsize=(18, 5),
    path_savefig: Path = None,
    separate_subset: bool = True,
    components_size_config: dict = None,
):
    """
    Plot annual GMST anomalies for all GCMs in subset and other GCMs per scenario.

    Parameters
    ----------
    scenarios : list of str
        List of scenario names.
    subset_gcms : list of pathlib.Path
        List of file paths to subset GCM NetCDF files.
    all_other_gcms_data : list of pathlib.Path
        List of file paths to other GCM NetCDF files.
    var : str
        Variable name in the NetCDF files to plot.
    time_dim : str
        Name of the time dimension (e.g., "year").
    reference_period : slice
        Reference period (e.g., slice("1871", "1900")) for anomaly calculation.
    color_palette : list, optional
        List of colors for plotting each GCM.
    figsize : tuple, optional
        Figure size.
    path_savefig : pathlib.Path, optional
        Path to directory where the figure will be saved.
    separate_subset : bool, optional
        If True, plot each subset GCM with a different color and label.
        If False, bundle all subset GCMs together with the same color and label "Subset GCMs".

    Returns
    -------
    None
    """
    annual_anomalies = {sc: [] for sc in scenarios}
    labels_dict = {sc: [] for sc in scenarios}
    colors_dict = {sc: [] for sc in scenarios}
    alphas_dict = {sc: [] for sc in scenarios}

    for sc in scenarios:
        # Other GCMs
        for file in [f for f in all_other_gcms_data if sc in f.name]:
            ds = (
                xr.open_dataset(file)
                .sortby(time_dim)
                .sel({time_dim: slice(reference_period.start, None)})
            )
            years, anomalies = compute_gmst_anomalies(
                ds, var, reference_period, time_dim
            )
            annual_anomalies[sc].append((years, anomalies.values))
            labels_dict[sc].append("All GCMs")
            colors_dict[sc].append(GREY)
            alphas_dict[sc].append(0.1)
            ds.close()
        # Subset GCMs
        subset_files = [f for f in subset_gcms if sc in f.name]
        for j, file in enumerate(subset_files):
            ds = (
                xr.open_dataset(file)
                .sortby(time_dim)
                .sel({time_dim: slice(reference_period.start, None)})
            )
            years, anomalies = compute_gmst_anomalies(
                ds, var, reference_period, time_dim
            )
            gcm = file.stem.split("_")[2]
            annual_anomalies[sc].append((years, anomalies.values))
            labels_dict[sc].append(gcm if separate_subset else "Subset GCMs")
            colors_dict[sc].append(
                COLORBLIND_PALETTE[j % len(COLORBLIND_PALETTE)]
                if separate_subset
                else COLORBLIND_PALETTE[0]
            )
            alphas_dict[sc].append(1.0 if separate_subset else 0.8)
            ds.close()

    fig, axes = plt.subplots(1, len(scenarios), figsize=figsize, sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    all_handles = []
    all_labels = []

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        for (years, anomalies), label, color, alpha in zip(
            annual_anomalies[sc], labels_dict[sc], colors_dict[sc], alphas_dict[sc]
        ):
            (handle,) = ax.plot(years, anomalies, color=color, alpha=alpha, label=label)
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)
        ax.set_xlabel(
            "Year",
            fontsize=components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS),
        )
        if i == 0:
            ax.set_ylabel(
                f"Annual GMST anomaly relative to {reference_period.start}-{reference_period.stop} (K)",
                fontsize=components_size_config.get("FONTSIZE_LABELS", FONTSIZE_LABELS),
            )
        ax.set_title(
            sc.upper(),
            fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
        )
        ax.tick_params(
            axis="both",
            labelsize=components_size_config.get("FONTSIZE_TICKS", FONTSIZE_TICKS),
        )

    fig.suptitle(
        "Annual GMST anomalies for selected and other GCMs (per scenario)",
        fontsize=components_size_config.get("FONTSIZE_TITLE", FONTSIZE_TITLE),
    )
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=components_size_config.get("FONTSIZE_LEGEND", FONTSIZE_LEGEND),
    )
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()