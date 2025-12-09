from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import seaborn as sns
import xarray as xr
import geopandas as gpd

from .analysis import (
    compute_gmst_anomalies,
    compute_percentage_change,
    compute_sign_agreement,
)

sns.set_style("whitegrid", {"grid.color": ".93"})

FONTSIZE_LEGEND = 18
FONTSIZE_TICKS = 16
FONTSIZE_LABELS = 19
FONTSIZE_TITLE = 22

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

CMAPS = {
    "pv": "plasma",
    "wind": "Greens",
    "hydropower": "PuBu",
    "HDD": "Blues",
    "CDD": "YlOrRd",
}


def plot_gmst_anomalies(
    scenarios: list[str],
    subset_gcms: list[Path],
    var: str,
    time_dim: str,
    reference_period: slice,
    figsize=(18, 5),
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
                fontsize=FONTSIZE_LABELS,
            )
        ax.set_title(f"{scenario.upper()}", fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)

    fig.suptitle(
        "Annual GMST anomalies for all GCMs in subset (per scenario)",
        fontsize=FONTSIZE_TITLE,
    )
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=FONTSIZE_LEGEND,
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
        ax.set_xlabel("Year", fontsize=FONTSIZE_LABELS)
        if i == 0:
            ax.set_ylabel(
                f"Annual GMST anomaly relative to {reference_period.start}-{reference_period.stop} (K)",
                fontsize=FONTSIZE_LABELS,
            )
        ax.set_title(sc.upper(), fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)

    fig.suptitle(
        "Annual GMST anomalies for selected and other GCMs (per scenario)",
        fontsize=FONTSIZE_TITLE,
    )
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=FONTSIZE_LEGEND,
    )
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def apply_land_mask(da: xr.DataArray) -> xr.DataArray:
    """
    Apply a land mask to the input DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with 'lat' and 'lon' dimensions.

    Returns
    -------
    xr.DataArray
        DataArray with NaN values where the land mask is False.
    """
    # Create a land mask using Natural Earth land polygons
    # Higher resolution options: land_50 (medium), land_10 (high)
    # land_10 is the highest resolution available in regionmask
    land = regionmask.defined_regions.natural_earth_v5_1_2.land_10

    # Example: mask mean_ref (repeat for other datasets as needed)
    # This will create a mask with 1 for land, NaN for ocean
    land_mask = land.mask(da)
    return da.where(~np.isnan(land_mask))


def create_shared_colorbar(
    fig,
    axes,
    idxs,
    im,
    label,
    orientation="horizontal",
    shrink=0.3,
    aspect=70,
    width_ratio=0.6,
):
    """
    Create a shared colorbar for a group of subplots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : array-like
        List or array of axes.
    idxs : list of int
        Indices of axes to share the colorbar.
    im : matplotlib.image.AxesImage or QuadMesh
        The mappable object (from imshow, pcolormesh, etc.).
    label : str
        Colorbar label.
    orientation : str, optional
        Orientation of the colorbar ("horizontal" or "vertical").
    fontsize : int, optional
        Font size for the colorbar label and ticks.
    """

    from matplotlib.transforms import Bbox

    selected_axes = [axes[i] for i in idxs]
    # Get position that spans all selected axes
    bboxes = [ax.get_position() for ax in selected_axes]
    bbox = Bbox.union(bboxes)
    if orientation == "horizontal":
        cbar_height = 0.015
        cbar_pad = 0.02
        cbar_width = bbox.width * width_ratio
        cbar_x0 = bbox.x0 + (bbox.width - cbar_width) / 2
        cbar_ax = fig.add_axes(
            [cbar_x0, bbox.y0 - cbar_pad - cbar_height, cbar_width, cbar_height]
        )
    else:
        # Place colorbar to the right of the axes, not too far
        cbar_width = 0.02
        cbar_pad = 0.01
        cbar_ax = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_width, bbox.height])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation=orientation, aspect=aspect, shrink=shrink
    )
    cbar.set_label(label, fontsize=FONTSIZE_LABELS)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)


def compute_vmin_vmax(
    ds: xr.Dataset,
    var_names: list[str],
    quantiles: tuple = None,
    center_zero: bool = False,
):
    """
    Compute a single vmin/vmax across all specified variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to compute vmin and vmax values for.
    var_names : list of str
        List of variable names to compute limits for.
    quantiles : tuple, optional
        Quantiles to use for min/max (e.g., (0.025, 0.975)). If None, use min/max.
    center_zero : bool, optional
        If True, make vmin/vmax symmetric around zero.

    Returns
    -------
    tuple
        (vmin, vmax) for all variables combined.
    """
    vals = []
    for v in var_names:
        if v in ds:
            vals.append(ds[v].values)
    if not vals:
        return None, None
    vals = np.concatenate([np.ravel(x) for x in vals])
    if quantiles is not None:
        vmin = float(np.nanquantile(vals, quantiles[0]))
        vmax = float(np.nanquantile(vals, quantiles[1]))
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if center_zero:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    return vmin, vmax


def plot_ensemble_mean(ds: xr.Dataset, figsize=(18, 5), country_border_width: float = 1.0, title: str = "Ensemble mean", path_savefig: Path = None):
    """
    Plot ensemble mean for each variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing variables with dimensions 'lat', 'lon', and 'model'.
    figsize : tuple, optional
        Figure size.
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    var_names = list(ds.data_vars.keys())
    var_titles = {
        "pv": "Solar Power Potential",
        "wind": "Wind Power Potential",
        "hydropower": "Hydropower Potential",
        "HDD": "Heating Degree Days",
        "CDD": "Cooling Degree Days",
    }
    var_clabel = {
        "pv": "Capacity Factor (-)",
        "wind": "Capacity Factor (-)",
        "hydropower": "Capacity Factor (-)",
        "HDD": "Degree-Days",
        "CDD": "Degree-Days",
    }
    n_vars = len(var_names)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_vars,
        figsize=(4 * n_vars, 6) if not figsize else figsize,
        subplot_kw={
            "projection": ccrs.Mercator(
                central_longitude=10.0, min_latitude=30.0, max_latitude=75.0
            )
        },
    )

    if n_vars == 1:
        axes = [axes]

    ens_mean = ds.mean(dim="model")

    for i, var in enumerate(var_names):
        ax = axes[i]
        im = ens_mean[var].plot(
            ax=ax,
            cmap=CMAPS.get(var, "viridis"),
            transform=ccrs.PlateCarree(),
            add_colorbar=True,
            vmax=ens_mean[var].quantile(0.99) if var in ["HDD", "CDD", "hydropower"] else None,
            cbar_kwargs={
                # "label": var_clabel.get(var, ""),
                "shrink": 0.7,
                "aspect": 20,
                "pad": 0.03,
                "orientation": "horizontal",
            },
        )
        cbar = im.colorbar
        cbar.set_label(var_clabel.get(var, ""), fontsize=FONTSIZE_LABELS-1)
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=country_border_width)
        ax.set_title(var_titles.get(var, var), fontsize=FONTSIZE_LABELS)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if title is not None:
        plt.suptitle(title, fontsize=FONTSIZE_TITLE, y=0.9)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ensemble_mean_percentage_change(
    ref_dataset: xr.Dataset,
    target_dataset: xr.Dataset,
    figsize=(18, 5),
    path_savefig: Path = None,
    title: str = "Ensemble mean percentage change",
    cmap="RdBu_r",
):
    """
    Plot ensemble mean percentage change for each dataset (variable), hatching where less than 2/3 of models agree on sign.

    Parameters
    ----------
    ref_dataset : xr.Dataset
        Dataset of variables averaged over the reference period, containing the following coordinates
        'lat', 'lon', and 'model'.
    target_dataset : xr.Dataset
        Dataset of variables averaged over the target period, containing the following coordinates
        'lat', 'lon', and 'model'.
    figsize : tuple, optional
        Figure size.
    path_savefig : pathlib.Path, optional
        Path to save the figure.
    vmin, vmax : float, optional
        Colorbar limits.
    cmap : str, optional
        Colormap.
    """
    var_names = list(ref_dataset.data_vars.keys())
    var_titles = {
        "pv": "Solar Power Potential",
        "wind": "Wind Power Potential",
        "hydropower": "Hydropower Potential",
        "HDD": "Heating Degree Days",
        "CDD": "Cooling Degree Days",
    }
    n_vars = len(var_names)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_vars,
        figsize=(4 * n_vars, 6) if not figsize else figsize,
        subplot_kw={
            "projection": ccrs.Mercator(
                central_longitude=10.0, min_latitude=30.0, max_latitude=75.0
            )
        },
    )

    if all(ref_dataset.model.values != target_dataset.model.values):
        raise ValueError(
            "Reference and target datasets must have the same 'model' coordinate."
        )

    # Compute percentage change for each model
    pct_change = compute_percentage_change(ref_dataset, target_dataset, round=2)
    # Ensemble mean
    ens_mean = pct_change.mean(dim="model")
    # Agreement mask: sign agreement across models
    sign_agree = compute_sign_agreement(pct_change, model_dim="model")
    n_models = pct_change.sizes["model"]
    agree_mask = ~(sign_agree >= (2 * n_models / 3))
    # Place NaNs in agree_mask where original data is NaN
    for var in var_names:
        agree_mask[var] = agree_mask[var].where(~np.isnan(ens_mean[var]))

    # Determine common vmin/vmax for PV/Wind and HDD/CDD groups
    pv_wind_vmin, pv_wind_vmax = compute_vmin_vmax(
        ens_mean, ["pv", "wind"], quantiles=None, center_zero=True
    )
    hydropower_vmin, hydropower_vmax = compute_vmin_vmax(
        ens_mean, ["hydropower"], quantiles=(0.05, 0.95), center_zero=True
    )
    hdd_vmin, hdd_vmax = compute_vmin_vmax(
        ens_mean, ["HDD"], quantiles=(0.05, 0.95), center_zero=True
    )
    cdd_vmin, cdd_vmax = compute_vmin_vmax(
        ens_mean, ["CDD"], quantiles=(0.05, 0.95), center_zero=True
    )

    for i, var in enumerate(var_names):
        ax = axes[i]
        im = ens_mean[var].plot(
            ax=ax,
            cmap=cmap,
            vmin=pv_wind_vmin
            if var in ["pv", "wind"]
            else (
                hydropower_vmin
                if var == "hydropower"
                else (hdd_vmin if var == "HDD" else cdd_vmin)
            ),
            vmax=pv_wind_vmax
            if var in ["pv", "wind"]
            else (
                hydropower_vmax
                if var == "hydropower"
                else (hdd_vmax if var == "HDD" else cdd_vmax)
            ),
            transform=ccrs.PlateCarree(),
            add_colorbar=True,
            cbar_kwargs={
                # "label": "%",
                "shrink": 0.7,
                "aspect": 20,
                "pad": 0.03,
                "orientation": "horizontal",
            },
        )
        cbar = im.colorbar
        cbar.set_label("% relative to reference", fontsize=FONTSIZE_LABELS-2)
        cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)
        # Add hatching where less than 2/3 agree using contourf
        hatch_overlay = ax.contourf(
            agree_mask[var]["lon"],
            agree_mask[var]["lat"],
            agree_mask[var],
            levels=[0.5, 1.5],  # one filled band where mask == 1
            hatches=["///"],
            colors="none",
            transform=ccrs.PlateCarree(),
        )
        for col in hatch_overlay.collections:
            col.set_facecolor("none")  # keep fill transparent
            # col.set_edgecolor((0, 0, 0, 0.8))  # RGBA, alpha=0.5
            col.set_edgecolor("black")  # hatches black
            col.set_linewidth(0)  # polygon outlines invisible
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
        ax.set_title(var_titles[var], fontsize=FONTSIZE_LABELS)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if title is not None:
        plt.suptitle(title, fontsize=FONTSIZE_TITLE, y=0.9)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_yearly_timeseries_ensemble(
    ds: xr.Dataset,
    selected_regions: list[str],
    time_dim: str = "time",
    model_dim: str = "model",
    region_dim: str = "mask",
    rolling: int = 5,
    figsize=(10, 5),
    title: str = "Regional mean (rolling) timeseries",
    path_savefig: Path = None,
):
    """
    Plot ensemble mean and range (min/max) timeseries for specified regions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the data variable and region mask.
    selected_regions : list of str
        List of region names to plot (must match region_dim values).
    time_dim : str, optional
        Name of the time dimension.
    model_dim : str, optional
        Name of the model dimension.
    region_dim : str, optional
        Name of the region mask dimension.
    rolling : int, optional
        Window size for rolling mean.
    figsize : tuple, optional
        Figure size.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    path_savefig : pathlib.Path, optional
        Path to save the figure.

    Returns
    -------
    None
    """
    var_names = list(ds.data_vars.keys())
    var_titles = {
        "pv": "Solar Power",
        "wind": "Wind Power",
        "hydropower": "Hydropower",
        "HDD": "Heating Degree Days",
        "CDD": "Cooling Degree Days",
    }
    var_ylabel = {
        "pv": "Mean Capacity Factor (-)",
        "wind": "Mean Capacity Factor (-)",
        "hydropower": "Mean Capacity Factor (-)",
        "HDD": "Total Degree-Days",
        "CDD": "Total Degree-Days",
    }
    n_vars = len(var_names)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_vars,
        figsize=(4 * n_vars, 6) if not figsize else figsize,
    )

    if n_vars == 1:
        axes = [axes]

    ds_region = ds.sel({region_dim: selected_regions})
    if rolling is not None:
        # Apply rolling mean if specified
        ds_region = ds_region.rolling({time_dim: rolling}, center=True).mean()
    mean_ds_region = ds_region.mean(dim=model_dim)
    min_ds_region = ds_region.min(dim=model_dim)
    max_ds_region = ds_region.max(dim=model_dim)
    wind_solar_vmin, wind_solar_vmax = compute_vmin_vmax(
        ds_region, ["pv", "wind"], quantiles=None, center_zero=False
    )

    for i, var in enumerate(var_names):
        ax = axes[i] if n_vars > 1 else axes
        for j, region in enumerate(selected_regions):
            mean_da = mean_ds_region[var].sel({region_dim: region})
            min_da = min_ds_region[var].sel({region_dim: region})
            max_da = max_ds_region[var].sel({region_dim: region})
            color = COLORBLIND_PALETTE[j % len(COLORBLIND_PALETTE)]
            ax.plot(mean_da[time_dim], mean_da, label=f"{region} (mean)", color=color)
            ax.fill_between(
                mean_da[time_dim].values,
                min_da.values,
                max_da.values,
                alpha=0.2,
                color=color,
                label=f"{region} (range)",
            )
            if var in ["pv", "wind"]:
                ax.set_ylim(wind_solar_vmin - 0.02, wind_solar_vmax + 0.02)
        ax.set_ylabel(var_ylabel.get(var, ""), fontsize=FONTSIZE_LABELS)
        ax.set_title(f"{var_titles.get(var, var)}", fontsize=FONTSIZE_LABELS)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        # Show only every other xtick to reduce clutter
        # Set x-axis limits to match data range to avoid empty space
        xticklocs = ax.get_xticks()
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        # Show only every other label
        new_labels = [label if (i+1) % 2 == 0 else "" for i, label in enumerate(xticklabels[:-1])]
        ax.set_xticks(xticklocs[:-1])
        ax.set_xticklabels(new_labels)



    handles, labels = ax.get_legend_handles_labels()
    fig = ax.get_figure()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(selected_regions),
        bbox_to_anchor=(0.5, -0.35),
        fontsize=FONTSIZE_LEGEND,
    )
    if title is not None:
        plt.suptitle(title, fontsize=FONTSIZE_TITLE)
    plt.tight_layout()
    if path_savefig is not None:
        path_savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_savefig, dpi=300, bbox_inches="tight")
    plt.show()


def plot_interval_strip_regions(
    ds: xr.Dataset,
    selected_regions: list[str],
    time_dim: str = "time",
    model_dim: str = "model",
    region_dim: str = "mask",
    window_length: int = 20,
    figsize=(10, 5),
    title: str = "Gradient",
    path_savefig: Path = None
):
    return