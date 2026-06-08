import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sns.set_style("whitegrid", {"grid.color": ".93"})

FONTSIZE_TICKS = 12
FONTSIZE_LABELS = 14
FONTSIZE_TITLE = 14


def plot_capacity_factor_and_percentage_change(
    mean_reference: xr.DataArray,
    percentage_change: xr.DataArray,
    selected_var: str,
    extent: Dict[str, float],
    ds_orog: xr.Dataset = None,
    path_figs: Path = None,
    europe: bool = False,
    vmin_mean: float = 0,
    vmax_mean: float = 0.25,
    vmin_change: float = -5.5,
    vmax_change: float = 5.5,
    cbar_label_mean: str = "Mean Capacity Factor",
    title_mean: str = "Mean Capacity Factor",
    title_change: str = "Percentage Change",
    cmap_mean: str = "plasma",
    save: bool = True,
    filename_prefix: str = "",
) -> None:
    """
    Plots the mean values for a reference period and percentage change for a given variable.

    Parameters
    ----------
    mean_reference: xr.DataArray
        Mean values for the reference period.
    percentage_change: xr.DataArray
        Percentage change data for the future period vs reference period.
    selected_var: str
        Selected variable name.
    extent: Dict[str, float]
        Dictionary containing the extent (lon_min, lon_max, lat_min, lat_max).
    ds_orog: xr.Dataset
        Dataset containing orography data.
    path_figs: Path
        Path to save the generated figures.
    vmin_mean : float, optional
        Minimum value for the color scale of the mean capacity factor plot. Default is 0.
    vmax_mean : float, optional
        Maximum value for the color scale of the mean capacity factor plot. Default is 0.25.
    vmin_change : float, optional
        Minimum value for the color scale of the percentage change plot. Default is -5.5.
    vmax_change : float, optional
        Maximum value for the color scale of the percentage change plot. Default is 5.5.
    cbar_label_mean : str, optional
        Label for the colorbar of the mean capacity factor plot. Default is "Mean Capacity Factor".
    title_mean : str, optional
        Title for the mean capacity factor plot. Default is "Mean Capacity Factor".
    title_change : str, optional
        Title for the percentage change plot. Default is "Percentage Change".
    cmap_mean : str, optional
        Colormap for the mean capacity factor plot. Default is "plasma".
    save : bool, optional
        Whether to save the figure. Default is True.
    filename_prefix : str, optional
        Prefix for the filename to save the figure. Default is an empty string.

    """
    if europe:
        # Define Mercator projection centered roughly over Europe
        proj = ccrs.Mercator(central_longitude=10.0, min_latitude=30.0, max_latitude=75.0)
    else:
        # Define Mercator projection centered roughly over Switzerland
        proj = ccrs.Mercator(central_longitude=8.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), subplot_kw={"projection": proj})

    # Helper to configure map features and extent
    def setup_axes(ax, title):
        ax.set_title(title, fontsize=FONTSIZE_TITLE)
        ax.set_extent(
            [
                extent["lon_min"],
                extent["lon_max"],
                extent["lat_min"],
                extent["lat_max"],
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.BORDERS, linestyle="-", edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Plot mean capacity factor
    plot = mean_reference.plot(
        ax=axes[0],
        cmap=cmap_mean,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
            "label": cbar_label_mean,
            "shrink": 0.8,
            "orientation": "horizontal",
            "pad": 0.05,
            "aspect": 20,
        },
        vmin=vmin_mean,
        vmax=vmax_mean,
    )

    # Modify colorbar font sizes
    colorbar = plot.colorbar
    colorbar.ax.tick_params(labelsize=FONTSIZE_TICKS)  # Set tick label size
    colorbar.set_label(
        colorbar.ax.get_ylabel(), fontsize=FONTSIZE_LABELS
    )  # Set label font size

    setup_axes(axes[0], title_mean)

    # Plot percentage change
    plot = percentage_change.plot(
        ax=axes[1],
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
            "label": "Percentage Change (%)",
            "shrink": 0.8,
            "orientation": "horizontal",
            "pad": 0.05,
            "aspect": 20,
        },
        vmin=vmin_change,
        vmax=vmax_change,
    )
    # Modify colorbar font sizes
    colorbar = plot.colorbar
    colorbar.ax.tick_params(labelsize=FONTSIZE_TICKS)  # Set tick label size
    colorbar.set_label(
        colorbar.ax.get_ylabel(), fontsize=FONTSIZE_LABELS
    )  # Set label font size

    setup_axes(axes[1], title_change)

    # Add elevation contours to both maps
    if ds_orog is not None:
        for ax in axes:
            cs = ax.contour(
                ds_orog.lon,
                ds_orog.lat,
                ds_orog.orog,
                levels=10,
                colors="grey",
                linewidths=0.5,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt="%d m")

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
        ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE_TICKS)

    plt.tight_layout()

    # Save figure
    if save:
        path_figs.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            path_figs
            / f"{filename_prefix}_{selected_var}_reference_mean_percentage_change.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_yearly_mean_boxplot(
    yearly_aggregated_future: List[xr.Dataset],
    scenarios: List[str],
    selected_var: str,
    regions_to_plot: list,
    path_figs: Path,
    filename_prefix: str = "",
) -> None:
    """
    Plots a boxplot of yearly mean PV capacity factor by region, decade, and scenario.

    Parameters
    ----------
    yearly_aggregated_future : List[xr.Dataset]
        List of xarray Datasets containing yearly aggregated data for different scenarios.
    selected_var : str
        Selected variable name to plot.
    regions_to_plot : list
        List of regions to include in the plot.
    path_figs : Path
        Path to save the generated figure.
    filename_prefix : str, optional
        Prefix for the filename to save the figure. Default is an empty string.
    """

    colors = sns.color_palette("colorblind")[0:2]

    # Add RCP26 data to the DataFrame
    yearly_aggregated_future = [
        ds.assign_coords(decade=(ds["time"].dt.year // 10) * 10)
        for ds in yearly_aggregated_future
    ]
    dfs_scenarios = [
        ds[selected_var].to_dataframe().reset_index() for ds in yearly_aggregated_future
    ]
    for i, scenario in enumerate(scenarios):
        dfs_scenarios[i]["scenario"] = scenario

    # Combine the data
    df_combined = pd.concat(dfs_scenarios, ignore_index=True)
    df_combined["decade_str"] = df_combined["decade"].astype(str) + "s"
    df_combined["region"] = df_combined["mask"].astype(str)

    # Filter data to include only relevant decades
    plot_df = df_combined[(df_combined["decade"] < 2050)].copy()

    if regions_to_plot is not None:
        plot_df = plot_df[plot_df["region"].isin(regions_to_plot)]

    # Create combined column for region and decade
    plot_df["region_decade"] = plot_df["decade_str"] + " - " + plot_df["region"]

    # Sort the values by region and decade
    plot_df = plot_df.sort_values(by=["region", "decade"])

    # Plot the boxplot
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(
        data=plot_df,
        x="region_decade",
        y=selected_var,
        hue="scenario",
        palette=colors,
        dodge=True,
        showfliers=False,
    )

    # Create a custom x-tick label that shows only the central decade
    xticks = ax.get_xticks()
    xtick_labels = ax.get_xticklabels()

    new_labels = []
    for i, label in enumerate(xtick_labels):
        label_text = label.get_text()
        decade, region = label_text.split(" - ")
        if (i + 2) % 3 == 0:
            new_labels.append(f"{decade}\n\n{region}")
        else:
            new_labels.append(f"{decade}")

    ax.set_xticklabels(new_labels, rotation=0, ha="center", fontsize=FONTSIZE_LABELS)

    # Add vertical lines between regions
    region_boundaries = [i + 0.5 for i in range(2, len(xticks), 3)]
    for boundary in region_boundaries:
        plt.axvline(x=boundary, color="gray", linestyle="--", linewidth=1)

    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_LABELS)
    ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE_LABELS)

    plt.ylim(0.125, 0.205)
    plt.title(
        "Yearly Mean PV Capacity Factor\nby Region, Decade, and Scenario",
        fontsize=FONTSIZE_LABELS,
    )
    plt.xlabel("")
    plt.ylabel("Mean Capacity Factor", fontsize=FONTSIZE_LABELS)
    plt.legend(
        title="Scenario",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=FONTSIZE_LABELS,
        title_fontsize=FONTSIZE_LABELS,
    )

    plt.tight_layout()

    path_figs.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path_figs / f"{filename_prefix}_{selected_var}_yearly_mean_decade_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
