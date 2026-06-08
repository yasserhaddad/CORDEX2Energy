from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from shapely.geometry import Point

EUROPEAN_COUNTRIES = [
    "Albania",
    "Austria",
    "Bosnia and Herzegovina",
    "Belgium",
    "Bulgaria",
    "Switzerland",
    "Cyprus",
    "Czech Republic",
    "Germany",
    "Denmark",
    "Estonia",
    "Spain",
    "Finland",
    "France",
    "Greece",
    "Croatia",
    "Hungary",
    "Ireland",
    "Italy",  # several bidding zones
    "Lithuania",
    "Luxembourg",
    "Latvia",
    "Montenegro",
    "Macedonia",
    "Netherlands",
    "Norway",  # several bidding zones
    "Poland",
    "Portugal",
    "Romania",
    "Serbia",
    "Sweden",  # several bidding zones
    "Slovenia",
    "Slovakia",
    "United Kingdom",
]

COUNTRIES_NATURAL_EARTH_TO_UPDATE = {
    "Czechia": "Czech Republic",
    "Bosnia and Herz.": "Bosnia and Herzegovina",
    "North Macedonia": "Macedonia",
}


def match_point_to_unique_polygon(
    gdf_points: gpd.GeoDataFrame,
    gdf_polygons: gpd.GeoDataFrame,
    id_column_name: str,
    buffer_distance: float = 0.01,
) -> dict:
    # Function to find the closest polygon for a point
    def find_closest_polygon(point, polygons):
        # Calculate the distance from the point to the centroid of each polygon
        distances = [
            (polygon["# idIntBus"], point.distance(polygon.geometry.centroid))
            for _, polygon in polygons.iterrows()
        ]
        # Return the polygon with the smallest distance
        return min(distances, key=lambda x: x[1])[0]

    # Create a mapping of points to their closest polygons
    point_to_polygon_map = {}
    for point in gdf_points.geometry:
        # Find all polygons that intersect with the point
        intersecting_polygons = gdf_polygons[gdf_polygons.geometry.intersects(point)]
        if len(intersecting_polygons) > 1:
            # If multiple polygons intersect, find the closest one
            closest_polygon = find_closest_polygon(point, intersecting_polygons)
            point_to_polygon_map[(point.y, point.x)] = closest_polygon
        elif len(intersecting_polygons) == 1:
            # If only one polygon intersects, assign it directly
            point_to_polygon_map[(point.y, point.x)] = intersecting_polygons.iloc[0][
                id_column_name
            ]

    # If not all polygons are matched, buffer the polygons and try again
    polygons_matched = np.unique(list(point_to_polygon_map.values()))
    if len(polygons_matched) != len(gdf_polygons):
        gdf_polygons_not_matched = gdf_polygons[
            ~gdf_polygons[id_column_name].isin(polygons_matched)
        ].copy()
        gdf_polygons_not_matched["geometry"] = gdf_polygons_not_matched.geometry.apply(
            lambda geom: geom.buffer(buffer_distance, cap_style=3, resolution=16)
        )
        gdf_points_not_matched = gdf_points[
            ~gdf_points["geometry"].apply(lambda point: (point.y, point.x)).isin(
            point_to_polygon_map.keys()
            )
        ].copy()
        point_to_polygon_map.update(
            match_point_to_unique_polygon(
                gdf_points_not_matched,
                gdf_polygons_not_matched,
                id_column_name,
                buffer_distance,
            )
        )

    return point_to_polygon_map

def create_regions_mask(gdf_regions: gpd.GeoDataFrame, names: str, da: xr.DataArray) -> xr.DataArray:
    regions = regionmask.from_geopandas(gdf_regions, names=names)
    regions_mask = regions.mask(da)

    return regions, regions_mask


def create_geodataframe_climate_grid(ds_climate_data: xr.Dataset) -> gpd.GeoDataFrame:
    # Create a list of points from the lat/lon grid
    points = [
        Point(lon, lat)
        for lat in ds_climate_data.lat.values
        for lon in ds_climate_data.lon.values
    ]

    # Convert the list of points into a GeoSeries
    gdf_climate_grid = (
        gpd.GeoSeries(points).to_frame(name="geometry").set_crs(crs="epsg:4326")
    )
    gdf_climate_grid["lat"] = [point.y for point in gdf_climate_grid.geometry]
    gdf_climate_grid["lon"] = [point.x for point in gdf_climate_grid.geometry]

    return gdf_climate_grid


def create_country_mask(ds: xr.Dataset) -> xr.DataArray:
    # Create a mask for the countries
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    mask_countries = countries.mask(ds, flag="names")
    mask_countries.name = "country"
    mask_countries.attrs["long_name"] = "Country"
    countries = countries.to_dataframe()
    countries["names"] = countries["names"].apply(lambda x: COUNTRIES_NATURAL_EARTH_TO_UPDATE.get(x, x))
    return mask_countries, countries


def process_eez_file(path_eez_file: Path) -> gpd.GeoDataFrame:
    gdf_eez = gpd.read_file(path_eez_file)
    gdf_eez = gdf_eez.to_crs("EPSG:4326")
    gdf_eez = gdf_eez[gdf_eez.GEONAME.str.contains("Exclusive")]
    gdf_eez = gdf_eez[gdf_eez["TERRITORY1"].isin(EUROPEAN_COUNTRIES)]
    return gdf_eez

def create_country_offshore_mask(ds: xr.Dataset, path_eez_file: Path) -> xr.DataArray:
    # Create a mask for the countries
    gdf_eez = process_eez_file(path_eez_file)
    eez_regions = regionmask.Regions(
        name="EEZs",
        numbers=range(len(gdf_eez)),
        names=gdf_eez["TERRITORY1"],
        outlines=gdf_eez.geometry.values,
    )
    mask_countries = eez_regions.mask(ds)
    mask_countries.name = "country"
    mask_countries.attrs["long_name"] = "Country"
    return mask_countries, eez_regions.to_dataframe()


def _aggregate_spatial(
    ds: xr.Dataset,
    mask: xr.DataArray,
    method: str,
    spatial_dims: list = None,
) -> xr.Dataset:
    """Aggregate *ds* over spatial dimensions using an integer region *mask*.

    The mask's ``.name`` attribute is used as the groupby dimension name and
    must be set before calling this function.

    Parameters
    ----------
    ds:
        Dataset with spatial dimensions to aggregate over.
    mask:
        Integer-valued DataArray whose ``.name`` identifies the output coordinate
        (e.g. ``"country"`` or ``"tyndp_zone"``).  Fill value -1 is treated as
        "no region" and must already be set before calling.
    method:
        One of ``"mean"``, ``"sum"``, or ``"median_and_better"``.
    spatial_dims:
        Spatial dimensions to reduce over.  Defaults to ``["lat", "lon"]``.

    Returns
    -------
    xr.Dataset
        Dataset with the spatial dimensions replaced by the region coordinate.
    """
    if spatial_dims is None:
        spatial_dims = ["lat", "lon"]

    if method not in ["mean", "median_and_better", "sum"]:
        raise ValueError(
            f"Method '{method}' is not supported. Use 'mean', 'sum' or 'median_and_better'."
        )

    dim = mask.name  # e.g. "country" or "tyndp_zone"

    if method == "mean":
        ds = ds.groupby(mask).mean(dim=spatial_dims, skipna=True)
    elif method == "sum":
        ds = ds.groupby(mask).sum(dim=spatial_dims, skipna=True)
    elif method == "median_and_better":
        mean_time = ds.mean(dim="time", skipna=True)
        if ds.isel(time=0).isnull().any():
            mean_time = mean_time.compute()
        ref = mean_time.groupby(mask).median(skipna=True)
        ref_broadcasted = xr.merge([
            xr.apply_ufunc(
                lambda region_id: ref[var].sel({dim: region_id}),
                mask,
                vectorize=True,
                output_dtypes=[float],
            ).rename(var)
            for var in ref.data_vars
        ])
        ds = ds.where(mean_time >= ref_broadcasted).groupby(mask).mean(
            dim=spatial_dims, skipna=True
        )

    return ds


def aggregate_data_per_country(
    ds: xr.Dataset,
    method: str = "mean",
    offshore: bool = False,
    path_eez_file: Path = None,
) -> xr.Dataset:
    mask, countries = (
        create_country_offshore_mask(ds, path_eez_file) if offshore else create_country_mask(ds)
    )
    mask = mask.fillna(-1)

    ds = _aggregate_spatial(ds, mask, method)

    ds = ds.sel(country=ds["country"].values[ds["country"].values != -1])
    ds["country"] = [countries.loc[number, "names"] for number in ds["country"].values]
    ds = ds.sel(country=[country for country in ds["country"].values if country in EUROPEAN_COUNTRIES])
    return ds


def aggregate_data_per_bidding_zone(
    ds: xr.Dataset,
    gdf_tyndp_zones: gpd.GeoDataFrame,
    method: str = "mean",
) -> xr.Dataset:
    """Aggregate *ds* to TYNDP 2024 bidding-zone level.

    Parameters
    ----------
    ds:
        Dataset with ``lat`` / ``lon`` coordinates in EPSG:4326.
    gdf_tyndp_zones:
        GeoDataFrame with a ``tyndp_zone`` column of zone codes and polygon
        geometries (output of ``tyndp.build_tyndp_zone_geometries``).
    method:
        Aggregation method — ``"mean"``, ``"sum"``, or ``"median_and_better"``.
    Returns
    -------
    xr.Dataset
        Dataset indexed by ``tyndp_zone`` (string zone codes).
    """
    regions, mask = create_regions_mask(gdf_tyndp_zones, names="tyndp_zone", da=ds)
    mask = mask.fillna(-1)
    mask.name = "tyndp_zone"

    ds = _aggregate_spatial(ds, mask, method)

    region_df = regions.to_dataframe()
    ds = ds.sel(tyndp_zone=ds["tyndp_zone"].values[ds["tyndp_zone"].values != -1])
    ds["tyndp_zone"] = [region_df.loc[i, "names"] for i in ds["tyndp_zone"].values]
    return ds


# ---------------------------------------------------------------------------
# Node-level spatial aggregation helpers
# ---------------------------------------------------------------------------

def assign_plants_to_nodes(
    gdf_plants: gpd.GeoDataFrame,
    gdf_voronoi_nodes: gpd.GeoDataFrame,
    proj_crs: str = "EPSG:3035",
) -> gpd.GeoDataFrame:
    """Assign power-plant points to Voronoi node polygons respecting country boundaries.

    Plants that fall in a polygon belonging to a different country (border cases)
    are reassigned to the nearest same-country node using planar distance in
    *proj_crs* (default: ETRS89-LAEA, suitable for Europe).

    Parameters
    ----------
    gdf_plants:
        GeoDataFrame with columns ``Name``, ``Country``, ``ISO``, ``Capacity``
        and a point geometry in EPSG:4326.
    gdf_voronoi_nodes:
        GeoDataFrame with columns ``BUS_I`` and ``COUNTRY`` (ISO-2 code) and a
        polygon geometry in EPSG:4326.
    proj_crs:
        CRS used for planar distance calculations when reassigning border plants.

    Returns
    -------
    GeoDataFrame
        Columns ``Name``, ``Country``, ``ISO``, ``Capacity``, ``BUS_I``,
        ``geometry`` (EPSG:4326).
    """
    gdf_joined = gpd.sjoin(
        gdf_plants,
        gdf_voronoi_nodes[["BUS_I", "COUNTRY", "geometry"]].rename(
            columns={"COUNTRY": "node_iso"}
        ),
        how="left",
        predicate="within",
    )

    correct = gdf_joined["node_iso"] == gdf_joined["ISO"]
    gdf_nodes = gdf_joined.loc[
        correct, ["Name", "Country", "ISO", "Capacity", "BUS_I", "geometry"]
    ].copy()

    border_mask = ~gdf_plants["Name"].isin(gdf_nodes["Name"])
    n_border = border_mask.sum()
    if n_border:
        print(f"{n_border} plant(s) near a border — reassigning to nearest same-country node.")
        gdf_border = gdf_plants.loc[border_mask].copy()

        gdf_nodes_proj = gdf_voronoi_nodes[["BUS_I", "COUNTRY", "geometry"]].to_crs(proj_crs)
        gdf_border_proj = gdf_border.to_crs(proj_crs)

        rows = []
        for _, plant in gdf_border_proj.iterrows():
            same_country = gdf_nodes_proj.loc[
                gdf_nodes_proj["COUNTRY"] == plant["ISO"], ["BUS_I", "geometry"]
            ].copy()
            if same_country.empty:
                print(f"  No nodes for ISO '{plant['ISO']}' — skipping {plant['Name']}")
                continue
            same_country["dist"] = same_country.geometry.distance(plant.geometry)
            nearest_bus = same_country.loc[same_country["dist"].idxmin(), "BUS_I"]
            rows.append({
                "Name": plant["Name"],
                "Country": plant["Country"],
                "ISO": plant["ISO"],
                "Capacity": plant["Capacity"],
                "BUS_I": nearest_bus,
                "geometry": gdf_border.loc[plant.name, "geometry"],
            })

        if rows:
            gdf_nodes = pd.concat(
                [gdf_nodes, gpd.GeoDataFrame(rows, crs="EPSG:4326")],
                ignore_index=True,
            )

    return gdf_nodes


def compute_node_capacity_weights(gdf_plant_nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute capacity-weighted shares for each ``(Country, BUS_I)`` pair.

    Parameters
    ----------
    gdf_plant_nodes:
        GeoDataFrame returned by :func:`assign_plants_to_nodes`.

    Returns
    -------
    pd.DataFrame
        Columns: ``Country``, ``BUS_I``, ``node_capacity``, ``node_weight``.
        Weights sum to 1.0 within each country.
    """
    node_cap = (
        gdf_plant_nodes.groupby(["Country", "BUS_I"])["Capacity"]
        .sum()
        .reset_index()
        .rename(columns={"Capacity": "node_capacity"})
    )
    node_cap["node_weight"] = (
        node_cap["node_capacity"]
        / node_cap.groupby("Country")["node_capacity"].transform("sum")
    )
    wsum = node_cap.groupby("Country")["node_weight"].sum()
    assert (wsum - 1.0).abs().max() < 1e-10, f"Node weights don't sum to 1:\n{wsum}"
    return node_cap


def distribute_country_timeseries_to_nodes(
    country_ts: pd.DataFrame,
    node_cap: pd.DataFrame,
    time_cols: list,
) -> pd.DataFrame:
    """Distribute country-level time series to nodes weighted by installed capacity.

    After distributing, a residual correction ensures country totals are
    preserved exactly (floating-point errors are absorbed by the heaviest node).

    Parameters
    ----------
    country_ts:
        DataFrame indexed by full country name; columns are *time_cols*.
    node_cap:
        Output of :func:`compute_node_capacity_weights` — must have columns
        ``Country``, ``BUS_I``, ``node_weight``.
    time_cols:
        Ordered list of time-step column names present in *country_ts*.

    Returns
    -------
    pd.DataFrame
        Indexed by time (``pd.DatetimeIndex``), columns named by ``BUS_I``.
    """
    node_ts = (
        node_cap[["Country", "BUS_I", "node_weight"]]
        .merge(country_ts, left_on="Country", right_index=True, how="left")
        .set_index(["Country", "BUS_I"])
    )

    missing = node_ts[time_cols].isna().all(axis=1)
    if missing.any():
        print(
            f"Warning: {missing.sum()} node(s) have no matching country time series:\n"
            + node_ts.loc[missing, []].reset_index()[["Country", "BUS_I"]].to_string()
        )

    df_nodes = node_ts[time_cols].multiply(node_ts["node_weight"], axis=0)

    # Residual correction: anchor node (largest weight) absorbs floating-point error
    for country, grp in node_cap.groupby("Country"):
        if country not in country_ts.index:
            continue
        idx = [(country, bus_i) for bus_i in grp["BUS_I"].tolist()]
        if len(idx) < 2:
            continue
        anchor_bus = grp.loc[grp["node_weight"].idxmax(), "BUS_I"]
        residual = country_ts.loc[country] - df_nodes.loc[idx].sum()
        df_nodes.loc[(country, anchor_bus)] += residual

    # Collapse cross-border contributions to the same node, then reshape to
    # (timesteps × BUS_I)
    df_nodes = df_nodes.groupby(level="BUS_I").sum().T
    df_nodes.index = pd.to_datetime(df_nodes.index)
    df_nodes.index.name = "time"
    df_nodes.columns.name = "BUS_I"
    return df_nodes


