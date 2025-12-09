from pathlib import Path
import geopandas as gpd
import numpy as np
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


def aggregate_data_per_country(ds: xr.Dataset, method: str = "mean", offshore: bool = False, path_eez_file: Path = None) -> xr.Dataset:
    mask, countries = create_country_offshore_mask(ds, path_eez_file) if offshore else create_country_mask(ds)
    mask = mask.fillna(-1)

    if method not in ["mean", "median_and_better", "sum"]:
        raise ValueError(f"Method '{method}' is not supported. Use 'mean', 'sum' or 'median_and_better'.")
    if method == "mean":
        ds = ds.groupby(mask).mean(dim=["lat", "lon"], skipna=True)
    elif method == "sum":
        ds = ds.groupby(mask).sum(dim=["lat", "lon"], skipna=True)
    elif method == "median_and_better":
        mean_time = ds.mean(dim="time", skipna=True)
        if ds.isel(time=0).isnull().any():
            mean_time = mean_time.compute()
        ref = mean_time.groupby(mask).median(skipna=True)  # median per country
        ref_broadcasted = xr.merge([xr.apply_ufunc(
            lambda country_id: ref[var].sel(country=country_id),
            mask,
            vectorize=True,
            output_dtypes=[float],
        ).rename(var) for var in ref.data_vars])
        ds = ds.where(mean_time >= ref_broadcasted).groupby(mask).mean(
            dim=["lat", "lon"], skipna=True
        )  # only average over locations that at least as good as the median on average

    ds = ds.sel(country=ds["country"].values[ds["country"].values != -1])
    ds["country"] = [countries.loc[number, "names"] for number in ds["country"].values]
    ds = ds.sel(country=[country for country in ds["country"].values if country in EUROPEAN_COUNTRIES])

    return ds