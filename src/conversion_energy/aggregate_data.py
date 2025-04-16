import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point


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
