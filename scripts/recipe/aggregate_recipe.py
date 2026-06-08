"""Aggregate climate-derived energy profiles to Voronoi nodes and save to HDF5.

For each requested year, this script:
  1. Disaggregates 3-hourly PV output to hourly using bias-corrected rsds.
  2. Groups PV and wind capacity factors by Voronoi node (spatial mean).
  3. Distributes country-level run-of-river and reservoir-inflow time series
     to nodes via installed-capacity weights.
  4. Distributes national heating / cooling demand to nodes via population share.
  5. Assembles all profiles into a single hourly DataFrame and saves it as HDF5.

Usage
-----
python aggregate_recipe.py paths_so4.json \\
    CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1 rcp85 \\
    [--year_start 2040] [--year_end 2060] [--wind_turbine E-126_7580]
"""

import argparse
import json
import os
import time
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import rioxarray as rxr
import xarray as xr
from dask.diagnostics import ProgressBar
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import box as shapely_box

from cordex2energy.conversion_energy.aggregate_data import (
    assign_plants_to_nodes,
    compute_node_capacity_weights,
    distribute_country_timeseries_to_nodes,
)
from cordex2energy.conversion_energy.disaggregate import disaggregate_dataset_radiation
from cordex2energy.utils.constants import REGRIDDED_CHUNKS

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Script-local helpers
# ---------------------------------------------------------------------------

def _fmt_bus(val) -> str:
    """Format a bus identifier: drop trailing `.0` for whole-number floats."""
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(val)


def _grouped_dataset_to_dataframe(
    ds: xr.Dataset,
    prefix: str,
    id_map: dict,
) -> pd.DataFrame:
    """Convert a node-grouped xarray Dataset to a flat DataFrame with prefixed columns."""
    var = list(ds.data_vars)[0]
    da = ds[var]
    return pd.DataFrame(
        da.values,
        index=pd.to_datetime(da.time.values),
        columns=[f"{prefix}.{_fmt_bus(id_map[int(n)])}" for n in da.node.values],
    ).rename_axis("time")

def _build_population_node_mask(
    da_pop: xr.DataArray,
    gdf_voronoi_nodes: gpd.GeoDataFrame,
) -> xr.DataArray:
    """Rasterize Voronoi polygons onto the population grid (fast scanline render)."""
    transform = da_pop.rio.transform()
    height, width = da_pop.shape
    shapes = [
        (geom.__geo_interface__, int(idx))
        for idx, geom in enumerate(gdf_voronoi_nodes.geometry)
    ]
    mask_arr = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype="int32",
        all_touched=False,
    )
    return xr.DataArray(
        np.where(mask_arr >= 0, mask_arr, np.nan).astype("float32"),
        dims=["y", "x"],
        coords={"y": da_pop.y, "x": da_pop.x},
        name="node",
    )


def _build_iso_to_fullname(df_hydro_generators: pd.DataFrame) -> dict:
    """Build ISO-2 → full country name mapping from the hydro-generator catalogue.

    A small supplement covers countries absent from the catalogue (e.g. Denmark,
    Netherlands) whose names must match the demand CSV exactly.
    """
    supplement = {
        "DK": "Denmark",
        "NL": "Netherlands",
        "XK": "Kosovo",
    }
    from_catalogue = (
        df_hydro_generators[["Country", "ISO"]]
        .dropna(subset=["ISO", "Country"])
        .drop_duplicates("ISO")
        .set_index("ISO")["Country"]
        .to_dict()
    )
    # Catalogue entries take precedence over the supplement
    return {**supplement, **from_catalogue}


def _distribute_demand(
    df_demand: pd.DataFrame,
    pop_df: pd.DataFrame,
    iso_to_fullname: dict,
) -> pd.DataFrame:
    """Distribute national demand to nodes weighted by population share.

    Parameters
    ----------
    df_demand:
        Wide-format DataFrame with columns ``country`` and one column per
        time step.
    pop_df:
        DataFrame with columns ``node_idx``, ``COUNTRY`` (ISO-2), and
        ``national_share`` produced by the population-mask step.
    iso_to_fullname:
        Mapping from ISO-2 code to the full country name used in *df_demand*.

    Returns
    -------
    pd.DataFrame
        Indexed by time (``pd.DatetimeIndex``), columns are integer node indices.
    """
    time_cols = [c for c in df_demand.columns if c != "country"]
    country_ts = df_demand.set_index("country")[time_cols]

    d = pop_df[["node_idx", "COUNTRY", "national_share"]].copy()
    d["full_country"] = d["COUNTRY"].map(iso_to_fullname)

    missing_iso = d.loc[d["full_country"].isna(), "COUNTRY"].unique()
    if len(missing_iso):
        print(
            f"  Warning: no ISO mapping for {missing_iso} "
            "— those nodes will be dropped (no demand data)."
        )
    d = d.dropna(subset=["full_country"])

    missing_countries = d.loc[
        ~d["full_country"].isin(country_ts.index), "full_country"
    ].unique()
    if len(missing_countries):
        print(f"  Warning: no demand time series for {missing_countries} — NaN assigned.")

    d = d.merge(country_ts, left_on="full_country", right_index=True, how="left")
    df_out = d[time_cols].multiply(d["national_share"].values, axis=0)
    df_out.index = d["node_idx"].values.astype(int)
    df_out = df_out.T
    df_out.index = pd.to_datetime(df_out.index)
    df_out.index.name = "time"
    df_out.columns.name = "node"
    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate climate-derived energy profiles to Voronoi nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths_file",
        help="Path to the JSON paths file (e.g. paths_so4.json).",
    )
    parser.add_argument(
        "climate_model",
        help="Climate model chain (e.g. CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1).",
    )
    parser.add_argument(
        "climate_scenario",
        help="Climate scenario (e.g. rcp85).",
    )
    parser.add_argument(
        "--year_start",
        type=int,
        default=2040,
        help="First year to process (default: 2040).",
    )
    parser.add_argument(
        "--year_end",
        type=int,
        default=2060,
        help="Last year to process, inclusive (default: 2060).",
    )
    parser.add_argument(
        "--wind_turbine",
        default="E-126_7580",
        help="Wind turbine identifier used in the zarr file name (default: E-126_7580).",
    )
    args = parser.parse_args()

    dask.config.set(num_workers=15, threads_per_worker=16)
    dask.config.set({"array.chunk-size": "400MiB", "logging.distributed": "error"})

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])
    path_data_recipe = path_data_processed / "recipe"
    path_energy_europe = path_data_processed / "energy" / "europe"

    climate_model = args.climate_model
    climate_scenario = args.climate_scenario
    wind_turbine = args.wind_turbine
    years = list(range(args.year_start, args.year_end + 1))
    _year_pattern = "|".join(str(y) for y in years)

    # Strip ensemble member (e.g. _r1i1p1) to get the c2e model name,
    # and convert "rcp85" → "85" for the file-name suffix.
    c2e_model = climate_model.rsplit("_", 1)[0]
    scen_suffix = climate_scenario.replace("rcp", "")
    path_c2e = path_data_processed / "c2e" / "outputs" / c2e_model / "output_variables"

    # ── Load gridded energy datasets ──────────────────────────────────────────
    print("Loading PV and wind zarr datasets …")
    ds_pv = xr.open_zarr(
        path_energy_europe
        / "pv"
        / f"{climate_model}_{climate_scenario}"
        / f"pv_{climate_model}_{climate_scenario}.zarr"
    ).sel(time=slice(str(years[0]), str(years[-1])))

    ds_wind = xr.open_zarr(
        path_energy_europe
        / "wind_power"
        / f"{climate_model}_{climate_scenario}"
        / f"wind_power_{climate_model}_{climate_scenario}_{wind_turbine}.zarr"
    ).sel(time=slice(str(years[0]), str(years[-1])))

    # ── Load tabular hydropower & demand CSVs ─────────────────────────────────
    print("Loading tabular hydropower and demand data …")

    def _load_csv(name: str) -> pd.DataFrame:
        df = pd.read_csv(path_c2e / name)
        time_cols = [c for c in df.columns if c != "country"]
        keep = [c for c in time_cols if any(str(y) in c for y in years)]
        return df[["country"] + keep]

    df_hydro_ror = _load_csv(
        f"hydro_ror_CORDEX_{c2e_model}_{scen_suffix}.csv"
    )
    df_hydro_inflow = _load_csv(
        f"hydro_inflow_CORDEX_{c2e_model}_{scen_suffix}.csv"
    )
    df_heating_demand = _load_csv(
        f"heating-demand_CORDEX_{c2e_model}_{scen_suffix}.csv"
    )
    df_cooling_demand = _load_csv(
        f"cooling-demand_CORDEX_{c2e_model}_{scen_suffix}.csv"
    )

    # ── Load spatial / static data ────────────────────────────────────────────
    print("Loading spatial data …")

    pop_path = (
        path_data_processed
        / "pop"
        / "gpw_v4"
        / "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"
    )
    da_pop = (
        rxr.open_rasterio(pop_path, masked=True)
        .squeeze("band", drop=True)
        .rio.clip_box(
            minx=float(ds_pv.lon.min()),
            miny=float(ds_pv.lat.min()),
            maxx=float(ds_pv.lon.max()),
            maxy=float(ds_pv.lat.max()),
        )
    )
    da_pop.name = "pop_count"

    df_hydro_generators = pd.read_excel(path_data_recipe / "hydro_generators.xlsx")

    _extent = shapely_box(
        float(ds_pv.lon.min()), float(ds_pv.lat.min()),
        float(ds_pv.lon.max()), float(ds_pv.lat.max()),
    )

    def _load_voronoi(csv_name: str) -> gpd.GeoDataFrame:
        df = pd.read_csv(path_data_recipe / csv_name)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
            crs="EPSG:4326",
        )
        return gdf[gdf.geometry.within(_extent)].reset_index(drop=True)

    gdf_voronoi_nodes = _load_voronoi("Node_demand_full.csv")
    gdf_voronoi_nodes_wind = _load_voronoi("Node_voronoi_cleaned.csv")

    # ── Build Voronoi node masks on the PV grid ───────────────────────────────
    print("Building node masks …")
    _pv_ref = ds_pv[list(ds_pv.data_vars)[0]].isel(time=0)

    nodes_regions = regionmask.Regions(
        name="nodes",
        numbers=range(len(gdf_voronoi_nodes)),
        names=gdf_voronoi_nodes["BUS_I"],
        outlines=gdf_voronoi_nodes.geometry.values,
    )
    mask_nodes = nodes_regions.mask(_pv_ref)
    mask_nodes.name = "node"

    nodes_wind_regions = regionmask.Regions(
        name="nodes",
        numbers=range(len(gdf_voronoi_nodes_wind)),
        names=gdf_voronoi_nodes_wind["BUS_I"],
        outlines=gdf_voronoi_nodes_wind.geometry.values,
        overlap=False,
    )
    mask_nodes_wind = nodes_wind_regions.mask(_pv_ref)
    mask_nodes_wind.name = "node"

    # ── PV: disaggregate 3-hourly → hourly using bias-corrected rsds ─────────
    print("Disaggregating PV to hourly (scipy cubic spline) …")
    rsds_3h = xr.open_zarr(
        path_data_processed
        / "cordex_processed"
        / f"{climate_model}_{climate_scenario}"
        / "bias_corrected"
        / "3hr"
        / "rsds.zarr",
        chunks="auto",
    )["rsds"].sel(time=slice(str(years[0]), str(years[-1])))

    pv_ref_da = ds_pv[list(ds_pv.data_vars)[0]]
    rsds_3h = rsds_3h.sel(
        lat=pv_ref_da.lat, lon=pv_ref_da.lon, method="nearest", tolerance=0.05
    )

    with ProgressBar():
        ds_pv_hourly = disaggregate_dataset_radiation(
            ds_pv.chunk(REGRIDDED_CHUNKS),
            rsds_3h.chunk(REGRIDDED_CHUNKS),
            method="scipy",
            threshold=5.0,
        ).compute()

    # ── Group PV and wind by Voronoi node ─────────────────────────────────────
    print("Grouping PV by demand node …")
    ds_pv_grouped = ds_pv_hourly.groupby(mask_nodes).mean(
        dim=["lat", "lon"], skipna=True
    )

    print("Interpolating wind to hourly and grouping by wind node …")
    with ProgressBar():
        ds_wind_grouped = (
            ds_wind.resample(time="h").interpolate("linear")
            .groupby(mask_nodes_wind).mean(dim=["lat", "lon"], skipna=True)
            .compute()
        )

    # ── Run-of-river hydropower ───────────────────────────────────────────────
    print("Processing run-of-river hydropower …")
    ror_gen = (
        df_hydro_generators.loc[
            df_hydro_generators["Technology"].str.contains("Run-of-River", na=False)
            & df_hydro_generators["Country"].isin(df_hydro_ror["country"])
        ]
        .dropna(subset=["Capacity"])
        .pipe(lambda d: d[d["Capacity"] > 0])
        .reset_index(drop=True)
    )

    gdf_ror = gpd.GeoDataFrame(
        ror_gen[["Name", "Country", "ISO", "Capacity"]],
        geometry=gpd.points_from_xy(ror_gen["lon"], ror_gen["lat"]),
        crs="EPSG:4326",
    )
    gdf_ror_nodes = assign_plants_to_nodes(gdf_ror, gdf_voronoi_nodes)
    node_cap_ror = compute_node_capacity_weights(gdf_ror_nodes)

    ror_time_cols = [c for c in df_hydro_ror.columns if c != "country"]
    ror_country_ts = df_hydro_ror.set_index("country")[ror_time_cols]
    df_ror_nodes = distribute_country_timeseries_to_nodes(
        ror_country_ts, node_cap_ror, ror_time_cols
    )

    # ── Reservoir inflows ─────────────────────────────────────────────────────
    print("Processing reservoir inflows …")
    inflow_gen = (
        df_hydro_generators.loc[
            ~df_hydro_generators["Technology"].str.contains("Run-of-River", na=False)
            & df_hydro_generators["Country"].isin(df_hydro_inflow["country"])
        ]
        .dropna(subset=["Capacity"])
        .pipe(lambda d: d[d["Capacity"] > 0])
        .reset_index(drop=True)
    )

    gdf_inflow = gpd.GeoDataFrame(
        inflow_gen[["Name", "Country", "ISO", "Capacity"]],
        geometry=gpd.points_from_xy(inflow_gen["lon"], inflow_gen["lat"]),
        crs="EPSG:4326",
    )
    gdf_inflow_nodes = assign_plants_to_nodes(gdf_inflow, gdf_voronoi_nodes)
    node_cap_inflow = compute_node_capacity_weights(gdf_inflow_nodes)

    inflow_time_cols = [c for c in df_hydro_inflow.columns if c != "country"]
    inflow_country_ts = df_hydro_inflow.set_index("country")[inflow_time_cols]
    df_inflow_nodes = distribute_country_timeseries_to_nodes(
        inflow_country_ts, node_cap_inflow, inflow_time_cols
    )

    # ── Heating and cooling demand ────────────────────────────────────────────
    print("Processing demand (heating and cooling) …")
    mask_pop = _build_population_node_mask(da_pop, gdf_voronoi_nodes)
    pop_per_node = da_pop.groupby(mask_pop).sum()

    pop_df = (
        pop_per_node.to_dataframe(name="population")
        .reset_index()
        .rename(columns={"node": "node_idx"})
    )
    pop_df["COUNTRY"] = pop_df["node_idx"].map(gdf_voronoi_nodes["COUNTRY"].to_dict())
    pop_df["national_population"] = pop_df.groupby("COUNTRY")["population"].transform("sum")
    pop_df["national_share"] = pop_df["population"] / pop_df["national_population"]

    iso_to_fullname = _build_iso_to_fullname(df_hydro_generators)
    df_heat_nodes = _distribute_demand(df_heating_demand, pop_df, iso_to_fullname)
    df_cool_nodes = _distribute_demand(df_cooling_demand, pop_df, iso_to_fullname)

    # ── Per-year assembly and save ────────────────────────────────────────────
    node_to_bus = gdf_voronoi_nodes["BUS_I"].to_dict()
    node_to_bus_w = gdf_voronoi_nodes_wind["BUS_I"].to_dict()

    output_folder = path_data_recipe / climate_model / climate_scenario
    output_folder.mkdir(parents=True, exist_ok=True)

    for year in years:
        t0 = time.perf_counter()
        print(f"\n{'='*60}\n  Assembling profiles for {year}\n{'='*60}")

        _wind_y = ds_wind_grouped.sel(time=str(year))
        if _wind_y.time.size == 0:
            print(f"  WARNING: no wind data for {year} — skipping.")
            continue

        var_wind = list(_wind_y.data_vars)[0]
        hourly_idx = pd.to_datetime(_wind_y[var_wind].time.values)
        leap_mask = (hourly_idx.month == 2) & (hourly_idx.day == 29)
        if leap_mask.any():
            print(f"  Dropping {leap_mask.sum()} leap-day hour(s).")
        hourly_idx = hourly_idx[~leap_mask]
        print(f"  Reference grid : {hourly_idx[0]}  →  {hourly_idx[-1]}  ({len(hourly_idx):,} steps)")

        # 1. PV — shift label by −30 min to align with interval start
        _pv_raw = _grouped_dataset_to_dataframe(
            ds_pv_grouped.sel(time=str(year)), "profile.generation.solar", node_to_bus
        )
        _pv_raw.index -= pd.Timedelta("30min")
        df_pv_1h = _pv_raw.reindex(hourly_idx)

        # 2. Wind
        df_wind_1h = _grouped_dataset_to_dataframe(
            _wind_y, "profile.generation.wind", node_to_bus_w
        ).reindex(hourly_idx)

        # 3. ROR (daily → uniform hourly by dividing by 24)
        _ror = df_ror_nodes.loc[str(year)].copy()
        _ror.index = _ror.index.normalize()
        df_ror_1h = (
            _ror.reindex(hourly_idx, method="ffill") / 24
        ).rename(columns=lambda c: f"profile.generation.ror.{_fmt_bus(c)}")

        # 4. Reservoir inflows (weekly → uniform hourly)
        _inf = df_inflow_nodes.loc[str(year)].sort_index().copy()
        _td = _inf.index.to_series().diff().shift(-1)
        _td.iloc[-1] = _td.iloc[-2]
        df_inflow_1h = (
            _inf.div(_td.dt.total_seconds() / 3600, axis=0)
            .reindex(hourly_idx, method="ffill")
            .ffill()
            .bfill()
        ).rename(columns=lambda c: f"profile.generation.inflow.{_fmt_bus(c)}")

        # 5. Heating demand (3-hourly → hourly, linear interpolation)
        df_heat_1h = (
            df_heat_nodes.loc[str(year)].resample("h").asfreq().interpolate("time")
            .reindex(hourly_idx)
            .rename(columns=lambda c: f"profile.demand.heating.{_fmt_bus(node_to_bus[int(c)])}")
        )

        # 6. Cooling demand (3-hourly → hourly, linear interpolation)
        df_cool_1h = (
            df_cool_nodes.loc[str(year)].resample("h").asfreq().interpolate("time")
            .reindex(hourly_idx)
            .rename(columns=lambda c: f"profile.demand.cooling.{_fmt_bus(node_to_bus[int(c)])}")
        )

        # 7. Concatenate
        df_profiles = pd.concat(
            [df_pv_1h, df_wind_1h, df_ror_1h, df_inflow_1h, df_heat_1h, df_cool_1h],
            axis=1,
        )
        df_profiles.index.name = "time"

        print(f"  Profile table  : {df_profiles.shape[0]:,} timesteps × {df_profiles.shape[1]:,} columns")
        print(f"  Time range     : {df_profiles.index[0]}  →  {df_profiles.index[-1]}")
        for pfx in [
            "profile.generation.solar",
            "profile.generation.wind",
            "profile.generation.ror",
            "profile.generation.inflow",
            "profile.demand.heating",
            "profile.demand.cooling",
        ]:
            cols = df_profiles.columns[df_profiles.columns.str.startswith(pfx)]
            nan_pct = df_profiles[cols].isna().mean().mean() * 100
            all_nan = int(df_profiles[cols].isna().all().sum())
            print(
                f"    {pfx:<35s}: {len(cols):>5} cols  "
                f"(avg NaN: {nan_pct:.1f}%,  fully-NaN: {all_nan})"
            )

        # 8. Save
        output_path = (
            output_folder / f"profiles_{climate_model}_{climate_scenario}_{year}.h5"
        )
        df_profiles.to_hdf(
            output_path, key="profiles", mode="w", complevel=5, complib="blosc"
        )
        elapsed = time.perf_counter() - t0
        print(
            f"\n  Saved  →  {output_path}"
            f"  ({output_path.stat().st_size / 1e6:.1f} MB)"
            f"  [{elapsed:.0f}s]"
        )

    print(f"\nDone — {len(years)} year(s) processed.")


if __name__ == "__main__":
    main()
