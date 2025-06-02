import argparse
import json
import xarray as xr

def interpolate_radiation(rad_3h):
    """Interpolate 3-hourly radiation to hourly using cubic spline."""
    return rad_3h.resample(time='1H').interpolate('cubic')

def normalize_radiation_weights(rad_1h, rad_3h):
    """Compute hourly radiation weights normalized to 3-hour block sums."""
    # Get 3-hourly block boundaries
    group = rad_1h.time.to_index().floor('3H')
    rad_1h_grouped = rad_1h.groupby(xr.DataArray(group, dims='time', coords={'time': rad_1h.time}))

    def scale_group(grouped_rad):
        total = grouped_rad.sum(dim='time')
        return xr.where(total > 0, grouped_rad / total, 0)

    weights = rad_1h_grouped.map(scale_group)
    return weights

def disaggregate_pv(pv_3h, rad_3h):
    """Disaggregate 3-hourly PV estimates into hourly values using radiation."""
    rad_1h = interpolate_radiation(rad_3h)
    weights = normalize_radiation_weights(rad_1h, rad_3h)

    # Forward-fill PV to hourly for shape
    pv_hourly_approx = pv_3h.resample(time='1H').ffill()

    # Apply weights to disaggregate
    pv_weighted = pv_hourly_approx * weights

    # Re-aggregate and scale to ensure sum match
    pv_reagg = pv_weighted.resample(time='3H').sum()
    scaling = xr.where(pv_reagg > 0, pv_3h / pv_reagg, 0)
    scaling_hourly = scaling.resample(time='1H').ffill()

    pv_hourly_scaled = pv_weighted * scaling_hourly
    return pv_hourly_scaled

def mask_night(pv_hourly, rad_1h, threshold=5):
    """Mask PV output during nighttime using a radiation threshold."""
    is_day = rad_1h > threshold
    return pv_hourly.where(is_day, 0)

def load_data(paths, climate_model_chain, climate_scenario):
    """Load PV and radiation data using a standardized dictionary."""
    pv_path = paths["pv_3h"][climate_model_chain][climate_scenario]
    rad_path = paths["radiation_3h"][climate_model_chain][climate_scenario]

    pv_3h = xr.open_dataset(pv_path, chunks={"time": 100})["pv"]
    rad_3h = xr.open_dataset(rad_path, chunks={"time": 100})["radiation"]

    return pv_3h, rad_3h

def save_output(pv_hourly, out_path):
    """Save the final hourly PV dataset."""
    ds = xr.Dataset({"pv_hourly": pv_hourly})
    ds.to_netcdf(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaggregate 3-hourly PV to hourly using radiation.")
    parser.add_argument("paths_file", type=str, help="Path to the JSON file containing paths.")
    parser.add_argument("climate_model_chain", type=str, help="Climate model chain.")
    parser.add_argument("climate_scenario", type=str, help="Climate scenario.")
    parser.add_argument("--output_path", type=str, default="pv_hourly.nc", help="Output NetCDF file path.")

    args = parser.parse_args()
    paths = json.load(open(args.paths_file))

    pv_3h, rad_3h = load_data(paths, args.climate_model_chain, args.climate_scenario)

    # Disaggregation pipeline
    pv_hourly_raw = disaggregate_pv(pv_3h, rad_3h)
    rad_1h = interpolate_radiation(rad_3h)
    pv_hourly_final = mask_night(pv_hourly_raw, rad_1h)

    save_output(pv_hourly_final, args.output_path)
