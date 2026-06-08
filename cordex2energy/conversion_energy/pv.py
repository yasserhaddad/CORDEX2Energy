import calendar
import json
import multiprocessing
import numba
from typing import Union
import numpy as np
import pandas as pd
import pvlib
import xarray as xr
from pvlib.irradiance import erbs
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

DEFAULT_PV_CONFIGURATION = {
    "temperature_model_parameters": TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ],
    "module_parameters": {
        "pdc0": 300,  # Peak DC capacity in watts (W)
        "gamma_pdc": -0.004,  # Power temperature coefficient (% per °C)
    },  # Module power and temp coefficient
    "inverter_parameters": {
        "pdc0": 3000,  # Maximum DC input capacity in watts (W)
        "eta_inv_nom": 0.96,  # Inverter efficiency (96%)
    },  # Inverter power and efficiency
    "racking_model": "open_rack",  # Typical for utility-scale PV
}

ROOFTOP_PV_CONFIGURATION = {
    "temperature_model_parameters": TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "close_mount_glass_glass"
    ],
    "module_parameters": {
        "pdc0": 400,  # Rooftop PV modules are commonly 380–450W
        "gamma_pdc": -0.0045,  # Rooftop panels are more affected by heat, slightly higher loss
    },
    "inverter_parameters": {
        "pdc0": 1200,  # A typical residential inverter size
        "eta_inv_nom": 0.97,  # Newer string inverters have 97% efficiency
    },
    "racking_model": "close_mount",  # Rooftop PV is usually close-mounted
}

NORMALISED_ROOFTOP_PV_CONFIGURATION = {
    "temperature_model_parameters": TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "close_mount_glass_glass"
    ],
    "module_parameters": {
        "pdc0": 1000,
        "gamma_pdc": -0.004,  # Typical heat coefficient for rooftop panels
    },
    "inverter_parameters": {
        "pdc0": 1200,
        "eta_inv_nom": 0.97,  # 97% inverter efficiency (modern string inverter)
    },
    "racking_model": "close_mount",  # Rooftop PV is usually close-mounted
}

# ---- wrappers that return plain ndarrays (dask/xr friendly) ----

def _solarpos_zen_azi(times_1d, lat, lon, altitude):
    # times_1d: 1D numpy datetime64[ns]; lat/lon/altitude: scalars
    times = pd.DatetimeIndex(times_1d)
    sp = pvlib.solarposition.spa_python(
        times,
        lat,
        lon,
        altitude=altitude,
    )
    return sp["zenith"].to_numpy(dtype=float), sp["azimuth"].to_numpy(dtype=float)


def _erbs_dni_dhi(ghi_1d, zenith_1d, times_1d):
    # all 1D over time, return 1D dni,dhi
    times = pd.DatetimeIndex(times_1d)
    out = pvlib.irradiance.erbs(ghi_1d, zenith_1d, times)
    return out["dni"].to_numpy(dtype=float), out["dhi"].to_numpy(dtype=float)

@numba.njit(parallel=True, fastmath=True)
def _erbs_fast_full(ghi, zenith, doy,
                   min_cos_zenith=0.065,
                   max_zenith=87.0,
                   max_clearness_index=1.0):
    """
    Estimate direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI) from global horizontal irradiance (GHI)
    using the Erbs model. Leverage numba to make it faster.

    Parameters
    ----------
    ghi : array_like
        Global horizontal irradiance [W/m²].
    zenith : array_like
        True (not refraction-corrected) solar zenith angles in decimal degrees.
    doy : array_like or scalar
        Day of year (1-366) or array of days of year.
    min_cos_zenith : float, optional
        Default is 0.065 (equivalent to zenith = 86.273 degrees).
    max_zenith : float, optional
        Maximum value of zenith (degrees) to allow in DNI calculation. DNI will be set to 0 for times
        with zenith values greater than max_zenith. Default is 87.0.
    max_clearness_index : float, optional
        Maximum value of clearness index kt. Default is 1.0.

    Returns
    -------
    dni : ndarray
        Direct normal irradiance [W/m²].
    dhi : ndarray
        Diffuse horizontal irradiance [W/m²].

    Notes
    -----
    Implements the Erbs et al. (1982) model for splitting GHI into DNI and DHI. Uses vectorized operations
    where possible and a loop for the Erbs diffuse fraction calculation.
    """
    n = ghi.size
    dni = np.zeros(n, dtype=np.float64)
    dhi = np.zeros(n, dtype=np.float64)
    kt = np.zeros(n, dtype=np.float64)

    # --- extraterrestrial irradiance ---
    Gsc = 1367.0
    B = 2.0 * np.pi * (doy - 1) / 365.0
    RoverR0sqrd = (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) +
                   0.000719 * np.cos(2*B) + 7.7e-05 * np.sin(2*B))
    dni_extra = Gsc * RoverR0sqrd

    # --- clearness index ---
    cosz = np.cos(np.radians(zenith))
    I0h = dni_extra * np.maximum(cosz, min_cos_zenith)
    kt[:] = ghi / I0h
    kt = np.clip(kt, 0.0, max_clearness_index)

    # --- Erbs diffuse fraction and split ---
    for i in numba.prange(n):
        if kt[i] <= 0.22:
            df = 1.0 - 0.09 * kt[i]
        elif kt[i] <= 0.8:
            df = (0.9511 - 0.1604*kt[i] + 4.388*kt[i]**2 -
                  16.638*kt[i]**3 + 12.336*kt[i]**4)
        else:
            df = 0.165

        dhi[i] = df * ghi[i]
        dni_i = (ghi[i] - dhi[i]) / cosz[i]

        if zenith[i] > max_zenith or ghi[i] < 0 or dni_i < 0:
            dni[i] = 0.0
            dhi[i] = ghi[i]
        else:
            dni[i] = dni_i

    return dni, dhi



def _poa_global_only(tilt, azimuth, dni_1d, ghi_1d, dhi_1d, zenith_1d, azimuth_sun_1d):
    out = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt),
        surface_azimuth=float(azimuth),
        dni=dni_1d,
        ghi=ghi_1d,
        dhi=dhi_1d,
        solar_zenith=zenith_1d,
        solar_azimuth=azimuth_sun_1d,
    )
    return out["poa_global"]


def compute_pv_from_config(
    ghi: xr.DataArray,  # (time, lat, lon) W/m²
    t_air: xr.DataArray,  # (time, lat, lon) °C
    wind: xr.DataArray,  # (time, lat, lon) m/s
    pv_configuration: dict,  # e.g. ROOFTOP_PV_CONFIGURATION
    tilt: float = 30.0,
    azimuth: float = 180.0,
    system_loss: float = 0.14,
    altitude: xr.DataArray | float | None = 0.0,
    dtype: str = "float32",
) -> xr.DataArray:
    """
    Fully vectorized PV calculation using PVWatts DC+AC and SAPM temperature model
    from pv_configuration.

    Returns AC power (W) as an xarray.DataArray with dims (time, lat, lon).
    """
    ghi = ghi.astype(dtype)
    t_air = t_air.astype(dtype)
    wind = wind.astype(dtype)

    # Extract config
    temp_params = pv_configuration["temperature_model_parameters"]
    pdc0 = pv_configuration["module_parameters"]["pdc0"]
    gamma_pdc = pv_configuration["module_parameters"]["gamma_pdc"]
    eta_inv_nom = pv_configuration["inverter_parameters"]["eta_inv_nom"]

    # --- 1) solar position (vectorized over lat/lon) ---
    zen, azi = xr.apply_ufunc(
        _solarpos_zen_azi,
        ghi["time"],
        ghi["lat"],
        ghi["lon"],
        altitude,
        input_core_dims=[["time"], [], [], []],
        output_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ghi.dtype, ghi.dtype],
    )
    night = (zen > 90) | (ghi < 1.0)

    # --- 2) ERBS (GHI → DNI/DHI) ---
    dni, dhi = xr.apply_ufunc(
        _erbs_fast_full,
        ghi,
        zen,
        ghi["time"].dt.dayofyear,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ghi.dtype, ghi.dtype],
    )
    dni = dni.clip(min=0)
    dhi = dhi.clip(min=0)

    # --- 3) POA global ---
    poa = xr.apply_ufunc(
        _poa_global_only,
        xr.DataArray(tilt),
        xr.DataArray(azimuth),
        dni,
        ghi,
        dhi,
        zen,
        azi,
        input_core_dims=[[], [], ["time"], ["time"], ["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ghi.dtype],
    ).where(~night, 0)

    # --- 4) Module temperature (SAPM cell) ---
    t_cell = xr.apply_ufunc(
        pvlib.temperature.sapm_cell,
        poa,
        t_air,
        wind,
        kwargs=temp_params,
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ghi.dtype],
    )

    # --- 5) PVWatts DC → AC ---
    pdc = (pdc0 * (poa / 1000.0) * (1 + gamma_pdc * (t_cell - 25.0))).astype(ghi.dtype)
    pac = xr.apply_ufunc(
        pvlib.inverter.pvwatts,
        pdc,
        pdc0,
        kwargs={"eta_inv_nom": eta_inv_nom},
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ghi.dtype],
    )

    pac = (pac.clip(min=0) * (1 - system_loss)).transpose("time", ...)
    pac.name = "ac_power"
    pac.attrs.update(
        dict(
            units="W",
            model="PVWatts + SAPM temperature",
            tilt=tilt,
            azimuth=azimuth,
            pv_configuration=pv_configuration,
            system_loss=system_loss,
            dtype=dtype,
        )
    )
    return pac.to_dataset()


def compute_pv_for_pixel(
    time: Union[pd.DatetimeIndex, np.ndarray[pd.DatetimeIndex]],
    lat: float,
    lon: float,
    rsds: float,
    temp: float,
    wind: float,
    tilt: float,
    azimuth: float,
    pv_configuration: dict,
    system_loss: float = 0.14,
    altitude: float = None,
) -> np.ndarray:
    """
    Compute solar power for a single pixel (lat, lon) over time.
    This function is vectorized over time.

    Parameters
    ----------
    time : Union[pd.DatetimeIndex, np.ndarray[pd.DatetimeIndex]]
        Time index for the data.
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    rsds : float
        Downward shortwave radiation data for the pixel (in W/m^2).
    temp : float
        Air temperature data for the pixel (in °C).
    wind : float
        Wind speed data for the pixel (in m/s).
    tilt : float
        Tilt angle of the PV panels.
    azimuth : float
        Azimuth angle of the PV panels.
    pv_configuration : dict
        Dictionary containing PV system configuration parameters.
    system_loss : float, optional
        System loss factor. Default is 0.14.
    altitude : float, optional
        Altitude of the location. Default is None.

    Returns
    -------
    np.ndarray
        Estimated AC power output over time for the given pixel.
    """

    # Create location object
    location = Location(lat, lon, altitude=altitude)

    # Define PV System
    system = PVSystem(
        module_parameters=pv_configuration["module_parameters"],
        inverter_parameters=pv_configuration["inverter_parameters"],
        racking_model=pv_configuration["racking_model"],
        temperature_model_parameters=pv_configuration["temperature_model_parameters"],
        surface_tilt=tilt,
        surface_azimuth=azimuth,
    )

    # Create ModelChain
    model_chain = ModelChain(system, location, aoi_model="martin_ruiz")

    # Compute solar position
    solar_position = location.get_solarposition(time)

    # Estimate DNI & DHI using Erbs model
    dni_dhi = erbs(rsds, solar_position["zenith"], time)
    dni, dhi = dni_dhi["dni"], dni_dhi["dhi"]

    # Compute POA Irradiance
    poa_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=dni,
        ghi=rsds,
        dhi=dhi,
        solar_zenith=solar_position["zenith"],
        solar_azimuth=solar_position["azimuth"],
    )

    # Compute Module Temperature
    module_temp = pvlib.temperature.sapm_cell(
        poa_irrad["poa_global"],
        temp,
        wind,
        **pv_configuration["temperature_model_parameters"],
    )

    # Run model
    weather_df = pd.DataFrame(
        {
            "ghi": rsds,
            "dni": dni,
            "dhi": dhi,
            "temp_air": temp,
            "wind_speed": wind,
            "temp_module": module_temp,
        }
    )

    model_chain.run_model(weather=weather_df)

    # Return AC power output
    return model_chain.results.ac.values * (1 - system_loss)


def compute_pv_for_pixel_all_timesteps(
    da_radiation: xr.DataArray,
    da_temperature: xr.DataArray,
    da_wind_speed: xr.DataArray,
    lat: float,
    lon: float,
    tilt: float = 30,
    azimuth: float = 180,
    pv_configuration: dict = DEFAULT_PV_CONFIGURATION,
    system_loss: float = 0.14,
    da_altitude: xr.DataArray = None,
) -> np.ndarray:
    """
    Compute solar power for a specific location (lat, lon) over all timesteps.

    Parameters
    ----------
    da_radiation : xr.DataArray
        Global horizontal irradiance data array with dimensions (time, lat, lon).
    da_temperature : xr.DataArray
        Air temperature data array with dimensions (time, lat, lon).
    da_wind_speed : xr.DataArray
        Wind speed data array with dimensions (time, lat, lon).
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    tilt : float, optional
        Tilt angle of the PV panels. Default is 30 degrees.
    azimuth : float, optional
        Azimuth angle of the PV panels. Default is 180 degrees.
    pv_configuration : dict
        Dictionary containing PV system configuration parameters.
        Default is DEFAULT_PV_CONFIGURATION.
    system_loss : float, optional
        System loss factor. Default is 0.14.
    da_altitude : xr.DataArray, optional
        Altitude data array with dimensions (lat, lon). Default is None.
    Returns
    -------
    np.ndarray
        Estimated AC power output over time for the given location.
    """
    # Extract data arrays for the current grid point
    rsds = da_radiation.sel(lat=lat, lon=lon).load().values
    temperature = da_temperature.sel(lat=lat, lon=lon).load().values
    wind_speed = da_wind_speed.sel(lat=lat, lon=lon).load().values
    altitude = (
        da_altitude.sel(lat=lat, lon=lon).values if da_altitude is not None else None
    )
    times = pd.to_datetime(da_radiation["time"].values)

    # Compute power for the location over all timesteps
    results = compute_pv_for_pixel(
        time=times,
        lat=lat,
        lon=lon,
        rsds=rsds,
        temp=temperature,
        wind=wind_speed,
        tilt=tilt,
        azimuth=azimuth,
        pv_configuration=pv_configuration,
        system_loss=system_loss,
        altitude=altitude,
    )

    da_results = xr.DataArray(
        results.reshape(-1, 1, 1),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [lat], "lon": [lon]},
        name="ac_power",
        attrs={
            "description": "Estimated AC power output over time",
            "units": "W",
            "tilt": tilt,
            "azimuth": azimuth,
            "pv_configuration": json.dumps(pv_configuration),
            "system_loss": system_loss,
        },
    )

    return da_results


def compute_pv_for_grid_worker(
    lat_lon: tuple,
    da_radiation: xr.DataArray,
    da_temperature: xr.DataArray,
    da_wind_speed: xr.DataArray,
    tilt: float,
    azimuth: float,
    pv_configuration: dict,
    system_loss: float,
    da_altitude: xr.DataArray = None,
) -> xr.DataArray:
    """
    Compute solar power for a specific location (lat, lon) over all timesteps.

    Parameters
    ----------
    lat_lon : tuple
        Tuple containing latitude and longitude (lat, lon).
    da_radiation : xr.DataArray
        Global horizontal irradiance data array with dimensions (time, lat, lon).
    da_temperature : xr.DataArray
        Air temperature data array with dimensions (time, lat, lon).
    da_wind_speed : xr.DataArray
        Wind speed data array with dimensions (time, lat, lon).
    tilt : float
        Tilt angle of the PV panels.
    azimuth : float
        Azimuth angle of the PV panels.
    pv_configuration : dict
        Dictionary containing PV system configuration parameters.
    system_loss : float
        System loss factor.
    da_altitude : xr.DataArray, optional
        Altitude data array with dimensions (lat, lon). Default is None.

    Returns
    -------
    xr.DataArray
        Estimated AC power output over time for the given location.
    """
    lat, lon = lat_lon
    return compute_pv_for_pixel_all_timesteps(
        da_radiation,
        da_temperature,
        da_wind_speed,
        lat=lat,
        lon=lon,
        tilt=tilt,
        azimuth=azimuth,
        pv_configuration=pv_configuration,
        system_loss=system_loss,
        da_altitude=da_altitude,
    )


def compute_pv_for_grid(
    da_radiation: xr.DataArray,
    da_temperature: xr.DataArray,
    da_wind_speed: xr.DataArray,
    tilt: float = 30,
    azimuth: float = 180,
    pv_configuration: dict = DEFAULT_PV_CONFIGURATION,
    system_loss: float = 0.14,
    da_altitude: xr.DataArray = None,
    num_workers: int = 4,
) -> xr.DataArray:
    """
    Compute solar power for all combinations of lat and lon using multiprocessing.

    Parameters
    ----------
    da_radiation : xr.DataArray
        Global horizontal irradiance data array with dimensions (time, lat, lon).
    da_temperature : xr.DataArray
        Air temperature data array with dimensions (time, lat, lon).
    da_wind_speed : xr.DataArray
        Wind speed data array with dimensions (time, lat, lon).
    tilt : float, optional
        Tilt angle of the PV panels. Default is 30 degrees.
    azimuth : float, optional
        Azimuth angle of the PV panels. Default is 180 degrees.
    pv_configuration : dict, optional
        Dictionary containing PV system configuration parameters.
        Default is DEFAULT_PV_CONFIGURATION.
    system_loss : float, optional
        System loss factor. Default is 0.14.
    da_altitude : xr.DataArray, optional
        Altitude data array with dimensions (lat, lon). Default is None.
    num_workers : int, optional
        Number of worker processes to use. Default is 4.

    Returns
    -------
    xr.DataArray
        Estimated AC power output over time for all locations.
    """
    lats = da_radiation.lat.values
    lons = da_radiation.lon.values

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(num_workers) as pool:
        results = pool.starmap(
            compute_pv_for_grid_worker,
            [
                (
                    lat_lon,
                    da_radiation,
                    da_temperature,
                    da_wind_speed,
                    tilt,
                    azimuth,
                    pv_configuration,
                    system_loss,
                    da_altitude,
                )
                for lat_lon in [(lat, lon) for lat in lats for lon in lons]
            ],
        )

    return xr.combine_by_coords(results)

# ---- PV disaggregation: 3-hourly → hourly --------------------------------

def compute_clearsky_profile(
    lat: float,
    lon: float,
    doy: int,
    tilt: float,
    azimuth: float,
) -> np.ndarray:
    """Compute a normalized clear-sky PV profile for one location and day of year.

    Uses pvlib's Ineichen clear-sky model to estimate plane-of-array (POA)
    irradiance for each of the 24 hours in the given day, then normalizes so
    hourly values sum to 1.  This is the scalar kernel called by
    :func:`generate_clearsky_profiles` through ``xr.apply_ufunc``.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (North positive).
    lon : float
        Longitude in decimal degrees (East positive).
    doy : int
        Day of year (1–366).
    tilt : float
        Panel tilt angle from horizontal (degrees).
    azimuth : float
        Panel azimuth angle (degrees; 0 = North, 90 = East, 180 = South,
        270 = West).

    Returns
    -------
    np.ndarray, shape (24,), dtype float32
        Normalized hourly PV profile.  Values sum to 1 when sunlight is
        present; all-zero for polar-winter days.
    """
    date = pd.Timestamp("2050-01-01") + pd.Timedelta(days=int(doy) - 1)
    times = pd.date_range(date, periods=24, freq="h", tz="UTC")
    location = Location(lat, lon, tz="UTC")
    solpos = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
    )
    profile = np.clip(poa["poa_global"].values, 0.0, None).astype(np.float32)
    total = profile.sum()
    return profile / total if total > 0.0 else np.zeros(24, dtype=np.float32)


def generate_clearsky_profiles(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    tilt: float,
    azimuth: float,
    n_days: int = 366,
    chunks: dict | None = None,
) -> xr.DataArray:
    """Generate normalized clear-sky PV profiles for a grid of locations.

    Calls :func:`compute_clearsky_profile` for every (lat, lon, day) cell via
    ``xr.apply_ufunc`` with ``dask="parallelized"``.  Pass ``chunks`` to
    obtain a lazy Dask-backed result that can be computed in parallel by any
    Dask scheduler (e.g. ``dask.distributed``).

    Parameters
    ----------
    lat_vals : array-like, shape (N_lat,)
        Latitude values in decimal degrees.
    lon_vals : array-like, shape (N_lon,)
        Longitude values in decimal degrees.
    tilt : float
        Panel tilt angle from horizontal (degrees).
    azimuth : float
        Panel azimuth angle (degrees; 0 = North, 90 = East, 180 = South,
        270 = West).
    n_days : int, optional
        Number of days to generate (use 365 for non-leap, 366 to cover all
        possible days of year).  Default is 366.
    chunks : dict or None, optional
        Dask chunk sizes along the first three dims, e.g.
        ``{"lat": 10, "lon": 10, "day": 30}``.  When *None* (default) the
        result is eagerly computed as a NumPy-backed DataArray.

    Returns
    -------
    xr.DataArray, dims (lat, lon, day, hour), dtype float32
        Normalized clear-sky PV profile.  ``day`` runs 1–n_days, ``hour``
        runs 0–23.  Each (lat, lon, day) slice sums to 1 when daylight is
        present, and to 0 for polar-winter days.
    """
    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = np.asarray(lon_vals, dtype=float)
    days = np.arange(1, n_days + 1)

    lat_grid, lon_grid, day_grid = np.meshgrid(lat_vals, lon_vals, days, indexing="ij")

    da_lat = xr.DataArray(lat_grid, dims=["lat", "lon", "day"])
    da_lon = xr.DataArray(lon_grid, dims=["lat", "lon", "day"])
    da_day = xr.DataArray(day_grid, dims=["lat", "lon", "day"])

    if chunks is not None:
        da_lat = da_lat.chunk(chunks)
        da_lon = da_lon.chunk(chunks)
        da_day = da_day.chunk(chunks)

    def _wrapper(lat, lon, doy):
        return compute_clearsky_profile(
            lat.item(), lon.item(), int(doy.item()), tilt, azimuth
        )

    profiles = xr.apply_ufunc(
        _wrapper,
        da_lat,
        da_lon,
        da_day,
        input_core_dims=[[], [], []],
        output_core_dims=[["hour"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"hour": 24}},
    )

    profiles = profiles.assign_coords(
        lat=lat_vals,
        lon=lon_vals,
        day=days,
        hour=np.arange(24),
    )
    profiles.name = "clearsky_pv_profile"
    profiles.attrs.update(
        {
            "description": "Normalized clear-sky PV profile (sums to 1 per day per location)",
            "tilt_deg": float(tilt),
            "azimuth_deg": float(azimuth),
            "units": "fraction",
        }
    )
    return profiles


def disaggregate_pv_to_hourly(
    pv_3h: xr.DataArray,
    profiles: xr.DataArray,
) -> xr.DataArray:
    """Disaggregate 3-hourly PV averages to hourly values using clear-sky profiles.

    Each 3-hourly mean is distributed over the three adjacent hours
    (center − 1 h, center, center + 1 h) proportionally to the clear-sky
    profile at those hours.  Energy is conserved: the sum of the three hourly
    outputs equals ``pv_3h * 3``.

    Parameters
    ----------
    pv_3h : xr.DataArray, dims (time, lat, lon)
        3-hourly PV averages with ``time`` centered at each 3-hour interval.
    profiles : xr.DataArray, dims (lat, lon, day, hour)
        Normalized hourly profiles from :func:`generate_clearsky_profiles`.
        Coordinate ``day`` must be day-of-year (1–366); ``hour`` must span
        0–23.

    Returns
    -------
    xr.DataArray, dims (time, lat, lon)
        Hourly PV values sorted in chronological order.
    """
    offsets = xr.DataArray([-1, 0, 1], dims="offset")

    # 1. New timestamps for each (time, offset) pair ----------------------
    # Use nanosecond-precision timedelta to avoid a UserWarning from xarray
    # about automatic precision up-casting.
    one_hour_ns = np.timedelta64(1, "h").astype("timedelta64[ns]")
    new_times = pv_3h.time + offsets * one_hour_ns  # (time, offset)

    # 2. Day-of-year and hour of day for every new timestamp --------------
    doy = new_times.dt.dayofyear  # (time, offset)
    hour = new_times.dt.hour      # (time, offset)

    # 3. Vectorized profile look-up ----------------------------------------
    # profiles: (lat, lon, day, hour)
    # Indexing with 2-D DataArray indexers yields (lat, lon, time, offset)
    weights = profiles.sel(day=doy, hour=hour, method="nearest")
    weights = weights.transpose("time", "offset", "lat", "lon")

    # 4. Normalize across the 3-hour window --------------------------------
    # Drop the offset dim from the sum so xarray can broadcast it back over
    # the offset dimension in weights.
    w_sum = weights.sum("offset")           # (time, lat, lon) — no offset dim
    w_sum = w_sum.where(w_sum > 0, other=1.0)
    weights = weights / w_sum              # broadcasts w_sum over offset dim

    # 5. Scale: mean × weight × 3 = hourly energy -------------------------
    pv_expanded = (
        pv_3h
        .expand_dims(offset=np.array([-1, 0, 1]))
        .transpose("time", "offset", "lat", "lon")
    )
    pv_hourly = pv_expanded * weights * 3  # (time, offset, lat, lon)

    # 6. Flatten (time, offset) → time with actual timestamps -------------
    # Stack first, then drop the stale MultiIndex level coords before assigning
    # the new flat timestamp coordinate.  This avoids the FutureWarning about
    # updating a PandasMultiIndex coordinate in an inconsistent state.
    hourly_ts = new_times.stack(new_time=("time", "offset")).values
    pv_hourly = (
        pv_hourly
        .stack(new_time=("time", "offset"))
        .drop_vars(["new_time", "time", "offset"])
        .assign_coords(new_time=hourly_ts)
        .rename(new_time="time")
        .sortby("time")
    )

    pv_hourly.name = pv_3h.name or "pv_hourly"
    pv_hourly.attrs = {
        "description": "Hourly PV disaggregated from 3-hourly averages",
        "units": pv_3h.attrs.get("units", ""),
        "method": "clear-sky profile weighting",
    }
    return pv_hourly


def disaggregate_pv_to_hourly_pd(
    pv_3h_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    config: str,
) -> pd.DataFrame:
    """Disaggregate 3-hourly PV averages to hourly values (pandas, single-site).

    A fully vectorized, loop-free implementation using NumPy broadcasting.
    Suitable for single-site or small-domain use cases.  For gridded data
    with Dask, prefer :func:`disaggregate_pv_to_hourly`.

    Parameters
    ----------
    pv_3h_df : pd.DataFrame
        Must contain columns ``'time'`` (datetime-like) and ``'pv'``.
    profiles_df : pd.DataFrame
        Normalised hourly profile table with 8 760 rows (one per hour of a
        non-leap year).  Must contain column *config*.  The integer
        hour-of-year index is read from a ``'time'`` column when present;
        otherwise the positional index is used.
    config : str
        Column in *profiles_df* to use as weights (e.g. ``'tilt-45_S'``).

    Returns
    -------
    pd.DataFrame
        Columns ``'time'`` (datetime) and ``'pv_hourly'`` (float).
        Returns ``3 × len(pv_3h_df)`` rows sorted by time.  Leap years are
        handled transparently by mapping to the equivalent non-leap year for
        the profile look-up.
    """
    pv_3h_df = pv_3h_df[["time", "pv"]].copy()
    pv_3h_df["time"] = pd.to_datetime(pv_3h_df["time"])

    year0 = pv_3h_df["time"].dt.year.iloc[0]
    is_leap = calendar.isleap(year0)
    if is_leap:
        pv_3h_df["time"] = pv_3h_df["time"].apply(
            lambda x: x.replace(year=x.year - 1)
        )

    # Build a flat integer-indexed lookup array (length 8760)
    if "time" in profiles_df.columns:
        profile_vals = profiles_df.set_index("time")[config].values
    else:
        profile_vals = profiles_df[config].values

    # Vectorized expansion: each 3h row → 3 hourly rows via broadcasting
    times_3h = pv_3h_df["time"].values       # numpy datetime64, shape (N,)
    pv_vals  = pv_3h_df["pv"].values         # shape (N,)

    deltas       = np.array([-1, 0, 1], dtype="timedelta64[h]")
    times_hourly = times_3h[:, None] + deltas  # (N, 3)
    pv_repeated  = np.repeat(pv_vals, 3)        # (3N,)
    times_flat   = times_hourly.ravel()         # (3N,)

    # Hour-of-year index (0 … 8759) for profile look-up
    dti = pd.DatetimeIndex(times_flat)
    hour_of_year = dti.hour + (dti.dayofyear - 1) * 24

    # Normalize weights within each 3-hour window
    weights      = profile_vals[hour_of_year].reshape(-1, 3)  # (N, 3)
    w_sum        = weights.sum(axis=1, keepdims=True)
    w_sum        = np.where(w_sum > 0, w_sum, 1.0)            # guard night
    weights_norm = (weights / w_sum).ravel()                   # (3N,)

    df_out = (
        pd.DataFrame(
            {"time": times_flat, "pv_hourly": pv_repeated * weights_norm * 3}
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    if is_leap:
        df_out["time"] = pd.DatetimeIndex(df_out["time"]).map(
            lambda x: x.replace(year=x.year + 1)
        )

    return df_out