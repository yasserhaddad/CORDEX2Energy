import json
import numba
from tqdm import tqdm
import multiprocessing
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
    # dni, dhi = xr.apply_ufunc(
    #     _erbs_dni_dhi,
    #     ghi,
    #     zen,
    #     ghi["time"],
    #     input_core_dims=[["time"], ["time"], ["time"]],
    #     output_core_dims=[["time"], ["time"]],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_dtypes=[ghi.dtype, ghi.dtype],
    # )
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
