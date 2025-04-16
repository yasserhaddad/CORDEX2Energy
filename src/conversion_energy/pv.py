import json
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
from multiprocessing import Pool

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
    rsds = da_radiation.sel(lat=lat, lon=lon).values
    temperature = da_temperature.sel(lat=lat, lon=lon).values
    wind_speed = da_wind_speed.sel(lat=lat, lon=lon).values
    altitude = da_altitude.sel(lat=lat, lon=lon).values if da_altitude is not None else None
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

    ctx = multiprocessing.get_context("spawn")
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
                    da_altitude
                )
                for lat_lon in [(lat, lon) for lat in lats for lon in lons]
            ],
        )

    da_pv = xr.combine_by_coords(results)

    return da_pv