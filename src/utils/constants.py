import zarr

REDUCED_NETCDF_ENCODINGS = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
}

COMPRESSED_ZARR_ENCODING = {
    "compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
}

REFERENCE_PERIOD = ["1991", "2020"]
FUTURE_PERIOD = ["2021", "2100"]
NEAR_FUTURE_PERIOD = ["2021", "2050"]
MID_CENTURY_PERIOD = ["2041", "2060"]

EXTENT_BC = {"lat_min": 30, "lat_max": 75, "lon_min": -30, "lon_max": 41}
EXTENT_BC_MRRO = {"lat_min": 20.8, "lat_max": 71.8, "lon_min": -51.7, "lon_max": 72.7}
EXTENT_CH = {"lat_min": 45.8, "lat_max": 47.9, "lon_min": 5.9, "lon_max": 10.5}
EXTENT_PREVAH = {"lat_min": 44.4, "lat_max": 48.6, "lon_min": 4.5, "lon_max": 16.5}

REGRIDDED_CHUNKS = {"time": -1, "lat": 20, "lon": 20}
BC_DAILY_CHUNKS = {"time": 365, "lat": 256, "lon": 256}
BC_SUBDAILY_CHUNKS = {"time": 2920, "lat": 128, "lon": 128}

EUROPE_TURBINES = ["E-126/7580", "SWT120/3600", "SWT142/3150"]
CH_TURBINES = ["E-82/2300", "E-92/2350", "V90/2000", "V112/3300"]