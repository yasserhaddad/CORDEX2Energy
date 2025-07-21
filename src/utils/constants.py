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

EXTENT_BC = {"lat_min": 30, "lat_max": 75, "lon_min": -30, "lon_max": 41}
EXTENT_BC_MRRO = {"lat_min": 20.8, "lat_max": 71.8, "lon_min": -51.7, "lon_max": 72.7}
EXTENT_CH = {"lat_min": 45.8, "lat_max": 47.9, "lon_min": 5.9, "lon_max": 10.5}