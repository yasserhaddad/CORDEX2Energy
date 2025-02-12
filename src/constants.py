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