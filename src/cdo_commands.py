import subprocess

REMAP_TYPE_MAP = {
    "conservative": "remapcon",
    "bilinear": "remapbil",
    "bicubic": "remapbic",
    "nearest_neighbor": "remapnn",
    "distance_weighted": "remapdis",
}

def cdo_remap(
    input_file_path: str, output_file_path: str, remap_type: str, target_grid_path: str
) -> None:
    """
    Remap the input file to the target grid using CDO.

    Parameters
    ----------
    input_file_path : str
        The path to the input file.
    output_file_path : str
        The path to the output file.
    remap_type : str
        The type of remapping to perform. Supported types are:
        - 'conservative'
        - 'bilinear'
        - 'bicubic'
        - 'nearest_neighbor'
        - 'distance_weighted'
    target_grid_path : str
        The path to the target grid file.
    """
    # Get the CDO remap command
    cdo_remap_command = REMAP_TYPE_MAP.get(remap_type.lower())
    if cdo_remap_command is None:
        raise ValueError(f"Unsupported remap type: {remap_type}")

    # Construct the CDO command
    cdo_command = [
        "cdo",
        f"{cdo_remap_command},{target_grid_path}",
        input_file_path,
        output_file_path,
    ]

    # Execute the CDO command
    subprocess.run(cdo_command, check=True)

def cdo_setgrid(input_file_path: str, grid_path: str, output_file_path: str) -> None:
    """
    Set the grid of the input file to the specified grid using CDO.

    Parameters
    ----------
    input_file_path : str
        The path to the input file.
    grid_path : str
        The path to the grid file.
    output_file_path : str
        The path to the output file.
    """
    # Construct the CDO command
    cdo_command = ["cdo", f"setgrid,{grid_path}", input_file_path, output_file_path]

    # Execute the CDO command
    subprocess.run(cdo_command, check=True)