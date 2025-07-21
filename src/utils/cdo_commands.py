import subprocess

REMAP_TYPE_MAP = {
    "conservative": "remapcon",
    "conservative2": "remapcon2",
    "bilinear": "remapbil",
    "bicubic": "remapbic",
    "nearest_neighbor": "remapnn",
    "distance_weighted": "remapdis",
}

def cdo_remap(
    input_file_path: str, output_file_path: str, remap_type: str, target_grid_path: str, input_grid_path: str = None
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
    ]

    if input_grid_path:
        cdo_command.append(f"-setgrid,{input_grid_path}")

    cdo_command += [
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


def save_cdo_griddes(input_path: str, output_path: str) -> None:
    """
    Save the grid definition of the input file to the specified output file.

    Parameters
    ----------
    input_path : str
        The path to the input file.
    output_path : str
        The path to the output file.
    """
    try:
        # Run the cdo griddes command
        result = subprocess.run(["cdo", "griddes", input_path], capture_output=True, text=True, check=True)

        # Write the output to the specified file
        with open(output_path, "w") as f:
            f.write(result.stdout)

        print(f"Grid definition saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running CDO: {e}")
    except FileNotFoundError:
        print("Error: CDO is not installed or not found in PATH.")