from pathlib import Path
from typing import Tuple

def get_components_climate_model_chain(
    climate_model_chain: str,
) -> Tuple[str, str, str, str]:
    institute = climate_model_chain.split("-")[0]
    rcm = climate_model_chain.split("_")[0]
    gcm = climate_model_chain.split("_")[1]
    ensemble = climate_model_chain.split("_")[-1]

    return institute, rcm, gcm, ensemble


def get_raw_cordex_data_paths(
    path_data_cordex: Path,
    climate_model_chain: str,
    climate_scenario: str,
    variable: str,
    temp_res: str,
) -> Tuple[Path, Path]:
    institute, rcm, gcm, ensemble = get_components_climate_model_chain(
        climate_model_chain
    )

    cordex_var = (
        path_data_cordex
        / climate_scenario
        / temp_res
        / variable
        / institute
        / gcm
        / rcm
        / ensemble
    )

    cordex_var_hist = (
        path_data_cordex
        / "historical"
        / temp_res
        / variable
        / institute
        / gcm
        / rcm
        / ensemble
    )

    return cordex_var, cordex_var_hist