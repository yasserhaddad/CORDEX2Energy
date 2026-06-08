import json
import pandas as pd
import xarray as xr
from pathlib import Path
import argparse

from cordex2energy.utils.constants import REDUCED_NETCDF_ENCODINGS


def save_combined_dataset(
    ds: xr.Dataset, output_dir: Path, filename: str, model_chain: str, scenario: str
):
    encoding = {var: REDUCED_NETCDF_ENCODINGS for var in ds.data_vars}
    output_path = output_dir / f"{model_chain}_{scenario}"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{filename}_{model_chain}_{scenario}.nc"
    ds.to_netcdf(output_file, encoding=encoding)
    print(f"Combined dataset saved to {output_file}")


def csv_to_dataset(path: Path, var_name: str, attrs: dict | None = None) -> xr.Dataset:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # If data is in wide format (one column per timestep), convert to long format
    if "country" in df.columns and not any(
        c in df.columns for c in ["time", "date", "datetime", "year", "Year"]
    ):
        candidate_cols = [c for c in df.columns if c != "country"]
        parsed_cols = pd.to_datetime(pd.Index(candidate_cols), errors="coerce")
        valid_time_cols = list(pd.Index(candidate_cols)[~parsed_cols.isna()])

        if valid_time_cols:
            df = df.melt(
                id_vars=["country"],
                value_vars=valid_time_cols,
                var_name="time",
                value_name=var_name,
            )

    # Detect time column
    time_candidates = ["time", "date", "datetime", "year", "Year"]
    time_col = next((c for c in time_candidates if c in df.columns), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Convert to xarray Dataset with dims: time, country
    if "country" in df.columns:
        # Long format: one row per (time, country)
        ds = (
            df.set_index([time_col, "country"])
            .rename_axis(index={time_col: "time"})
            .to_xarray()
        )
    else:
        # Wide format: one column per country
        wide = df.set_index(time_col).rename_axis("time")
        da = wide.stack().rename(var_name)
        da.index.names = ["time", "country"]
        ds = da.to_xarray().to_dataset(name=var_name)

    ds[var_name].attrs["long_name"] = (
        var_name.replace("-", " ").replace("_", " ").title()
    )
    ds.attrs["source_file"] = str(path)
    if attrs:
        ds.attrs.update(attrs)

    return ds


def combine_hydro(files, scenario, hydropower_type):
    datasets = []
    if hydropower_type not in ["ror", "inflows"]:
        raise ValueError(
            f"Invalid hydropower type: {hydropower_type}. Must be 'ror' or 'inflows'."
        )
    selected_files = [
        f
        for f in files
        if hydropower_type in f.name and f.name.endswith(f"{scenario}.csv")
    ]
    for file in selected_files:
        model_name = (
            file.parent.parent.name
        )  # Assuming structure: model_chain/output_variables/file.csv
        stem_parts = file.stem.split("_CORDEX_")
        var_name = stem_parts[0]

        meta_parts = stem_parts[1].split("_") if len(stem_parts) > 1 else []
        scenario_code = meta_parts[-1] if meta_parts else ""
        gcm = meta_parts[-2] if len(meta_parts) >= 2 else ""
        rcm = "_".join(meta_parts[:-2]) if len(meta_parts) >= 3 else ""
        scenario = f"rcp{scenario_code}" if scenario_code.isdigit() else scenario_code
        ds = csv_to_dataset(
            file,
            var_name=var_name,
            attrs={
                "model_chain": model_name,
                "rcm": rcm,
                "driving_model": gcm,
                "scenario": scenario,
            },
        )
        datasets.append(ds)

    combined_ds = xr.concat(datasets, dim="time").sortby(
        "time"
    )  # Concatenate along time dimension and sort
    return combined_ds


def combine_demand(files: list[Path], scenario: str) -> xr.Dataset:
    datasets = []
    selected_files = [
        f for f in files if "demand" in f.name and f.name.endswith(f"{scenario}.csv")
    ]
    for file in selected_files:
        model_name = (
            file.parent.parent.name
        )  # Assuming structure: model_chain/output_variables/file.csv
        stem_parts = file.stem.split("_CORDEX_")
        var_name = stem_parts[0]
        if "fully_electrified" in file.name:
            var_name += "_fully_electrified"

        meta_parts = stem_parts[1].split("_") if len(stem_parts) > 1 else []
        scenario_code = meta_parts[-1] if meta_parts else ""
        gcm = meta_parts[-2] if len(meta_parts) >= 2 else ""
        rcm = "_".join(meta_parts[:-2]) if len(meta_parts) >= 3 else ""
        scenario = f"rcp{scenario_code}" if scenario_code.isdigit() else scenario_code
        ds = csv_to_dataset(
            file,
            var_name=var_name,
            attrs={
                "model_chain": model_name,
                "rcm": rcm,
                "driving_model": gcm,
                "scenario": scenario,
            },
        )
        datasets.append(ds)

    combined_ds = xr.merge(
        datasets, compat="override"
    )  # Merge datasets, allowing for different variables

    return combined_ds


def main():
    parser = argparse.ArgumentParser(
        description="Combine outputs from clim2energy conversion."
    )
    parser.add_argument(
        "--paths_file",
        type=str,
        required=True,
        help="Path to the JSON file containing paths.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the parent directory containing the climate2energy data to combine.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    paths = json.load(open(args.paths_file))
    path_data_processed = Path(paths["path_data_processed"])
    path_data_energy = path_data_processed / "energy"
    path_data_energy.mkdir(parents=True, exist_ok=True)

    # Load all individual output files and combine them into a single dataset
    models = json.load(open(Path.cwd().parent.parent / "models.json"))["model_chains"]
    model_chains = {
        f"{model['rcm']}_{model['gcm']}": {
            "scenarios": model["scenarios"],
            "ensemble": model["ensemble"],
        }
        for model in models
    }

    for model in input_dir.glob("*/"):
        if model.name in model_chains:
            files = list(model.glob("output_variables/*.csv"))
            if len(files) > 0:
                for scenario in model_chains[model.name]["scenarios"]:
                    print(
                        f"\nCombining outputs for model chain {model.name} and scenario {scenario}..."
                    )
                    ds_demand = combine_demand(
                        files, scenario=scenario.split("rcp")[-1]
                    )

                    save_combined_dataset(
                        ds_demand,
                        output_dir=path_data_energy / "demand",
                        filename="c2e_demand",
                        climate_model_chain=model.name + f'_{model_chains[model.name]["ensemble"]}',
                        scenario=scenario,
                    )

                    ds_hydro_ror = combine_hydro(
                        files, scenario=scenario.split("rcp")[-1], hydropower_type="ror"
                    )

                    save_combined_dataset(
                        ds_hydro_ror,
                        output_dir=path_data_energy / "hydropower",
                        filename="c2e_hydropower_ror",
                        climate_model_chain=model.name + f'_{model_chains[model.name]["ensemble"]}',
                        scenario=scenario,
                    )

                    ds_hydro_inflows = combine_hydro(
                        files,
                        scenario=scenario.split("rcp")[-1],
                        hydropower_type="inflows",
                    )

                    save_combined_dataset(
                        ds_hydro_inflows,
                        output_dir=path_data_energy / "hydropower",
                        filename="c2e_hydropower_inflows",
                        climate_model_chain=model.name + f'_{model_chains[model.name]["ensemble"]}',
                        scenario=scenario,
                    )
