#!/bin/bash

# Bias correct missing variables for climate model chains
# This script checks for missing bias-corrected variables and runs corrections

# Configuration
PATHS_FILE="../../paths_so4.json"
NR_PROCESSES=35
MEMORY_LIMIT="auto"

# Climate model chains (comment out ones you don't want to process)
CLIMATE_MODEL_CHAINS=(
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    # "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    # "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    "CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
    "CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
    "CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"
    "MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    "MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
    # "MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
)

CLIMATE_SCENARIOS=("rcp26" "rcp85")

# Variable definitions
VARIABLES_3HR=("tas" "hurs" "ps" "sfcWind" "rsds")
VARIABLES_DAY=("pr" "mrro" "clt" "psl")

# Variable mapping: variable -> reference_variable temp_res height
declare -A VARIABLE_MAPPING
VARIABLE_MAPPING["tas"]="2t 3hr 2"
VARIABLE_MAPPING["hurs"]="2r 3hr 2"
VARIABLE_MAPPING["ps"]="sp 3hr"
VARIABLE_MAPPING["sfcWind"]="10si 3hr 10"
VARIABLE_MAPPING["rsds"]="ssrd 3hr"
VARIABLE_MAPPING["pr"]="tp day"
VARIABLE_MAPPING["mrro"]="ro day"
VARIABLE_MAPPING["clt"]="tcc day"
VARIABLE_MAPPING["psl"]="msl day"

# Get data processed path from JSON
PATH_DATA_PROCESSED=$(python3 -c "import json; print(json.load(open('$PATHS_FILE'))['path_data_processed'])")

echo "================================================================================"
echo "CHECKING FOR MISSING BIAS-CORRECTED VARIABLES"
echo "================================================================================"

# Array to store missing jobs
declare -a MISSING_JOBS=()

# Check for missing variables
for climate_model_chain in "${CLIMATE_MODEL_CHAINS[@]}"; do
    for climate_scenario in "${CLIMATE_SCENARIOS[@]}"; do
        bias_corrected_dir="${PATH_DATA_PROCESSED}/cordex_processed/${climate_model_chain}_${climate_scenario}/bias_corrected"

        if [ -d "$bias_corrected_dir" ]; then
            echo ""
            echo "Checking ${climate_model_chain} - ${climate_scenario}"

            for temp_res in "3hr" "day"; do
                # Determine expected variables
                if [ "$temp_res" = "3hr" ]; then
                    expected_vars=("${VARIABLES_3HR[@]}")
                else
                    expected_vars=("${VARIABLES_DAY[@]}")
                fi

                # Get present variables
                present_vars=()
                if [ -d "${bias_corrected_dir}/${temp_res}" ]; then
                    for zarr_path in "${bias_corrected_dir}/${temp_res}"/*.zarr; do
                        if [ -e "$zarr_path" ]; then
                            var_name=$(basename "$zarr_path" .zarr)
                            present_vars+=("$var_name")
                        fi
                    done
                fi

                # Find missing variables
                missing_vars=()
                for var in "${expected_vars[@]}"; do
                    found=0
                    for present in "${present_vars[@]}"; do
                        if [ "$var" = "$present" ]; then
                            found=1
                            break
                        fi
                    done
                    if [ $found -eq 0 ]; then
                        missing_vars+=("$var")
                    fi
                done

                if [ ${#missing_vars[@]} -gt 0 ]; then
                    echo "  ${temp_res}: Missing ${missing_vars[*]}"
                    for var in "${missing_vars[@]}"; do
                        # Store job info: model|scenario|variable|mapping_info
                        MISSING_JOBS+=("${climate_model_chain}|${climate_scenario}|${var}|${VARIABLE_MAPPING[$var]}")
                    done
                else
                    echo "  ${temp_res}: All variables present"
                fi
            done
        fi
    done
done

if [ ${#MISSING_JOBS[@]} -eq 0 ]; then
    echo ""
    echo "✓ All variables are bias-corrected for all models and scenarios!"
    exit 0
fi

echo ""
echo "================================================================================"
echo "FOUND ${#MISSING_JOBS[@]} MISSING BIAS CORRECTIONS"
echo "================================================================================"

read -p "Do you want to run bias corrections for these missing variables? (yes/no): " response

if [[ "$response" =~ ^(yes|y|Y|YES)$ ]]; then
    echo ""
    echo "================================================================================"
    echo "RUNNING BIAS CORRECTIONS"
    echo "================================================================================"

    job_num=0
    for job in "${MISSING_JOBS[@]}"; do
        ((job_num++))

        # Parse job info
        IFS='|' read -r model scenario variable mapping_info <<< "$job"
        IFS=' ' read -r variable_ref temp_res height_ref <<< "$mapping_info"

        echo ""
        echo "[${job_num}/${#MISSING_JOBS[@]}] Processing:"
        echo "  Model: $model"
        echo "  Scenario: $scenario"
        echo "  Variable: $variable -> $variable_ref"
        echo "  Temp Res: $temp_res"

        # Build command
        cmd="nice -n 10 python -u bias_correct_climate_model_data.py $PATHS_FILE $variable $model $scenario $temp_res $variable_ref quantile_mapping --default_parameters --nr_processes $NR_PROCESSES --memory_limit $MEMORY_LIMIT"

        if [ -n "$height_ref" ]; then
            cmd="$cmd --height_ref $height_ref"
        fi

        # Create log directory
        log_dir="logs/${model}_${scenario}"
        mkdir -p "$log_dir"
        log_file="${log_dir}/bias_correct_climate_model_data_${variable}_${temp_res}.log"

        echo "  Command: $cmd"
        $cmd 2>&1 | tee "$log_file"
    done

    echo ""
    echo "================================================================================"
    echo "BIAS CORRECTION COMPLETE"
    echo "================================================================================"
else
    echo ""
    echo "Skipping bias corrections. You can run them manually using the bash script."
fi
