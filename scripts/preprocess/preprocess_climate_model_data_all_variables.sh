#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
scenario="rcp85"

# Define variables and temporal resolutions
variables_3hr=("tas" "hurs" "rsds" "sfcWind" "psl" "wind100m" "wind150m")
variables_day=("pr" "mrro" "clt")

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

# Loop through each temporal resolution and its corresponding variables
for temp_res in "3hr" "day"; do
    if [ "$temp_res" == "3hr" ]; then
        variables=("${variables_3hr[@]}")
    else
        variables=("${variables_day[@]}")
    fi

    for variable in "${variables[@]}"; do
        echo "Processing variable: $variable for model: $model, scenario: $scenario, and temporal resolution: $temp_res"
        python -u preprocess_climate_model_data.py "$paths_file" "$model" "$scenario" "$variable" "$temp_res" 2>&1 | tee "logs/${model}_${scenario}/preprocess_climate_model_data_${variable}_${temp_res}.log"
    done
done

echo "All variables processed."
