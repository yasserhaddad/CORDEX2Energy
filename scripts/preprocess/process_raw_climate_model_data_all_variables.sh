#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp26"
variables=("tas" "huss" "hurs" "rsds" "sfcWind" "psl")
temp_res="3hr"
variables=("wind100m" "wind150m")
temp_res="3hr"
# variables=("pr" "mrro" "clt")
# temp_res="day")
# variables=("mrro")
# temp_res="day"

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

# Loop through each variable and run the script
for variable in "${variables[@]}"; do
    echo "Processing variable: $variable for model: $model and scenario: $scenario"
    python -u process_raw_climate_model_data.py "$paths_file" "$model" "$scenario" "$variable" "$temp_res" 2>&1 | tee "logs/${model}_${scenario}/process_raw_climate_model_data_${variable}.log"
done

echo "All variables processed."
