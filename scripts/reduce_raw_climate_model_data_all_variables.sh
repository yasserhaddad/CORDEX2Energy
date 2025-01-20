#!/bin/bash
# Define the arguments
paths_file="../paths.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
temp_res="3hr"
variables=("tas" "pr" "huss" "rsds" "sfcWind")

# Loop through each variable and run the script
for variable in "${variables[@]}"; do
    echo "Processing variable: $variable"
    python reduce_raw_climate_model_data.py "$paths_file" "$model" "$scenario" "$variable" "$temp_res"
done

echo "All variables processed."
