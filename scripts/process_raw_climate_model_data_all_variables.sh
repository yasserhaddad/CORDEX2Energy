#!/bin/bash
# Define the arguments
paths_file="../paths.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
# variables=("tas" "huss" "hurs" "rsds" "sfcWind" "psl")
variables=("wind100m" "wind150m" "wind200m")
temp_res="3hr"
# variables=("pr")
# temp_res="day"

# Loop through each variable and run the script
for variable in "${variables[@]}"; do
    echo "Processing variable: $variable"
    python process_raw_climate_model_data.py "$paths_file" "$model" "$scenario" "$variable" "$temp_res"
done

echo "All variables processed."
