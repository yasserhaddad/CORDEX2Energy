#!/bin/bash
# Define the arguments
paths_file="../paths.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
temp_res="3hr"
variables=("tas" "hurs" "pr" "rsds" "psl")
# variables=("huss" "sfcWind")
variables_ref=("2t" "2r" "tp" "ssrd" "sp")
# variables_ref=("q" "10si")
temp_res_array=("3hr" "3hr" "day" "3hr" "3hr")
height_ref_array=(2 2 "None" "None" "None")

for i in "${!variables[@]}"; do
    variable="${variables[$i]}"
    variable_ref="${variables_ref[$i]}"
    temp_res="${temp_res_array[$i]}"
    height_ref="${height_ref_array[$i]}"
    echo "Processing variable: $variable and variable_ref: $variable_ref"

    if [ "$height_ref" == "None" ]; then
        nice -n 10 python -u bias_correct_climate_model_data.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            "quantile_mapping"
    else
        nice -n 10 python -u bias_correct_climate_model_data.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            "quantile_mapping" \
            --height_ref "$height_ref"
    fi
done