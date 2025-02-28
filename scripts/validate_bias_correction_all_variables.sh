#!/bin/bash
# Define the arguments

paths_file="../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
temp_res="3hr"

# Run on exo
variables=("tas")
variables_ref=("2t")
temp_res_array=("3hr")
height_ref_array=(2)
nr_processes=10
memory_limit="60GB"

for i in "${!variables[@]}"; do
    variable="${variables[$i]}"
    variable_ref="${variables_ref[$i]}"
    temp_res="${temp_res_array[$i]}"
    height_ref="${height_ref_array[$i]}"
    echo "Processing variable: $variable and variable_ref: $variable_ref"

    if [ "$height_ref" == "None" ]; then
        nice -n 10 python -u validate_bias_correction.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            --nr_processes ${nr_processes} \
            --memory_limit ${memory_limit}
    else
        nice -n 10 python -u validate_bias_correction.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            --height_ref "$height_ref" \
            --nr_processes ${nr_processes} \
            --memory_limit ${memory_limit}
    fi
done