#!/bin/bash
# Define the arguments

paths_file="../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
temp_res="3hr"

# # Run on exo
# variables=("tas" "hurs" "huss")
# variables_ref=("2t" "2r" "q")
# temp_res_array=("3hr" "3hr" "3hr")
# height_ref_array=(2 2 2)
# nr_processes=20
# memory_limit="auto"

# variables=("huss")
# variables_ref=("q")
# temp_res_array=("3hr")
# height_ref_array=(2)
# nr_processes=20
# memory_limit="60GB"

# Run on so4
# variables=("pr" "psl")
# variables_ref=("tp" "sp")
# temp_res_array=("day" "3hr")
# height_ref_array=("None" "None")
# nr_processes=20
# memory_limit="auto"

# Run on iacdipl-7
# variables=("wind100m" "wind150m" "wind200m" "sfcWind")
# variables_ref=("ws" "ws" "ws" "10si")
# temp_res_array=("3hr" "3hr" "3hr" "3hr")
# height_ref_array=(100 150 200 10)
# nr_processes=30
# memory_limit="auto"

# variables=("wind150m" "wind200m" "sfcWind")
# variables_ref=("ws" "ws" "10si")
# temp_res_array=("3hr" "3hr" "3hr")
# height_ref_array=(150 200 10)
# nr_processes=20
# memory_limit="60GB"

# Run on iacdipl-8
variables=("rsds")
variables_ref=("ssrd")
temp_res_array=("3hr")
height_ref_array=("None")
nr_processes=20
memory_limit="60GB"


for i in "${!variables[@]}"; do
    variable="${variables[$i]}"
    variable_ref="${variables_ref[$i]}"
    temp_res="${temp_res_array[$i]}"
    height_ref="${height_ref_array[$i]}"
    parametric="${parametric_dist[$i]}"
    echo "Processing variable: $variable and variable_ref: $variable_ref"

    if [ "$height_ref" == "None" ]; then
        nice -n 10 python -u bias_correct_climate_model_data_xclim.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            "quantile_mapping" \
            --default_parameters \
            --nr_processes ${nr_processes} \
            --memory_limit ${memory_limit}
    else
        nice -n 10 python -u bias_correct_climate_model_data_xclim.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            "quantile_mapping" \
            --height_ref "$height_ref" \
            --default_parameters \
            --nr_processes ${nr_processes} \
            --memory_limit ${memory_limit}
    fi
done