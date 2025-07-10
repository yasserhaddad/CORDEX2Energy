#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"
# model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
wind_multiple_heights="False"
scenario="rcp85"

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

# Define all variables, references, temporal resolutions, and heights
if [ "$wind_multiple_heights" = "False" ]; then
    variables=("tas" "hurs" "psl" "sfcWind" "rsds" "pr" "mrro" "clt")
    variables_ref=("2t" "2r" "sp" "10si" "ssrd" "tp" "ro" "tcc")
    temp_res_array=("3hr" "3hr" "3hr" "3hr" "3hr" "day" "day" "day")
    height_ref_array=(2 2 "None" 10 "None" "None" "None" "None")
else
    variables=("tas" "hurs" "psl" "wind100m" "wind150m" "sfcWind" "rsds" "pr" "mrro" "clt")
    variables_ref=("2t" "2r" "sp" "ws" "ws" "10si" "ssrd" "tp" "ro" "tcc")
    temp_res_array=("3hr" "3hr" "3hr" "3hr" "3hr" "3hr" "3hr" "day" "day" "day")
    height_ref_array=(2 2 "None" 100 150 10 "None" "None" "None" "None")
fi

# Only process mrro and clt
variables=("mrro" "clt")
variables_ref=("ro" "tcc")
temp_res_array=("day" "day")
height_ref_array=("None" "None")

nr_processes=20
memory_limit="auto"

# Process variables in the defined order
for i in "${!variables[@]}"; do
    variable="${variables[$i]}"
    variable_ref="${variables_ref[$i]}"
    temp_res="${temp_res_array[$i]}"
    height_ref="${height_ref_array[$i]}"

    echo "Processing variable: $variable and variable_ref: $variable_ref for model: $model and scenario: $scenario"

    if [ "$height_ref" == "None" ]; then
        nice -n 10 python -u bias_correct_climate_model_data.py \
            "$paths_file" \
            "$variable" \
            "$model" \
            "$scenario" \
            "$temp_res" \
            "$variable_ref" \
            "quantile_mapping" \
            --default_parameters \
            --nr_processes ${nr_processes} \
            --memory_limit ${memory_limit} 2>&1 | tee "logs/${model}_${scenario}/bias_correct_climate_model_data_${variable}.log"
    else
        nice -n 10 python -u bias_correct_climate_model_data.py \
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
            --memory_limit ${memory_limit} 2>&1 | tee "logs/${model}_${scenario}/bias_correct_climate_model_data_${variable}.log"
    fi
done