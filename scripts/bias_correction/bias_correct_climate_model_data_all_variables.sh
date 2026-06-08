#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"
# model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

# model="MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
# model="MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
# model="MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"

wind_multiple_heights="True"
scenario="rcp85"

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

# Define all variables, references, temporal resolutions, and heights
if [ "$wind_multiple_heights" = "False" ]; then
    variables=("tas" "hurs" "ps" "sfcWind" "rsds" "pr" "mrro" "clt" "psl")
    variables_ref=("2t" "2r" "sp" "10si" "ssrd" "tp" "ro" "tcc" "msl")
    temp_res_array=("3hr" "3hr" "3hr" "3hr" "3hr" "day" "day" "day" "day" )
    height_ref_array=(2 2 "None" 10 "None" "None" "None" "None" "None")
else
    variables=("tas" "hurs" "ps" "sfcWind" "rsds" "pr" "mrro" "clt" "psl" "ps" "wind100m" "wind150m")
    variables_ref=("2t" "2r" "sp""10si" "ssrd" "tp" "ro" "tcc" "msl" "sp" "ws" "ws")
    temp_res_array=("3hr" "3hr" "3hr" "3hr" "3hr" "day" "day" "day" "day" "day" "3hr" "3hr")
    height_ref_array=(2 2 "None" 10 "None" "None" "None" "None" "None" "None" 100 150)
fi

variables=("hurs" "sfcWind" "rsds" "pr" "mrro" "clt" "ps" "psl")
variables_ref=("2r" "10si" "ssrd" "tp" "ro" "tcc" "sp" "msl")
temp_res_array=("3hr" "3hr" "3hr" "day" "day" "day" "day" "day")
height_ref_array=(2 10 "None" "None" "None" "None" "None" "None")

variables=("rsds")
variables_ref=("ssrd")
temp_res_array=("3hr" )
height_ref_array=("None")

# variables=("rsds")
# variables_ref=("ssrd_daily")
# temp_res_array=("day" )
# height_ref_array=("None")

variables=("mrro")
variables_ref=("ro")
temp_res_array=("day")
height_ref_array=("None")

variables=("wind100m" "wind150m")
variables_ref=("ws" "ws")
temp_res_array=("3hr" "3hr")
height_ref_array=(100 150)

nr_processes=35
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
            --memory_limit ${memory_limit} 2>&1 | tee "logs/${model}_${scenario}/bias_correct_climate_model_data_${variable}_${temp_res}.log"
    fi
done