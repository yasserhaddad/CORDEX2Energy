#!/bin/bash
# Define the arguments
paths_file="../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
temp_res="3hr"

# Run on exo
variables=("tas" "hurs" "huss")
variables_ref=("2t" "2r" "q")
temp_res_array=("3hr" "3hr" "3hr")
height_ref_array=(2 2 2)
parametric_dist=(false true false)

# Run on so4
variables=("pr" "rsds" "psl")
variables_ref=("tp" "ssrd" "sp")
temp_res_array=("day" "3hr" "3hr")
height_ref_array=("None" "None" "None")
parametric_dist=(true false false)



for i in "${!variables[@]}"; do
    variable="${variables[$i]}"
    variable_ref="${variables_ref[$i]}"
    temp_res="${temp_res_array[$i]}"
    height_ref="${height_ref_array[$i]}"
    parametric="${parametric_dist[$i]}"
    echo "Processing variable: $variable and variable_ref: $variable_ref"

    if [ "$height_ref" == "None" ]; then
        if [ "$parametric" == true ]; then
            nice -n 10 python -u bias_correct_climate_model_data.py \
                "$paths_file" \
                "$variable" \
                "$model" \
                "$scenario" \
                "$temp_res" \
                "$variable_ref" \
                "quantile_mapping" \
                --parametric_existing_settings \
                --nr_processes 20
        else
            nice -n 10 python -u bias_correct_climate_model_data.py \
                "$paths_file" \
                "$variable" \
                "$model" \
                "$scenario" \
                "$temp_res" \
                "$variable_ref" \
                "quantile_mapping" \
                --nr_processes 20
        fi
    else
        if [ "$parametric" == true ]; then
            nice -n 10 python -u bias_correct_climate_model_data.py \
                "$paths_file" \
                "$variable" \
                "$model" \
                "$scenario" \
                "$temp_res" \
                "$variable_ref" \
                "quantile_mapping" \
                --height_ref "$height_ref" \
                --parametric_existing_settings \
                --nr_processes 20
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
                --nr_processes 20
        fi
    fi
done
done