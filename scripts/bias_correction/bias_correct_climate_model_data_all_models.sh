#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"

models=("CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
        "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
        "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
        "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
        "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
        "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
        "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
        "MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
        "MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
        "CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
        "CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
        "CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"
        "MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1")
scenarios=(
    "rcp26"
    "rcp45"
    "rcp85"
    "rcp26"
    "rcp85"
    "rcp26"
    "rcp85"
    "rcp85"
    "rcp85"
    "rcp85"
    "rcp85"
    "rcp85"
    "rcp85"
)

models=(
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    # "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    # "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
)
scenarios=(
    # "rcp26"
    # "rcp85"
    # "rcp26"
    # "rcp85"
    "rcp26"
    # "rcp45"
    "rcp85"
)


# Create a directory for logs if it doesn't exist

# variables=("clt")
# variables_ref=("tcc")
# temp_res_array=("day" )
# height_ref_array=("None")

# variables=("psl")
# variables_ref=("msl")
# temp_res_array=("day" )
# height_ref_array=("None")

variables=("ps")
variables_ref=("sp_daily")
temp_res_array=("day" )
height_ref_array=("None")

# variables=("hurs")
# variables_ref=("2r_daily")
# temp_res_array=("day")
# height_ref_array=(2)

nr_processes=35
memory_limit="auto"

# Process variables in the defined order
for m_idx in "${!models[@]}"; do
    model="${models[$m_idx]}"
    scenario="${scenarios[$m_idx]}"
    mkdir -p "logs/${model}_${scenario}"
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
                --memory_limit ${memory_limit} 2>&1 | tee "logs/${model}_${scenario}/bias_correct_climate_model_data_${variable}_${temp_res}.log"
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
done