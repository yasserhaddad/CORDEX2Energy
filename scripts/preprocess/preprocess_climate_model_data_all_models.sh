#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"

models=(
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    # "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    # "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
    # "CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
    # "CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"
    # "MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
    # "MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
)
scenarios=(
    # "rcp26"
    # "rcp85"
    # "rcp26"
    # "rcp85"
    "rcp26"
    # "rcp45"
    "rcp85"
    # "rcp85"
    # "rcp85"
    # "rcp85"
    # "rcp85"
    # "rcp85"
    # "rcp85"
)

# models=(
#     "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
#     "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
#     "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
#     "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# )
# scenarios=(
#     "rcp26"
#     "rcp85"
#     "rcp26"
#     "rcp85"
# )

models=(
    # "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    # "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
)
scenarios=(
    # "rcp26"
    # "rcp26"
    # "rcp85"
)

# Create a directory for logs if it doesn't exist
variables=("hurs")
temp_res_array=("day")

# Process variables in the defined order
for m_idx in "${!models[@]}"; do
    model="${models[$m_idx]}"
    scenario="${scenarios[$m_idx]}"
    mkdir -p "logs/${model}_${scenario}"
    for i in "${!variables[@]}"; do
        variable="${variables[$i]}"
        temp_res="${temp_res_array[$i]}"

        echo "Processing variable: $variable for model: $model, scenario: $scenario, and temporal resolution: $temp_res"
        python -u preprocess_climate_model_data.py "$paths_file" "$model" "$scenario" "$variable" "$temp_res" 2>&1 | tee "logs/${model}_${scenario}/preprocess_climate_model_data_${variable}_${temp_res}.log"
    done
done