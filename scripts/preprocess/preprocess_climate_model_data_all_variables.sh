#!/bin/bash
# Define the arguments
paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

# model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
# model="MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
# model="MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
scenario="rcp85"
wind_multiple_heights="True"

# Define variables and temporal resolutions
## Add or remove wind100m and wind150m from variables_3hr if wind_multiple_heights is present
if [ "$wind_multiple_heights" = "False" ]; then
    variables_3hr=("tas" "hurs" "rsds" "sfcWind" "psl")
else
    variables_3hr=("tas" "hurs" "rsds" "sfcWind" "psl" "wind100m" "wind150m")
fi

variables_day=("pr" "mrro" "clt")

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

# Loop through each temporal resolution and its corresponding variables
for temp_res in "3hr" "day"; do
    if [ "$temp_res" == "3hr" ]; then
        variables=("${variables_3hr[@]}")
    else
        variables=("${variables_day[@]}")
    fi

    for variable in "${variables[@]}"; do
        echo "Processing variable: $variable for model: $model, scenario: $scenario, and temporal resolution: $temp_res"
        python -u preprocess_climate_model_data.py "$paths_file" "$model" "$scenario" "$variable" "$temp_res" 2>&1 | tee "logs/${model}_${scenario}/preprocess_climate_model_data_${variable}_${temp_res}.log"
    done
done

# Remap orography
echo "Remapping orography for model: $model and scenario: $scenario"
python remap_orography.py "$paths_file" "$model" "$scenario"

echo "All variables processed."
