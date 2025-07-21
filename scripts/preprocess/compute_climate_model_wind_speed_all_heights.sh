paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
scenario="rcp85"
temp_res="3hr"
heights=(100 150)
mkdir -p "logs/${model}_${scenario}/"

# Loop through each height and scenario and run the script
for height in "${heights[@]}"; do
    echo "Processing model: $model, scenario: $scenario and height: $height"
    python -u compute_climate_model_wind_speed.py "$paths_file" "$model" "$scenario" "$height" "$temp_res" 2>&1 | tee "logs/${model}_${scenario}/get_climate_model_wind_speeds_${height}.log"
done

echo "All heights and scenarios processed."
