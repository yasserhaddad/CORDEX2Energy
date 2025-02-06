paths_file="../paths.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"
temp_res="3hr"
heights=(100 150)

# Loop through each height and scenario and run the script
for height in "${heights[@]}"; do
    echo "Processing height: $height and scenario: $scenario"
    python get_climate_model_wind_speed.py "$paths_file" "$model" "$scenario" "$height" "$temp_res"
done

echo "All heights and scenarios processed."
