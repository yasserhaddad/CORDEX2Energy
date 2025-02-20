#!/bin/bash
# Define the arguments
paths_file="../paths_so4.json"
variables=("2t" "tp" "ssrd" "ws" "q" "2r" "sp")
# variables=("10si")
variables=("ssrd")

# Loop through each variable and run the script
for variable in "${variables[@]}"; do
    echo "Processing variable: $variable"
    python process_raw_reanalysis_data.py "$paths_file" "$variable"
done

echo "All variables processed."
