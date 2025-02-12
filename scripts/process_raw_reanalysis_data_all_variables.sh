#!/bin/bash
# Define the arguments
paths_file="../paths.json"
# variables=("2t" "tp" "ssrd" "ws" "10si" "q" "2r" "skt" "sp")
variables=("q")

# Loop through each variable and run the script
for variable in "${variables[@]}"; do
    echo "Processing variable: $variable"
    python process_raw_reanalysis_data.py "$paths_file" "$variable"
done

echo "All variables processed."
