paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

# model="MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# model="MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
# model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
# model="MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"

scenario="rcp26"

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

current_date=$(date +"%Y%m%d")
echo "Postprocessing bias corrected data, model ${model} and scenario ${scenario}"
echo ""

# echo "Computing bias corrected tas min and max..."
# nice -n 10 python -u compute_bias_corrected_tas_min_max.py  "$paths_file" "$model" "$scenario" "3hr" 2>&1 | tee "logs/${model}_${scenario}/tas_min_max_${current_date}.log"

# echo "Aggregating bias corrected data..."
# nice -n 10 python -u aggregate_bias_corrected_data.py  "$paths_file" "$model" "$scenario" "3hr" 2>&1 | tee "logs/${model}_${scenario}/aggregate_${current_date}.log"

# echo "Computing bias corrected specific humidity..."
# nice -n 10 python -u compute_bias_corrected_specific_humidity.py  "$paths_file" "$model" "$scenario" "3hr" 2>&1 | tee "logs/${model}_${scenario}/specific_humidity_${current_date}.log"

echo "Postprocessing bias corrected runoff..."
# postprocess_runoff.py takes the model/scenario as OPTIONS, not positional arguments:
# omitting them switches it to CERRA mode.
nice -n 10 python -u postprocess_runoff.py  "$paths_file" --climate_model_chain "$model" --climate_scenario "$scenario" 2>&1 | tee "logs/${model}_${scenario}/runoff_${current_date}.log"

# echo "Prepare daily data from sub-daily data for PREVAH..."
# nice -n 10 python -u prepare_daily_data_prevah.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${model}_${scenario}/daily_data_${current_date}.log"

# echo "Prepare demand data for Switzerland..."
# nice -n 10 python -u prepare_data_demand_ch.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${model}_${scenario}/demand_data_ch_${current_date}.log"