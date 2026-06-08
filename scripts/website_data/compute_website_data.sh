paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp45"

current_date=$(date +"%Y%m%d")

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${model}_${scenario}"

echo "Computing demand and temperature anomaly data for Switzerland for model ${model} and scenario ${scenario}..."
# echo "Computing heating and cooling days per month"
# nice -10 python -u compute_heating_cooling_days.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${model}_${scenario}/heating_cooling_days_${current_date}.log"

# echo "Computing heating and cooling days per month with temperature weighting"
# nice -10 python -u compute_heating_cooling_days.py  "$paths_file" "$model" "$scenario" --temperature_weighted 2>&1 | tee "logs/${model}_${scenario}/heating_cooling_days_temp_weighted_${current_date}.log"

# echo "Computing decadal temperature anomalies for Europe (per country) and for Switzerland (per canton)"
# nice -10 python -u compute_temperature_anomalies.py  "$paths_file" "$model" "$scenario" --temporal-resolution yearly 2>&1 | tee "logs/${model}_${scenario}/decadal_temperature_anomalies_yearly_${current_date}.log"

echo "Computing decadal temperature anomalies for Europe (per country) and for Switzerland (per canton)"
nice -10 python -u compute_temperature_anomalies.py  "$paths_file" "$model" "$scenario" --temporal-resolution monthly 2>&1 | tee "logs/${model}_${scenario}/decadal_temperature_anomalies_monthly_${current_date}.log"