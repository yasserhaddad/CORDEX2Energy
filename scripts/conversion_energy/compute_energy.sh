# Create a directory for logs if it doesn't exist
paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
scenario="rcp85"
scope="ch"

mkdir -p "logs/${scope}/${model}_${scenario}"

current_date=$(date +"%Y%m%d")
echo "Computing energy quantites for scope ${scope}, model ${model} and scenario ${scenario}"
echo ""
# echo "Computing wind power..."
# nice -10 python -u compute_wind_power_${scope}.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/wind_power_${current_date}.log"

echo "Computing PV..."
nice -10 python -u compute_pv_${scope}.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/pv_${current_date}.log"

# echo "Computing Demand..."
# nice -10 python -u compute_demand.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/demand_${current_date}.log"