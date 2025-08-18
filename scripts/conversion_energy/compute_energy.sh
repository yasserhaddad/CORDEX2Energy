paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
# model="MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
# model="MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
# model="MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"

scenario="rcp85"


current_date=$(date +"%Y%m%d")
echo "Computing energy quantites, model ${model} and scenario ${scenario}"
echo ""

scope="ch"

# Create a directory for logs if it doesn't exist
mkdir -p "logs/${scope}/${model}_${scenario}"

# echo "Computing wind power for scope ${scope}..."
# nice -10 python -u compute_wind_power_${scope}.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/wind_power_${current_date}.log"

# echo "Computing PV for scope ${scope}..."
# nice -10 python -u compute_pv_${scope}.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/pv_${current_date}.log"


scope="europe"
# Create a directory for logs if it doesn't exist
mkdir -p "logs/${scope}/${model}_${scenario}"

echo "Computing wind power for scope ${scope}..."
nice -10 python -u compute_wind_power_${scope}.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/wind_power_${current_date}.log"

echo "Computing PV for scope ${scope}..."
nice -10 python -u compute_pv_${scope}.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/pv_${current_date}.log"

# echo "Computing hydropower..."
# nice -10 python -u compute_hydropower.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/hydropower_${current_date}.log"

echo "Computing Demand..."
nice -10 python -u compute_demand.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/demand_${current_date}.log"