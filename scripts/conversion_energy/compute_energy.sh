# Create a directory for logs if it doesn't exist
paths_file="../../paths_so4.json"
model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
scenario="rcp85"

mkdir -p "logs/${model}_${scenario}"

current_date=$(date +"%Y%m%d")
echo "Computing wind power..."
# nice -10 python -u compute_wind.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${model}_${scenario}/wind_power_${current_date}.log"
echo "Computing PV..."
nice -10 python -u compute_pv.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${model}_${scenario}/pv_${current_date}.log"
