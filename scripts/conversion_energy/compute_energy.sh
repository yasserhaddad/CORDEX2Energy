paths_file="../../paths_so4.json"
# model="CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
model="CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
# model="CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
# model="CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"

# model="MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
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
# nice -10 python -u compute_wind_power.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/wind_power_${current_date}.log"

# echo "Computing PV for scope ${scope}..."
# nice -10 python -u compute_pv.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/pv_${current_date}.log"

scope="europe"
# Create a directory for logs if it doesn't exist
mkdir -p "logs/${scope}/${model}_${scenario}"

# echo "Computing wind power for scope ${scope}..."
# nice -10 python -u compute_wind_power.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/wind_power_${current_date}.log"

# echo "Computing PV for scope ${scope}..."
# nice -10 python -u compute_pv.py  "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}_${scenario}/pv_${current_date}.log"

# echo "Computing hydropower..."
# nice -10 python -u compute_hydropower.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/hydropower_${current_date}.log"

# echo "Computing degree days..."
# nice -10 python -u compute_degree_days.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/degree_days_${current_date}.log"

# echo "Computing demand (JRC degree-day model; needs demand_coefficients.csv from derive_demand_coefficients.py)..."
# nice -10 python -u compute_demand.py  "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}_${scenario}/demand_${current_date}.log"

# --- CERRA reanalysis baseline ---
model="cerra"
scenario="none"
scope="europe"
mkdir -p "logs/${scope}/${model}"

# echo "Computing wind power for CERRA..."
# nice -10 python -u compute_wind_power.py "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}/wind_power_${current_date}.log"

# echo "Computing PV for CERRA..."
# nice -10 python -u compute_pv.py "$paths_file" "$model" "$scenario" "$scope" 2>&1 | tee "logs/${scope}/${model}/pv_${current_date}.log"

# echo "Computing degree days for CERRA..."
# nice -10 python -u compute_degree_days.py "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}/degree_days_${current_date}.log"

# Coefficients are derived once from CERRA degree days (needs compute_degree_days for CERRA first):
# nice -10 python -u derive_demand_coefficients.py "$paths_file" 2>&1 | tee "logs/${scope}/${model}/derive_coeffs_${current_date}.log"

echo "Computing demand for CERRA (JRC degree-day model)..."
nice -10 python -u compute_demand.py "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}/demand_${current_date}.log"

# echo "Aggregating TYNDP zones for CERRA..."
# nice -10 python -u compute_tyndp_aggregates.py "$paths_file" "$model" "$scenario" 2>&1 | tee "logs/${scope}/${model}/tyndp_agg_${current_date}.log"