paths_file="../../paths_so4.json"

models=(
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    "MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
    "MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
    "CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
    "CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
    "CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"
    "MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
)

scenario=("rcp85")

demand_scenario="DE"
demand_year=2050
supply_scenario="HIGH"
supply_year=2050

for model in "${models[@]}"; do
    echo "Running TYNDP projections for model: $model, scenario: $scenario"
    python compute_tyndp_projections_extremes.py \
        --paths_file "$paths_file" \
        --climate_model "$model" \
        --climate_scenario "$scenario" \
        --demand_scenario "$demand_scenario" \
        --demand_year "$demand_year" \
        --supply_scenario "$supply_scenario" \
        --supply_year "$supply_year"
done
