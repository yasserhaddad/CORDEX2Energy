#!/bin/bash

# Validate all bias-corrected variables for climate model chains
# Usage: ./validate_bias_corrected_all_variables.sh [climate_model_chain] [climate_scenario] [variables...] [--temp_res 3hr|day]

# Climate model chains
CLIMATE_MODEL_CHAINS=(
    "MOHC-HadREM3-GA7-05_MOHC-HadGEM2-ES_r1i1p1"
    "MOHC-HadREM3-GA7-05_ICHEC-EC-EARTH_r12i1p1"
    "CNRM-ALADIN63_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "CNRM-ALADIN63_MOHC-HadGEM2-ES_r1i1p1"
    # "CNRM-ALADIN63_NCC-NorESM1-M_r1i1p1"
    # "CNRM-ALADIN63_MPI-M-MPI-ESM-LR_r1i1p1"
    # "MOHC-HadREM3-GA7-05_CNRM-CERFACS-CNRM-CM5_r1i1p1"
    # "MOHC-HadREM3-GA7-05_MPI-M-MPI-ESM-LR_r1i1p1"
    # "MOHC-HadREM3-GA7-05_NCC-NorESM1-M_r1i1p1"
)

# Climate scenarios
CLIMATE_SCENARIOS=("rcp26" "rcp85")

# All variables to validate
VARIABLES_3HR=("tas" "hurs" "ps" "sfcWind" "rsds")
VARIABLES_DAY=("pr" "mrro" "clt" "psl")
ALL_VARIABLES="${VARIABLES_3HR[@]} ${VARIABLES_DAY[@]}"

PATHS_FILE="../../paths_so4.json"

# Parse command line arguments
TEMP_RES=""
SELECTED_VARIABLES=""

# Extract --temp_res if present
for arg in "$@"; do
    if [ "$prev_arg" = "--temp_res" ]; then
        TEMP_RES="$arg"
        break
    fi
    prev_arg="$arg"
done

# Remove --temp_res and its value from arguments
args=()
skip_next=false
for arg in "$@"; do
    if [ "$skip_next" = true ]; then
        skip_next=false
        continue
    fi
    if [ "$arg" = "--temp_res" ]; then
        skip_next=true
        continue
    fi
    args+=("$arg")
done

if [ ${#args[@]} -eq 0 ]; then
    # No arguments: validate all model chains and scenarios
    SELECTED_CHAINS=("${CLIMATE_MODEL_CHAINS[@]}")
    SELECTED_SCENARIOS=("${CLIMATE_SCENARIOS[@]}")
    SELECTED_VARIABLES="$ALL_VARIABLES"
elif [ ${#args[@]} -eq 1 ]; then
    # One argument: specific model chain, all scenarios
    SELECTED_CHAINS=("${args[0]}")
    SELECTED_SCENARIOS=("${CLIMATE_SCENARIOS[@]}")
    SELECTED_VARIABLES="$ALL_VARIABLES"
elif [ ${#args[@]} -eq 2 ]; then
    # Two arguments: specific model chain and scenario
    SELECTED_CHAINS=("${args[0]}")
    SELECTED_SCENARIOS=("${args[1]}")
    SELECTED_VARIABLES="$ALL_VARIABLES"
else
    # Three+ arguments: specific model chain, scenario, and variables
    SELECTED_CHAINS=("${args[0]}")
    SELECTED_SCENARIOS=("${args[1]}")
    SELECTED_VARIABLES="${args[@]:2}"
fi


echo "================================================================================"
echo "VALIDATING BIAS-CORRECTED VARIABLES"
echo "================================================================================"
echo "Climate Model Chains: ${#SELECTED_CHAINS[@]} chain(s)"
echo "Climate Scenarios: ${#SELECTED_SCENARIOS[@]} scenario(s)"
echo "Variables: $SELECTED_VARIABLES"
if [ -n "$TEMP_RES" ]; then
    echo "Temporal Resolution: $TEMP_RES (override)"
else
    echo "Temporal Resolution: from config"
fi
echo "================================================================================"
echo ""

# Counter for progress
total_jobs=$((${#SELECTED_CHAINS[@]} * ${#SELECTED_SCENARIOS[@]}))
current_job=0

# Loop through all combinations
for CLIMATE_MODEL_CHAIN in "${SELECTED_CHAINS[@]}"; do
    for CLIMATE_SCENARIO in "${SELECTED_SCENARIOS[@]}"; do
        ((current_job++))

        echo ""
        echo "--------------------------------------------------------------------------------"
        echo "[$current_job/$total_jobs] Processing: $CLIMATE_MODEL_CHAIN - $CLIMATE_SCENARIO"
        echo "--------------------------------------------------------------------------------"

        # Create log directory
        LOG_DIR="logs/${CLIMATE_MODEL_CHAIN}_${CLIMATE_SCENARIO}"
        mkdir -p "$LOG_DIR"
        LOG_FILE="${LOG_DIR}/validate_bias_corrected_all_variables.log"

        # Build command with optional --temp_res
        CMD="nice -n 10 python validate_bias_corrected_variables.py \
            \"$PATHS_FILE\" \
            \"$CLIMATE_MODEL_CHAIN\" \
            \"$CLIMATE_SCENARIO\" \
            --variables $SELECTED_VARIABLES"

        if [ -n "$TEMP_RES" ]; then
            CMD="$CMD --temp_res $TEMP_RES"
        fi

        # Run validation
        eval $CMD 2>&1 | tee "$LOG_FILE"

        if [ $? -eq 0 ]; then
            echo "✓ Validation completed successfully for $CLIMATE_MODEL_CHAIN - $CLIMATE_SCENARIO"
        else
            echo "✗ Validation failed for $CLIMATE_MODEL_CHAIN - $CLIMATE_SCENARIO"
        fi
    done
done

echo ""
echo "================================================================================"
echo "VALIDATION COMPLETE"
echo "Processed $total_jobs model chain/scenario combinations"
echo "Logs saved in logs/ directory"
echo "================================================================================"
