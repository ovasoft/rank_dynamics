#!/bin/bash
# probe_all_ranks.sh — Run probe_dynamics for all LR rank runs against FR.
#
# Usage:
#   bash probe_all_ranks.sh configs/cifar10.yaml
#   bash probe_all_ranks.sh configs/babylm_strict_small.yaml
#
# Discovers all LR_rN directories under output_root and probes each one.
# Skips any run whose probes/probes.csv already exists.

set -euo pipefail

CONFIG="${1:?Usage: bash probe_all_ranks.sh <config.yaml>}"

# Read output_root from config
OUTPUT_ROOT=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(c.get('output_root', 'outputs'))
")

echo "Config:       $CONFIG"
echo "Output root:  $OUTPUT_ROOT"
echo ""

FR_DIR="${OUTPUT_ROOT}/FR"

if [[ ! -d "$FR_DIR" ]]; then
    echo "ERROR: FR run directory not found: $FR_DIR"
    exit 1
fi

# Discover all LR_rN primary runs (no seed suffix, no dsn suffix)
RANKS=()
for d in "${OUTPUT_ROOT}"/LR_r*/; do
    name=$(basename "$d")
    # Match LR_rN exactly — skip LR_rN_seed*, LR_rN_dsn_*
    if [[ "$name" =~ ^LR_r[0-9]+$ ]]; then
        RANKS+=("$name")
    fi
done

if [[ ${#RANKS[@]} -eq 0 ]]; then
    echo "No LR_rN directories found under $OUTPUT_ROOT"
    exit 1
fi

echo "Found ${#RANKS[@]} LR run(s): ${RANKS[*]}"
echo ""

for run_name in "${RANKS[@]}"; do
    run_dir="${OUTPUT_ROOT}/${run_name}"
    probe_out="${run_dir}/probes/probes.csv"

    if [[ -f "$probe_out" ]]; then
        echo "[SKIP] $run_name — probes already exist"
        continue
    fi

    echo "══ Probing $run_name ══════════════════════════════════════"
    python3 probe_dynamics.py \
        --config   "$CONFIG" \
        --run_dir  "$run_dir" \
        --ref_run_dir "$FR_DIR"
    echo ""
done

echo "All ranks probed."