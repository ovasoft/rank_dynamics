#!/bin/bash
# ── Rank Dynamics Pipeline ────────────────────────────────────────────────
# Master runner. All stages are controlled by a single YAML config.
# Each stage skips gracefully if its output already exists (--skip-existing).
#
# Usage:
#   bash run_pipeline.sh configs/cifar10.yaml
#   bash run_pipeline.sh configs/babylm_strict_small.yaml
#   bash run_pipeline.sh configs/babylm_strict_small.yaml --stage 2   # one stage only
#   bash run_pipeline.sh configs/babylm_strict_small.yaml --from 3    # resume from stage
#
# Stages:
#   0  Validate config and environment
#   1  Train FR + LR primary runs
#   2  Probe FR + LR (spectral + CKA self)
#   3  Train baseline seeds (FR seed1, FR seed2, LR seed1) for CKA reference
#   4  Probe cross-run CKA baselines
#   5  Train DSN variants (static, dynamic, flat)
#   6  Probe DSN variants
#   7  Epoch-1 intervention
#   8  Freeze-and-finetune ablation (10ep)
#   9  Full finetune to convergence (30ep)
#   10 Rank sweep (LR_r16, LR_r32)
#   11 Collate all results

set -euo pipefail

CONFIG="${1:-configs/cifar10.yaml}"
STAGE_ONLY=""
FROM_STAGE=0

# Parse optional flags
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE_ONLY="$2"; shift 2 ;;
        --from)  FROM_STAGE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Read config fields via Python ─────────────────────────────────────────
cfg() { python3 -c "
import yaml, sys
with open('$CONFIG') as f: c = yaml.safe_load(f)
keys = '$1'.split('.')
v = c
for k in keys: v = v[k]
print(v)
" 2>/dev/null || echo "$2"; }

TASK=$(cfg task "cifar10")
OUT=$(cfg output_root "outputs/$TASK")
RES=$(cfg results_dir "results/$TASK")
RANK=$(cfg primary_rank "8")
EPOCHS=$(cfg epochs "20")
SEEDS_RAW=$(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(str(s) for s in c.get('seeds', [0,1,2])))
" 2>/dev/null || echo "0 1 2")
read -ra SEEDS <<< "$SEEDS_RAW"
PRIMARY_SEED=${SEEDS[0]}

DSN_MODES_RAW=$(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(c.get('dsn_modes', ['static','dynamic','flat'])))
" 2>/dev/null || echo "static dynamic flat")
read -ra DSN_MODES <<< "$DSN_MODES_RAW"

RANKS_RAW=$(python3 -c "
import yaml
with open('$CONFIG') as f: c = yaml.safe_load(f)
print(' '.join(str(r) for r in c.get('ranks', [8,16,32])))
" 2>/dev/null || echo "8 16 32")
read -ra ALL_RANKS <<< "$RANKS_RAW"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="cuda:0"

echo "════════════════════════════════════════════════════════════"
echo "  Rank Dynamics Pipeline"
echo "  Config:  $CONFIG"
echo "  Task:    $TASK"
echo "  Output:  $OUT"
echo "  Results: $RES"
echo "  GPU:     $GPU"
echo "════════════════════════════════════════════════════════════"

run_stage() {
    local n="$1"; local desc="$2"; shift 2
    [[ -n "$STAGE_ONLY" && "$STAGE_ONLY" != "$n" ]] && return 0
    [[ "$n" -lt "$FROM_STAGE" ]] && return 0
    echo ""
    echo "── Stage $n: $desc ──────────────────────────────────────────"
    "$@"
}

skip_if_exists() {
    local path="$1"; shift
    if [[ -e "$path" ]]; then
        echo "  [SKIP] $path already exists"
        return 0
    fi
    "$@"
}

# ── Stage 0: Validate ─────────────────────────────────────────────────────
run_stage 0 "Validate config and environment" python3 -c "
from task_registry import load_config, get_task
cfg = load_config('$CONFIG')
task = get_task(cfg['task'])
print(f'  Task: {cfg[\"task\"]}')
print(f'  Primary rank: {cfg.get(\"primary_rank\", 8)}')
print(f'  Epochs: {cfg.get(\"epochs\", 20)}')
print('  Config OK')
"

# ── Stage 1: Train primary FR and LR runs ────────────────────────────────
run_stage 1 "Train FR + LR_r${RANK} (seed ${PRIMARY_SEED})" bash -c "
  skip_if_exists() { [[ -e \"\$1\" ]] && echo \"  [SKIP] \$1\" && return 0; shift; \"\$@\"; }
  skip_if_exists ${OUT}/FR/final/model.pt \
    python3 train.py --config $CONFIG --run_type FR --seed $PRIMARY_SEED
  skip_if_exists ${OUT}/LR_r${RANK}/final/model.pt \
    python3 train.py --config $CONFIG --run_type LR --rank $RANK --seed $PRIMARY_SEED
"

# ── Stage 2: Probe FR and LR ─────────────────────────────────────────────
run_stage 2 "Probe FR + LR_r${RANK}" bash -c "
  skip_if_exists ${OUT}/FR/probes/probes.csv \
    python3 probe_dynamics.py --config $CONFIG \
      --run_dir ${OUT}/FR --ref_run_dir ${OUT}/LR_r${RANK}
  skip_if_exists ${OUT}/LR_r${RANK}/probes/probes.csv \
    python3 probe_dynamics.py --config $CONFIG \
      --run_dir ${OUT}/LR_r${RANK} --ref_run_dir ${OUT}/FR
"

# ── Stage 3: Train baseline seeds ────────────────────────────────────────
run_stage 3 "Train baseline seeds for CKA reference" bash -c "
  skip_if_exists ${OUT}/FR_seed1/final/model.pt \
    python3 train.py --config $CONFIG --run_type FR --seed 1
  skip_if_exists ${OUT}/FR_seed2/final/model.pt \
    python3 train.py --config $CONFIG --run_type FR --seed 2
  skip_if_exists ${OUT}/LR_r${RANK}_seed1/final/model.pt \
    python3 train.py --config $CONFIG --run_type LR --rank $RANK --seed 1
"

# ── Stage 4: Cross-run CKA baselines ─────────────────────────────────────
run_stage 4 "Probe cross-run CKA baselines (FR-FR, LR-LR)" bash -c "
  skip_if_exists ${OUT}/FR_seed1/probes/cka_cross.csv \
    python3 probe_dynamics.py --config $CONFIG \
      --run_dir ${OUT}/FR_seed1 --ref_run_dir ${OUT}/FR_seed2
  skip_if_exists ${OUT}/LR_r${RANK}/probes/cka_cross.csv \
    python3 probe_dynamics.py --config $CONFIG \
      --run_dir ${OUT}/LR_r${RANK} --ref_run_dir ${OUT}/LR_r${RANK}_seed1
"

# ── Stage 5: Train DSN variants ───────────────────────────────────────────
run_stage 5 "Train DSN variants (${DSN_MODES[*]})" bash -c "
  for mode in ${DSN_MODES[*]}; do
    dir=${OUT}/LR_r${RANK}_dsn_\${mode}
    skip_if_exists \${dir}/final/model.pt \
      python3 train_dsn.py --config $CONFIG --rank $RANK --dsn_mode \${mode}
  done
"

# ── Stage 6: Probe DSN variants ───────────────────────────────────────────
run_stage 6 "Probe DSN variants" bash -c "
  for mode in ${DSN_MODES[*]}; do
    dir=${OUT}/LR_r${RANK}_dsn_\${mode}
    skip_if_exists \${dir}/probes/probes.csv \
      python3 probe_dynamics.py --config $CONFIG \
        --run_dir \${dir} --ref_run_dir ${OUT}/FR
  done
"

# ── Stage 7: Epoch-1 intervention ────────────────────────────────────────
run_stage 7 "Epoch-1 static intervention" bash -c "
  skip_if_exists ${OUT}/interventions/epoch1_intervention.csv \
    python3 epoch1_intervention.py --config $CONFIG \
      --fr_ckpt ${OUT}/FR/checkpoints/epoch_00/model.pt \
      --out_dir ${OUT}/interventions
"

# ── Stage 8: Freeze-and-finetune ablation (10ep) ─────────────────────────
run_stage 8 "Freeze-and-finetune ablation (10ep)" bash -c "
  skip_if_exists ${OUT}/ablation/ablation_results.csv \
    python3 ablate_early_layers.py --config $CONFIG \
      --lr_run_dir ${OUT}/LR_r${RANK} \
      --out_dir ${OUT}/ablation \
      --finetune_epochs 10
"

# ── Stage 9: Full finetune to convergence (30ep) ─────────────────────────
run_stage 9 "Full finetune to convergence (30ep)" bash -c "
  skip_if_exists ${OUT}/ablation_convergence/ablation_results.csv \
    python3 ablate_early_layers.py --config $CONFIG \
      --lr_run_dir ${OUT}/LR_r${RANK} \
      --out_dir ${OUT}/ablation_convergence \
      --finetune_epochs 30 --finetune_only
"

# ── Stage 10: Rank sweep ─────────────────────────────────────────────────
run_stage 10 "Rank sweep (${ALL_RANKS[*]})" bash -c "
  for r in ${ALL_RANKS[*]}; do
    [[ \"\$r\" == \"$RANK\" ]] && continue   # primary rank already done
    dir=${OUT}/LR_r\${r}
    skip_if_exists \${dir}/final/model.pt \
      python3 train.py --config $CONFIG --run_type LR --rank \${r} --seed $PRIMARY_SEED
    skip_if_exists \${dir}/probes/probes.csv \
      python3 probe_dynamics.py --config $CONFIG \
        --run_dir \${dir} --ref_run_dir ${OUT}/FR
  done
"

# ── Stage 11: Collate ─────────────────────────────────────────────────────
run_stage 11 "Collate all results" \
  python3 collate_results.py --config $CONFIG \
    --base_dir "$OUT" --out_dir "$RES"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Pipeline complete. Results: $RES"
echo "════════════════════════════════════════════════════════════"