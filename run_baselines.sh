#!/bin/bash
# run_baselines.sh — launches the three remaining experiments
# CKA baseline: two independent FR runs with different seeds
# Full finetune to convergence: LR checkpoint, all layers unfrozen, 30 epochs

set -euo pipefail

# Run from the repo root and make src/, probing/ and analysis/ importable.
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD/src/vision:$PWD/probing:$PWD/analysis${PYTHONPATH:+:$PYTHONPATH}"

DATASET=cifar10
LR_RUN=outputs_cifar/cifar10/LR_r8
OUT_DIR=outputs_cifar/cifar10

# ── 1. FR seed 1 (already exists as FR, this is seed 2 and seed 3) ──────
echo "=== Training FR seed 1 ==="
python src/vision/train_cifar.py --dataset $DATASET --seed 1 \
    --run_dir ${OUT_DIR}/FR_seed1

echo "=== Training FR seed 2 ==="
python src/vision/train_cifar.py --dataset $DATASET --seed 2 \
    --run_dir ${OUT_DIR}/FR_seed2

# ── 2. Probe both to get cka_self, then cross-probe them against each other
echo "=== Probing FR_seed1 ==="
python probing/probe_dynamics.py \
    --run_dir ${OUT_DIR}/FR_seed1 \
    --dataset $DATASET \
    --ref_run_dir ${OUT_DIR}/FR_seed2

echo "=== Probing FR_seed2 ==="
python probing/probe_dynamics.py \
    --run_dir ${OUT_DIR}/FR_seed2 \
    --dataset $DATASET \
    --ref_run_dir ${OUT_DIR}/FR_seed1

# Also probe LR vs LR (need a second LR run for the LR-LR baseline)
echo "=== Training LR_r8 seed 1 ==="
python src/vision/train_cifar.py --dataset $DATASET --seed 1 --rank 8 \
    --run_dir ${OUT_DIR}/LR_r8_seed1

echo "=== Probing LR_r8 vs LR_r8_seed1 (LR-LR baseline) ==="
python probing/probe_dynamics.py \
    --run_dir ${OUT_DIR}/LR_r8 \
    --dataset $DATASET \
    --ref_run_dir ${OUT_DIR}/LR_r8_seed1

# ── 3. Full finetune to convergence ──────────────────────────────────────
echo "=== Full finetune to convergence ==="
python src/ablate_early_layers.py \
    --dataset $DATASET \
    --rank 8 \
    --lr_run_dir $LR_RUN \
    --out_dir ${OUT_DIR}/ablation_convergence \
    --finetune_epochs 30 \
    --finetune_only   # flag to skip the k-sweep and only run full_finetune

echo "=== Done ==="