# Rank Dynamics: Gradient Starvation in Low-Rank Pretraining

Code for the paper on OpenReview: **https://openreview.net/forum?id=Aw1itOSwhx**

This repository studies *when* and *why* low-rank (LR) pretrained models diverge from
their full-rank (FR) counterparts. We train matched FR and rank-constrained models on
language (BabyLM 2026 Strict-Small, GPT-2-style decoder) and vision (small ViT on
CIFAR-10), checkpoint them densely, and track four axes over training:

1. **Parameter**: drift from initialisation
2. **Spectral**: effective rank, stable rank and singular-value power-law exponent of weights
3. **Optimisation**: effective rank of gradients and weight updates
4. **Representational**: CKA and mutual-kNN alignment against FR, and self-CKA over time

## Main findings

- **The FR–LR split happens early and abruptly.** Most of the representational divergence
  is present by the first checkpoint and barely changes afterwards.
- **Gradient starvation.** LR models receive gradients with a much lower effective rank
  than FR models, and this ratio stays roughly constant through training rather than closing.
- **LR models settle because they run out of room.** LR models reach a stable state sooner
  than FR, but because they have exhausted their capacity, not because they converged faster.
- **Spectral geometry at initialisation matters but is not enough.** Matching the
  initial spectrum (`LR_specmatch8`) or correcting the spectrum during training (DSN)
  does not close the gap.
- **Representational alignment has a functional cost that perplexity misses.** DSN
  variants cut LR perplexity by about 75% but fall to chance on BLiMP. Untreated LR_r8 stays
  above chance despite much worse perplexity.

Final results on BabyLM Strict-Small (rank 8, seed 0; from [`docs/table1_main.md`](docs/table1_main.md)):

| Model | CKA to FR | BLiMP | Val PPL | Grad eRank growth | Self-CKA |
|---|---|---|---|---|---|
| FR | 0.9800 | 0.6338 | 46.98 | 5.98× | 1.0000 |
| LR_r8 | 0.0349 | 0.5563 | 456.60 | 0.65× | 0.7050 |
| GradTrunc8 | 0.0645 | 0.5775 | 72.81 | 2.48× | 0.8499 |
| ReLoRA_r8 | 0.0527 | 0.4950 | 604.58 | 1.91× | 0.8839 |
| LR_specmatch8 | 0.0287 | 0.5325 | 438.75 | 4.73× | 0.6758 |
| LR_r8_tuned | 0.0398 | 0.5200 | 138.24 | 0.62× | 0.7300 |
| LR_r8_dsn_static | 0.0313 | 0.5150 | 117.18 | 0.93× | 0.9507 |
| LR_r8_dsn_dynamic | 0.0400 | 0.5144 | 114.73 | 0.94× | 0.9556 |
| LR_r8_dsn_flat | 0.0328 | 0.4875 | 109.67 | 0.83× | 0.9496 |

## Repository layout

```
configs/      YAML configs: one per task (cifar10, babylm_strict_small, babylm_debug)
src/          Training
  train.py                         unified FR / LR trainer (epoch or token-budget mode)
  task_registry.py                 task → data loaders, model, LowRankLinear replacement
  train_galore.py, train_relora.py gradient-projection / merge-and-restart baselines
  train_gradrank.py                direct gradient-eRank truncation (GradTrunc)
  generate_spectrum_matched_init.py, resume_training_from_checkpoint.py
                                   spectrum-matched LR initialisation control
  sweep_lr_hyperparameters.py, train_lr_tuned.py
                                   LR-favourable hyperparameter sweep + full run
  epoch1_intervention.py           single-axis interventions after epoch 1
  ablate_early_layers.py           freeze-and-finetune early-layer ablation
  vision/train_cifar.py, vision/train_cifar_dsn.py
                                   ViT training and Dynamic Spectral Normalisation (DSN)
probing/      Checkpoint probes
  probe_dynamics.py                spectral / optimisation / parameter / CKA probes
  probe_gradient_erank.py          true gradient eRank from backward passes
  probe_mutual_knn.py              mutual-kNN alignment (Huh et al., 2024)
  probe_transfer.py                BLiMP minimal-pair evaluation
  generate_step0_checkpoints.py, save_step0_cka.py, patch_step0.py
                                   zero-update (step-0) CKA baseline
analysis/     Collation, tables and figures (analysis/visualization/ for per-aim plots)
docs/         Result tables (main table, BLiMP-vs-CKA) and rebuttal notes
rebuttal_tables_r2w1/  GaLore / ReLoRA / GradTrunc comparison tables
run_pipeline.sh        end-to-end staged pipeline driven by a config
```

`data/`, `outputs/`, `results/`, `figures/` and `babylm-eval/` are git-ignored. The scripts
regenerate them.

## Setup

Tested with Python 3, PyTorch 2.8 (CUDA 12.8) and `transformers` 4.57.

```bash
pip install torch torchvision transformers datasets pyyaml numpy pandas matplotlib
```

The BabyLM task loads the `BabyLM-community/BabyLM-2026-Strict-Small` dataset and the
architecture from `BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction` on the
Hugging Face Hub. CIFAR-10 is downloaded with `torchvision`.

The scripts import each other as top-level modules (for example, `task_registry`
and `probe_dynamics`). Put all three code directories on the path before running anything:

```bash
export PYTHONPATH=$PWD/src:$PWD/src/vision:$PWD/probing:$PWD/analysis
```

## Running

**Full pipeline.** Stages 0–11 cover training, probing, seed baselines, DSN, epoch-1
intervention, ablations, the rank sweep and collation:

```bash
bash run_pipeline.sh configs/babylm_strict_small.yaml
bash run_pipeline.sh configs/cifar10.yaml
bash run_pipeline.sh configs/babylm_strict_small.yaml --stage 2   # one stage
bash run_pipeline.sh configs/babylm_strict_small.yaml --from 3    # resume
```

> `run_pipeline.sh` was written for a flat layout. It calls scripts like `train.py`
> and `probe_dynamics.py` from the repo root, so run it from a directory where those
> names resolve, or adjust the paths. For example, use `python3 src/train.py`.

**Individual steps:**

```bash
# Train full-rank and rank-8 models
python src/train.py --config configs/babylm_strict_small.yaml --run_type FR --seed 0
python src/train.py --config configs/babylm_strict_small.yaml --run_type LR --rank 8 --seed 0

# Probe every checkpoint, with CKA against the FR run
python probing/probe_dynamics.py --config configs/babylm_strict_small.yaml \
    --run_dir outputs/babylm_strict_small/LR_r8 \
    --ref_run_dir outputs/babylm_strict_small/FR

# Probe all LR_rN runs found under output_root
bash probe_all_ranks.sh configs/babylm_strict_small.yaml

# BLiMP evaluation on final checkpoints
python probing/probe_transfer.py --config configs/babylm_strict_small.yaml \
    --run_dirs outputs/babylm_strict_small/FR outputs/babylm_strict_small/LR_r8 \
    --checkpoint final --max_pairs_per_phenomenon 200 --all_phenomena

# Collate per-aim CSVs and build the main table
python analysis/collate_results.py --config configs/babylm_strict_small.yaml
python analysis/generate_results_table.py --help
```

Use `--help` on any script for its full options. Each module docstring describes the
experiment it implements.

## Citation

If you use this code, please cite the paper (see the OpenReview page for the up-to-date
BibTeX):

```bibtex
@misc{rank_dynamics,
  title  = {TODO: paper title},
  author = {TODO},
  note   = {OpenReview},
  url    = {https://openreview.net/forum?id=Aw1itOSwhx}
}
```
