# Gradient Starvation in Low-Rank Language Model Pretraining

**Victor Adelakun Omolaoye and Gerard de Melo** (HPI / University of Potsdam)

[Paper (PDF)](docs/gradient_starvation_low_rank_pretraining.pdf) · [OpenReview](https://openreview.net/forum?id=Aw1itOSwhx)

Low-rank (LR) pretraining underperforms full-rank (FR) pretraining, but why the gap
persists remains unclear. Using GPT-2 on BabyLM, we show the representational split
between them is not fixed at initialisation. Spectrum-matched initialisation eliminates
the representational gap at step zero, but it is reasserted within a few training steps,
implicating training dynamics over initialisation. We further show gradient
dimensionality is a causally separable driver of this split, not merely a restatement of
the weight factorisation. Restricting or expanding gradient dimensionality of LR models
directly shifts representational alignment with the FR counterpart independent of weight
rank. Finally, optimisation success and representational alignment dissociate: spectral
normalisation and other interventions improve perplexity without recovering FR–LR
similarity, and representational alignment predicts downstream competence better than
perplexity does.

This repository contains all code needed to reproduce the paper. That includes training,
the probing pipeline, and the analysis and figure scripts.

## Setup

```bash
git clone https://github.com/ovasoft/rank_dynamics.git
cd rank_dynamics
pip install -r requirements.txt
```

We used Python 3.9, PyTorch 2.8 and CUDA 12.8 on NVIDIA A100 80GB GPUs. Language runs
use BF16. The BabyLM task downloads the corpus (`BabyLM-community/BabyLM-2026-Strict-Small`)
and the GPT-2 architecture and tokenizer
(`BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction`) from the Hugging Face
Hub. CIFAR-10 is downloaded through `torchvision`.

Run scripts from the repository root. Configs and output paths are relative to it. Each
script adds `src/`, `probing/` and `analysis/` to its own import path, so no `PYTHONPATH`
setup is needed.

## Quick check

`configs/babylm_debug.yaml` runs the same pipeline on a 200K-token budget and writes to
`outputs/babylm_debug/`. It takes a few minutes on one GPU:

```bash
for s in 0 1 2 3 4 5 6 7 11; do bash run_pipeline.sh configs/babylm_debug.yaml --stage $s; done
```

## Reproducing the paper

### 1. Core pipeline

`run_pipeline.sh` runs every experiment that depends only on a config. Each stage skips
any run whose output already exists, so it can be stopped and resumed safely.

```bash
bash run_pipeline.sh configs/babylm_strict_small.yaml          # language (primary)
bash run_pipeline.sh configs/cifar10.yaml                      # vision (cross-domain)
bash run_pipeline.sh configs/babylm_strict_small.yaml --stage 2   # a single stage
bash run_pipeline.sh configs/babylm_strict_small.yaml --from 5    # resume from a stage
```

| Stage | What it does | Paper |
|---|---|---|
| 0 | Validate config and environment | |
| 1 | Train FR and LR_r8 (seed 0) | §3.1–3.2 |
| 2 | Probe spectra, gradients and CKA at every checkpoint | §3.3, Aims 1–3 |
| 3–4 | Second-seed FR and LR runs, plus FR–FR and LR–LR CKA baselines | Aim 1, App. I |
| 5–6 | Train and probe the three DSN variants | §3.4, Aim 4 (O4.2) |
| 7 | Spectral correction at t* (drift-scaled, alpha-flat, combined) | Aim 4 (O4.1), Table 2 |
| 8–9 | Layer reinitialisation (k-sweep) and full fine-tune to convergence | Aim 6 |
| 10 | Rank sweep: train and probe every rank in `ranks` | Aims 1, 3, 5 |
| 11 | Collate per-aim CSVs into `results/<task>/` | |

`run_pipeline.sh` uses `CUDA_VISIBLE_DEVICES`, so you can pin it to one GPU:
`CUDA_VISIBLE_DEVICES=1 bash run_pipeline.sh ...`.

### 2. Gradient-dimensionality interventions (§3.4, Aim 3)

```bash
# GradTrunc: full-rank weights, gradients projected onto a fixed rank-8 subspace
python src/train_gradrank.py --config configs/babylm_strict_small.yaml --grad_rank 8

# ReLoRA: rank-8 adapter merged and reinitialised every 5M tokens
python src/train_relora.py --config configs/babylm_strict_small.yaml \
    --rank 8 --merge_every_tokens 5000000
```

### 3. Spectrum-matched initialisation (§3.2, Aim 4 O4.3)

```bash
python src/generate_spectrum_matched_init.py --config configs/babylm_strict_small.yaml \
    --fr_reference_ckpt outputs/babylm_strict_small/FR/checkpoints/tokens_0000000000/model.pt \
    --rank 8 --run_dir outputs/babylm_strict_small/LR_specmatch8
python src/resume_training_from_checkpoint.py --config configs/babylm_strict_small.yaml \
    --resume_from outputs/babylm_strict_small/LR_specmatch8/checkpoints/tokens_0000000000/model.pt \
    --rank 8 --run_dir outputs/babylm_strict_small/LR_specmatch8
```

### 4. LR-tuned hyperparameters (§3.4, Table 3)

```bash
python src/sweep_lr_hyperparameters.py --config configs/babylm_strict_small.yaml \
    --rank 8 --sweep_token_budget 10000000 \
    --lr_mults 0.5 1.0 2.0 3.0 --init_scale_mults 0.5 1.0 2.0
python src/train_lr_tuned.py --config configs/babylm_strict_small.yaml \
    --rank 8 --lr_mult 3.0 --init_scale_mult 0.5 \
    --run_dir outputs/babylm_strict_small/LR_r8_tuned
```

### 5. Probing the extra runs

Run these for each extra run directory (`GradTrunc8`, `ReLoRA_r8`, `LR_specmatch8`,
`LR_r8_tuned`, the DSN runs):

```bash
RUN=outputs/babylm_strict_small/GradTrunc8
FR=outputs/babylm_strict_small/FR
python probing/probe_dynamics.py       --config configs/babylm_strict_small.yaml --run_dir $RUN --ref_run_dir $FR
python probing/probe_gradient_erank.py --config configs/babylm_strict_small.yaml --run_dir $RUN
python probing/probe_mutual_knn.py     --config configs/babylm_strict_small.yaml --run_dir $RUN --ref_run_dir $FR --k 10
```

BLiMP accuracy (App. J.2), 200 pairs per phenomenon, 1,600 pairs in total:

```bash
python probing/probe_transfer.py --config configs/babylm_strict_small.yaml \
    --run_dirs outputs/babylm_strict_small/{FR,LR_r8,GradTrunc8,ReLoRA_r8,LR_specmatch8,LR_r8_tuned,LR_r8_dsn_static,LR_r8_dsn_dynamic,LR_r8_dsn_flat} \
    --checkpoint final --max_pairs_per_phenomenon 200
```

### 6. Tables and figures

```bash
# Table 1 (main results)
python analysis/generate_results_table.py \
    --blimp_summary results/babylm_strict_small/blimp_summary.csv \
    --run FR=outputs/babylm_strict_small/FR --run LR_r8=outputs/babylm_strict_small/LR_r8 \
    ...  # one --run NAME=DIR per row
    --out results/babylm_strict_small/table1_main

# Table 5 (BLiMP vs CKA vs perplexity)
python analysis/combine_blimp_cka_table.py --blimp_summary results/babylm_strict_small/blimp_summary.csv \
    --run ... --fr_fr_baseline outputs/babylm_strict_small/FR_seed1 \
    --out results/babylm_strict_small/combined_blimp_cka.md

# Cross-domain collation, then figures
python analysis/collate_results.py --config configs/cifar10.yaml --config2 configs/babylm_strict_small.yaml \
    --out_dir results/cross_domain
python analysis/visualization/visualize_aim1.py   # likewise aim2, aim3, aim4, aim5_aim6
```

Every script supports `--help`, and its module docstring describes the experiment and its
outputs. The tables reported in the paper are also saved in [`docs/`](docs/).

## Main results

BabyLM Strict-Small, GPT-2, rank 8 (paper Table 1). eRank growth is the ratio of final to
initial mean gradient effective rank. Self-CKA is the mean CKA between consecutive
checkpoints of the same run.

| Model | CKA to FR | BLiMP | PPL | eRank growth | Self-CKA |
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
configs/                 one YAML per task; babylm_debug.yaml is a fast smoke test
src/
  task_registry.py       task → data loaders, model, LowRankLinear (W = BA) replacement
  train.py               FR / LR training (epoch mode for vision, token budget for language)
  train_dsn.py           Dynamic Spectral Normalisation (static / dynamic / flat)
  train_gradrank.py      GradTrunc: fixed rank-r gradient projection, full-rank weights
  train_relora.py        ReLoRA: periodic merge-and-reinit of a rank-r adapter
  train_galore.py        GaLore comparison (extra; not reported in the paper)
  generate_spectrum_matched_init.py, resume_training_from_checkpoint.py
  sweep_lr_hyperparameters.py, train_lr_tuned.py
  epoch1_intervention.py spectral correction at t*
  ablate_early_layers.py layer reinitialisation and full fine-tuning
  vision/train_cifar.py  ViT definition and standalone CIFAR training
probing/
  probe_dynamics.py      CKA (cross / self / init), weight spectra, update and drift metrics
  probe_gradient_erank.py  gradient effective rank from a backward pass on the probe batch
  probe_mutual_knn.py    mutual k-NN alignment
  probe_transfer.py      BLiMP minimal-pair accuracy
  generate_step0_checkpoints.py, save_step0_cka.py, verify_checkpoint_loading.py
analysis/                collation, tables, summaries; visualization/ holds the figure scripts
docs/                    paper PDF and result tables
```

`data/`, `outputs/`, `results/` and `figures/` are generated and git-ignored.

## Citation

```bibtex
@misc{omolaoye2026gradient,
  title  = {Gradient Starvation in Low-Rank Language Model Pretraining},
  author = {Omolaoye, Victor Adelakun and de Melo, Gerard},
  year   = {2026},
  url    = {https://openreview.net/forum?id=Aw1itOSwhx}
}
```
