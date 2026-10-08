"""
collate_grad_erank.py — Collate gradient eRank results across FR, LR, and
DSN runs and produce:

  1. grad_erank_ratio.csv — FR/LR eRank ratio over training (Aim 3 extension)
  2. dsn_grad_erank.csv   — eRank under each DSN mode vs untreated LR and FR
  3. Figure: gradient SVD distribution shapes (FR vs LR_r8 at early/late)
  4. Figure: gradient eRank under DSN vs FR vs LR

These answer:
  - Does DSN raise gradient eRank?
  - What do the SVD distributions of FR vs LR gradient matrices look like?

Usage:
    python analysis/collate_grad_erank.py --config configs/babylm_strict_small.yaml

Expects probe_gradient_erank.py to have been run for all relevant run dirs:
    outputs/babylm_strict_small/FR/probes/grad_erank.csv
    outputs/babylm_strict_small/LR_r8/probes/grad_erank.csv
    outputs/babylm_strict_small/LR_r{16,32,64,128,256}/probes/grad_erank.csv
    outputs/babylm_strict_small/DSN_{static,dynamic,flat}_r8/probes/grad_erank.csv
    (also FR/probes/grad_svd/ and LR_r8/probes/grad_svd/ if --save_svd was used)

Outputs written to:
    results/babylm_strict_small/grad_erank_ratio.csv
    results/babylm_strict_small/dsn_grad_erank.csv
    results/babylm_strict_small/fig_grad_svd_distribution.png
    results/babylm_strict_small/fig_dsn_grad_erank.png
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from task_registry import load_config
except ImportError:
    def load_config(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


# ── Helpers ───────────────────────────────────────────────────────────────

def load_grad_erank(run_dir):
    """Load grad_erank.csv for a run; return DataFrame or None."""
    path = os.path.join(run_dir, 'probes', 'grad_erank.csv')
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    df = pd.read_csv(path)
    return df


def mean_erank_by_checkpoint(df):
    """Average grad_erank across all weight matrices per checkpoint."""
    return (df.groupby(['checkpoint_name', 'tokens_or_epoch'])
              ['grad_erank'].mean()
              .reset_index()
              .sort_values('tokens_or_epoch'))


def load_grad_svd(run_dir, ckpt_name, weight_pattern='transformer.h.0'):
    """
    Load gradient SVD spectrum for a specific checkpoint and weight pattern.
    Returns dict {weight_name: singular_values_tensor} for matching weights.
    """
    svd_dir = os.path.join(run_dir, 'probes', 'grad_svd')
    if not os.path.isdir(svd_dir):
        return {}
    result = {}
    for fname in os.listdir(svd_dir):
        if not fname.endswith('_grad.pt'):
            continue
        if not fname.startswith(ckpt_name):
            continue
        weight_name = fname[len(ckpt_name) + 1: -len('_grad.pt')].replace('_', '.')
        if weight_pattern not in weight_name:
            continue
        path = os.path.join(svd_dir, fname)
        result[weight_name] = torch.load(path, weights_only=True)
    return result


# ── Figure 1: Gradient SVD distribution shapes ────────────────────────────

def plot_svd_distributions(fr_dir, lr_dir, output_path, rank=8):
    """
    Plot gradient singular value distributions for FR vs LR_r8
    at early (first real checkpoint) and late (final checkpoint) training.
    Two rows (early/late) × two columns (FR/LR), log-scale y-axis.

    Shows whether starvation is
    about a hard ceiling on the number of directions or a flatter distribution
    truncated at a low-rank horizon.
    """
    import torch

    # Find the two checkpoints to compare
    def get_first_and_last(run_dir):
        svd_dir = os.path.join(run_dir, 'probes', 'grad_svd')
        if not os.path.isdir(svd_dir):
            return None, None
        ckpts = sorted(set(
            f.split('_transformer')[0] if '_transformer' in f else
            f.split('_hf_model')[0] if '_hf_model' in f else None
            for f in os.listdir(svd_dir)
            if f.endswith('_grad.pt')
        ) - {None})
        if not ckpts:
            return None, None
        # Sort by token count
        def sort_key(name):
            try:
                return int(name.split('_')[-1])
            except ValueError:
                return 0
        ckpts.sort(key=sort_key)
        # Skip tokens_0 (step-0) for the "early" comparison — use first real update
        real_ckpts = [c for c in ckpts if not c.endswith('0000000000')]
        first = real_ckpts[0] if real_ckpts else ckpts[0]
        last  = ckpts[-1]
        return first, last

    fr_first, fr_last = get_first_and_last(fr_dir)
    lr_first, lr_last = get_first_and_last(lr_dir)

    if fr_first is None or lr_first is None:
        print("  [SVD distribution] SVD files not found — run probe_gradient_erank.py "
              "with --save_svd first.")
        return

    # Layer to visualise: use layer 0 attention projection
    # Try several common weight name patterns
    weight_patterns = [
        'transformer.h.0.attn.c_attn',
        'transformer.h.0.attn.c_proj',
        'blocks.0',
    ]

    def find_svd(run_dir, ckpt_name, patterns):
        svd_dir = os.path.join(run_dir, 'probes', 'grad_svd')
        for pat in patterns:
            d = load_grad_svd(run_dir, ckpt_name, weight_pattern=pat)
            if d:
                # Return first matching weight's SVD
                k = sorted(d.keys())[0]
                return k, d[k]
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle('Gradient singular value distributions: FR vs LR_r8\n'
                 '(early training vs final checkpoint)', fontsize=11)

    configs = [
        (0, 0, fr_dir,  fr_first, f'FR — early ({fr_first})',  '#1f77b4'),
        (0, 1, lr_dir,  lr_first, f'LR_r{rank} — early ({lr_first})', '#d62728'),
        (1, 0, fr_dir,  fr_last,  f'FR — final ({fr_last})',   '#1f77b4'),
        (1, 1, lr_dir,  lr_last,  f'LR_r{rank} — final ({lr_last})',  '#d62728'),
    ]

    for row, col, run_dir, ckpt, title, color in configs:
        ax = axes[row][col]
        wname, S = find_svd(run_dir, ckpt, weight_patterns)
        if S is None:
            ax.text(0.5, 0.5, 'SVD not found\nRun with --save_svd',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')
            ax.set_title(title, fontsize=9)
            continue

        S_np = S.numpy()
        S_np = S_np / S_np.sum()   # normalise to probability distribution
        ax.semilogy(np.arange(1, len(S_np) + 1), S_np,
                    color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Singular value index', fontsize=8)
        ax.set_ylabel('Normalised magnitude', fontsize=8)
        ax.tick_params(labelsize=7)

        # Annotate eRank
        from probe_gradient_erank import erank as _erank
        import torch as _torch
        er = _erank(_torch.tensor(S_np * S_np.sum()))
        ax.text(0.97, 0.97, f'eRank={er:.0f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        # Mark effective rank cutoff
        cumvar = np.cumsum(S_np)
        idx90  = np.searchsorted(cumvar, 0.90)
        ax.axvline(idx90, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(idx90 + 1, S_np[0] * 0.5, f'90%\n@{idx90}',
                fontsize=6, color='gray', va='top')

        # Weight name annotation
        ax.text(0.03, 0.03, wname.split('.')[-2] + '.' + wname.split('.')[-1],
                transform=ax.transAxes, fontsize=6, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SVD distribution figure: {output_path}")


# ── Figure 2: Gradient eRank under DSN ───────────────────────────────────

def plot_dsn_grad_erank(fr_mean, lr_mean, dsn_dfs, output_path, rank=8):
    """
    Plot mean gradient eRank over training for:
      FR (reference ceiling), LR untreated (floor), and each DSN mode.

    If DSN raises gradient eRank toward FR levels → starvation account
    is complicated (something else explains DSN's perplexity improvement).
    If DSN leaves eRank near LR floor → starvation account is supported
    (DSN improves perplexity via a different mechanism than expanding eRank).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # FR and LR untreated baselines
    ax.plot(fr_mean['tokens_or_epoch'], fr_mean['grad_erank'],
            color='#1f77b4', linewidth=2, label='FR (full-rank)', zorder=5)
    ax.plot(lr_mean['tokens_or_epoch'], lr_mean['grad_erank'],
            color='#d62728', linewidth=2, linestyle='--',
            label=f'LR_r{rank} (untreated)', zorder=5)

    # DSN modes
    dsn_styles = {
        'static':  ('#ff7f0e', '-.',  'DSN-static'),
        'dynamic': ('#2ca02c', ':',   'DSN-dynamic'),
        'flat':    ('#9467bd', (0,(3,1,1,1)), 'DSN-flat'),
    }
    for mode, (color, ls, label) in dsn_styles.items():
        if mode not in dsn_dfs or dsn_dfs[mode] is None:
            continue
        m = mean_erank_by_checkpoint(dsn_dfs[mode])
        ax.plot(m['tokens_or_epoch'], m['grad_erank'],
                color=color, linewidth=1.5, linestyle=ls, label=label)

    ax.set_xlabel('Tokens seen', fontsize=10)
    ax.set_ylabel('Mean gradient eRank', fontsize=10)
    ax.set_title(f'Gradient eRank under DSN vs FR vs LR_r{rank}\n'
                 f'(BabyLM Strict-Small / GPT-2)', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ DSN gradient eRank figure: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    cfg         = load_config(args.config)
    output_root = cfg.get('output_root', 'outputs/babylm_strict_small')
    results_dir = cfg.get('results_dir', 'results/babylm_strict_small')
    os.makedirs(results_dir, exist_ok=True)

    ranks = cfg.get('ranks', [8, 16, 32, 64, 128, 256])

    # ── Load FR ──────────────────────────────────────────────────────────
    fr_dir = os.path.join(output_root, 'FR')
    fr_df  = load_grad_erank(fr_dir)
    if fr_df is None:
        print("ERROR: FR grad_erank.csv not found. Run probe_gradient_erank.py "
              "on the FR run first.")
        return
    fr_mean = mean_erank_by_checkpoint(fr_df)

    # ── Load LR ranks ────────────────────────────────────────────────────
    lr_dfs = {}
    for r in ranks:
        lr_dir = os.path.join(output_root, f'LR_r{r}')
        df = load_grad_erank(lr_dir)
        if df is not None:
            lr_dfs[r] = (df, mean_erank_by_checkpoint(df))

    # ── Gradient eRank ratio table ────────────────────────────────────────
    ratio_rows = []
    for r, (df, lr_mean) in lr_dfs.items():
        merged = fr_mean.merge(lr_mean, on='tokens_or_epoch', suffixes=('_fr', '_lr'))
        merged['erank_ratio'] = merged['grad_erank_fr'] / merged['grad_erank_lr'].clip(lower=1e-6)
        merged['rank'] = r
        ratio_rows.append(merged[['tokens_or_epoch', 'rank',
                                   'grad_erank_fr', 'grad_erank_lr', 'erank_ratio']])

    if ratio_rows:
        ratio_df = pd.concat(ratio_rows, ignore_index=True)
        ratio_csv = os.path.join(results_dir, 'grad_erank_ratio.csv')
        ratio_df.to_csv(ratio_csv, index=False)
        print(f"✓ Gradient eRank ratio table: {ratio_csv}")

        # Print summary for paper
        print("\nGradient eRank ratio summary (FR / LR) at final checkpoint:")
        final = ratio_df.groupby('rank')['erank_ratio'].last()
        for r, val in final.items():
            print(f"  LR_r{r}: {val:.1f}x")

    # ── Load DSN runs ─────────────────────────────────────────────────────
    dsn_modes = cfg.get('dsn_modes', ['static', 'dynamic', 'flat'])
    primary_rank = cfg.get('primary_rank', 8)
    dsn_dfs = {}
    for mode in dsn_modes:
        dsn_dir = os.path.join(output_root, f'DSN_{mode}_r{primary_rank}')
        df = load_grad_erank(dsn_dir)
        dsn_dfs[mode] = df

    n_dsn_found = sum(1 for d in dsn_dfs.values() if d is not None)
    print(f"\nDSN runs found: {n_dsn_found}/{len(dsn_modes)}")

    # ── DSN eRank table ───────────────────────────────────────────────────
    dsn_rows = []
    if primary_rank in lr_dfs:
        lr_mean_primary = lr_dfs[primary_rank][1]
        for mode, df in dsn_dfs.items():
            if df is None:
                continue
            m = mean_erank_by_checkpoint(df)
            m['run'] = f'DSN_{mode}'
            dsn_rows.append(m)
        fr_row = fr_mean.copy(); fr_row['run'] = 'FR'
        lr_row = lr_mean_primary.copy(); lr_row['run'] = f'LR_r{primary_rank}'
        all_rows = [fr_row, lr_row] + dsn_rows
        dsn_df = pd.concat(all_rows, ignore_index=True)
        dsn_csv = os.path.join(results_dir, 'dsn_grad_erank.csv')
        dsn_df.to_csv(dsn_csv, index=False)
        print(f"✓ DSN gradient eRank table: {dsn_csv}")

        # Print summary
        print("\nGradient eRank at final checkpoint (mean across layers):")
        final_summary = dsn_df.groupby('run')['grad_erank'].last()
        for run, val in final_summary.items():
            print(f"  {run}: {val:.1f}")

    # ── Figure: SVD distributions ─────────────────────────────────────────
    lr8_dir = os.path.join(output_root, f'LR_r{primary_rank}')
    svd_fig_path = os.path.join(results_dir, 'fig_grad_svd_distribution.png')
    print(f"\nGenerating SVD distribution figure...")
    plot_svd_distributions(fr_dir, lr8_dir, svd_fig_path, rank=primary_rank)

    # ── Figure: DSN gradient eRank ────────────────────────────────────────
    if primary_rank in lr_dfs:
        dsn_fig_path = os.path.join(results_dir, 'fig_dsn_grad_erank.png')
        print(f"Generating DSN gradient eRank figure...")
        plot_dsn_grad_erank(
            fr_mean, lr_dfs[primary_rank][1],
            dsn_dfs, dsn_fig_path, rank=primary_rank)

    print("\nAll done.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    main(p.parse_args())