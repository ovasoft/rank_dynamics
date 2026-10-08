"""
collate_galore_relora_erank.py — Compare gradient eRank trajectories for
GaLore and ReLoRA runs against the FR / LR_r8 baselines.

collate_grad_erank.py's run discovery is hardcoded to 'FR', 'LR_r{rank}',
and 'DSN_{mode}_r{primary_rank}' naming, so it has no path for GaLore or
ReLoRA run directories. Rather than modifying that file, this script
reuses its two core functions directly and adds the GaLore/ReLoRA
comparison on top.

The discriminating question: GaLore keeps weights full-rank but
projects GRADIENTS to rank r; ReLoRA periodically merges a rank-r adapter
back to full-rank. If either shows a gradient eRank trajectory closer to
LR_r8's (flat, near its structural ceiling) than to FR's (expanding
throughout training), that supports gradient-subspace restriction as the
real mechanism, independent of weight factorization. If either instead
tracks FR — expanding over training rather than staying flat — that
would suggest the static LR_r8 result is at least partly attributable to
the weight factorization itself, not gradient-subspace restriction alone.

Reuses:
  - load_grad_erank, mean_erank_by_checkpoint   (collate_grad_erank.py)
  - load_config                                  (task_registry.py)

Usage:
    python analysis/collate_galore_relora_erank.py \\
        --config configs/babylm_strict_small.yaml \\
        --galore_dir outputs/babylm_strict_small/GaLore_proj8 \\
        --relora_dir outputs/babylm_strict_small/ReLoRA_r8 \\
        --rank 8

Expects probe_gradient_erank.py to have already been run for FR, LR_r{rank},
GaLore_proj{rank}, and ReLoRA_r{rank} (i.e. each has probes/grad_erank.csv).

Outputs:
    results/babylm_strict_small/galore_relora_grad_erank.csv
    results/babylm_strict_small/fig_galore_relora_grad_erank.png
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from collate_grad_erank import load_grad_erank, mean_erank_by_checkpoint
from task_registry import load_config


def main(args):
    cfg = load_config(args.config)
    output_root = cfg.get('output_root', 'outputs/babylm_strict_small')
    results_dir = cfg.get('results_dir', 'results/babylm_strict_small')
    os.makedirs(results_dir, exist_ok=True)

    rank = args.rank

    runs = {
        'FR':               os.path.join(output_root, 'FR'),
        f'LR_r{rank}':      os.path.join(output_root, f'LR_r{rank}'),
        f'GaLore_proj{rank}': args.galore_dir or os.path.join(
            output_root, f'GaLore_proj{rank}'),
        f'ReLoRA_r{rank}':  args.relora_dir or os.path.join(
            output_root, f'ReLoRA_r{rank}'),
    }

    means = {}
    for label, run_dir in runs.items():
        df = load_grad_erank(run_dir)
        if df is None:
            print(f"  [missing] {label}: no probes/grad_erank.csv at {run_dir} "
                  f"— run probe_gradient_erank.py on this run first")
            continue
        means[label] = mean_erank_by_checkpoint(df)

    if 'FR' not in means:
        print("ERROR: FR grad_erank.csv not found — cannot build comparison.")
        return

    # ── Comparison table: eRank ratio FR/{run} at each checkpoint ──
    fr_mean = means['FR']
    rows = []
    for label, m in means.items():
        if label == 'FR':
            continue
        merged = fr_mean.merge(m, on='tokens_or_epoch', suffixes=('_fr', '_other'))
        merged['erank_ratio_fr_over_run'] = (
            merged['grad_erank_fr'] / merged['grad_erank_other'].clip(lower=1e-6))
        merged['run'] = label
        rows.append(merged[['tokens_or_epoch', 'run', 'grad_erank_fr',
                            'grad_erank_other', 'erank_ratio_fr_over_run']])

    if not rows:
        print("No comparison runs found besides FR — nothing to write.")
        return

    comparison_df = pd.concat(rows, ignore_index=True)
    out_csv = os.path.join(results_dir, 'galore_relora_grad_erank.csv')
    comparison_df.to_csv(out_csv, index=False)
    print(f"✓ Comparison table: {out_csv}")

    print("\nGradient eRank ratio (FR / run) at final checkpoint:")
    final = comparison_df.groupby('run')['erank_ratio_fr_over_run'].last()
    for run, val in final.items():
        print(f"  {run}: {val:.1f}x")
    if f'LR_r{rank}' in means:
        lr_ratio = final.get(f'LR_r{rank}')
        print(f"\n  Reference — LR_r{rank} ratio: {lr_ratio:.1f}x "
              f"(compare GaLore/ReLoRA ratios above against this)")

    # ── Figure: absolute gradient eRank, all runs on one plot ──
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {
        'FR':                 ('#1f77b4', '-',  2.0),
        f'LR_r{rank}':        ('#d62728', '--', 2.0),
        f'GaLore_proj{rank}': ('#2ca02c', '-.', 1.5),
        f'ReLoRA_r{rank}':    ('#ff7f0e', ':',  1.5),
    }
    for label, m in means.items():
        color, ls, lw = styles.get(label, ('#7f7f7f', '-', 1.5))
        ax.plot(m['tokens_or_epoch'], m['grad_erank'],
                color=color, linestyle=ls, linewidth=lw, label=label)

    ax.set_xlabel('Tokens seen', fontsize=10)
    ax.set_ylabel('Mean gradient eRank', fontsize=10)
    ax.set_title(f'Gradient eRank: FR vs LR_r{rank} vs GaLore vs ReLoRA\n'
                 f'(BabyLM Strict-Small / GPT-2)', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'fig_galore_relora_grad_erank.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison figure: {fig_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--rank', type=int, default=8,
                   help='Rank to compare (matches LR_r{rank}, GaLore_proj{rank}, '
                        'ReLoRA_r{rank})')
    p.add_argument('--galore_dir', default=None,
                   help='Override GaLore run dir (default: '
                       '<output_root>/GaLore_proj{rank})')
    p.add_argument('--relora_dir', default=None,
                   help='Override ReLoRA run dir (default: '
                       '<output_root>/ReLoRA_r{rank})')
    main(p.parse_args())