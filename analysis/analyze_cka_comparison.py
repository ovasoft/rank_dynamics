"""
analyze_cka_comparison.py — Compare cross-run CKA-vs-FR trajectories across
an arbitrary named set of runs (LR_r8, GaLore, ReLoRA, DSN variants, ...).

This is the CKA counterpart to analyze_grad_erank_approx.py. Question it
answers: does GaLore/ReLoRA show the same early representational
split and low, flat CKA-vs-FR trajectory the paper reports for LR_r8 (Aim
1's "split exists before meaningful learning" finding), or does either
method's periodic subspace rotation let it track FR more closely in
representation space too — not just in gradient eRank?

collate_results.py's aim1_divergence performs the equivalent collation but
is hardcoded to discover directories named 'FR' and 'LR_r{rank}' via
discover_runs()/is_lr_primary(). It has no path for arbitrarily-named runs
like 'GaLore_proj8' or 'ReLoRA_r8', so this script reads the same
cka_cross.csv file (written by probe_dynamics.py when given --ref_run_dir)
directly, for any named set of runs.

Reuses:
  - load_config              (task_registry.py)
  - checkpoint_tokens parsing convention (same as analyze_grad_erank_approx.py)

Prerequisite: each run_dir must already have probes/cka_cross.csv, i.e.
probe_dynamics.py must have been run for it with --ref_run_dir <FR_dir>.

Usage:
    python analysis/analyze_cka_comparison.py --config configs/babylm_strict_small.yaml \\
        --run LR_r8=outputs/babylm_strict_small/LR_r8 \\
        --run GaLore_proj8=outputs/babylm_strict_small/GaLore_proj8 \\
        --run ReLoRA_r8=outputs/babylm_strict_small/ReLoRA_r8 \\
        --fr_fr_baseline outputs/babylm_strict_small/FR_seed1

--fr_fr_baseline is optional: if given, it should be a second FR run's
cka_cross.csv (probed with --ref_run_dir pointing at the primary FR run),
providing the FR-vs-FR reference level shown in the paper's Figure 1(a) —
useful context for how far below "two independently trained FR models"
each comparison run sits.

Outputs:
    <results_dir>/cka_comparison.csv          — long-format per-checkpoint,
                                                per-run mean CKA-vs-FR
    <results_dir>/cka_comparison_wide.csv     — wide-format, one column per run
    <results_dir>/fig_cka_comparison.png
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from task_registry import load_config


def checkpoint_tokens(checkpoint_name):
    """Parse the token count out of 'tokens_0010000000' / 'epoch_05' etc."""
    try:
        return int(checkpoint_name.split('_')[1])
    except (IndexError, ValueError):
        return 0


def load_run_cka(run_dir):
    """
    Load <run_dir>/probes/cka_cross.csv, mean cka_vs_ref across layers per
    checkpoint. Returns a Series indexed by checkpoint_name, sorted by
    token count, or None if the file is missing.
    """
    path = os.path.join(run_dir, 'probes', 'cka_cross.csv')
    if not os.path.exists(path):
        print(f"  [missing] {path} — run probe_dynamics.py with --ref_run_dir "
             f"first")
        return None
    df = pd.read_csv(path)
    trend = df.groupby('checkpoint_name')['cka_vs_ref'].mean()
    trend = trend.reindex(sorted(trend.index, key=checkpoint_tokens))
    return trend


def main(args):
    cfg = load_config(args.config) if args.config else {}
    results_dir = args.results_dir or cfg.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    runs = {}
    for spec in args.run:
        if '=' not in spec:
            raise ValueError(f"--run must be NAME=PATH, got: {spec}")
        name, path = spec.split('=', 1)
        runs[name] = path

    print("Loading CKA-vs-FR trends...")
    trends = {}
    for name, path in runs.items():
        t = load_run_cka(path)
        if t is None:
            continue
        trends[name] = t
        last = t.dropna()
        print(f"  [ok] {name}: {len(t)} checkpoints, "
             f"first={last.iloc[0]:.4f}, final={last.iloc[-1]:.4f}"
             if len(last) else f"  [ok] {name}: empty")

    fr_fr_trend = None
    if args.fr_fr_baseline:
        fr_fr_trend = load_run_cka(args.fr_fr_baseline)
        if fr_fr_trend is not None:
            last = fr_fr_trend.dropna()
            print(f"  [ok] FR_vs_FR baseline: {len(fr_fr_trend)} checkpoints, "
                 f"first={last.iloc[0]:.4f}, final={last.iloc[-1]:.4f}"
                 if len(last) else "  [ok] FR_vs_FR baseline: empty")

    if not trends:
        print("ERROR: no run had a usable cka_cross.csv — nothing to compare.")
        return

    # ── Long-format table ──
    long_rows = []
    for name, t in trends.items():
        for ckpt, val in t.items():
            long_rows.append({
                'checkpoint_name': ckpt,
                'tokens': checkpoint_tokens(ckpt),
                'run': name,
                'mean_cka_vs_fr': val,
            })
    long_df = pd.DataFrame(long_rows).sort_values(['run', 'tokens'])
    long_path = os.path.join(results_dir, 'cka_comparison.csv')
    long_df.to_csv(long_path, index=False)
    print(f"\n✓ Long-format comparison table: {long_path}")

    # ── Wide-format table ──
    wide_df = pd.DataFrame({name: t for name, t in trends.items()})
    if fr_fr_trend is not None:
        wide_df['FR_vs_FR_baseline'] = fr_fr_trend
    wide_df.index.name = 'checkpoint_name'
    wide_df['tokens'] = [checkpoint_tokens(c) for c in wide_df.index]
    wide_df = wide_df.sort_values('tokens').reset_index()
    wide_path = os.path.join(results_dir, 'cka_comparison_wide.csv')
    wide_df.to_csv(wide_path, index=False)
    print(f"✓ Wide-format table: {wide_path}")

    # ── Console summary ──
    print("\nFirst-checkpoint vs final-checkpoint mean CKA-vs-FR:")
    for name, t in trends.items():
        valid = t.dropna()
        if len(valid) == 0:
            continue
        print(f"  {name:<20} first={valid.iloc[0]:.4f}  final={valid.iloc[-1]:.4f}")
    if fr_fr_trend is not None:
        valid = fr_fr_trend.dropna()
        if len(valid):
            print(f"  {'FR_vs_FR (baseline)':<20} first={valid.iloc[0]:.4f}  "
                 f"final={valid.iloc[-1]:.4f}")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, t in trends.items():
        ax.plot([checkpoint_tokens(c) for c in t.index], t.values,
               '--', marker='o', markersize=3, linewidth=1.6, label=name)
    if fr_fr_trend is not None:
        ax.plot([checkpoint_tokens(c) for c in fr_fr_trend.index], fr_fr_trend.values,
               '-', color='black', linewidth=2, label='FR_vs_FR (baseline)')
    ax.set_xlabel('Tokens seen', fontsize=10)
    ax.set_ylabel('Mean CKA vs FR', fontsize=10)
    ax.set_title('Cross-run CKA vs FR', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'fig_cka_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison figure: {fig_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compare cross-run CKA-vs-FR across an arbitrary named '
                   'set of runs.')
    p.add_argument('--config', default=None)
    p.add_argument('--results_dir', default=None)
    p.add_argument('--run', action='append', required=True,
                  help="NAME=PATH, repeatable, e.g. --run LR_r8=outputs/.../LR_r8")
    p.add_argument('--fr_fr_baseline', default=None,
                  help='Optional: run_dir of a second FR seed probed with '
                      '--ref_run_dir pointing at the primary FR, for the '
                      'FR-vs-FR reference level')
    main(p.parse_args())