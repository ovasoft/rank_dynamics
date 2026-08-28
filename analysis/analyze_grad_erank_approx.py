"""
analyze_grad_erank_approx.py — Compare weight-diff-based gradient eRank
(grad_erank_approx, from probe_dynamics.py's probes.csv) across an
arbitrary set of runs — FR, LR_r8, GaLore, ReLoRA, or any combination.

Why this metric and not probe_gradient_erank.py's true gradient: for
methods like GaLore, a fresh backward pass on a probe batch only sees the
RAW gradient before any internal optimizer-side projection — it is blind
to what the optimizer actually does with that gradient before applying
the update. grad_erank_approx instead measures the SVD entropy of the
actual applied update (W_curr - W_prev) between consecutive checkpoints,
which reflects what happened during training regardless of whether the
constraint lives in the forward graph (LowRankLinear, LoRA/ReLoRA
adapters) or inside the optimizer (GaLore-style gradient projection).
This is also the metric the paper's own aim3_grad_erank uses (filtering
weight names containing "attn"), so ratios computed here are directly
comparable to the paper's headline eRank-ratio numbers.

Reuses:
  - load_config                (task_registry.py)
Everything else here is new — no existing script compares grad_erank_approx
across an arbitrary named set of runs; collate_grad_erank.py is hardcoded
to FR/LR_r{rank}/DSN_{mode}_r{rank} naming and to probe_gradient_erank.py's
grad_erank.csv (a different file, computed by a different method).

Usage:
    python analyze_grad_erank_approx.py --config configs/babylm_strict_small.yaml \\
        --run FR=outputs/babylm_strict_small/FR \\
        --run LR_r8=outputs/babylm_strict_small/LR_r8 \\
        --run GaLore_proj8=outputs/babylm_strict_small/GaLore_proj8 \\
        --run ReLoRA_r8=outputs/babylm_strict_small/ReLoRA_r8 \\
        --baseline FR \\
        --weight_filter attn

Expects each run_dir to have probes/probes.csv (written by probe_dynamics.py).

Outputs:
    <results_dir>/grad_erank_approx_comparison.csv   — long-format per-checkpoint,
                                                        per-run values and ratios
    <results_dir>/grad_erank_approx_wide.csv         — wide-format, one column
                                                        per run, easy to eyeball
    <results_dir>/fig_grad_erank_approx_comparison.png
"""

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from task_registry import load_config


def checkpoint_tokens(checkpoint_name):
    """
    Parse the token count out of a checkpoint_name like 'tokens_0010000000'
    or 'epoch_05'. Returns an int used purely for sorting/x-axis position.
    """
    try:
        return int(checkpoint_name.split('_')[1])
    except (IndexError, ValueError):
        return 0


def load_run_trend(run_dir, weight_filter='attn', metric='grad_erank_approx'):
    """
    Load <run_dir>/probes/probes.csv, filter to weight_name containing
    `weight_filter` (default 'attn' — matches both c_attn and c_proj, same
    convention the paper's aim3_grad_erank uses), and return a Series
    indexed by checkpoint_name with the mean `metric` per checkpoint,
    sorted by token count.

    Returns None if the file is missing.
    """
    path = os.path.join(run_dir, 'probes', 'probes.csv')
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None

    df = pd.read_csv(path)
    if weight_filter:
        df = df[df['weight_name'].str.contains(weight_filter)]

    trend = df.groupby('checkpoint_name')[metric].mean()
    trend = trend.reindex(sorted(trend.index, key=checkpoint_tokens))
    return trend


def main(args):
    cfg = load_config(args.config) if args.config else {}
    results_dir = args.results_dir or cfg.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Parse --run NAME=PATH arguments into an ordered dict
    runs = {}
    for spec in args.run:
        if '=' not in spec:
            raise ValueError(f"--run must be in the form NAME=PATH, got: {spec}")
        name, path = spec.split('=', 1)
        runs[name] = path

    if args.baseline not in runs:
        raise ValueError(f"--baseline '{args.baseline}' not found among --run names: "
                         f"{list(runs.keys())}")

    print(f"Loading trends (weight_filter='{args.weight_filter}', "
         f"metric='{args.metric}')...")
    trends = {}
    for name, path in runs.items():
        t = load_run_trend(path, weight_filter=args.weight_filter, metric=args.metric)
        if t is None:
            print(f"  [skip] {name}: no probes.csv found at {path}")
            continue
        trends[name] = t
        print(f"  [ok] {name}: {len(t)} checkpoints, "
             f"final={t.dropna().iloc[-1]:.2f}" if len(t.dropna()) else f"  [ok] {name}: empty")

    if args.baseline not in trends:
        print(f"ERROR: baseline run '{args.baseline}' has no data — cannot compute ratios.")
        return

    baseline_trend = trends[args.baseline]

    # ── Long-format table: checkpoint, run, value, baseline_value, ratio ──
    long_rows = []
    for name, t in trends.items():
        if name == args.baseline:
            continue
        merged = pd.merge(
            baseline_trend.rename('baseline_value'),
            t.rename('value'),
            left_index=True, right_index=True, how='inner')
        for ckpt, row in merged.iterrows():
            long_rows.append({
                'checkpoint_name': ckpt,
                'tokens': checkpoint_tokens(ckpt),
                'run': name,
                'baseline_run': args.baseline,
                'value': row['value'],
                'baseline_value': row['baseline_value'],
                'ratio_baseline_over_run': (row['baseline_value'] / row['value']
                                           if row['value'] else None),
            })

    long_df = pd.DataFrame(long_rows).sort_values(['run', 'tokens'])
    long_path = os.path.join(results_dir, 'grad_erank_approx_comparison.csv')
    long_df.to_csv(long_path, index=False)
    print(f"\n✓ Long-format comparison table: {long_path}")

    # ── Wide-format table: one row per checkpoint, one column per run ──
    wide_df = pd.DataFrame({name: t for name, t in trends.items()})
    wide_df.index.name = 'checkpoint_name'
    wide_df['tokens'] = [checkpoint_tokens(c) for c in wide_df.index]
    wide_df = wide_df.sort_values('tokens').reset_index()
    wide_path = os.path.join(results_dir, 'grad_erank_approx_wide.csv')
    wide_df.to_csv(wide_path, index=False)
    print(f"✓ Wide-format table: {wide_path}")

    # ── Console summary: ratio at final common checkpoint per run ──
    print(f"\nFinal-checkpoint ratio ({args.baseline} / run), weight_filter="
         f"'{args.weight_filter}':")
    for name in trends:
        if name == args.baseline:
            continue
        sub = long_df[long_df['run'] == name]
        if sub.empty:
            print(f"  {name}: no overlapping checkpoints with baseline")
            continue
        last = sub.sort_values('tokens').iloc[-1]
        print(f"  {name:<20} value={last['value']:.2f}  "
             f"{args.baseline}={last['baseline_value']:.2f}  "
             f"ratio={last['ratio_baseline_over_run']:.2f}x")

    # ── Figure: all runs on one plot ──
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, t in trends.items():
        style = '-' if name == args.baseline else '--'
        ax.plot([checkpoint_tokens(c) for c in t.index], t.values,
               style, marker='o', markersize=3, linewidth=1.6, label=name)
    ax.set_xlabel('Tokens seen', fontsize=10)
    ax.set_ylabel(f'Mean {args.metric} ({args.weight_filter} layers)', fontsize=10)
    ax.set_title(f'Weight-diff gradient eRank comparison\n'
                f'baseline={args.baseline}', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'fig_grad_erank_approx_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison figure: {fig_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compare grad_erank_approx (weight-diff proxy) across an '
                   'arbitrary named set of runs.')
    p.add_argument('--config', default=None,
                  help='YAML config (used only for results_dir default)')
    p.add_argument('--results_dir', default=None,
                  help='Override output directory (default: config results_dir '
                      'or "results")')
    p.add_argument('--run', action='append', required=True,
                  help="NAME=PATH, repeatable. e.g. --run FR=outputs/.../FR "
                      "--run LR_r8=outputs/.../LR_r8")
    p.add_argument('--baseline', required=True,
                  help='Which --run NAME to use as the reference for ratios '
                      '(e.g. FR)')
    p.add_argument('--weight_filter', default='attn',
                  help="Substring filter on weight_name (default 'attn', "
                      "matches the paper's own aim3_grad_erank convention — "
                      "includes both c_attn and c_proj). Use 'attn.c_attn' "
                      "for c_attn only.")
    p.add_argument('--metric', default='grad_erank_approx',
                  help='Column to average (default: grad_erank_approx)')
    main(p.parse_args())