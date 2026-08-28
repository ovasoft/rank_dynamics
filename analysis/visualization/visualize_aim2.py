"""
visualize_aim2.py — Generate 4 separate PNG figures for Aim 2 (Stabilisation).

Outputs:
  figures/aim2a_stab_checkpoint.png  — Stabilisation point vs rank, both domains
  figures/aim2b_mechanism.png        — Exhaustion vs convergence by run and domain
  figures/aim2c_init_drift.png       — CKA-vs-init at stabilisation point
  figures/aim2d_layer_profile.png    — Per-layer mechanism and stab checkpoint (vision)

Usage:
    python visualize_aim2.py
    python visualize_aim2.py \
        --vision results/cifar10/aim2_stabilisation_cifar10.csv \
        --language results/babylm_strict_small/aim2_stabilisation_babylm_strict_small.csv \
        --out figures/
"""

import argparse
import csv
import os
import collections
import statistics

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          10,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linewidth':     0.5,
    'figure.dpi':         150,
})

VISION_COLOR  = '#378ADD'
LANG_COLOR    = '#1D9E75'
EXHAUS_COLOR  = '#A32D2D'
CONV_COLOR    = '#378ADD'
NEVER_COLOR   = '#888780'
FR_MARKER     = 'D'

RANK_COLORS = {
    'FR':  '#042C53',
    '8':   '#A32D2D',
    '16':  '#D4537E',
    '32':  '#BA7517',
    '64':  '#639922',
    '128': '#378ADD',
    '256': '#1D9E75',
}

DEFAULT_VISION   = 'results/cifar10/aim2_stabilisation_cifar10.csv'
DEFAULT_LANGUAGE = 'results/babylm_strict_small/aim2_stabilisation_babylm_strict_small.csv'
DEFAULT_OUT      = 'figures'


# ── Data loading ──────────────────────────────────────────────────────────

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'domain':          r['domain'],
                'run':             r['run'],
                'rank':            r['rank'],
                'layer':           int(r['layer']),
                'stab_checkpoint': float(r['stab_checkpoint']) if r['stab_checkpoint'] else None,
                'cka_vs_init_at_stab': float(r['cka_vs_init_at_stab']) if r['cka_vs_init_at_stab'] else None,
                'final_cka_vs_prev':   float(r['final_cka_vs_prev'])   if r['final_cka_vs_prev']   else None,
                'final_cka_vs_init':   float(r['final_cka_vs_init'])   if r['final_cka_vs_init']   else None,
                'mechanism':       r.get('mechanism', ''),
            })
    return rows


def by_run(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r['run']].append(r)
    return d


def run_summary(run_rows):
    stabs   = [r['stab_checkpoint']      for r in run_rows if r['stab_checkpoint'] is not None]
    inits   = [r['cka_vs_init_at_stab']  for r in run_rows if r['cka_vs_init_at_stab'] is not None]
    finits  = [r['final_cka_vs_init']    for r in run_rows if r['final_cka_vs_init'] is not None]
    mechs   = [r['mechanism']            for r in run_rows if r['mechanism']]
    return {
        'mean_stab':     statistics.mean(stabs)  if stabs  else None,
        'mean_init':     statistics.mean(inits)  if inits  else None,
        'mean_finit':    statistics.mean(finits) if finits else None,
        'exhaustion_pct': 100 * mechs.count('exhaustion')  / len(mechs) if mechs else 0,
        'convergence_pct':100 * mechs.count('convergence') / len(mechs) if mechs else 0,
        'never_pct':      100 * mechs.count('never_stable')/ len(mechs) if mechs else 0,
        'rank':          run_rows[0]['rank'],
        'run':           run_rows[0]['run'],
    }


def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'  Saved: {path}')
    plt.close(fig)


def rank_sort_key(s):
    return 9999 if s == 'FR' else int(s)


# ── Panel A — Stabilisation checkpoint vs rank ───────────────────────────

def plot_a(v_rows, l_rows, out_dir):
    """
    O2.1 / O2.3: When does each run stabilise, and how does it scale with rank?
    Higher rank stabilises earlier in both domains.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, rows, domain_label, color, ck_label_fn in [
        (axes[0], v_rows, 'Vision (CIFAR-10)',   VISION_COLOR, lambda x: f'ep{int(x)}'),
        (axes[1], l_rows, 'Language (BabyLM)',   LANG_COLOR,
            lambda x: f'{int(x)//1_000_000}M' if x >= 1e6 else f'{int(x)//1000}K'),
    ]:
        summaries = {run: run_summary(rrows)
                     for run, rrows in by_run(rows).items()}
        runs_sorted = sorted(summaries.keys(),
                             key=lambda r: rank_sort_key(summaries[r]['rank']))

        x_pos  = range(len(runs_sorted))
        stabs  = [summaries[r]['mean_stab'] for r in runs_sorted]
        ranks  = [summaries[r]['rank'] for r in runs_sorted]
        colors = [RANK_COLORS.get(rk, '#888') for rk in ranks]

        bars = ax.bar(x_pos, stabs, color=colors, alpha=0.85, width=0.6)

        # Value labels on bars
        for bar, val in zip(bars, stabs):
            if val:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        ck_label_fn(val),
                        ha='center', va='bottom', fontsize=7)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(runs_sorted, rotation=30, ha='right')
        ax.set_ylabel('stabilisation checkpoint')
        ax.set_title(f'{domain_label}\nstabilisation point by run')

        # FR reference line
        if 'FR' in summaries and summaries['FR']['mean_stab']:
            ax.axhline(summaries['FR']['mean_stab'],
                       color=RANK_COLORS['FR'], lw=1.2, ls='--', alpha=0.7,
                       label=f'FR ({ck_label_fn(summaries["FR"]["mean_stab"])})')
            ax.legend(fontsize=7)

    fig.suptitle('Aim 2 — O2.1/O2.3: When do models stabilise?\n'
                 'Higher rank → earlier stabilisation in both domains',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim2a_stab_checkpoint.png')


# ── Panel B — Mechanism (exhaustion vs convergence) ───────────────────────

def plot_b(v_rows, l_rows, out_dir):
    """
    O2.2: Is LR stabilisation exhaustion or convergence?
    Stacked bars: exhaustion / convergence / never_stable per run.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, rows, domain_label in [
        (axes[0], v_rows, 'Vision (CIFAR-10)'),
        (axes[1], l_rows, 'Language (BabyLM)'),
    ]:
        summaries = {run: run_summary(rrows)
                     for run, rrows in by_run(rows).items()}
        runs_sorted = sorted(summaries.keys(),
                             key=lambda r: rank_sort_key(summaries[r]['rank']))

        x_pos = np.arange(len(runs_sorted))
        exh   = [summaries[r]['exhaustion_pct']  for r in runs_sorted]
        conv  = [summaries[r]['convergence_pct'] for r in runs_sorted]
        never = [summaries[r]['never_pct']        for r in runs_sorted]

        ax.bar(x_pos, exh,   color=EXHAUS_COLOR, alpha=0.85, label='exhaustion',   width=0.6)
        ax.bar(x_pos, conv,  color=CONV_COLOR,   alpha=0.85, label='convergence',
               bottom=exh, width=0.6)
        bot2 = [e+c for e,c in zip(exh, conv)]
        ax.bar(x_pos, never, color=NEVER_COLOR,  alpha=0.6,  label='never stable',
               bottom=bot2, width=0.6)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(runs_sorted, rotation=30, ha='right')
        ax.set_ylabel('% of layers')
        ax.set_ylim(0, 115)
        ax.set_title(f'{domain_label}\nmechanism by run')
        ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('Aim 2 — O2.2: Exhaustion vs convergence\n'
                 'Vision LR_r128 = convergence; all others = exhaustion',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim2b_mechanism.png')


# ── Panel C — CKA-vs-init at stabilisation point ─────────────────────────

def plot_c(v_rows, l_rows, out_dir):
    """
    O2.2: How far have representations moved from init when they stabilise?
    Low value = exhaustion (moved far, capacity used up).
    High value = convergence (still close to init, found a good solution early).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, rows, domain_label, color in [
        (axes[0], v_rows, 'Vision (CIFAR-10)', VISION_COLOR),
        (axes[1], l_rows, 'Language (BabyLM)', LANG_COLOR),
    ]:
        summaries = {run: run_summary(rrows)
                     for run, rrows in by_run(rows).items()}
        runs_sorted = sorted(summaries.keys(),
                             key=lambda r: rank_sort_key(summaries[r]['rank']))

        x_pos     = np.arange(len(runs_sorted))
        init_vals = [summaries[r]['mean_init']  for r in runs_sorted]
        finit_vals= [summaries[r]['mean_finit'] for r in runs_sorted]
        colors    = [RANK_COLORS.get(summaries[r]['rank'], '#888') for r in runs_sorted]

        # Bars for init_at_stab
        ax.bar(x_pos - 0.18, init_vals,  0.35, color=colors, alpha=0.85,
               label='at stabilisation')
        # Dots for final
        ax.scatter(x_pos + 0.18, finit_vals, s=40, color=colors,
                   marker='o', zorder=5, label='final checkpoint')
        # Line connecting the two for each run
        for xi, iv, fv in zip(x_pos, init_vals, finit_vals):
            if iv and fv:
                ax.plot([xi - 0.18, xi + 0.18], [iv, fv],
                        color='#aaa', lw=0.8, zorder=3)

        ax.axhline(0.65, color='#888', lw=0.9, ls=':',
                   label='exhaustion threshold (0.65)')
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(runs_sorted, rotation=30, ha='right')
        ax.set_ylabel('CKA vs initialisation')
        ax.set_ylim(0, 0.85)
        ax.set_title(f'{domain_label}\ndrift from init at stabilisation point')
        ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle('Aim 2 — O2.2: How far from init when stabilised?\n'
                 'Below dashed line = exhaustion; above = convergence',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim2c_init_drift.png')


# ── Panel D — Per-layer mechanism profile (vision) ───────────────────────

def plot_d(v_rows, out_dir):
    """
    O2.4: Does the mechanism vary by layer?
    Vision FR: shallow layers converge, deep layers exhaust.
    Vision LR_r128: all layers converge (enough capacity).
    Vision LR_r8: all layers exhaust (insufficient capacity).
    """
    runs_to_show = ['FR', 'LR_r128', 'LR_r64', 'LR_r32', 'LR_r16', 'LR_r8']
    run_data = by_run(v_rows)

    # Filter to runs we have
    runs_to_show = [r for r in runs_to_show if r in run_data]
    n_runs = len(runs_to_show)
    n_layers = max(r['layer'] for r in v_rows) + 1

    fig, axes = plt.subplots(1, n_runs, figsize=(2.2 * n_runs, 4), sharey=True)
    if n_runs == 1:
        axes = [axes]

    mech_color = {'exhaustion': EXHAUS_COLOR, 'convergence': CONV_COLOR,
                  'never_stable': NEVER_COLOR, '': '#ccc'}

    for ax, run in zip(axes, runs_to_show):
        rows_r = sorted(run_data[run], key=lambda x: x['layer'])
        rank   = rows_r[0]['rank']

        for r in rows_r:
            layer  = r['layer']
            mech   = r['mechanism']
            stab   = r['stab_checkpoint']
            col    = mech_color.get(mech, '#ccc')
            # Bar length = stab checkpoint (normalised)
            ax.barh(layer, stab or 0, color=col, alpha=0.75, height=0.7)

        ax.set_title(f'{run}\n(r={rank})', fontsize=8)
        ax.set_xlabel('stab epoch', fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel('layer')
        ax.set_yticks(range(n_layers))

    # Legend
    patches = [
        mpatches.Patch(color=EXHAUS_COLOR, alpha=0.75, label='exhaustion'),
        mpatches.Patch(color=CONV_COLOR,   alpha=0.75, label='convergence'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.suptitle('Aim 2 — O2.4: Per-layer stabilisation mechanism (vision)\n'
                 'Bar length = stabilisation epoch; colour = mechanism',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim2d_layer_profile.png')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vision',   default=DEFAULT_VISION)
    p.add_argument('--language', default=DEFAULT_LANGUAGE)
    p.add_argument('--out',      default=DEFAULT_OUT)
    args = p.parse_args()

    for path in [args.vision, args.language]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'CSV not found: {path}\n'
                'Run collate_results.py for each domain first.')

    v_rows = load(args.vision)
    l_rows = load(args.language)
    print(f'Loaded {len(v_rows)} vision rows, {len(l_rows)} language rows')
    print(f'Output dir: {args.out}\n')

    plot_a(v_rows, l_rows, args.out)
    plot_b(v_rows, l_rows, args.out)
    plot_c(v_rows, l_rows, args.out)
    plot_d(v_rows,         args.out)

    print('\nDone. 4 figures written.')

if __name__ == '__main__':
    main()