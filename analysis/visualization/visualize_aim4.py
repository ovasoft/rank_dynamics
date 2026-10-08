"""
visualize_aim4.py — 4 separate PNG figures for Aim 4 (Spectral correction).

Outputs:
  figures/aim4a_intervention_cka.png     — CKA at t* by intervention variant
  figures/aim4b_intervention_metric.png  — Task metric at t* by intervention variant
  figures/aim4c_dsn_trajectory.png       — DSN CKA trajectory over training
  figures/aim4d_dsn_metric_vs_cka.png    — DSN: task metric vs final CKA (scatter)

Usage:
    python analysis/visualization/visualize_aim4.py
    python analysis/visualization/visualize_aim4.py \
        --vision   results/cifar10/aim4_dsn_cifar10.csv \
        --language results/babylm_strict_small/aim4_dsn_babylm_strict_small.csv \
        --out      figures/
"""

import argparse, csv, os
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
    'lines.linewidth':    1.6,
    'figure.dpi':         150,
})

VARIANT_COLORS = {
    'baseline':    '#888780',
    'drift_scaled':'#BA7517',
    'alpha_flat':  '#A32D2D',
    'combined':    '#D4537E',
    'static':      '#378ADD',
    'dynamic':     '#1D9E75',
    'flat':        '#BA7517',
}
VARIANT_LABELS = {
    'baseline':    'Baseline LR',
    'drift_scaled':'Drift scaled',
    'alpha_flat':  'Alpha flat\n(spectral)',
    'combined':    'Combined',
    'static':      'DSN static',
    'dynamic':     'DSN dynamic',
    'flat':        'DSN flat',
}
VISION_COLOR = '#378ADD'
LANG_COLOR   = '#1D9E75'
FR_COLOR     = '#042C53'

DEFAULT_VISION   = 'results/cifar10/aim4_dsn_cifar10.csv'
DEFAULT_LANGUAGE = 'results/babylm_strict_small/aim4_dsn_babylm_strict_small.csv'
DEFAULT_OUT      = 'figures'

# LR_r8 baseline reference values (from aim1/aim5 collation)
LR_BASELINE = {
    'cifar10':  {'cka': 0.1346, 'metric': 0.3298, 'metric_name': 'val_acc',
                 'higher_better': True,  'fr_metric': 0.7968},
    'babylm':   {'cka': 0.0336, 'metric': 456.60,  'metric_name': 'val_ppl',
                 'higher_better': False, 'fr_metric': 46.98},
}


def sf(x):
    try: return float(x)
    except: return None

def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'  Saved: {path}')
    plt.close(fig)

def fmt_token(t):
    t = int(float(t))
    if t >= 1_000_000: return f'{t//1_000_000}M'
    if t >= 1_000:     return f'{t//1_000}K'
    return str(t)

def intervention_rows(rows):
    return {r['variant']: r for r in rows if r.get('source') == 'intervention'}

def dsn_traj_rows(rows, mode):
    return sorted(
        [r for r in rows if r.get('source') == 'dsn_trajectory'
         and r.get('dsn_mode') == mode],
        key=lambda r: float(r['checkpoint'])
    )

def dsn_best(rows, mode):
    bests = [r for r in rows if r.get('source') == 'dsn_best_metric'
             and r.get('dsn_mode') == mode]
    return sf(bests[0]['val_metric']) if bests else None


# ── Panel A — Intervention CKA at t* ──────────────────────────────────────

def plot_a(v_rows, l_rows, out_dir):
    variants = ['baseline', 'drift_scaled', 'alpha_flat', 'combined']
    x     = np.arange(len(variants))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, rows, dom_key, title in [
        (axes[0], v_rows, 'cifar10',  'Vision (CIFAR-10)'),
        (axes[1], l_rows, 'babylm',   'Language (BabyLM)'),
    ]:
        interv = intervention_rows(rows)
        colors = [VARIANT_COLORS[v] for v in variants]

        mean_cka = [sf(interv.get(v, {}).get('mean_cka')) or 0 for v in variants]
        cka_l0   = [sf(interv.get(v, {}).get('cka_L0'))   or 0 for v in variants]

        ax.bar(x - width/2, mean_cka, width, color=colors, alpha=0.85,
               label='mean CKA (all layers)')
        ax.bar(x + width/2, cka_l0,   width, color=colors, alpha=0.5,
               hatch='//', label='CKA L0 (shallowest)')

        for i, (bar_x, val) in enumerate(zip(x - width/2, mean_cka)):
            ax.text(bar_x, val + 0.005, f'{val:.3f}', ha='center',
                    va='bottom', fontsize=7)

        # LR baseline reference
        lb_cka = LR_BASELINE[dom_key]['cka']
        ax.axhline(lb_cka, color='#555', lw=1, ls=':',
                   label=f'LR r=8 final CKA ({lb_cka:.3f})')

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], fontsize=8)
        ax.set_ylabel('CKA vs FR')
        ax.set_ylim(0, max(max(mean_cka), max(cka_l0)) * 1.3)
        ax.set_title(f'{title}\nCKA alignment at t* by intervention variant')
        ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle('Aim 4 — O4.1: Spectral correction at t* improves FR alignment\n'
                 'alpha_flat is most effective in both domains', fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim4a_intervention_cka.png')


# ── Panel B — Intervention task metric ────────────────────────────────────

def plot_b(v_rows, l_rows, out_dir):
    variants = ['baseline', 'drift_scaled', 'alpha_flat', 'combined']
    x   = np.arange(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, rows, dom_key, title in [
        (axes[0], v_rows, 'cifar10', 'Vision (CIFAR-10)'),
        (axes[1], l_rows, 'babylm',  'Language (BabyLM)'),
    ]:
        interv  = intervention_rows(rows)
        lb      = LR_BASELINE[dom_key]
        hb      = lb['higher_better']
        colors  = [VARIANT_COLORS[v] for v in variants]
        metrics = [sf(interv.get(v, {}).get('val_metric')) for v in variants]

        bars = ax.bar(x, metrics, color=colors, alpha=0.85, width=0.6)

        # Baseline and FR reference lines
        ax.axhline(lb['metric'], color='#555', lw=1, ls='--',
                   label=f'LR r=8 baseline ({lb["metric"]:.1f})')
        ax.axhline(lb['fr_metric'], color=FR_COLOR, lw=1.5, ls='-.',
                   label=f'FR ({lb["fr_metric"]:.1f})', alpha=0.7)

        for bar, val in zip(bars, metrics):
            if val:
                va = 'bottom' if hb else 'top'
                yd = val * 1.01 if hb else val * 0.99
                ax.text(bar.get_x() + bar.get_width()/2, yd,
                        f'{val:.1f}', ha='center', va=va, fontsize=7.5)

        # Y zoom
        valid = [m for m in metrics if m]
        all_ref = [lb['metric'], lb['fr_metric']] + valid
        pad = 0.12
        if hb:
            ax.set_ylim(min(all_ref) * (1-pad), max(all_ref) * (1+pad))
        else:
            ax.set_ylim(min(all_ref) * (1-pad), max(all_ref) * (1+pad))

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], fontsize=8)
        ax.set_ylabel(lb['metric_name'])
        ax.set_title(f'{title}\nTask metric at t* by intervention')
        ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle('Aim 4 — O4.1: Spectral correction improves task performance\n'
                 'Vision: +13.5% acc | Language: -61% perplexity (alpha_flat)',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim4b_intervention_metric.png')


# ── Panel C — DSN trajectory ───────────────────────────────────────────────

def plot_c(v_rows, l_rows, out_dir):
    """
    O4.2: Does continuous DSN sustain the early alignment gains?
    DSN improves task metric but CKA does not stay elevated vs LR baseline.
    """
    modes = ['static', 'dynamic', 'flat']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, rows, dom_key, title, ck_fn in [
        (axes[0], v_rows, 'cifar10', 'Vision (CIFAR-10)',
         lambda r: int(float(r['checkpoint']))),
        (axes[1], l_rows, 'babylm',  'Language (BabyLM)',
         lambda r: int(float(r['checkpoint']))),
    ]:
        lb_cka = LR_BASELINE[dom_key]['cka']

        # LR baseline reference (flat line)
        ax.axhline(lb_cka, color='#555', lw=1.2, ls='--',
                   label=f'LR r=8 final CKA ({lb_cka:.3f})', zorder=5)

        for mode in modes:
            traj = dsn_traj_rows(rows, mode)
            if not traj: continue
            cks  = [ck_fn(r) for r in traj]
            ckas = [sf(r['mean_cka']) for r in traj]
            x    = range(len(cks))

            # X labels
            if dom_key == 'cifar10':
                xlabs = [f'ep{c}' for c in cks]
            else:
                xlabs = [fmt_token(c) for c in cks]

            ax.plot(x, ckas, color=VARIANT_COLORS[mode],
                    label=VARIANT_LABELS[mode], marker='o', markersize=3)

        step = max(1, len(cks)//7) if traj else 1
        ax.set_xticks(range(0, len(cks), step))
        ax.set_xticklabels(xlabs[::step], rotation=25, ha='right')
        ax.set_ylabel('mean CKA vs FR')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{title}\nDSN CKA vs FR over training')
        ax.legend(framealpha=0.9)

    fig.suptitle('Aim 4 — O4.2: DSN does not sustain representational alignment\n'
                 'Task metric improves but CKA stays near or below LR baseline',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim4c_dsn_trajectory.png')


# ── Panel D — DSN: task metric vs final CKA (scatter) ────────────────────

def plot_d(v_rows, l_rows, out_dir):
    """
    The dissociation: DSN improves task metric WITHOUT improving CKA.
    Each point is a training configuration (LR baseline, DSN modes, FR).
    X = final CKA vs FR, Y = best task metric (normalised).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, rows, dom_key, title in [
        (axes[0], v_rows, 'cifar10', 'Vision (CIFAR-10)'),
        (axes[1], l_rows, 'babylm',  'Language (BabyLM)'),
    ]:
        lb  = LR_BASELINE[dom_key]
        hb  = lb['higher_better']

        # Collect points: (cka, metric, label, color)
        points = [
            (lb['cka'],       lb['metric'],       'LR r=8',   '#888780'),
            (1.0,             lb['fr_metric'],    'FR',        FR_COLOR),
        ]
        for mode in ['static', 'dynamic', 'flat']:
            traj = dsn_traj_rows(rows, mode)
            bm   = dsn_best(rows, mode)
            if traj and bm:
                final_cka = sf(traj[-1]['mean_cka'])
                points.append((final_cka, bm, VARIANT_LABELS[mode],
                               VARIANT_COLORS[mode]))

        for cka, metric, label, color in points:
            # Normalise metric: 0=LR_baseline, 1=FR
            lb_m, fr_m = lb['metric'], lb['fr_metric']
            if hb:
                norm = (metric - lb_m) / (fr_m - lb_m)
            else:
                norm = (lb_m - metric) / (lb_m - fr_m)
            marker = 'D' if label == 'FR' else ('s' if label == 'LR r=8' else 'o')
            size   = 120 if label in ('FR', 'LR r=8') else 80
            ax.scatter(cka, norm, color=color, s=size, marker=marker,
                       zorder=5, edgecolors='white', linewidths=0.5)
            ax.annotate(label, xy=(cka, norm),
                        xytext=(cka + 0.005, norm + 0.03),
                        fontsize=7, color=color)

        ax.axhline(0, color='#555', lw=0.8, ls=':', alpha=0.5)
        ax.axhline(1, color=FR_COLOR, lw=0.8, ls=':', alpha=0.5)
        ax.set_xlabel('final CKA vs FR')
        ax.set_ylabel('task metric\n(0=LR baseline, 1=FR)')
        ax.set_ylim(-0.2, 1.3)
        ax.set_title(f'{title}\ntask metric vs representational alignment')

    fig.suptitle('Aim 4 — Dissociation: DSN improves task metric WITHOUT improving CKA\n'
                 'Better performance ≠ more FR-like representations',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim4d_dsn_metric_vs_cka.png')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vision',   default=DEFAULT_VISION)
    p.add_argument('--language', default=DEFAULT_LANGUAGE)
    p.add_argument('--out',      default=DEFAULT_OUT)
    args = p.parse_args()

    v_rows = load(args.vision)
    l_rows = load(args.language)
    print(f'Loaded {len(v_rows)} vision rows, {len(l_rows)} language rows')

    # Fallback to hardcoded if CSVs missing
    if not v_rows:
        v_rows = [
            {'source':'intervention','variant':'baseline',    'val_metric':'0.2229','mean_cka':'0.3178','cka_L0':'0.2976'},
            {'source':'intervention','variant':'drift_scaled','val_metric':'0.2336','mean_cka':'0.3619','cka_L0':'0.3108'},
            {'source':'intervention','variant':'alpha_flat',  'val_metric':'0.2529','mean_cka':'0.5157','cka_L0':'0.4961'},
            {'source':'intervention','variant':'combined',    'val_metric':'0.2488','mean_cka':'0.4698','cka_L0':'0.5434'},
        ]
    if not l_rows:
        l_rows = [
            {'source':'intervention','variant':'baseline',    'val_metric':'489.71','mean_cka':'0.0815','cka_L0':'0.0834'},
            {'source':'intervention','variant':'drift_scaled','val_metric':'478.48','mean_cka':'0.1040','cka_L0':'0.0891'},
            {'source':'intervention','variant':'alpha_flat',  'val_metric':'188.24','mean_cka':'0.3183','cka_L0':'0.2373'},
            {'source':'intervention','variant':'combined',    'val_metric':'268.16','mean_cka':'0.1975','cka_L0':'0.1522'},
        ]

    print(f'Output dir: {args.out}\n')
    plot_a(v_rows, l_rows, args.out)
    plot_b(v_rows, l_rows, args.out)
    plot_c(v_rows, l_rows, args.out)
    plot_d(v_rows, l_rows, args.out)
    print('\nDone. 4 figures written.')

if __name__ == '__main__':
    main()