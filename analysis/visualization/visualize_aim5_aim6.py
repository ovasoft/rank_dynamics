"""
visualize_aim5_aim6.py — 4 separate PNG figures for Aims 5 and 6.

Outputs:
  figures/aim5a_gap_vs_rank.png          — Normalised performance gap vs rank, both domains
  figures/aim5b_absolute_performance.png — Raw task metric vs rank, both domains
  figures/aim6a_k_sweep.png              — Gap closed by k-sweep (layer reinit), both domains
  figures/aim6b_residual_gap.png         — Residual gap after fine-tuning, both domains

Usage:
    python analysis/visualization/visualize_aim5_aim6.py
    python analysis/visualization/visualize_aim5_aim6.py \
        --aim5  results/cross_domain/aim5_performance_gap_combined.csv \
        --aim6v results/cifar10/aim6_recovery_cifar10.csv \
        --aim6l results/babylm_strict_small/aim6_recovery_babylm_strict_small.csv \
        --out   figures/
"""

import argparse, csv, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

VISION_COLOR = '#378ADD'
LANG_COLOR   = '#1D9E75'
FR_COLOR     = '#042C53'
SHORT_COLOR  = '#BA7517'
CONV_COLOR   = '#A32D2D'

DEFAULT_AIM5  = 'results/cross_domain/aim5_performance_gap_combined.csv'
DEFAULT_AIM6V = 'results/cifar10/aim6_recovery_cifar10.csv'
DEFAULT_AIM6L = 'results/babylm_strict_small/aim6_recovery_babylm_strict_small.csv'
DEFAULT_OUT   = 'figures'


def sf(x):
    try: return float(x)
    except: return None

def load(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'  Saved: {path}')
    plt.close(fig)


# ── Panel 5A — Normalised gap vs rank ────────────────────────────────────

def plot_5a(aim5_rows, out_dir):
    """
    O5.2/5.3/5.4: Normalised performance gap vs rank in both domains.
    Vision: non-monotonic (r128 anomaly). Language: smooth monotonic.
    Both reach comparable normalised gap at very different rank values.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for domain, color, label, marker in [
        ('cifar10',             VISION_COLOR, 'Vision',   'o'),
        ('babylm_strict_small', LANG_COLOR,   'Language', 's'),
    ]:
        rows = sorted(
            [r for r in aim5_rows if r['domain'] == domain],
            key=lambda r: int(r['rank'])
        )
        if not rows: continue
        ranks = [int(r['rank'])      for r in rows]
        gaps  = [sf(r['normalised_gap']) for r in rows]

        ax.plot(ranks, gaps, color=color, marker=marker, markersize=7,
                label=label, lw=1.8)

        # Annotate each point
        for rk, gap in zip(ranks, gaps):
            ax.annotate(f'r={rk}', xy=(rk, gap),
                        xytext=(rk, gap + 0.15),
                        fontsize=6.5, ha='center', color=color)

    ax.axhline(0, color='#aaa', lw=0.8, ls=':')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('rank (r)  [log₂ scale]')
    ax.set_ylabel('normalised performance gap\n(FR_metric - LR_metric) / FR_metric')
    ax.set_title('Aim 5 — O5.2/5.3/5.4: Performance gap vs rank\n'
                 'Language gap is larger and decays more smoothly than vision')
    ax.legend(framealpha=0.9)

    # Annotate vision anomaly
    ax.annotate('r=128 anomaly\n(worse than r=64)',
                xy=(128, 0.310), xytext=(90, 0.45),
                fontsize=7, color=VISION_COLOR,
                arrowprops=dict(arrowstyle='->', color=VISION_COLOR, lw=0.7))

    fig.tight_layout()
    save_fig(fig, out_dir, 'aim5a_gap_vs_rank.png')


# ── Panel 5B — Absolute task metric vs rank ───────────────────────────────

def plot_5b(aim5_rows, out_dir):
    """
    O5.1: Raw task metric vs rank — shows the practical performance cost.
    Two y-axes: val_acc (vision, left) and val_ppl (language, right, inverted).
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    for domain, color, label, marker, ax, ylab in [
        ('cifar10',             VISION_COLOR, 'Vision (val_acc)',   'o', ax1,
         'val_acc (higher = better)'),
        ('babylm_strict_small', LANG_COLOR,   'Language (val_ppl)', 's', ax2,
         'val_ppl (lower = better, log scale)'),
    ]:
        rows = sorted(
            [r for r in aim5_rows if r['domain'] == domain],
            key=lambda r: int(r['rank'])
        )
        if not rows: continue
        ranks   = [int(r['rank'])    for r in rows]
        metrics = [sf(r['lr_best'])  for r in rows]
        fr_val  = sf(rows[0]['fr_best'])

        ax.plot(ranks, metrics, color=color, marker=marker, markersize=7,
                label=label, lw=1.8)
        ax.axhline(fr_val, color=color, lw=1.2, ls='--', alpha=0.6,
                   label=f'FR ({fr_val:.2f})')
        ax.set_ylabel(ylab, color=color)
        ax.tick_params(axis='y', labelcolor=color)

    ax2.set_yscale('log')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('rank (r)  [log₂ scale]')
    ax1.set_title('Aim 5 — O5.1: Task metric vs rank\n'
                  'Both domains: performance improves with rank but gap persists')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, out_dir, 'aim5b_absolute_performance.png')


# ── Panel 6A — k-sweep gap closed ────────────────────────────────────────

def plot_6a(v6_rows, l6_rows, out_dir):
    """
    O6.2: How much of the gap is closed by reinitialising the first k blocks?
    Shows the non-monotonic vision pattern vs smooth language recovery.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, rows, domain_label, color in [
        (axes[0], v6_rows, 'Vision (CIFAR-10)', VISION_COLOR),
        (axes[1], l6_rows, 'Language (BabyLM)', LANG_COLOR),
    ]:
        short = sorted(
            [r for r in rows if r.get('budget') == 'short'
             and r.get('condition') not in ('lr_baseline',)
             and r.get('k_reinit', '') != ''],
            key=lambda r: (r['k_reinit'] == 'all',
                           int(r['k_reinit']) if r['k_reinit'] != 'all' else 9999)
        )
        if not short: continue

        labels    = [r['condition'] for r in short]
        gap_closed = [sf(r['gap_closed']) * 100 for r in short]
        x = np.arange(len(labels))

        bars = ax.bar(x, gap_closed, color=color, alpha=0.85, width=0.6)

        # Full-finetune reference (short budget)
        ff_short = next((sf(r['gap_closed'])*100 for r in rows
                         if r.get('budget')=='short'
                         and r.get('condition')=='full_finetune'), None)
        ff_conv  = next((sf(r['gap_closed'])*100 for r in rows
                         if r.get('budget')=='convergence'
                         and r.get('condition')=='full_finetune'), None)
        if ff_short:
            ax.axhline(ff_short, color='#555', lw=1, ls='--',
                       label=f'full fine-tune short ({ff_short:.1f}%)')
        if ff_conv:
            ax.axhline(ff_conv, color='#333', lw=1, ls='-.',
                       label=f'full fine-tune conv ({ff_conv:.1f}%)')

        # Value labels
        for bar, val in zip(bars, gap_closed):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('gap closed (%)')
        ax.set_ylim(0, max(gap_closed) * 1.25)
        ax.set_title(f'{domain_label}\ngap closed by layer reinitialisation (k-sweep)')
        ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle('Aim 6 — O6.2: Which layers need reinitialising to recover performance?\n'
                 'Vision: non-monotonic (k=9 jump). Language: smooth monotonic increase.',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim6a_k_sweep.png')


# ── Panel 6B — Residual gap ───────────────────────────────────────────────

def plot_6b(v6_rows, l6_rows, out_dir):
    """
    O6.3: What residual gap persists after best-case fine-tuning?
    Shows that FR and LR end up in different basins.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, rows, domain_label, color, hb in [
        (axes[0], v6_rows, 'Vision (CIFAR-10)', VISION_COLOR, True),
        (axes[1], l6_rows, 'Language (BabyLM)', LANG_COLOR,   False),
    ]:
        fr  = sf(rows[0]['fr_ceiling'])   if rows else None
        lb  = sf(rows[0]['lr_baseline'])  if rows else None
        if not fr or not lb: continue

        # Key conditions to show
        conditions = []
        for cond_name, budget in [
            ('LR baseline',     'short'),
            ('k=2 (short)',     'short'),
            ('k=4 (short)',     'short'),
            ('k=all (short)',   'short'),
            ('k=all (conv)',    'convergence'),
        ]:
            cond_key = {
                'LR baseline':   'lr_baseline',
                'k=2 (short)':   'k2',
                'k=4 (short)':   'k4',
                'k=all (short)': 'full_finetune',
                'k=all (conv)':  'full_finetune',
            }[cond_name]
            match = next((sf(r['metric']) for r in rows
                          if r.get('budget') == budget
                          and r.get('condition') == cond_key), None)
            if match is not None:
                conditions.append((cond_name, match))

        # Add FR ceiling
        conditions.append(('FR ceiling', fr))

        x      = np.arange(len(conditions))
        labels = [c[0] for c in conditions]
        vals   = [c[1] for c in conditions]
        colors = []
        for lab in labels:
            if 'FR' in lab:        colors.append(FR_COLOR)
            elif 'baseline' in lab:colors.append('#888780')
            elif 'conv' in lab:    colors.append(CONV_COLOR)
            else:                  colors.append(color)

        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)

        # FR and LR reference lines
        ax.axhline(fr, color=FR_COLOR, lw=1.5, ls='-.', alpha=0.7,
                   label=f'FR ({fr:.1f})')
        ax.axhline(lb, color='#888', lw=1, ls='--',
                   label=f'LR baseline ({lb:.1f})')

        for bar, val in zip(bars, vals):
            va = 'bottom' if hb else 'top'
            yd = val * (1.01 if hb else 0.99)
            ax.text(bar.get_x() + bar.get_width()/2, yd,
                    f'{val:.2f}', ha='center', va=va, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right')
        ax.set_ylabel('val_acc' if hb else 'val_ppl')
        if not hb: ax.set_yscale('log')
        ax.set_title(f'{domain_label}\nresidual gap after fine-tuning')
        ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle('Aim 6 — O6.3: Substantial residual gap persists in both domains\n'
                 'FR and LR pretraining lead to different parameter space basins',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim6b_residual_gap.png')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aim5',  default=DEFAULT_AIM5)
    p.add_argument('--aim6v', default=DEFAULT_AIM6V)
    p.add_argument('--aim6l', default=DEFAULT_AIM6L)
    p.add_argument('--out',   default=DEFAULT_OUT)
    args = p.parse_args()

    aim5_rows = load(args.aim5)
    v6_rows   = load(args.aim6v)
    l6_rows   = load(args.aim6l)
    print(f'Aim5: {len(aim5_rows)} rows | Aim6 vision: {len(v6_rows)} | '
          f'Aim6 language: {len(l6_rows)}')
    print(f'Output dir: {args.out}\n')

    plot_5a(aim5_rows, args.out)
    plot_5b(aim5_rows, args.out)
    plot_6a(v6_rows, l6_rows, args.out)
    plot_6b(v6_rows, l6_rows, args.out)

    print('\nDone. 4 figures written.')

if __name__ == '__main__':
    main()