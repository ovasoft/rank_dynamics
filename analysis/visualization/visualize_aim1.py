"""
visualize_aim1.py — Generate 5 separate PNG figures for Aim 1.

Outputs:
  figures/aim1a_vision_trajectories.png
  figures/aim1b_language_trajectories.png
  figures/aim1c_normalised_init.png
  figures/aim1d_normalised_final.png
  figures/aim1e_frfr_trend.png

Usage:
    python analysis/visualization/visualize_aim1.py
    python analysis/visualization/visualize_aim1.py --csv results/cross_domain/aim1_divergence_combined.csv
    python analysis/visualization/visualize_aim1.py --out figures/
"""

import argparse
import csv
import os

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

# ── Palette ───────────────────────────────────────────────────────────────

RANK_COLORS = {
    8:   '#A32D2D',
    16:  '#D4537E',
    32:  '#BA7517',
    64:  '#639922',
    128: '#378ADD',
    256: '#1D9E75',
}
FF_COLOR     = '#042C53'
VISION_COLOR = '#378ADD'
LANG_COLOR   = '#1D9E75'

SHARED_RANKS = [8, 16, 32, 64, 128]


# ── Data helpers ──────────────────────────────────────────────────────────

def load(csv_path):
    data = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ck  = int(float(r['checkpoint']))
            key = (r['domain'], r['comparison'], r['rank'], ck)
            data[key] = float(r['mean_cka']) if r['mean_cka'] else None
    return data


def traj(data, domain, comparison, rank_str):
    pts = [(ck, v)
           for (dom, comp, rk, ck), v in data.items()
           if dom == domain and comp == comparison
           and rk == rank_str and v is not None]
    return sorted(pts)


def fmt_token(t):
    if t >= 1_000_000: return f'{t // 1_000_000}M'
    if t >= 1_000:     return f'{t // 1_000}K'
    return str(t)


def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'  Saved: {path}')
    plt.close(fig)


# ── Panel A — Vision trajectories ────────────────────────────────────────

def plot_a(data, out_dir):
    domain = 'cifar10'
    ranks  = [128, 64, 32, 16, 8]
    fig, ax = plt.subplots(figsize=(6, 4))

    ff = traj(data, domain, 'FR_vs_FR', 'FR')
    if ff:
        xs, ys = zip(*ff)
        pos    = range(len(xs))
        ax.plot(pos, ys, color=FF_COLOR, lw=2, ls='--',
                label='FR vs FR (baseline)', zorder=5)
        step = max(1, len(xs) // 8)
        ax.set_xticks(range(0, len(xs), step))
        ax.set_xticklabels([f'ep{x}' for x in xs][::step])

    for rank in ranks:
        t = traj(data, domain, f'FR_vs_LR_r{rank}', str(rank))
        if not t: continue
        _, ys_r = zip(*t)
        ax.plot(range(len(ys_r)), ys_r,
                color=RANK_COLORS[rank], label=f'r={rank}')

    ax.axvline(0, color='#aaa', lw=0.8, ls=':')
    ax.text(0.3, 0.02, 't*=0: split pre-exists training',
            fontsize=7, color='#666', transform=ax.get_xaxis_transform())
    ax.set_xlabel('epoch')
    ax.set_ylabel('mean CKA')
    ax.set_ylim(0, 0.95)
    ax.set_title('Vision — FR vs LR alignment by rank\n'
                 'All ranks split from FR_vs_FR at epoch 0')
    ax.legend(loc='center right', framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim1a_vision_trajectories.png')


# ── Panel B — Language trajectories ──────────────────────────────────────

def plot_b(data, out_dir):
    domain = 'babylm_strict_small'
    ranks  = [256, 128, 64, 32, 16, 8]
    fig, ax = plt.subplots(figsize=(6, 4))

    ff = traj(data, domain, 'FR_vs_FR', 'FR')
    if ff:
        xs, ys = zip(*ff)
        pos    = range(len(xs))
        ax.plot(pos, ys, color=FF_COLOR, lw=2, ls='--',
                label='FR vs FR (baseline)', zorder=5)
        step = max(1, len(xs) // 8)
        ax.set_xticks(range(0, len(xs), step))
        ax.set_xticklabels([fmt_token(x) for x in xs][::step],
                           rotation=25, ha='right')

    for rank in ranks:
        t = traj(data, domain, f'FR_vs_LR_r{rank}', str(rank))
        if not t: continue
        _, ys_r = zip(*t)
        ls = '-.' if rank == 256 else '-'
        ax.plot(range(len(ys_r)), ys_r,
                color=RANK_COLORS[rank], ls=ls, label=f'r={rank}')

    if ff:
        ax.annotate('r=256 ≥ FR_vs_FR\nat 50M tokens',
                    xy=(len(xs) - 1, 0.096),
                    xytext=(len(xs) - 4, 0.145),
                    fontsize=7, color=RANK_COLORS[256],
                    arrowprops=dict(arrowstyle='->', color=RANK_COLORS[256], lw=0.8))

    ax.set_xlabel('tokens seen')
    ax.set_ylabel('mean CKA')
    ax.set_ylim(0, 0.20)
    ax.set_title('Language — FR vs LR alignment by rank\n'
                 'r=256 converges to FR_vs_FR by 50M tokens')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim1b_language_trajectories.png')


# ── Panel C — Normalised divergence at t=0 ───────────────────────────────

def plot_c(data, out_dir):
    domains = [
        ('cifar10',             'Vision',   VISION_COLOR),
        ('babylm_strict_small', 'Language', LANG_COLOR),
    ]
    x     = np.arange(len(SHARED_RANKS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, (domain, label, color) in enumerate(domains):
        ff = traj(data, domain, 'FR_vs_FR', 'FR')
        if not ff: continue
        ff0, ck0 = ff[0][1], ff[0][0]
        vals = []
        for rank in SHARED_RANKS:
            t = traj(data, domain, f'FR_vs_LR_r{rank}', str(rank))
            if not t:
                vals.append(0); continue
            ck_map = dict(t)
            raw    = ck_map.get(ck0) or min(t, key=lambda p: abs(p[0]-ck0))[1]
            vals.append(raw / ff0 if ff0 else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.axhline(1.0, color='#888', lw=0.9, ls=':', label='FR_vs_FR = 1.0')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in SHARED_RANKS])
    ax.set_ylabel('FR_vs_LR / FR_vs_FR')
    ax.set_ylim(0, 1.15)
    ax.set_title('Normalised divergence at t=0\n'
                 'Rank ordering consistent across domains')
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim1c_normalised_init.png')


# ── Panel D — Normalised divergence at final checkpoint ──────────────────

def plot_d(data, out_dir):
    domains = [
        ('cifar10',             'Vision',   VISION_COLOR),
        ('babylm_strict_small', 'Language', LANG_COLOR),
    ]
    all_ranks = SHARED_RANKS
    x     = np.arange(len(all_ranks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for i, (domain, label, color) in enumerate(domains):
        ff = traj(data, domain, 'FR_vs_FR', 'FR')
        if not ff: continue
        ffl, ckl = ff[-1][1], ff[-1][0]
        vals = []
        for rank in all_ranks:
            t = traj(data, domain, f'FR_vs_LR_r{rank}', str(rank))
            if not t:
                vals.append(0); continue
            ck_map = dict(t)
            raw    = ck_map.get(ckl) or min(t, key=lambda p: abs(p[0]-ckl))[1]
            vals.append(raw / ffl if ffl else 0)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    # r=256 language-only bar
    ff_l = traj(data, 'babylm_strict_small', 'FR_vs_FR', 'FR')
    if ff_l:
        ffl, ckl = ff_l[-1][1], ff_l[-1][0]
        t256 = traj(data, 'babylm_strict_small', 'FR_vs_LR_r256', '256')
        if t256:
            raw256 = dict(t256).get(ckl) or min(t256, key=lambda p: abs(p[0]-ckl))[1]
            n256   = raw256 / ffl
            bx     = len(all_ranks) + 0.2
            ax.bar(bx, n256, width, color=LANG_COLOR, alpha=0.85, hatch='//',
                   label='r=256 (lang)')
            ax.annotate(f'{n256:.2f} ≥ 1.0',
                        xy=(bx, n256), xytext=(bx - 0.3, n256 + 0.07),
                        fontsize=7, color=LANG_COLOR,
                        arrowprops=dict(arrowstyle='->', color=LANG_COLOR, lw=0.7))
            all_ticks = list(x) + [bx]
            all_labs  = [f'r={r}' for r in all_ranks] + ['r=256\n(lang)']
            ax.set_xticks(all_ticks)
            ax.set_xticklabels(all_labs)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f'r={r}' for r in all_ranks])

    ax.axhline(1.0, color='#888', lw=0.9, ls=':', label='FR_vs_FR = 1.0')
    ax.set_ylabel('FR_vs_LR / FR_vs_FR')
    ax.set_ylim(0, 1.35)
    ax.set_title('Normalised divergence at final checkpoint\n'
                 'Language r=256 indistinguishable from FR_vs_FR')
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim1d_normalised_final.png')


# ── Panel E — FR_vs_FR trend (normalised) ────────────────────────────────

def plot_e(data, out_dir):
    """
    Both series normalised to 1.0 at t=0.
    Vision rising above 1.0 = two FR seeds converge.
    Language falling below 1.0 = two FR seeds diverge.
    """
    domain_styles = [
        ('cifar10',             'Vision',   VISION_COLOR, '-'),
        ('babylm_strict_small', 'Language', LANG_COLOR,   '--'),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))

    for domain, label, color, ls in domain_styles:
        t = traj(data, domain, 'FR_vs_FR', 'FR')
        if not t: continue
        xs, ys   = zip(*t)
        ys       = np.array(ys)
        norm_y   = ys / ys[0]            # relative to t=0
        norm_x   = np.linspace(0, 1, len(xs))

        ax.plot(norm_x, norm_y, color=color, ls=ls, lw=2, label=label)

        # Shade diverging/converging regions
        ax.fill_between(norm_x, 1.0, norm_y,
                        where=(norm_y >= 1.0), color=color, alpha=0.08)
        ax.fill_between(norm_x, norm_y, 1.0,
                        where=(norm_y < 1.0),  color=color, alpha=0.08)

        final   = norm_y[-1]
        arrow_y = final + (0.04 if final > 1.0 else -0.06)
        txt     = f'{label}: {final:.2f}×\n({"↑ converges" if final>1 else "↓ diverges"})'
        ax.annotate(txt,
                    xy=(1.0, final), xytext=(0.72, arrow_y),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.axhline(1.0, color='#aaa', lw=0.9, ls=':', label='t=0 baseline')
    ax.set_xlabel('training progress (normalised)')
    ax.set_ylabel('FR_vs_FR CKA relative to t=0')
    ax.set_xlim(0, 1.05)
    ax.set_title('FR_vs_FR trend — are two FR seeds converging or diverging?\n'
                 'Vision converges; language diverges')
    ax.legend(loc='center left', framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim1e_frfr_trend.png')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',
                   default='results/cross_domain/aim1_divergence_combined.csv')
    p.add_argument('--out', default='figures')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f'CSV not found: {args.csv}\n'
            'Run collate_results.py with --config and --config2 first.')

    data = load(args.csv)
    print(f'Loaded {len(data)} rows from {args.csv}')
    print(f'Output dir: {args.out}\n')

    plot_a(data, args.out)
    plot_b(data, args.out)
    plot_c(data, args.out)
    plot_d(data, args.out)
    plot_e(data, args.out)

    print('\nDone. 5 figures written.')

if __name__ == '__main__':
    main()