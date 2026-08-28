"""
visualize_aim3.py — Generate 4 separate PNG figures for Aim 3.

Outputs:
  figures/aim3a_ratio_over_training.png  — Gradient eRank ratio FR/LR over training (both domains)
  figures/aim3b_fr_grad_erank.png        — FR absolute gradient eRank over training
  figures/aim3c_ratio_vs_rank.png        — Ratio at final checkpoint vs rank, both domains
  figures/aim3d_erank_structural.png     — Structural eRank (weight matrix) vs grad eRank

Usage:
    python visualize_aim3.py
    python visualize_aim3.py \
        --vision   results/cifar10/aim3_grad_erank_cifar10.csv \
        --language results/babylm_strict_small/aim3_grad_erank_babylm_strict_small.csv \
        --out      figures/
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

RANK_COLORS = {
    8:   '#A32D2D',
    16:  '#D4537E',
    32:  '#BA7517',
    64:  '#639922',
    128: '#378ADD',
    256: '#1D9E75',
}
VISION_COLOR = '#378ADD'
LANG_COLOR   = '#1D9E75'
FR_COLOR     = '#042C53'

DEFAULT_VISION   = 'results/cifar10/aim3_grad_erank_cifar10.csv'
DEFAULT_LANGUAGE = 'results/babylm_strict_small/aim3_grad_erank_babylm_strict_small.csv'
DEFAULT_OUT      = 'figures'


def sf(x):
    try: v = float(x); return None if v == 0 else v
    except: return None

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'  Saved: {path}')
    plt.close(fig)

def fmt_token(t):
    t = int(float(t))
    if t >= 1_000_000: return f'{t // 1_000_000}M'
    if t >= 1_000:     return f'{t // 1_000}K'
    return str(t)

def get_ranks(rows):
    """Discover which ranks are present from column names."""
    ranks = []
    if not rows: return ranks
    for k in rows[0]:
        if k.startswith('grad_erank_ratio_r'):
            try: ranks.append(int(k.replace('grad_erank_ratio_r', '')))
            except: pass
    return sorted(ranks)


# ── Panel A — Ratio over training ─────────────────────────────────────────

def plot_a(v_rows, l_rows, out_dir):
    """
    O3.1: Is the gradient eRank ratio constant over training?
    Vision: ~constant (flat lines). Language: growing (FR gains directions, LR cannot).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, rows, domain_label, ck_fn, ck_label in [
        (axes[0], v_rows, 'Vision (CIFAR-10)',
         lambda r: int(r['checkpoint']),
         lambda r: f"ep{r['checkpoint']}"),
        (axes[1], l_rows, 'Language (BabyLM)',
         lambda r: int(float(r['checkpoint'])),
         lambda r: fmt_token(r['checkpoint'])),
    ]:
        if not rows: continue
        ranks   = get_ranks(rows)
        valid   = [r for r in rows if sf(r['fr_grad_erank'])]
        if not valid: continue

        cks     = [ck_fn(r) for r in valid]
        x       = range(len(cks))
        xlabels = [ck_label(r) for r in valid]

        for rank in ranks:
            col  = f'grad_erank_ratio_r{rank}'
            vals = [sf(r.get(col)) for r in valid]
            if not any(v for v in vals): continue
            # Fill None with NaN for plotting gaps
            yvals = [v if v is not None else float('nan') for v in vals]
            ax.plot(x, yvals, color=RANK_COLORS.get(rank, '#888'),
                    label=f'r={rank}', marker='o', markersize=3)

        ax.axhline(1.0, color='#aaa', lw=0.9, ls=':')
        step = max(1, len(cks) // 7)
        ax.set_xticks(range(0, len(cks), step))
        ax.set_xticklabels(xlabels[::step], rotation=25, ha='right')
        ax.set_ylabel('gradient eRank ratio  FR / LR')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{domain_label}\ngradient eRank ratio over training')
        ax.legend(loc='upper right', framealpha=0.9)

    axes[0].set_title('Vision — ratio approx. constant\n(gap established at first update)')
    axes[1].set_title('Language — ratio grows over training\n(FR gains directions; LR cannot)')

    fig.suptitle('Aim 3 — O3.1: Is the gradient eRank gap constant?', fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim3a_ratio_over_training.png')


# ── Panel B — FR absolute gradient eRank ──────────────────────────────────

def plot_b(v_rows, l_rows, out_dir):
    """
    Supporting: FR's absolute gradient eRank over training.
    Vision: starts high (~76), decays slightly to ~62.
    Language: starts lower (~306 at 32K), grows to ~510 at 40M then falls.
    This explains why the language ratio grows — FR is actively expanding
    its gradient dimensionality while LR is fixed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, rows, domain_label, color, ck_fn, ck_label in [
        (axes[0], v_rows, 'Vision (CIFAR-10)', VISION_COLOR,
         lambda r: int(r['checkpoint']),
         lambda r: f"ep{r['checkpoint']}"),
        (axes[1], l_rows, 'Language (BabyLM)', LANG_COLOR,
         lambda r: int(float(r['checkpoint'])),
         lambda r: fmt_token(r['checkpoint'])),
    ]:
        if not rows: continue
        valid = [r for r in rows if sf(r['fr_grad_erank'])]
        if not valid: continue

        cks    = [ck_fn(r)     for r in valid]
        fr_ge  = [sf(r['fr_grad_erank']) for r in valid]
        xlabels= [ck_label(r)  for r in valid]
        x      = range(len(cks))

        ax.plot(x, fr_ge, color=color, lw=2, label='FR grad eRank')
        ax.fill_between(x, 0, fr_ge, color=color, alpha=0.1)

        # LR_r8 for reference
        ranks = get_ranks(rows)
        if ranks:
            r8 = min(ranks)
            r8_col = f'lr_r{r8}_grad_erank'
            r8_vals = [sf(r.get(r8_col)) for r in valid]
            if any(v for v in r8_vals):
                ax.plot(x, [v or float('nan') for v in r8_vals],
                        color=RANK_COLORS.get(r8, '#A32D2D'), lw=1.4,
                        ls='--', label=f'LR r={r8} grad eRank')

        step = max(1, len(cks) // 7)
        ax.set_xticks(range(0, len(cks), step))
        ax.set_xticklabels(xlabels[::step], rotation=25, ha='right')
        ax.set_ylabel('gradient eRank')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{domain_label}\nFR gradient eRank (solid) vs LR_r{min(get_ranks(rows))} (dashed)')
        ax.legend(framealpha=0.9)

    fig.suptitle('Aim 3 — FR gradient eRank over training\n'
                 'Language FR grows; LR stays flat — widening gap', fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim3b_fr_grad_erank.png')


# ── Panel C — Ratio vs rank at final checkpoint ───────────────────────────

def plot_c(v_rows, l_rows, out_dir):
    """
    O3.2/O3.3: How does the ratio scale with rank at the final checkpoint?
    Both domains: higher rank → smaller ratio (closer to 1).
    Language ratios are much larger than vision at equivalent ranks.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Vision final row
    vfinal = [r for r in v_rows if sf(r['fr_grad_erank'])]
    vfinal = vfinal[-1] if vfinal else None

    # Language final row
    lfinal = [r for r in l_rows if sf(r['fr_grad_erank'])]
    lfinal = lfinal[-1] if lfinal else None

    shared_ranks = [8, 16, 32, 64, 128]

    if vfinal:
        vranks = get_ranks(v_rows)
        vr = [sf(vfinal.get(f'grad_erank_ratio_r{r}')) for r in shared_ranks
              if r in vranks]
        vx = [r for r in shared_ranks if r in vranks and
              sf(vfinal.get(f'grad_erank_ratio_r{r}'))]
        ax.plot(vx, [sf(vfinal.get(f'grad_erank_ratio_r{r}')) for r in vx],
                color=VISION_COLOR, marker='o', markersize=6,
                label='Vision (final epoch)', lw=1.8)

    if lfinal:
        lranks = get_ranks(l_rows)
        lx = [r for r in sorted(lranks)
              if sf(lfinal.get(f'grad_erank_ratio_r{r}'))]
        ly = [sf(lfinal.get(f'grad_erank_ratio_r{r}')) for r in lx]
        ax.plot(lx, ly, color=LANG_COLOR, marker='s', markersize=6,
                label='Language (50M tokens)', lw=1.8)
        # Annotate r=256
        if 256 in lx:
            idx = lx.index(256)
            ax.annotate('r=256', xy=(256, ly[idx]),
                        xytext=(220, ly[idx] + 2), fontsize=7,
                        color=LANG_COLOR,
                        arrowprops=dict(arrowstyle='->', color=LANG_COLOR, lw=0.7))

    ax.axhline(1.0, color='#aaa', lw=0.9, ls=':', label='ratio = 1 (equal grad eRank)')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('rank (r)')
    ax.set_ylabel('gradient eRank ratio  FR / LR')
    ax.set_title('O3.2/O3.3: Gradient eRank ratio vs rank at final checkpoint\n'
                 'Both domains: higher rank → smaller ratio; language ratios much larger')
    ax.legend(framealpha=0.9)

    # Mark where ratio ≈ 1 (language between r128 and r256)
    if lfinal and sf(lfinal.get('grad_erank_ratio_r128')) and \
       sf(lfinal.get('grad_erank_ratio_r256')):
        ax.axhspan(1.0, 2.0, alpha=0.04, color=LANG_COLOR)
        ax.text(220, 1.3, 'approaching parity\n(r≈256 language)', fontsize=7,
                color=LANG_COLOR, ha='right')

    fig.tight_layout()
    save_fig(fig, out_dir, 'aim3c_ratio_vs_rank.png')


# ── Panel D — Structural eRank vs gradient eRank ─────────────────────────

def plot_d(v_rows, l_rows, out_dir):
    """
    Supporting: weight matrix eRank (structural) vs gradient eRank (dynamic).
    LR structural eRank is fixed by the rank constraint; gradient eRank is not.
    FR structural eRank decreases over training (spectral consolidation);
    FR gradient eRank is higher and more variable.
    Shows the two metrics capture different things.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, rows, domain_label, color, ck_label_fn in [
        (axes[0], v_rows, 'Vision (CIFAR-10)', VISION_COLOR,
         lambda r: f"ep{r['checkpoint']}"),
        (axes[1], l_rows, 'Language (BabyLM)', LANG_COLOR,
         lambda r: fmt_token(r['checkpoint'])),
    ]:
        valid = [r for r in rows if sf(r['fr_grad_erank'])]
        if not valid: continue

        x       = range(len(valid))
        xlabels = [ck_label_fn(r) for r in valid]

        # FR: structural eRank (weight matrix) vs gradient eRank
        fr_erank = [sf(r['fr_erank'])      for r in valid]
        fr_ge    = [sf(r['fr_grad_erank']) for r in valid]

        ax2 = ax.twinx()
        ax.plot(x, fr_erank, color=color, lw=1.8, ls='-',
                label='FR weight eRank (left)')
        ax2.plot(x, fr_ge, color=color, lw=1.8, ls='--',
                 label='FR grad eRank (right)', alpha=0.7)

        # LR_r8 structural (flat line — fixed by construction)
        ranks = get_ranks(rows)
        if ranks:
            r8 = min(ranks)
            lr_erank = [sf(r.get(f'lr_r{r8}_erank')) for r in valid]
            ax.plot(x, [v or float('nan') for v in lr_erank],
                    color=RANK_COLORS[r8], lw=1.4, ls='-',
                    label=f'LR r={r8} weight eRank (left)')
            lr_ge = [sf(r.get(f'lr_r{r8}_grad_erank')) for r in valid]
            ax2.plot(x, [v or float('nan') for v in lr_ge],
                     color=RANK_COLORS[r8], lw=1.4, ls='--',
                     label=f'LR r={r8} grad eRank (right)', alpha=0.7)

        step = max(1, len(valid) // 7)
        ax.set_xticks(range(0, len(valid), step))
        ax.set_xticklabels(xlabels[::step], rotation=25, ha='right')
        ax.set_ylabel('weight matrix eRank (solid)')
        ax2.set_ylabel('gradient eRank (dashed)', color='#555')
        ax.set_title(f'{domain_label}\nstructural vs gradient eRank')

        # Combined legend
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7,
                  loc='upper right', framealpha=0.9)

    fig.suptitle('Aim 3 — Structural vs gradient eRank\n'
                 'Structural eRank fixed for LR; gradient eRank varies dynamically',
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_dir, 'aim3d_erank_structural.png')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vision',   default=DEFAULT_VISION)
    p.add_argument('--language', default=DEFAULT_LANGUAGE)
    p.add_argument('--out',      default=DEFAULT_OUT)
    args = p.parse_args()

    for path in [args.vision, args.language]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'CSV not found: {path}')

    v_rows = load(args.vision)
    l_rows = load(args.language)
    print(f'Loaded {len(v_rows)} vision rows, {len(l_rows)} language rows')
    print(f'Output dir: {args.out}\n')

    plot_a(v_rows, l_rows, args.out)
    plot_b(v_rows, l_rows, args.out)
    plot_c(v_rows, l_rows, args.out)
    plot_d(v_rows, l_rows, args.out)

    print('\nDone. 4 figures written.')

if __name__ == '__main__':
    main()