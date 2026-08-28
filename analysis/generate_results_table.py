"""
generate_results_table.py — Build the raw-results table (model, CKA to FR,
BLiMP, PPL, eRank growth, self-CKA) directly from each run's probe outputs,
instead of hand-copying numbers into LaTeX.

Reuses:
  - load_run_cka            (analyze_cka_comparison.py)
  - load_final_perplexity   (combine_blimp_cka_table.py)

New in this file:
  - load_erank_growth(): final/initial mean gradient eRank (attn layers) from
    grad_erank.csv -- a ratio, e.g. 6.0x growth or 0.65x (shrinkage)
  - load_self_cka(): mean cka_vs_prev across checkpoints from cka_self.csv --
    within-run representational stability, independent of FR

Usage:
    python generate_results_table.py \\
        --blimp_summary results/babylm_strict_small/blimp_summary.csv \\
        --run FR=outputs/babylm_strict_small/FR \\
        --run LR_r8=outputs/babylm_strict_small/LR_r8 \\
        --run GradTrunc8=outputs/babylm_strict_small/GradTrunc8 \\
        --run ReLoRA_r8=outputs/babylm_strict_small/ReLoRA_r8 \\
        --run LR_specmatch8=outputs/babylm_strict_small/LR_specmatch8 \\
        --run LR_r8_tuned=outputs/babylm_strict_small/LR_r8_tuned \\
        --run LR_r8_dsn_static=outputs/babylm_strict_small/LR_r8_dsn_static \\
        --run LR_r8_dsn_dynamic=outputs/babylm_strict_small/LR_r8_dsn_dynamic \\
        --run LR_r8_dsn_flat=outputs/babylm_strict_small/LR_r8_dsn_flat \\
        --out results_table

Any run missing a given file is reported (not silently skipped) and shown
as "--" in the table, rather than guessed at.

Outputs:
    <out>.csv    -- full raw table, one row per run
    <out>.md     -- markdown version
    <out>.tex    -- LaTeX booktabs table
"""

import os, argparse
import pandas as pd

from analyze_cka_comparison import load_run_cka
from combine_blimp_cka_table import load_final_perplexity


def checkpoint_tokens(checkpoint_name):
    try:
        return int(checkpoint_name.split('_')[1])
    except (IndexError, ValueError):
        return 0


def load_erank_growth(run_dir, weight_filter='attn'):
    """
    Final / initial mean gradient eRank (filtered to weight_filter), from
    grad_erank.csv. Returns (ratio, first_val, final_val) or (None, None, None)
    if the file is missing or has fewer than 2 checkpoints.
    """
    path = os.path.join(run_dir, 'probes', 'grad_erank.csv')
    if not os.path.exists(path):
        return None, None, None
    df = pd.read_csv(path)
    if weight_filter:
        df = df[df['weight_name'].str.contains(weight_filter)]
    trend = df.groupby('checkpoint_name')['grad_erank'].mean()
    trend = trend.reindex(sorted(trend.index, key=checkpoint_tokens))
    valid = trend.dropna()
    if len(valid) < 2:
        return None, None, None
    first_val, final_val = valid.iloc[0], valid.iloc[-1]
    ratio = final_val / first_val if first_val else None
    return ratio, first_val, final_val


def load_self_cka(run_dir):
    """
    Mean cka_vs_prev across all checkpoints (excluding the first, which has
    no previous checkpoint to compare against) from cka_self.csv. This is a
    within-run representational stability measure, independent of FR.
    Returns None if the file is missing.
    """
    path = os.path.join(run_dir, 'probes', 'cka_self.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    valid = df['cka_vs_prev'].dropna()
    if not len(valid):
        return None
    return valid.mean()


def main(args):
    blimp = pd.read_csv(args.blimp_summary) if args.blimp_summary else None

    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        runs[name] = path

    rows = []
    for name, path in runs.items():
        print(f"Processing {name}...")

        cka_trend = load_run_cka(path)
        cka_final = None
        if cka_trend is not None:
            valid = cka_trend.dropna()
            if len(valid):
                cka_final = valid.iloc[-1]
        else:
            print(f"  [missing] cka_cross.csv at {path}/probes/")

        blimp_acc = None
        if blimp is not None:
            row = blimp[blimp['run'] == name]
            if not row.empty:
                blimp_acc = row.iloc[0]['overall_accuracy']
            else:
                print(f"  [missing] '{name}' not found in blimp_summary.csv")

        final_ppl, best_ppl, metric_name = load_final_perplexity(path)
        if metric_name is None:
            print(f"  [missing] log.csv at {path}")

        erank_ratio, erank_first, erank_final = load_erank_growth(path)
        if erank_ratio is None:
            print(f"  [missing] grad_erank.csv (or <2 checkpoints) at {path}/probes/")

        self_cka = load_self_cka(path)
        if self_cka is None:
            print(f"  [missing] cka_self.csv at {path}/probes/")

        rows.append({
            'model': name,
            'cka_to_fr': cka_final,
            'blimp': blimp_acc,
            'ppl': final_ppl,
            'erank_growth': erank_ratio,
            'erank_first': erank_first,
            'erank_final': erank_final,
            'self_cka': self_cka,
        })

    df = pd.DataFrame(rows)

    # ---- CSV ----
    csv_path = f"{args.out}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\u2713 Written: {csv_path}")

    # ---- Markdown ----
    def fmt(v, decimals=4, suffix=''):
        return f"{v:.{decimals}f}{suffix}" if pd.notna(v) else "\u2014"

    md_lines = ["| Model | CKA to FR | BLiMP | PPL | eRank growth | Self-CKA |",
               "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        eg = f"{r.erank_growth:.2f}\u00d7" if pd.notna(r.erank_growth) else "\u2014"
        md_lines.append(
            f"| {r.model} | {fmt(r.cka_to_fr)} | {fmt(r.blimp)} | "
            f"{fmt(r.ppl, 2)} | {eg} | {fmt(r.self_cka)} |")
    md_path = f"{args.out}.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\u2713 Written: {md_path}")
    print("\n" + "\n".join(md_lines))

    # ---- LaTeX ----
    def fmt_tex(v, decimals=4):
        return f"{v:.{decimals}f}" if pd.notna(v) else "--"

    tex_lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"\textbf{Model} & \textbf{CKA to FR} & \textbf{BLiMP} & \textbf{PPL} & "
        r"\textbf{eRank growth} & \textbf{Self-CKA} \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        eg = f"{r.erank_growth:.2f}$\\times$" if pd.notna(r.erank_growth) else "--"
        model_tex = r.model.replace('_', r'\_')
        tex_lines.append(
            f"{model_tex} & {fmt_tex(r.cka_to_fr)} & {fmt_tex(r.blimp)} & "
            f"{fmt_tex(r.ppl, 2)} & {eg} & {fmt_tex(r.self_cka)} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}",
                 r"\caption{Raw results across all evaluated models. "
                 r"eRank growth is the ratio of final to initial mean "
                 r"gradient effective rank (attention layers); self-CKA is "
                 r"the mean within-run CKA between consecutive checkpoints.}",
                 r"\label{tab:raw_results}", r"\end{table*}"]
    tex_path = f"{args.out}.tex"
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"\n\u2713 Written: {tex_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--blimp_summary', default=None,
                  help='Path to blimp_summary.csv (optional; BLiMP column '
                      'left blank if omitted)')
    p.add_argument('--run', action='append', required=True,
                  help="NAME=PATH, repeatable")
    p.add_argument('--out', default='results_table',
                  help='Output filename stem (writes .csv, .md, .tex)')
    main(p.parse_args())