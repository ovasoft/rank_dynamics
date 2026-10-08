"""
generate_comparison_tables.py — Generate markdown comparison tables from
existing probe outputs (grad_erank.csv, cka_cross.csv) across any set of
named runs.

Built to grow incrementally: add a
new --run NAME=PATH entry (and optionally a --grad_csv / --cka_csv override
for irregular layouts) whenever a new experiment finishes, rather than
hand-rebuilding comparison tables from scratch each time. Missing files for
a given run are skipped gracefully (not an error), so this can be run at
any point with whatever subset of experiments is currently done.

Reuses the same checkpoint-token parsing convention as
analyze_grad_erank_approx.py / analyze_cka_comparison.py.

Usage:
    python analysis/generate_comparison_tables.py \\
        --run FR=outputs/babylm_strict_small/FR \\
        --run LR_r8=outputs/babylm_strict_small/LR_r8 \\
        --run ReLoRA_r8=outputs/babylm_strict_small/ReLoRA_r8 \\
        --run GradTrunc8=outputs/babylm_strict_small/GradTrunc8 \\
        --run GradTrunc16=outputs/babylm_strict_small/GradTrunc16 \\
        --run LR_specmatch8=outputs/babylm_strict_small/LR_specmatch8 \\
        --fr_fr_baseline outputs/babylm_strict_small/FR_seed1 \\
        --baseline FR \\
        --gap_reference LR_r8 \\
        --out_dir results/comparison_tables

Produces (in --out_dir):
    table1_grad_erank.md   — final-checkpoint true gradient eRank per run
    table2_cka_trajectory.md — full CKA-vs-FR trajectory, all runs + FR-FR baseline
    table3_gap_decomposition.md — final-checkpoint % of FR-FR baseline and
                                  share of --gap_reference's gap explained
    all_tables.md          — all three concatenated, ready to paste as one block
"""

import os, argparse
import pandas as pd


def checkpoint_tokens(checkpoint_name):
    try:
        return int(checkpoint_name.split('_')[1])
    except (IndexError, ValueError):
        return 0


def load_grad_erank_trend(run_dir, weight_filter=None, csv_path=None):
    path = csv_path or os.path.join(run_dir, 'probes', 'grad_erank.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if weight_filter:
        df = df[df['weight_name'].str.contains(weight_filter)]
    trend = df.groupby('checkpoint_name')['grad_erank'].mean()
    return trend.reindex(sorted(trend.index, key=checkpoint_tokens))


def load_cka_trend(run_dir, csv_path=None):
    path = csv_path or os.path.join(run_dir, 'probes', 'cka_cross.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    trend = df.groupby('checkpoint_name')['cka_vs_ref'].mean()
    return trend.reindex(sorted(trend.index, key=checkpoint_tokens))


def fmt_tokens(tok):
    if tok >= 1_000_000:
        return f"{tok/1_000_000:.0f}M" if tok % 1_000_000 == 0 else f"{tok:,}"
    if tok >= 1_000:
        return f"{tok/1_000:.0f}K" if tok % 1_000 == 0 else f"{tok:,}"
    return f"{tok:,}"


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        runs[name] = path

    grad_trends = {}
    cka_trends = {}
    for name, path in runs.items():
        g = load_grad_erank_trend(path, weight_filter=args.weight_filter)
        c = load_cka_trend(path)
        if g is not None:
            grad_trends[name] = g
            print(f"  [ok] {name}: grad_erank.csv loaded ({len(g)} checkpoints)")
        else:
            print(f"  [skip] {name}: no grad_erank.csv found")
        if c is not None:
            cka_trends[name] = c
            print(f"  [ok] {name}: cka_cross.csv loaded ({len(c)} checkpoints)")
        else:
            print(f"  [skip] {name}: no cka_cross.csv found")

    frfr_trend = None
    if args.fr_fr_baseline:
        frfr_trend = load_cka_trend(args.fr_fr_baseline)
        if frfr_trend is not None:
            print(f"  [ok] FR-FR baseline: cka_cross.csv loaded "
                 f"({len(frfr_trend)} checkpoints)")

    all_blocks = []

    # ── Table 1: final-checkpoint true gradient eRank ──
    if grad_trends:
        lines = ["## Table 1: True gradient effective rank (final checkpoint)",
                "", "| Run | Final gradient eRank |", "|---|---|"]
        for name, trend in grad_trends.items():
            valid = trend.dropna()
            if len(valid):
                lines.append(f"| {name} | {valid.iloc[-1]:.2f} |")
        table1 = "\n".join(lines)
        with open(os.path.join(args.out_dir, "table1_grad_erank.md"), "w") as f:
            f.write(table1 + "\n")
        all_blocks.append(table1)
        print(f"\n✓ Table 1 written")

    # ── Table 2: full CKA trajectory ──
    if cka_trends:
        all_tokens = sorted(set().union(*[set(t.index) for t in cka_trends.values()]),
                            key=checkpoint_tokens)
        header_names = list(cka_trends.keys())
        if frfr_trend is not None:
            header_names = header_names + ["FR-FR baseline"]

        lines = ["## Table 2: CKA vs. FR across training", "",
                "| Tokens | " + " | ".join(header_names) + " |",
                "|---|" + "---|" * len(header_names)]
        for ckpt in all_tokens:
            row = [fmt_tokens(checkpoint_tokens(ckpt))]
            for name in cka_trends:
                val = cka_trends[name].get(ckpt)
                row.append(f"{val:.4f}" if pd.notna(val) else "—")
            if frfr_trend is not None:
                val = frfr_trend.get(ckpt)
                row.append(f"{val:.4f}" if pd.notna(val) else "—")
            lines.append("| " + " | ".join(row) + " |")
        table2 = "\n".join(lines)
        with open(os.path.join(args.out_dir, "table2_cka_trajectory.md"), "w") as f:
            f.write(table2 + "\n")
        all_blocks.append(table2)
        print(f"✓ Table 2 written")

    # ── Table 3: final-checkpoint gap decomposition ──
    if cka_trends and frfr_trend is not None:
        frfr_valid = frfr_trend.dropna()
        if len(frfr_valid):
            frfr_final = frfr_valid.iloc[-1]
            lines = ["## Table 3: Final-checkpoint gap decomposition", "",
                    "| Run | CKA vs. FR | % of FR-FR baseline"]
            ref_gap = None
            if args.gap_reference and args.gap_reference in cka_trends:
                ref_valid = cka_trends[args.gap_reference].dropna()
                if len(ref_valid):
                    ref_pct = 100 * ref_valid.iloc[-1] / frfr_final
                    ref_gap = 100 - ref_pct
                    lines[-1] += f" | Share of {args.gap_reference}'s gap explained"
            lines.append("|---|---|---|" + ("---|" if ref_gap is not None else ""))

            lines.append(f"| FR-FR baseline | {frfr_final:.4f} | 100% |" +
                        (" — |" if ref_gap is not None else ""))
            for name, trend in cka_trends.items():
                valid = trend.dropna()
                if not len(valid):
                    continue
                final_val = valid.iloc[-1]
                pct = 100 * final_val / frfr_final
                row = f"| {name} | {final_val:.4f} | {pct:.1f}% |"
                if ref_gap is not None:
                    gap = 100 - pct
                    row += f" {gap/ref_gap*100:.0f}% |" if name != args.gap_reference else " 100% (reference) |"
                lines.append(row)
            table3 = "\n".join(lines)
            with open(os.path.join(args.out_dir, "table3_gap_decomposition.md"), "w") as f:
                f.write(table3 + "\n")
            all_blocks.append(table3)
            print(f"✓ Table 3 written")

    if all_blocks:
        with open(os.path.join(args.out_dir, "all_tables.md"), "w") as f:
            f.write("\n\n".join(all_blocks) + "\n")
        print(f"\n✓ Combined: {args.out_dir}/all_tables.md")
    else:
        print("\nNo tables generated — no matching probe files found for any run.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True,
                  help="NAME=PATH, repeatable")
    p.add_argument('--fr_fr_baseline', default=None,
                  help="Run dir of a second FR seed (cka_cross.csv vs primary FR) "
                      "for the FR-FR natural baseline")
    p.add_argument('--gap_reference', default=None,
                  help="Run name (from --run) to use as the 100%% reference gap "
                      "in Table 3, e.g. LR_r8")
    p.add_argument('--weight_filter', default='attn',
                  help="Substring filter on weight_name for grad_erank.csv "
                      "(default 'attn')")
    p.add_argument('--out_dir', default='results/comparison_tables')
    main(p.parse_args())