"""
combine_blimp_cka_table.py — Join BLiMP accuracy (from probe_transfer.py's
blimp_summary.csv) with final CKA-vs-FR (from each run's cka_cross.csv)
into a single table, for R2-W3 / R3-W1: does lower CKA correlate with
lower functional (downstream task) performance?

Reuses:
  - load_run_cka                  (analyze_cka_comparison.py)

Usage:
    python combine_blimp_cka_table.py \\
        --blimp_summary results/babylm_strict_small/blimp_summary.csv \\
        --run FR=outputs/babylm_strict_small/FR \\
        --run LR_r8=outputs/babylm_strict_small/LR_r8 \\
        --run LR_r8_dsn_static=outputs/babylm_strict_small/LR_r8_dsn_static \\
        --run LR_r8_dsn_dynamic=outputs/babylm_strict_small/LR_r8_dsn_dynamic \\
        --run LR_r8_dsn_flat=outputs/babylm_strict_small/LR_r8_dsn_flat \\
        --run GradTrunc8=outputs/babylm_strict_small/GradTrunc8 \\
        --run LR_r8_tuned=outputs/babylm_strict_small/LR_r8_tuned \\
        --run LR_specmatch8=outputs/babylm_strict_small/LR_specmatch8 \\
        --run ReLoRA_r8=outputs/babylm_strict_small/ReLoRA_r8 \\
        --fr_fr_baseline outputs/babylm_strict_small/FR_seed1 \\
        --out combined_blimp_cka.md

The blimp_summary.csv 'run' column must match the --run NAME you pass here
(i.e. the run directory's basename, as produced by probe_transfer.py).
"""

import os, argparse
import pandas as pd

from analyze_cka_comparison import load_run_cka


def load_final_perplexity(run_dir):
    """
    Read <run_dir>/log.csv and return (final_val_metric, best_val_metric,
    metric_name) from the last row. Works for both val_ppl and val_acc
    tasks (metric column name varies by task, so this reads whichever of
    val_ppl/val_acc is actually present).
    """
    log_path = os.path.join(run_dir, 'log.csv')
    if not os.path.exists(log_path):
        return None, None, None
    df = pd.read_csv(log_path)
    if df.empty:
        return None, None, None
    metric_name = 'val_ppl' if 'val_ppl' in df.columns else (
        'val_acc' if 'val_acc' in df.columns else None)
    if metric_name is None:
        return None, None, None
    last_row = df.iloc[-1]
    final_val = last_row.get(metric_name)
    best_val = last_row.get(f'best_{metric_name}')
    return final_val, best_val, metric_name


def main(args):
    blimp = pd.read_csv(args.blimp_summary)

    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        runs[name] = path

    frfr_final = None
    if args.fr_fr_baseline:
        frfr_trend = load_run_cka(args.fr_fr_baseline)
        if frfr_trend is not None:
            valid = frfr_trend.dropna()
            if len(valid):
                frfr_final = valid.iloc[-1]
                print(f"FR-FR baseline final CKA: {frfr_final:.4f}")

    rows = []
    for name, path in runs.items():
        blimp_row = blimp[blimp['run'] == name]
        if blimp_row.empty:
            print(f"  [warn] {name}: no matching row in blimp_summary.csv "
                 f"(check the 'run' column matches this name exactly)")
            acc, n_pairs, flag = None, None, None
        else:
            r = blimp_row.iloc[0]
            acc, n_pairs = r['overall_accuracy'], r['n_total_pairs']
            flag = r.get('cross_check_flag', None)

        cka_trend = load_run_cka(path)
        cka_final = None
        if cka_trend is not None:
            valid = cka_trend.dropna()
            if len(valid):
                cka_final = valid.iloc[-1]
        else:
            print(f"  [warn] {name}: no cka_cross.csv found at {path}")

        pct_baseline = (100 * cka_final / frfr_final
                       if cka_final is not None and frfr_final else None)

        final_ppl, best_ppl, metric_name = load_final_perplexity(path)
        if metric_name is None:
            print(f"  [warn] {name}: no log.csv found (or no recognizable "
                 f"metric column) at {path}")

        rows.append({
            'run': name,
            'blimp_accuracy': acc,
            'n_pairs': n_pairs,
            'final_cka_vs_fr': cka_final,
            'pct_of_frfr_baseline': pct_baseline,
            'final_val_metric': final_ppl,
            'best_val_metric': best_ppl,
            'metric_name': metric_name,
            'cross_check_flag': flag,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('final_cka_vs_fr', ascending=False, na_position='last')

    # ── Console + markdown table ──
    lines = ["## BLiMP accuracy, final CKA-vs-FR, and perplexity (triangulation)", "",
            "| Run | BLiMP accuracy | n pairs | Final CKA vs FR | % of FR-FR baseline | "
            "Final metric | Best metric | Checkpoint check |",
            "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        acc_s = f"{r.blimp_accuracy:.4f}" if pd.notna(r.blimp_accuracy) else "—"
        n_s = f"{int(r.n_pairs)}" if pd.notna(r.n_pairs) else "—"
        cka_s = f"{r.final_cka_vs_fr:.4f}" if pd.notna(r.final_cka_vs_fr) else "—"
        pct_s = f"{r.pct_of_frfr_baseline:.1f}%" if pd.notna(r.pct_of_frfr_baseline) else "—"
        final_ppl_s = (f"{r.final_val_metric:.2f} ({r.metric_name})"
                      if pd.notna(r.final_val_metric) else "—")
        best_ppl_s = f"{r.best_val_metric:.2f}" if pd.notna(r.best_val_metric) else "—"
        flag_s = r.cross_check_flag if pd.notna(r.cross_check_flag) else "—"
        lines.append(f"| {r.run} | {acc_s} | {n_s} | {cka_s} | {pct_s} | "
                     f"{final_ppl_s} | {best_ppl_s} | {flag_s} |")
    table_md = "\n".join(lines)

    print("\n" + table_md)

    if args.out:
        with open(args.out, 'w') as f:
            f.write(table_md + "\n")
        print(f"\n✓ Written: {args.out}")

    csv_out = os.path.splitext(args.out)[0] + ".csv" if args.out else "combined_blimp_cka.csv"
    df.to_csv(csv_out, index=False)
    print(f"✓ Written: {csv_out}")

    # Quick correlation checks, if enough rows have both values.
    # Rows in --exclude_from_corr (e.g. FR itself) are dropped from the
    # correlation math but still shown in the table -- a model's CKA
    # against itself (or against whatever the FR reference happens to be)
    # isn't a "treatment condition" data point for the question this
    # correlation is asking (does representational cost predict functional
    # cost across LR-family interventions), and including it can distort
    # the result if that value is stale/mismeasured (as it was found to be
    # for FR in this project -- FR's own cka_cross.csv read ~0.039, not
    # the ~1.0 a true self-comparison or ~0.0935 FR-FR-baseline value
    # would give, indicating it was leftover from being probed against an
    # unrelated run at some point).
    excluded = set(args.exclude_from_corr or [])
    if excluded:
        print(f"\nExcluding from correlation computation (shown in table, "
             f"but not treated as data points): {sorted(excluded)}")
    df_corr = df[~df['run'].isin(excluded)]

    valid_ba = df_corr.dropna(subset=['blimp_accuracy', 'final_cka_vs_fr'])
    if len(valid_ba) >= 3:
        corr = valid_ba['blimp_accuracy'].corr(valid_ba['final_cka_vs_fr'])
        print(f"\nPearson correlation (BLiMP accuracy, CKA vs FR): {corr:.3f} "
             f"(n={len(valid_ba)})")

    valid_bp = df_corr.dropna(subset=['blimp_accuracy', 'final_val_metric'])
    if len(valid_bp) >= 3:
        corr = valid_bp['blimp_accuracy'].corr(valid_bp['final_val_metric'])
        print(f"Pearson correlation (BLiMP accuracy, perplexity): {corr:.3f} "
             f"(n={len(valid_bp)})  [expect NEGATIVE: lower ppl = better = higher BLiMP]")

    valid_cp = df_corr.dropna(subset=['final_cka_vs_fr', 'final_val_metric'])
    if len(valid_cp) >= 3:
        corr = valid_cp['final_cka_vs_fr'].corr(valid_cp['final_val_metric'])
        print(f"Pearson correlation (CKA vs FR, perplexity): {corr:.3f} "
             f"(n={len(valid_cp)})  [expect NEGATIVE: lower ppl = better = usually higher CKA]")

    # Flag the case that actually occurred in this project: FR's own CKA
    # reading far from ~100% of the FR-FR baseline usually means
    # FR/probes/cka_cross.csv is stale or was generated against an
    # unintended reference run. Only checked when an 'FR' row is actually
    # present in the table -- FR is the reference, not a required
    # treatment condition, so it's fine (and often preferable, per
    # --fr_fr_baseline usage) to omit it entirely via --run.
    if 'FR' in df['run'].values and 'FR' not in excluded:
        fr_rows = df[df.run == 'FR']
        if not fr_rows.empty and pd.notna(fr_rows['pct_of_frfr_baseline'].iloc[0]):
            fr_pct = fr_rows['pct_of_frfr_baseline'].iloc[0]
            if not (95 <= fr_pct <= 105):
                print(f"\n[WARNING] The 'FR' row's own CKA is {fr_pct:.1f}% of the "
                     f"FR-FR baseline, not ~100%. This usually means "
                     f"FR/probes/cka_cross.csv is stale or was generated against "
                     f"an unintended reference run -- consider re-generating it, "
                     f"or pass --exclude_from_corr FR to omit it from the "
                     f"correlation numbers above.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--blimp_summary', required=True,
                  help='Path to blimp_summary.csv from probe_transfer.py')
    p.add_argument('--run', action='append', required=True,
                  help="NAME=PATH, repeatable. NAME must match the 'run' "
                      "column in blimp_summary.csv")
    p.add_argument('--fr_fr_baseline', default=None,
                  help='Run dir of a second FR seed for the FR-FR baseline '
                      '(enables the "% of FR-FR baseline" column)')
    p.add_argument('--out', default='combined_blimp_cka.md')
    p.add_argument('--exclude_from_corr', nargs='*', default=None,
                  help="Run names to exclude from correlation computation "
                      "(kept in the table). Typically the FR reference run "
                      "itself, e.g. --exclude_from_corr FR -- FR's own "
                      "cka_cross.csv doesn't represent a treatment condition "
                      "and can be stale/mismeasured (see WARNING check).")
    main(p.parse_args())