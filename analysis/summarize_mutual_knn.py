"""
summarize_mutual_knn.py — Print mean mutual-kNN alignment per checkpoint
for a run, from probe_mutual_knn.py's mutual_knn_cross.csv.

Mirrors summarize_grad_erank.py's design: a small, reliably-pasteable
summary instead of sharing a full CSV.

Usage:
    python analysis/summarize_mutual_knn.py outputs/babylm_strict_small/LR_r8
    python analysis/summarize_mutual_knn.py outputs/babylm_strict_small/GradTrunc8 --k 10
"""

import os, sys, argparse
import pandas as pd


def checkpoint_tokens(checkpoint_name):
    """Parse the token/epoch count out of 'tokens_0010000000' / 'epoch_05'."""
    try:
        return int(checkpoint_name.split('_')[1])
    except (IndexError, ValueError):
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', help='Run directory, e.g. outputs/babylm_strict_small/LR_r8')
    p.add_argument('--k', type=int, default=None,
                  help="Optional: restrict to rows with this mutual_knn_k value "
                      "(useful if the CSV mixes multiple k settings)")
    p.add_argument('--csv_path', default=None,
                  help="Override path to mutual_knn_cross.csv (default: "
                      "<run_dir>/probes/mutual_knn_cross.csv)")
    args = p.parse_args()

    csv_path = args.csv_path or os.path.join(args.run_dir, 'probes', 'mutual_knn_cross.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if args.k is not None:
        before = len(df)
        df = df[df['mutual_knn_k'] == args.k]
        print(f"Filtered to k={args.k}: {before} -> {len(df)} rows")

    trend = df.groupby('checkpoint_name')['mutual_knn_alignment'].mean()
    trend = trend.reindex(sorted(trend.index, key=checkpoint_tokens))

    print(f"\nRun: {args.run_dir}")
    print(f"Source: {csv_path}")
    print(f"\n{'checkpoint':>22}{'tokens/epoch':>14}{'mean mutual_knn':>18}")
    for name, val in trend.items():
        print(f"{name:>22}{checkpoint_tokens(name):>14,}{val:>18.4f}")

    valid = trend.dropna()
    if len(valid):
        print(f"\nFirst: {valid.iloc[0]:.4f}   Final: {valid.iloc[-1]:.4f}   "
             f"(n_checkpoints={len(valid)})")


if __name__ == '__main__':
    main()