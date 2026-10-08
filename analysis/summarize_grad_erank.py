"""
summarize_grad_erank.py — Print mean grad_erank per checkpoint for a run.

Small utility so results can be shared as a short, reliable text summary
instead of pasting a full grad_erank.csv (which has been failing to come
through as an attachment).

Usage:
    python analysis/summarize_grad_erank.py outputs/babylm_strict_small/ReLoRA_r8
    python analysis/summarize_grad_erank.py outputs/babylm_strict_small/LR_r8 --weight_filter attn
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
    p.add_argument('run_dir', help='Run directory, e.g. outputs/babylm_strict_small/ReLoRA_r8')
    p.add_argument('--weight_filter', default=None,
                  help="Optional substring filter on weight_name, e.g. 'attn'")
    p.add_argument('--csv_path', default=None,
                  help="Override path to grad_erank.csv (default: <run_dir>/probes/grad_erank.csv)")
    args = p.parse_args()

    csv_path = args.csv_path or os.path.join(args.run_dir, 'probes', 'grad_erank.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if args.weight_filter:
        before = len(df)
        df = df[df['weight_name'].str.contains(args.weight_filter)]
        print(f"Filtered by '{args.weight_filter}': {before} -> {len(df)} rows")

    trend = df.groupby('checkpoint_name')['grad_erank'].mean()
    trend = trend.reindex(sorted(trend.index, key=checkpoint_tokens))

    print(f"\nRun: {args.run_dir}")
    print(f"Source: {csv_path}")
    print(f"\n{'checkpoint':>22}{'tokens/epoch':>14}{'mean grad_erank':>18}")
    for name, val in trend.items():
        print(f"{name:>22}{checkpoint_tokens(name):>14,}{val:>18.2f}")

    valid = trend.dropna()
    if len(valid):
        print(f"\nFirst: {valid.iloc[0]:.2f}   Final: {valid.iloc[-1]:.2f}   "
              f"(n_checkpoints={len(valid)})")


if __name__ == '__main__':
    main()