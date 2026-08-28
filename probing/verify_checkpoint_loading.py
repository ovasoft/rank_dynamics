"""
verify_checkpoint_loading.py — Reload a checkpoint fresh (the same way
probe_transfer.py does) and check whether it reproduces its own
training-log perplexity on the actual validation set.

This isolates two very different failure modes that look similar from
outside (both produce "wrong perplexity"):
  1. The checkpoint fails to load correctly (architecture/shape mismatch,
     wrong config, etc.) -- fresh-load perplexity will be wildly different
     from the training-log value.
  2. The checkpoint loads fine, but something specific to the BLiMP
     scoring path (tokenization, sequence_logprob, phenomenon loading)
     is broken -- fresh-load perplexity on the VALIDATION set will match
     the training log closely, even though BLiMP numbers look wrong.

Reuses:
  - load_config, get_task                (task_registry.py)
  - load_model_from_checkpoint            (probe_dynamics.py)
  - evaluate                              (train.py)

Usage:
    python verify_checkpoint_loading.py --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/LR_r16 \\
        --checkpoint final
"""

import os, argparse
import torch

from task_registry import load_config, get_task
from probe_dynamics import _fix_conv1d_shapes
from train import evaluate


def load_training_log_value(run_dir, checkpoint):
    """Pull the training-log val_ppl for the given checkpoint, for comparison."""
    import pandas as pd
    log_path = os.path.join(run_dir, 'log.csv')
    if not os.path.exists(log_path):
        return None, None
    df = pd.read_csv(log_path)
    metric_col = 'val_ppl' if 'val_ppl' in df.columns else (
        'val_acc' if 'val_acc' in df.columns else None)
    if metric_col is None or df.empty:
        return None, metric_col
    if checkpoint == 'final':
        return df.iloc[-1][metric_col], metric_col
    return None, metric_col


def diagnose_load(ckpt_path, task, cfg, device):
    """
    Build the model shell the same way load_model_from_checkpoint does,
    but capture and print the exact missing_keys/unexpected_keys from
    load_state_dict, instead of discarding them -- this pinpoints exactly
    which parameters failed to populate, rather than inferring it
    indirectly from a downstream perplexity gap.
    """
    from task_registry import replace_with_low_rank

    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    has_lr_keys = any(k.endswith('.A') or k.endswith('.B') for k in sd)
    rank = None
    if has_lr_keys:
        for k, v in sd.items():
            if k.endswith('.A'):
                rank = v.shape[1]
                break

    model = task['make_model'](cfg or {})
    if has_lr_keys and rank is not None:
        replace_with_low_rank(model, rank, verbose=False)
    model = model.to(device)

    sd_fixed = _fix_conv1d_shapes(sd, model)
    target = model.hf_model if hasattr(model, 'hf_model') else model
    result = target.load_state_dict(sd_fixed, strict=False)

    missing = getattr(result, 'missing_keys', [])
    unexpected = getattr(result, 'unexpected_keys', [])
    print(f"\n--- load_state_dict diagnostic ---")
    print(f"has_lr_keys detected: {has_lr_keys}  (rank={rank})")
    print(f"missing_keys: {len(missing)}")
    if missing:
        print(f"  first 10: {missing[:10]}")
    print(f"unexpected_keys: {len(unexpected)}")
    if unexpected:
        print(f"  first 10: {unexpected[:10]}")
    if not missing and not unexpected:
        print("  No missing/unexpected keys -- load_state_dict matched everything. "
             "If perplexity still doesn't match, the issue is in the VALUES "
             "loaded (e.g. wrong checkpoint file content), not the key names.")

    # Confirmatory test: LowRankLinear never has a bias (forward() has no +bias
    # term), so materialize_low_rank's bias=False reconstruction is not
    # dropping a trained value -- it's correctly representing a bias fixed at
    # zero. If that's the whole story, explicitly zeroing exactly the missing
    # bias parameters (rather than leaving them at the shell's random init)
    # should recover perplexity close to the training log.
    if missing and all(k.endswith('.bias') for k in missing):
        print(f"\n--- confirmatory test: zeroing {len(missing)} missing bias params ---")
        state = target.state_dict()
        with torch.no_grad():
            for k in missing:
                if k in state:
                    state[k].zero_()
        print("Missing bias parameters set to zero (matching LowRankLinear's "
             "true bias-free forward pass).")

    return model


def main(args):
    cfg = load_config(args.config)
    task = get_task(cfg['task'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(args.run_dir, args.checkpoint, 'model.pt')
    print(f"Loading checkpoint: {ckpt_path}")

    model = diagnose_load(ckpt_path, task, cfg, device)

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    print("\nRunning fresh evaluation on the actual validation set...")
    val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    print(f"  Fresh-load {metric_name}: {val_metric:.4f}")

    log_value, log_metric_name = load_training_log_value(args.run_dir, args.checkpoint)
    if log_value is not None:
        print(f"  Training-log {log_metric_name}: {log_value:.4f}")
        ratio = val_metric / log_value if log_value else None
        if ratio is not None:
            print(f"  Ratio (fresh-load / training-log): {ratio:.2f}x")
            if 0.9 <= ratio <= 1.1:
                print("  -> MATCH. Checkpoint loads correctly; the bug is elsewhere "
                     "(likely the BLiMP scoring path specifically).")
            else:
                print("  -> MISMATCH. Checkpoint is NOT loading correctly for this run.")
    else:
        print("  [warn] Could not read training-log value from log.csv for comparison.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--checkpoint', default='final')
    main(p.parse_args())