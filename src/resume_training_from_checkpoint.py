"""
resume_training_from_checkpoint.py — Continue training from an existing
checkpoint under the normal AdamW token-budget schedule.

R2-W2 follow-up: spectrum-matched init produces a step-0 CKA at parity
with the FR-FR baseline (expected by construction -- confirms the init
worked, doesn't yet answer the reviewer's actual question). The decisive
test is whether this parity persists through training or the split
re-emerges once ordinary rank-8 gradient updates start flowing. This
script resumes normal training from the spectrum-matched
tokens_0000000000 checkpoint (or any other existing checkpoint) so that
trajectory can be measured against LR_r8's own, checkpoint-for-checkpoint.

Mirrors train.py's own token-budget training loop structure (as reused
verbatim in train_galore.py / train_gradrank.py) rather than assuming a
specific monolithic training-loop function name/signature exists and is
importable -- this keeps the script consistent with the rest of the
project's pattern of reusing get_lr_by_tokens / save_checkpoint /
evaluate as building blocks, since those are the pieces confirmed
importable and reused throughout this project's other new scripts.

Only the optimizer here is plain AdamW -- identical to normal LR_r8
training -- so the ONLY difference from a fresh LR_r8 run is the initial
weights loaded before the loop starts. Optimizer state is NOT resumed
(fresh AdamW moments), matching how the original LR_r8 run itself also
started with fresh optimizer state; this keeps the comparison apples-to-
apples (both begin training with zero optimizer state, differing only in
initial weights).

Reuses:
  - load_config, get_task, replace_with_low_rank   (task_registry.py)
  - get_lr_by_tokens, save_checkpoint, evaluate     (train.py)

Usage:
    python resume_training_from_checkpoint.py \\
        --config configs/babylm_strict_small.yaml \\
        --resume_from outputs/babylm_strict_small/LR_specmatch8/checkpoints/tokens_0000000000/model.pt \\
        --rank 8 \\
        --run_dir outputs/babylm_strict_small/LR_specmatch8
"""

import os, time, csv, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_registry import load_config, get_task, replace_with_low_rank
from train import get_lr_by_tokens, save_checkpoint, evaluate


def load_resume_weights(model, resume_from, device):
    """
    Load an existing checkpoint's weights into `model` (already built with
    the correct architecture, e.g. replace_with_low_rank already applied).
    Prints a clear warning if any keys fail to match, since a silent
    mismatch here would mean training "resumes" from an unintended
    (partially random) state -- exactly the failure mode this project has
    hit before with rank-detection bugs in the probing scripts.
    """
    sd = torch.load(resume_from, map_location=device, weights_only=True)
    target = model.hf_model if hasattr(model, 'hf_model') else model
    result = target.load_state_dict(sd, strict=False)

    missing = getattr(result, 'missing_keys', [])
    unexpected = getattr(result, 'unexpected_keys', [])
    print(f"  Loaded {resume_from}")
    print(f"  missing_keys: {len(missing)}   unexpected_keys: {len(unexpected)}")
    if missing:
        print(f"  WARNING: {len(missing)} parameters were NOT loaded from the "
             f"checkpoint and remain at their fresh random initialization. "
             f"First few: {missing[:5]}")
        print(f"  This usually means the model architecture built here doesn't "
             f"match how the checkpoint was saved -- verify --rank matches the "
             f"checkpoint's actual rank, and that the checkpoint is in "
             f"materialized (plain .weight) form, not raw .A/.B factors.")
    return model


# ── Training loop ────────────────────────────────────────────────────────
# Mirrors train.py's own token-budget loop / train_galore.py's variant, but
# with plain AdamW (identical optimizer to a normal LR_r8 run) and weights
# loaded from an existing checkpoint instead of a fresh random init.

def train_resume_token_budget(model, train_loader, val_loader, device, loss_fn,
                              cfg, run_dir, log):
    token_budget  = int(cfg.get('token_budget', 10_000_000))
    warmup_tokens = int(cfg.get('warmup_tokens', 500_000))
    peak_lr       = cfg.get('lr', 3e-4)
    min_lr        = cfg.get('lr_min', 1e-6)
    weight_decay  = cfg.get('weight_decay', 0.1)
    grad_clip     = cfg.get('grad_clip', 1.0)

    ckpt_tokens = sorted(int(t) for t in cfg.get('checkpoint_tokens', [token_budget]))
    next_ckpt = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                  weight_decay=weight_decay)

    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    tokens_seen, step, t0 = 0, 0, time.time()
    model.train()

    print(f"[Resume] token_budget={token_budget:,} (fresh optimizer state, "
         f"resumed weights)")

    while tokens_seen < token_budget:
        for batch in train_loader:
            if tokens_seen >= token_budget:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            cur_tokens = x.numel()

            cur_lr = get_lr_by_tokens(tokens_seen, token_budget, warmup_tokens,
                                      peak_lr, min_lr)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                if loss_fn == 'cross_entropy_cls':
                    loss = F.cross_entropy(logits, y)
                else:
                    B, T, V = logits.shape
                    loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            tokens_seen += cur_tokens
            step += 1

            if step % 100 == 0:
                pct = tokens_seen / token_budget * 100
                print(f'tokens {tokens_seen:>10,} ({pct:5.1f}%) | '
                     f'loss {loss.item():.4f} | lr {cur_lr:.2e} | '
                     f'{time.time()-t0:.0f}s')

            if next_ckpt < len(ckpt_tokens) and tokens_seen >= ckpt_tokens[next_ckpt]:
                t_label = ckpt_tokens[next_ckpt]
                val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)
                is_best = (val_metric < best_metric if loss_fn != 'cross_entropy_cls'
                          else val_metric > best_metric)
                if is_best:
                    best_metric = val_metric
                    save_checkpoint(model, os.path.join(run_dir, 'best'))
                ckpt_name = f'tokens_{t_label:010d}'
                save_checkpoint(model, os.path.join(run_dir, 'checkpoints', ckpt_name))
                print(f'  \u2713 checkpoint @ {tokens_seen:,} tokens | '
                     f'val_loss={val_loss:.4f} | {metric_name}={val_metric:.4f} | '
                     f'best={best_metric:.4f}')
                log.writerow([next_ckpt, tokens_seen, f'{tokens_seen:,}',
                             f'{val_loss:.4f}', f'{val_metric:.4f}',
                             f'{best_metric:.4f}', f'{time.time()-t0:.0f}'])
                next_ckpt += 1

    return best_metric


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    cfg  = load_config(args.config)
    task = get_task(cfg['task'])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Task: {cfg['task']}  |  Resuming rank={args.rank}")

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    train_cfg = dict(cfg); train_cfg['_role'] = 'train'
    model = task['make_model'](train_cfg)
    replace_with_low_rank(model, args.rank, verbose=True)
    model = model.to(device)

    print(f"\nLoading checkpoint to resume from: {args.resume_from}")
    load_resume_weights(model, args.resume_from, device)

    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun dir: {run_dir}")

    log_path = os.path.join(run_dir, 'log.csv')
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    log = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    if write_header:
        log.writerow(['checkpoint_idx', 'tokens_seen', 'tokens_label',
                      'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])

    best = train_resume_token_budget(
        model, train_loader, val_loader, device, loss_fn, cfg, run_dir, log)

    save_checkpoint(model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume_from', required=True,
                  help='Path to the checkpoint model.pt to resume from '
                      '(e.g. the spectrum-matched tokens_0000000000 checkpoint)')
    p.add_argument('--rank', type=int, default=8,
                  help='Rank to build the model shell with (must match the '
                      'checkpoint being resumed)')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())