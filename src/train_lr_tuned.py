"""
train_lr_tuned.py — Full-budget training of LR_r{rank} under the
best-found hyperparameters from the LR sweep (sweep_lr_hyperparameters.py),
producing a properly checkpointed, probeable run.

sweep_lr_hyperparameters.py found that the paper's
original fixed hyperparameters (shared with FR) were substantially
suboptimal for LR_r8 -- the best-found configuration (lr_mult=3.0,
init_scale_mult=0.5 by default; override via --lr_mult/--init_scale_mult)
reduced perplexity by 72.5% relative to the fixed baseline, at a reduced
10M-token sweep budget. This script retrains that configuration at the
FULL token budget with the standard checkpoint schedule, so it can be
probed (CKA, gradient eRank) the same way as every other run in this
project -- the necessary next step to determine whether the paper's
REPRESENTATIONAL claims survive under properly-tuned hyperparameters,
separately from the raw perplexity gap.

Reuses:
  - load_config, get_task, replace_with_low_rank   (task_registry.py)
  - get_lr_by_tokens, save_checkpoint, evaluate     (train.py)
  - rescale_lr_init_                                (sweep_lr_hyperparameters.py)

Usage:
    python src/train_lr_tuned.py --config configs/babylm_strict_small.yaml \\
        --rank 8 --lr_mult 3.0 --init_scale_mult 0.5 \\
        --run_dir outputs/babylm_strict_small/LR_r8_tuned
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, time, csv, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_registry import load_config, get_task, replace_with_low_rank
from train import get_lr_by_tokens, save_checkpoint, evaluate
from sweep_lr_hyperparameters import rescale_lr_init_


# ── Training loop (full budget, standard checkpoint schedule) ──────────────
# Mirrors train.py's / train_galore.py's token-budget loop, with the peak
# LR scaled by lr_mult and the low-rank init rescaled by init_scale_mult
# applied once before training starts.

def train_tuned_token_budget(model, train_loader, val_loader, device, loss_fn,
                             cfg, run_dir, log, lr_mult):
    token_budget  = int(cfg.get('token_budget', 50_000_000))
    warmup_tokens = int(cfg.get('warmup_tokens', 500_000))
    base_lr       = cfg.get('lr', 3e-4)
    peak_lr       = base_lr * lr_mult
    min_lr        = cfg.get('lr_min', 1e-6) * lr_mult
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

    print(f"[LR_tuned] token_budget={token_budget:,} lr_mult={lr_mult} "
         f"peak_lr={peak_lr:.2e}")

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
    cfg = load_config(args.config)
    task = get_task(cfg['task'])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Task: {cfg['task']}  |  rank={args.rank}  "
         f"lr_mult={args.lr_mult}  init_scale_mult={args.init_scale_mult}")

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    train_cfg = dict(cfg); train_cfg['_role'] = 'train'
    model = task['make_model'](train_cfg)
    replace_with_low_rank(model, args.rank, verbose=True)
    rescale_lr_init_(model, args.init_scale_mult)
    model = model.to(device)

    run_dir = args.run_dir or os.path.join(
        cfg['output_root'], f'LR_r{args.rank}_tuned')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    log_path = os.path.join(run_dir, 'log.csv')
    log_file = open(log_path, 'w', newline='')
    log = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    log.writerow(['checkpoint_idx', 'tokens_seen', 'tokens_label',
                 'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])

    best = train_tuned_token_budget(
        model, train_loader, val_loader, device, loss_fn, cfg, run_dir, log,
        lr_mult=args.lr_mult)

    save_checkpoint(model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best:.4f}')
    print(f"\nNext steps:")
    print(f"  python probing/probe_dynamics.py --config {args.config} \\")
    print(f"      --run_dir {run_dir} --ref_run_dir <FR run dir>")
    print(f"  python probing/probe_gradient_erank.py --config {args.config} \\")
    print(f"      --run_dir {run_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--lr_mult', type=float, default=3.0,
                  help='Best-found LR multiplier from the sweep (default: 3.0)')
    p.add_argument('--init_scale_mult', type=float, default=0.5,
                  help='Best-found init-scale multiplier from the sweep (default: 0.5)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run_dir', default=None)
    main(p.parse_args())