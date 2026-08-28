"""
sweep_lr_hyperparameters.py — LR-favorable hyperparameter sweep for R1-W3.

R1-W3: "The comparison between FR and LR may be affected by fixed
hyperparameters. Using the same AdamW setup for all ranks may disadvantage
LR models. More tuning for LR learning rate, initialization scale, and
optimizer settings is needed before attributing the gap mainly to rank
constraints."

This script trains LR_r{rank} under a small grid of learning-rate
multipliers and initialization-scale multipliers (relative to the paper's
fixed baseline hyperparameters), under a REDUCED token budget (to keep the
sweep tractable ahead of a rebuttal deadline), and reports final
validation perplexity for every configuration. The question this answers:
does the FR-LR performance gap shrink substantially once LR's own
hyperparameters are tuned in its favor, or does it persist regardless of
the specific learning rate / init scale chosen?

Reuses:
  - load_config, get_task, replace_with_low_rank, LowRankLinear
                                                    (task_registry.py)
  - get_lr_by_tokens, save_checkpoint, evaluate     (train.py)

New in this file:
  - rescale_lr_init_(): rescales every LowRankLinear's A, B by
    sqrt(init_scale_mult), so the effective weight A@B.T scales by
    init_scale_mult without changing its spectral shape
  - a reduced-budget training loop parameterized by an LR multiplier
    (train.py's loop hardcodes a single fixed lr; a variant is needed to
    sweep it)
  - a sweep driver that runs every (lr_mult, init_scale_mult) combination
    and writes a summary CSV/table

Usage:
    python sweep_lr_hyperparameters.py --config configs/babylm_strict_small.yaml \\
        --rank 8 --sweep_token_budget 10000000 \\
        --lr_mults 0.5 1.0 2.0 3.0 \\
        --init_scale_mults 0.5 1.0 2.0
"""

import os, time, csv, argparse, random, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_registry import load_config, get_task, replace_with_low_rank, LowRankLinear
from train import get_lr_by_tokens, save_checkpoint, evaluate


# ── Initialization-scale rescaling ──────────────────────────────────────────

def rescale_lr_init_(model, init_scale_mult):
    """
    In-place: rescale every LowRankLinear's A, B by sqrt(init_scale_mult),
    so the effective weight A @ B.T scales by init_scale_mult overall
    while its spectral SHAPE (relative singular value ratios) is
    unchanged -- isolating "initialization scale" as the reviewer names
    it, distinct from rank or spectral shape.
    """
    if init_scale_mult == 1.0:
        return model
    root = model.hf_model if hasattr(model, 'hf_model') else model
    factor = init_scale_mult ** 0.5
    n_rescaled = 0
    for module in root.modules():
        if isinstance(module, LowRankLinear):
            module.A.data.mul_(factor)
            module.B.data.mul_(factor)
            n_rescaled += 1
    print(f"    Rescaled {n_rescaled} LowRankLinear layers by "
         f"sqrt({init_scale_mult}) = {factor:.4f}")
    return model


# ── Training loop (reduced budget, LR-multiplier parameterized) ────────────
# Mirrors train.py's / train_galore.py's token-budget loop, with the peak
# LR scaled by lr_mult -- train.py's own loop hardcodes a single fixed lr,
# so a variant is needed to sweep it.

def train_sweep_token_budget(model, train_loader, val_loader, device, loss_fn,
                             cfg, lr_mult, token_budget):
    warmup_tokens = int(cfg.get('warmup_tokens', 500_000))
    base_lr       = cfg.get('lr', 3e-4)
    peak_lr       = base_lr * lr_mult
    min_lr        = cfg.get('lr_min', 1e-6) * lr_mult
    weight_decay  = cfg.get('weight_decay', 0.1)
    grad_clip     = cfg.get('grad_clip', 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                  weight_decay=weight_decay)

    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    tokens_seen, step, t0 = 0, 0, time.time()
    model.train()

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

            if step % 200 == 0:
                pct = tokens_seen / token_budget * 100
                print(f'    tokens {tokens_seen:>10,} ({pct:5.1f}%) | '
                     f'loss {loss.item():.4f} | lr {cur_lr:.2e}')

    val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)
    elapsed = time.time() - t0
    print(f'    Final: val_loss={val_loss:.4f}  {metric_name}={val_metric:.4f}  '
         f'({elapsed:.0f}s)')
    return val_metric, metric_name


# ── Sweep driver ─────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    task = get_task(cfg['task'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    results_dir = cfg.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, 'lr_hyperparam_sweep.csv')
    csv_file = open(out_csv, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['rank', 'lr_mult', 'init_scale_mult', 'peak_lr',
                     'metric_name', 'val_metric', 'sweep_token_budget'])

    print(f"Sweep: rank={args.rank}  token_budget={args.sweep_token_budget:,}  "
         f"lr_mults={args.lr_mults}  init_scale_mults={args.init_scale_mults}\n")

    combos = list(itertools.product(args.lr_mults, args.init_scale_mults))
    for i, (lr_mult, scale_mult) in enumerate(combos):
        print(f"[{i+1}/{len(combos)}] lr_mult={lr_mult}  init_scale_mult={scale_mult}")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

        train_cfg = dict(cfg); train_cfg['_role'] = 'train'
        model = task['make_model'](train_cfg)
        replace_with_low_rank(model, args.rank, verbose=False)
        rescale_lr_init_(model, scale_mult)
        model = model.to(device)

        val_metric, metric_name = train_sweep_token_budget(
            model, train_loader, val_loader, device, loss_fn, cfg,
            lr_mult=lr_mult, token_budget=args.sweep_token_budget)

        writer.writerow([args.rank, lr_mult, scale_mult,
                        cfg.get('lr', 3e-4) * lr_mult,
                        metric_name, val_metric, args.sweep_token_budget])
        csv_file.flush()

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nDone. Sweep results: {out_csv}")

    # Print a quick summary sorted by val_metric (lower is better for ppl)
    import pandas as pd
    df = pd.read_csv(out_csv)
    higher_better = df['metric_name'].iloc[0] == 'val_acc'
    df_sorted = df.sort_values('val_metric', ascending=not higher_better)
    print("\nBest configurations (best first):")
    print(df_sorted.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--sweep_token_budget', type=int, default=10_000_000,
                  help='Reduced token budget for the sweep (default: 10M, vs '
                      'the full 50M budget, to keep the grid tractable)')
    p.add_argument('--lr_mults', type=float, nargs='+',
                  default=[0.5, 1.0, 2.0, 3.0],
                  help='Learning-rate multipliers relative to the config lr')
    p.add_argument('--init_scale_mults', type=float, nargs='+',
                  default=[0.5, 1.0, 2.0],
                  help='Initialization-scale multipliers for A@B.T')
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())