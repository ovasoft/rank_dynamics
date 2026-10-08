"""
train_galore.py — GaLore comparison: gradient-projection with full-rank weights.

Tests whether gradient starvation is a separate mechanism or merely a
restatement of the low-rank weight constraint.

GaLore (Zhao et al. 2024) keeps weights FULL-RANK and instead projects
GRADIENTS into a low-rank subspace before the optimizer update — the
structural opposite of this paper's LowRankLinear (which factorizes the
WEIGHT, leaving gradient shape unconstrained). If a GaLore-trained model
shows the same FR-LR CKA split and exhaustion-type stabilization as
LR_r8, that is evidence for gradient-subspace restriction as a real,
separable mechanism. If it does not, the paper's causal attribution to
"gradient starvation" needs qualification.

Uses the `galore-torch` package (pip install galore-torch) for the
low-rank gradient projection and optimizer step — none of that math is
reimplemented here; this file only adds the orchestration galore-torch
doesn't provide (splitting params into galore/non-galore groups matching
the paper's own LR-eligibility rule, and a token-budget training loop
matching train.py's, since train.py hardcodes plain AdamW internally).

Reuses:
  - load_config, get_task, _is_replaceable       (task_registry.py)
  - get_lr_by_tokens, save_checkpoint, evaluate  (train.py)

Usage:
    pip install galore-torch
    python src/train_galore.py --config configs/babylm_strict_small.yaml \\
        --rank 8 --update_proj_gap 200 --galore_scale 0.25
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, time, csv, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_registry import load_config, get_task, _is_replaceable
from train import get_lr_by_tokens, save_checkpoint, evaluate

try:
    from galore_torch import GaLoreAdamW
except ImportError as e:
    raise ImportError(
        "galore-torch is required for this script: pip install galore-torch"
    ) from e


# ── Parameter grouping ──────────────────────────────────────────────────────

def split_galore_params(model):
    """
    Split model parameters into (galore_params, non_galore_params, names)
    using task_registry._is_replaceable — the SAME eligibility rule used
    to decide which layers get factorized under LR training. This keeps
    the comparison apples-to-apples: GaLore projects gradients for
    exactly the matrices this paper's LR condition factorizes (attention
    Q/K/V/O and both feed-forward projections), leaving embeddings,
    biases, layernorm, and lm_head untouched in both conditions.
    """
    root = model.hf_model if hasattr(model, 'hf_model') else model
    galore_params, non_galore_params, galore_names = [], [], []

    for name, module in root.named_modules():
        is_target = _is_replaceable(name, module)
        for pname, p in module.named_parameters(recurse=False):
            full_name = f"{name}.{pname}" if name else pname
            if is_target and pname == 'weight':
                galore_params.append(p)
                galore_names.append(full_name)
            else:
                non_galore_params.append(p)

    print(f"GaLore split: {len(galore_params)} gradient-projected weight "
          f"matrices, {len(non_galore_params)} other parameters "
          f"(embeddings, biases, layernorm, lm_head)")
    return galore_params, non_galore_params, galore_names


# ── Training loop ────────────────────────────────────────────────────────────
# Mirrors train.train_token_budget_mode exactly (checkpoint schedule, LR
# schedule via get_lr_by_tokens, eval via evaluate, save via
# save_checkpoint — all reused directly). The only structural difference
# is the optimizer: train.py hardcodes plain AdamW inside the function
# body, so that loop can't be reused as-is without this variant.

def train_galore_token_budget(model, train_loader, val_loader, device, loss_fn,
                              cfg, run_dir, log, rank, update_proj_gap,
                              galore_scale, proj_type='std'):
    token_budget  = int(cfg.get('token_budget', 10_000_000))
    warmup_tokens = int(cfg.get('warmup_tokens', 500_000))
    peak_lr       = cfg.get('lr', 3e-4)
    min_lr        = cfg.get('lr_min', 1e-6)
    weight_decay  = cfg.get('weight_decay', 0.1)
    grad_clip     = cfg.get('grad_clip', 1.0)

    ckpt_tokens = sorted(int(t) for t in cfg.get('checkpoint_tokens', [token_budget]))
    next_ckpt = 0

    galore_params, non_galore_params, _ = split_galore_params(model)
    param_groups = [
        {'params': non_galore_params},
        {'params': galore_params, 'rank': rank, 'update_proj_gap': update_proj_gap,
         'scale': galore_scale, 'proj_type': proj_type},
    ]
    optimizer = GaLoreAdamW(param_groups, lr=peak_lr, weight_decay=weight_decay)

    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    tokens_seen, step, t0 = 0, 0, time.time()
    model.train()

    print(f"[GaLore] token_budget={token_budget:,} rank={rank} "
          f"update_proj_gap={update_proj_gap} scale={galore_scale} "
          f"proj_type={proj_type}")

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
    print(f"Device: {device}  |  Task: {cfg['task']}  |  GaLore rank={args.rank}")

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    train_cfg = dict(cfg); train_cfg['_role'] = 'train'
    model = task['make_model'](train_cfg)   # FULL-RANK — no replace_with_low_rank call
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters (full-rank, GaLore projects gradients only): {n_params:,}")

    # Naming note: deliberately avoids '_r<N>' in the run name — the
    # existing probing scripts (probe_dynamics.py / probe_gradient_erank.py)
    # pattern-match run directory names like 'LR_r8' to decide whether to
    # call replace_with_low_rank before loading a checkpoint. Since GaLore
    # checkpoints are full-rank (no .A/.B keys), a name containing '_r8'
    # would risk being misread as a low-rank run. 'GaLore_proj{rank}' is
    # safe under both the state-dict-based detection in
    # probe_dynamics.load_model_from_checkpoint (checks for .A/.B keys
    # directly, unaffected either way) and the directory-name-based
    # detection in probe_gradient_erank.py (no '_r' substring present).
    run_dir = args.run_dir or os.path.join(
        cfg['output_root'], f'GaLore_proj{args.rank}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    log_path = os.path.join(run_dir, 'log.csv')
    log_file = open(log_path, 'w', newline='')
    log = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    log.writerow(['checkpoint_idx', 'tokens_seen', 'tokens_label',
                  'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])

    best = train_galore_token_budget(
        model, train_loader, val_loader, device, loss_fn, cfg, run_dir, log,
        rank=args.rank, update_proj_gap=args.update_proj_gap,
        galore_scale=args.galore_scale, proj_type=args.proj_type)

    save_checkpoint(model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--rank', type=int, default=8,
                   help='GaLore gradient projection rank — match LR_r8 for '
                        'direct comparison')
    p.add_argument('--update_proj_gap', type=int, default=200,
                   help='Steps between gradient subspace re-estimation (SVD)')
    p.add_argument('--galore_scale', type=float, default=0.25)
    p.add_argument('--proj_type', default='std',
                   choices=['std', 'reverse_std', 'right', 'left', 'full'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run_dir', default=None)
    main(p.parse_args())