"""
train.py — Unified training script for the rank dynamics pipeline.

Supports two training modes controlled by config:
  epoch       — fixed number of epochs (vision default)
  token_budget — train until a token budget is exhausted, saving checkpoints
                 at specified token counts (NLP default)

Token-budget mode is the scientifically cleaner design for NLP: it lets us
ask "at what token count does FR-LR divergence occur" rather than the
dataset-size-dependent "at what epoch", enabling direct comparison across
datasets and model sizes.

Usage:
    python train.py --config configs/cifar10.yaml --run_type FR
    python train.py --config configs/babylm_strict_small.yaml --run_type LR --rank 8
"""

import os, math, time, csv, argparse, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from task_registry import (
    load_config, get_task,
    replace_with_low_rank, materialize_low_rank
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # silence tokenizer warning for NLP tasks

# ── LR schedule ───────────────────────────────────────────────────────────

def get_lr_by_step(step, total_steps, warmup_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * t))


def get_lr_by_tokens(tokens_seen, token_budget, warmup_tokens, peak_lr, min_lr):
    if tokens_seen < warmup_tokens:
        return peak_lr * tokens_seen / max(1, warmup_tokens)
    t = (tokens_seen - warmup_tokens) / max(1, token_budget - warmup_tokens)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * t))


# ── Checkpoint ────────────────────────────────────────────────────────────

def save_checkpoint(model, path):
    os.makedirs(path, exist_ok=True)
    m2 = copy.deepcopy(model).cpu()
    materialize_low_rank(m2)
    torch.save(m2.state_dict(), os.path.join(path, 'model.pt'))


# ── Evaluate ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = total_correct = total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            if loss_fn == 'cross_entropy_cls':
                loss = F.cross_entropy(logits, y, reduction='sum')
                total_correct += (logits.argmax(1) == y).sum().item()
                total_tokens  += y.size(0)
            else:
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B*T, V), y.reshape(B*T), reduction='sum')
                total_tokens += B * T
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / max(1, total_tokens)
    if loss_fn == 'cross_entropy_cls':
        return avg_loss, total_correct / max(1, total_tokens)
    return avg_loss, math.exp(min(avg_loss, 20))


# ── Epoch-mode training ──────────────────────────────────────────────────

def train_epoch_mode(model, train_loader, val_loader, device, loss_fn,
                     cfg, run_dir, log):
    epochs       = cfg.get('epochs', 20)
    peak_lr      = cfg.get('lr', 1e-3)
    min_lr       = cfg.get('lr_min', 1e-6)
    warmup_frac  = cfg.get('warmup_frac', 0.05)
    weight_decay = cfg.get('weight_decay', 0.05)
    grad_clip    = cfg.get('grad_clip', 1.0)

    total_steps  = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_frac))
    optimizer    = AdamW(model.parameters(), lr=peak_lr,
                         weight_decay=weight_decay)

    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    step, t0    = 0, time.time()
    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            cur_lr = get_lr_by_step(step, total_steps, warmup_steps,
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
                    loss = F.cross_entropy(
                        logits.reshape(B*T, V), y.reshape(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if step % 200 == 0:
                print(f'ep {epoch:2d} step {step:5d} | '
                      f'loss {loss.item():.4f} | lr {cur_lr:.2e} | '
                      f'{time.time()-t0:.0f}s')
            step += 1

        val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)
        is_best = (val_metric < best_metric if loss_fn != 'cross_entropy_cls'
                   else val_metric > best_metric)
        if is_best:
            best_metric = val_metric
            save_checkpoint(model, os.path.join(run_dir, 'best'))
        print(f'  ep {epoch:2d} | val_loss={val_loss:.4f} | '
              f'{metric_name}={val_metric:.4f} | best={best_metric:.4f}')
        log.writerow([epoch, step, '', f'{val_loss:.4f}',
                      f'{val_metric:.4f}', f'{best_metric:.4f}',
                      f'{time.time()-t0:.0f}'])
        save_checkpoint(model,
                        os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:02d}'))

    return best_metric


# ── Token-budget training ─────────────────────────────────────────────────

def train_token_budget_mode(model, train_loader, val_loader, device, loss_fn,
                             cfg, run_dir, log):
    token_budget    = int(cfg.get('token_budget', 10_000_000))
    warmup_tokens   = int(cfg.get('warmup_tokens', 500_000))
    peak_lr         = cfg.get('lr', 3e-4)
    min_lr          = cfg.get('lr_min', 1e-6)
    weight_decay    = cfg.get('weight_decay', 0.1)
    grad_clip       = cfg.get('grad_clip', 1.0)
    block_size      = cfg.get('model', {}).get('block_size', 128)
    batch_size      = cfg.get('batch_size', 64)
    tokens_per_batch = batch_size * block_size

    # Checkpoint schedule — sorted list of token counts
    ckpt_tokens = sorted(int(t) for t in
                         cfg.get('checkpoint_tokens', [token_budget]))
    ckpt_set    = set(ckpt_tokens)
    next_ckpt   = 0   # index into ckpt_tokens

    optimizer   = AdamW(model.parameters(), lr=peak_lr,
                        weight_decay=weight_decay)

    metric_name  = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric  = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    tokens_seen  = 0
    step         = 0
    t0           = time.time()
    model.train()

    ckpt_str = ", ".join(f"{t:,}" for t in ckpt_tokens)
    print(f"Token budget: {token_budget:,}  |  "
        f"Checkpoints at: [{ckpt_str}]")

    # ── Step-0 checkpoint: before any gradient update ─────────────────────
    # Addresses reviewer request for zero-update CKA baseline.
    # Saved as tokens_0000000000 so probe_dynamics.py sorts it first.
    step0_ckpt = os.path.join(run_dir, 'checkpoints', 'tokens_0000000000')
    if not os.path.exists(os.path.join(step0_ckpt, 'model.pt')):
        print("Saving step-0 checkpoint (before any gradient update)...")
        save_checkpoint(model, step0_ckpt)
        print(f"  ✓ step-0 checkpoint saved: {step0_ckpt}")
    else:
        print(f"  step-0 checkpoint already exists: {step0_ckpt}")

    pass_num = 0
    while tokens_seen < token_budget:
        pass_num += 1
        pass_tokens_start = tokens_seen
        print(f"  Pass {pass_num} | tokens so far: {tokens_seen:,} / {token_budget:,}")
        for batch in train_loader:
            if tokens_seen >= token_budget:
                break

            x, y = batch[0].to(device), batch[1].to(device)
            cur_tokens = x.numel()   # actual tokens in this batch

            cur_lr = get_lr_by_tokens(tokens_seen, token_budget,
                                      warmup_tokens, peak_lr, min_lr)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                if loss_fn == 'cross_entropy_cls':
                    loss = F.cross_entropy(logits, y)
                else:
                    B, T, V = logits.shape
                    loss = F.cross_entropy(
                        logits.reshape(B*T, V), y.reshape(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            tokens_seen += cur_tokens
            step        += 1

            if step % 100 == 0:
                pct = tokens_seen / token_budget * 100
                print(f'tokens {tokens_seen:>10,} ({pct:5.1f}%) | '
                      f'loss {loss.item():.4f} | lr {cur_lr:.2e} | '
                      f'{time.time()-t0:.0f}s')

            # Checkpoint at scheduled token counts
            if (next_ckpt < len(ckpt_tokens) and
                    tokens_seen >= ckpt_tokens[next_ckpt]):

                t_label = ckpt_tokens[next_ckpt]
                val_loss, val_metric = evaluate(
                    model, val_loader, device, loss_fn)
                is_best = (val_metric < best_metric
                           if loss_fn != 'cross_entropy_cls'
                           else val_metric > best_metric)
                if is_best:
                    best_metric = val_metric
                    save_checkpoint(model, os.path.join(run_dir, 'best'))

                ckpt_name = f'tokens_{t_label:010d}'
                save_checkpoint(model,
                                os.path.join(run_dir, 'checkpoints', ckpt_name))
                print(f'  ✓ checkpoint @ {tokens_seen:,} tokens | '
                      f'val_loss={val_loss:.4f} | '
                      f'{metric_name}={val_metric:.4f} | '
                      f'best={best_metric:.4f}')
                log.writerow([next_ckpt, tokens_seen, f'{tokens_seen:,}',
                              f'{val_loss:.4f}', f'{val_metric:.4f}',
                              f'{best_metric:.4f}', f'{time.time()-t0:.0f}'])
                next_ckpt += 1

    return best_metric


# ── Main ─────────────────────────────────────────────────────────────────

def train(args):
    cfg  = load_config(args.config)
    task = get_task(cfg['task'])

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_mode = cfg.get('training_mode', 'epoch')
    print(f"Device: {device}  |  Task: {cfg['task']}  |  "
          f"Mode: {training_mode}  |  run_type: {args.run_type}  |  "
          f"seed: {seed}")

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    model = task['make_model'](cfg)
    if args.run_type == 'LR':
        replace_with_low_rank(model, args.rank)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_name = 'FR' if args.run_type == 'FR' else f'LR_r{args.rank}'
        if seed != 0:
            run_name += f'_seed{seed}'
        run_dir = os.path.join(cfg['output_root'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Logging
    log_path = os.path.join(run_dir, 'log.csv')
    log_file = open(log_path, 'w', newline='')
    log      = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'

    if training_mode == 'token_budget':
        log.writerow(['checkpoint_idx', 'tokens_seen', 'tokens_label',
                      'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])
        best = train_token_budget_mode(
            model, train_loader, val_loader, device, loss_fn,
            cfg, run_dir, log)
    else:
        log.writerow(['epoch', 'step', 'tokens_seen',
                      'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])
        best = train_epoch_mode(
            model, train_loader, val_loader, device, loss_fn,
            cfg, run_dir, log)

    save_checkpoint(model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',   required=True)
    p.add_argument('--run_type', choices=['FR', 'LR'], default='FR')
    p.add_argument('--rank',     type=int, default=8)
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--run_dir',  default=None)
    train(p.parse_args())