"""
train_cifar_dsn.py — Low-rank ViT training with Dynamic Spectral Normalisation (DSN).

After each optimiser step, the singular value spectrum of every LowRankLinear
weight matrix is re-projected to maintain a target power-law exponent (alpha),
while preserving:
  - the rank constraint (number of non-zero singular values)
  - the total Frobenius norm (energy)
  - the singular vector directions (U and Vh unchanged)

This separates two properties of LR training that are normally conflated:
  - HOW MANY directions the model uses  (rank)
  - HOW ENERGY IS DISTRIBUTED across those directions (spectral shape / alpha)

DSN holds the spectral shape fixed to match FR's alpha trajectory, allowing us
to ask: is the FR-LR functional divergence caused by the rank constraint alone,
or also by the different spectral geometry that the rank constraint induces?

Three DSN variants are trained:
  dsn_static:   target alpha fixed to FR's epoch-1 value (0.43) throughout
  dsn_dynamic:  target alpha follows FR's observed alpha trajectory
                (loaded from fr_spectral.csv produced by collate_results.py)
  dsn_flat:     target alpha = 0 (fully flat spectrum, all SVs equal — orthogonal init)

All variants save checkpoints in the same format as train_cifar.py, so
probe_dynamics.py can be run on them without modification.

Usage:
    # Static DSN
    python train_cifar_dsn.py --dataset cifar10 --rank 8 --dsn_mode static

    # Dynamic DSN (requires FR spectral CSV)
    python train_cifar_dsn.py --dataset cifar10 --rank 8 --dsn_mode dynamic \
        --fr_spectral results/cifar10/summary_spectral.csv

    # Flat (orthogonal) DSN
    python train_cifar_dsn.py --dataset cifar10 --rank 8 --dsn_mode flat
"""

import os, math, time, argparse, csv, copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from task_registry import (
    load_config, get_task,
    replace_with_low_rank, materialize_low_rank, LowRankLinear
)

# ── Hyperparameters (overridden by config when --config provided) ─────────
BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 1e-3
WARMUP_FRAC  = 0.05
WEIGHT_DECAY = 0.05
GRAD_CLIP    = 1.0
LOG_EVERY    = 50
VAL_EVERY    = 200

# FR alpha trajectory from summary_spectral.csv (fallback if no CSV provided)
FR_ALPHA_TRAJECTORY = {
    0:  0.396, 1:  0.407, 2:  0.420, 3:  0.435, 4:  0.451,
    5:  0.467, 6:  0.481, 7:  0.495, 8:  0.506, 9:  0.515,
    10: 0.523, 11: 0.528, 12: 0.533, 13: 0.536, 14: 0.538,
    15: 0.540, 16: 0.541, 17: 0.542, 18: 0.542, 19: 0.543,
}

# ── Dynamic Spectral Normalisation ───────────────────────────────────────

def project_spectrum(W, target_alpha, strength=1.0):
    """
    Soft-project the singular value spectrum of W toward a target power law.
    strength=1.0 is full projection; strength=0.0 is no-op.

    Rather than replacing SVs with the target shape, we interpolate between
    the current SVs and the target-shaped SVs (both normalised to same energy).
    This preserves the learned relative magnitudes while nudging the shape.
    """
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = len(S)
    if target_alpha == 0.0:
        target_S = torch.ones(r, device=S.device, dtype=S.dtype)
    else:
        ranks    = torch.arange(1, r+1, dtype=torch.float32, device=S.device)
        target_S = ranks.pow(-target_alpha)
    # Rescale target to same Frobenius norm as current
    target_S = target_S * (S.norm() / target_S.norm().clamp(min=1e-8))
    # Soft interpolation
    S_new = (1.0 - strength) * S + strength * target_S
    W_new = (U * S_new.unsqueeze(0)) @ Vh
    return W_new.to(W.dtype)


def reproject_to_low_rank(module, target_alpha, strength=1.0):
    """
    Apply spectral reprojection to a single LowRankLinear module in-place.
    Materialises W = A@B.T, reprojects, re-factorises back to A, B.
    """
    with torch.no_grad():
        W   = module.A @ module.B.T
        W_p = project_spectrum(W, target_alpha, strength)
        r   = module.A.shape[1]
        U, S, Vh = torch.linalg.svd(W_p, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        sqrt_S = S_r.sqrt().clamp(min=1e-8)
        module.A.copy_(U_r  * sqrt_S.unsqueeze(0))
        module.B.copy_((Vh_r * sqrt_S.unsqueeze(1)).T)


def apply_dsn(model, target_alpha, strength=1.0):
    """Apply DSN to all LowRankLinear layers in model."""
    for module in model.modules():
        if type(module).__name__ == 'LowRankLinear':
            reproject_to_low_rank(module, target_alpha, strength)

# ── LR schedule ──────────────────────────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, peak_lr, min_lr=1e-6):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * t))


def get_lr_by_tokens(tokens_seen, token_budget, warmup_tokens, peak_lr, min_lr=1e-6):
    if tokens_seen < warmup_tokens:
        return peak_lr * tokens_seen / max(1, warmup_tokens)
    t = (tokens_seen - warmup_tokens) / max(1, token_budget - warmup_tokens)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * t))

# ── Evaluate ──────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    model.train()
    return correct / total

# ── Checkpoint ────────────────────────────────────────────────────────────

def save_checkpoint(model, path):
    os.makedirs(path, exist_ok=True)
    m2 = copy.deepcopy(model).cpu()
    materialize_low_rank(m2)
    torch.save(m2.state_dict(), os.path.join(path, 'model.pt'))

# ── Alpha schedule ────────────────────────────────────────────────────────

def build_alpha_schedule(mode, fr_spectral_path=None, n_epochs=None):
    """
    Returns a dict {epoch -> target_alpha} for 0..n_epochs-1.
    mode: 'static'  → use FR epoch-1 alpha throughout
          'dynamic' → follow FR alpha trajectory
          'flat'    → alpha=0 throughout (equal SVs)
    """
    n = n_epochs or EPOCHS
    if mode == 'flat':
        return {e: 0.0 for e in range(n)}

    if mode == 'static':
        return {e: FR_ALPHA_TRAJECTORY[1] for e in range(n)}

    # dynamic: load from CSV or use hardcoded trajectory
    if fr_spectral_path and os.path.exists(fr_spectral_path):
        schedule = {}
        with open(fr_spectral_path) as f:
            for row in csv.DictReader(f):
                if row['run'] == 'FR':
                    schedule[int(row['epoch'])] = float(row['alpha_mean'])
        for e in range(n):
            if e not in schedule:
                schedule[e] = FR_ALPHA_TRAJECTORY.get(e, 0.43)
        return schedule
    else:
        return {e: FR_ALPHA_TRAJECTORY.get(e, 0.43) for e in range(n)}

# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # ── Config resolution ─────────────────────────────────────────────────
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    task_name    = args.dataset or cfg.get('task', 'cifar10')
    epochs       = cfg.get('epochs',       EPOCHS)
    peak_lr      = cfg.get('lr',           LR)
    min_lr       = cfg.get('lr_min',       1e-6)
    warmup_frac  = cfg.get('warmup_frac',  WARMUP_FRAC)
    weight_decay = cfg.get('weight_decay', WEIGHT_DECAY)
    grad_clip    = cfg.get('grad_clip',    GRAD_CLIP)
    out_root     = cfg.get('output_root',  f'outputs_cifar/{task_name}')

    task      = get_task(task_name)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Task: {task_name}  |  '
          f'DSN mode: {args.dsn_mode}  |  rank: {args.rank}')

    # Alpha schedule — uses epochs from config
    alpha_schedule = build_alpha_schedule(args.dsn_mode, args.fr_spectral,
                                          n_epochs=epochs)
    print(f'Alpha schedule (first 5 epochs): '
          f'{[(e, round(alpha_schedule[e],3)) for e in range(min(5,epochs))]}')

    # Data and model
    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn  = meta['loss_fn']
    model    = task['make_model'](cfg)
    replace_with_low_rank(model, args.rank)

    # Apply DSN at initialisation (epoch 0 target, full strength)
    apply_dsn(model, alpha_schedule[0], strength=1.0)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # Run directory
    run_name = f'LR_r{args.rank}_dsn_{args.dsn_mode}'
    if args.seed != 0:
        run_name += f'_seed{args.seed}'
    run_dir = args.run_dir or os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'Run dir: {run_dir}')

    # ── Shared training infrastructure ───────────────────────────────────
    training_mode   = cfg.get('training_mode', 'epoch')
    token_budget    = int(cfg.get('token_budget', 0))
    warmup_tokens   = int(cfg.get('warmup_tokens', 500_000))
    dsn_freq_tokens = int(cfg.get('dsn_freq_tokens', 10_000_000))
    ckpt_tokens     = sorted(int(t) for t in
                             cfg.get('checkpoint_tokens', []))

    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)

    log_path = os.path.join(run_dir, 'log.csv')
    log_file = open(log_path, 'w', newline='')
    log      = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    log.writerow(['step', 'epoch_or_tokens', 'train_loss', 'lr',
                  'val_loss', metric_name, f'best_{metric_name}',
                  'elapsed_s', 'target_alpha'])

    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    t0          = time.time()
    model.train()

    def _forward_loss(x, y):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if loss_fn == 'cross_entropy_cls':
                return F.cross_entropy(model(x), y)
            else:
                logits = model(x)
                B, T, V = logits.shape
                return F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))

    def _validate():
        model.eval()
        vl_sum = vc = vt = 0
        with torch.no_grad():
            for vb in val_loader:
                xv, yv = vb[0].to(device), vb[1].to(device)
                lg = model(xv)
                if loss_fn == 'cross_entropy_cls':
                    vl_sum += F.cross_entropy(lg, yv, reduction='sum').item()
                    vc     += (lg.argmax(1) == yv).sum().item()
                    vt     += yv.size(0)
                else:
                    B, T, V = lg.shape
                    vl_sum += F.cross_entropy(
                        lg.reshape(B*T,V), yv.reshape(B*T), reduction='sum').item()
                    vt += B * T
        model.train()
        avg = vl_sum / max(1, vt)
        return avg, (vc/vt if loss_fn=='cross_entropy_cls' else math.exp(min(avg,20)))

    # ── Epoch mode (vision) ───────────────────────────────────────────────
    if training_mode == 'epoch':
        total_steps  = epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * warmup_frac))
        step = 0
        for epoch in range(epochs):
            target_alpha = alpha_schedule.get(epoch, 0.43)
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                cur_lr = get_lr(step, total_steps, warmup_steps, peak_lr, min_lr)
                for g in optimizer.param_groups: g['lr'] = cur_lr
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_loss(x, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if step % LOG_EVERY == 0:
                    print(f'ep {epoch:2d} step {step:5d} | loss {loss.item():.4f}'
                          f' | lr {cur_lr:.2e} | α={target_alpha:.3f}'
                          f' | {time.time()-t0:.0f}s')
                step += 1
            # DSN at epoch boundary
            if (epoch + 1) % args.dsn_freq == 0:
                apply_dsn(model, target_alpha, strength=args.dsn_strength)
                print(f'  [DSN] ep {epoch} α={target_alpha:.3f} '
                      f'strength={args.dsn_strength:.2f}')
            val_loss, val_metric = _validate()
            is_best = (val_metric > best_metric if loss_fn=='cross_entropy_cls'
                       else val_metric < best_metric)
            if is_best:
                best_metric = val_metric
                save_checkpoint(model, os.path.join(run_dir, 'best'))
            print(f'  ep {epoch:2d} | val_loss={val_loss:.4f}'
                  f' | {metric_name}={val_metric:.4f} | best={best_metric:.4f}')
            log.writerow([step, epoch, f'{loss.item():.4f}', f'{cur_lr:.2e}',
                          f'{val_loss:.4f}', f'{val_metric:.4f}',
                          f'{best_metric:.4f}', f'{time.time()-t0:.0f}',
                          f'{target_alpha:.3f}'])
            log_file.flush()
            save_checkpoint(model, os.path.join(run_dir, 'checkpoints',
                                                f'epoch_{epoch:02d}'))

    # ── Token-budget mode (NLP) ───────────────────────────────────────────
    else:
        tokens_seen      = 0
        next_ckpt_idx    = 0
        next_dsn_tokens  = dsn_freq_tokens
        step             = 0
        pass_num         = 0
        print(f"Token budget: {token_budget:,} | "
              f"DSN every {dsn_freq_tokens:,} tokens")

        while tokens_seen < token_budget:
            pass_num += 1
            print(f"  Pass {pass_num} | tokens: {tokens_seen:,}/{token_budget:,}")
            for batch in train_loader:
                if tokens_seen >= token_budget:
                    break
                x, y = batch[0].to(device), batch[1].to(device)
                cur_tokens = x.numel()
                cur_lr = get_lr_by_tokens(
                    tokens_seen, token_budget, warmup_tokens, peak_lr, min_lr)
                for g in optimizer.param_groups: g['lr'] = cur_lr
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_loss(x, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                tokens_seen += cur_tokens
                step        += 1
                if step % LOG_EVERY == 0:
                    pct = tokens_seen / token_budget * 100
                    print(f'tokens {tokens_seen:>12,} ({pct:5.1f}%)'
                          f' | loss {loss.item():.4f} | lr {cur_lr:.2e}'
                          f' | {time.time()-t0:.0f}s')

                # DSN at token intervals
                if tokens_seen >= next_dsn_tokens:
                    epoch_equiv = tokens_seen // dsn_freq_tokens
                    target_alpha = alpha_schedule.get(epoch_equiv, 0.43)
                    apply_dsn(model, target_alpha, strength=args.dsn_strength)
                    print(f'  [DSN] {tokens_seen:,} tokens → α={target_alpha:.3f}'
                          f' strength={args.dsn_strength:.2f}')
                    next_dsn_tokens += dsn_freq_tokens

                # Checkpoint at scheduled token counts
                if (next_ckpt_idx < len(ckpt_tokens) and
                        tokens_seen >= ckpt_tokens[next_ckpt_idx]):
                    t_label = ckpt_tokens[next_ckpt_idx]
                    val_loss, val_metric = _validate()
                    is_best = (val_metric > best_metric
                               if loss_fn=='cross_entropy_cls'
                               else val_metric < best_metric)
                    if is_best:
                        best_metric = val_metric
                        save_checkpoint(model, os.path.join(run_dir, 'best'))
                    ckpt_name = f'tokens_{t_label:010d}'
                    save_checkpoint(model, os.path.join(
                        run_dir, 'checkpoints', ckpt_name))
                    print(f'  ✓ {ckpt_name} | val_loss={val_loss:.4f}'
                          f' | {metric_name}={val_metric:.4f}'
                          f' | best={best_metric:.4f}')
                    log.writerow([step, tokens_seen, f'{loss.item():.4f}',
                                  f'{cur_lr:.2e}', f'{val_loss:.4f}',
                                  f'{val_metric:.4f}', f'{best_metric:.4f}',
                                  f'{time.time()-t0:.0f}',
                                  f'{alpha_schedule.get(tokens_seen//dsn_freq_tokens, 0.43):.3f}'])
                    log_file.flush()
                    next_ckpt_idx += 1

    save_checkpoint(model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best_metric:.4f}')
    print(f'Run: {run_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default=None,
                   help='YAML config path')
    p.add_argument('--dataset',    default=None)
    p.add_argument('--rank',       type=int,  default=None)
    p.add_argument('--dsn_mode',   default='static',
                   choices=['static','dynamic','flat'])
    p.add_argument('--fr_spectral', default=None)
    p.add_argument('--seed',       type=int,  default=0)
    p.add_argument('--run_dir',    default=None)
    p.add_argument('--dsn_freq',    type=int,   default=None)
    p.add_argument('--dsn_strength', type=float, default=None)
    args = p.parse_args()
    # Apply config defaults
    if args.config:
        try:
            import yaml
            with open(args.config) as f: cfg = yaml.safe_load(f)
            out_root = cfg.get('output_root', 'outputs')
            if args.dataset     is None: args.dataset     = cfg.get('task', 'cifar10')
            if args.rank        is None: args.rank        = cfg.get('primary_rank', 8)
            if args.dsn_freq    is None: args.dsn_freq    = cfg.get('dsn_freq', 1)
            if args.dsn_strength is None: args.dsn_strength = cfg.get('dsn_strength', 0.3)
            if args.fr_spectral is None and args.dsn_mode == 'dynamic':
                args.fr_spectral = os.path.join(
                    cfg.get('results_dir', 'results'), 'summary_spectral.csv')
            if args.run_dir is None:
                args.run_dir = os.path.join(
                    out_root, f'LR_r{args.rank}_dsn_{args.dsn_mode}')
        except Exception as e:
            print(f"Warning: could not load config: {e}")
    # Fallback defaults
    if args.dataset      is None: args.dataset      = 'cifar10'
    if args.rank         is None: args.rank         = 8
    if args.dsn_freq     is None: args.dsn_freq     = 1
    if args.dsn_strength is None: args.dsn_strength = 0.3
    main(args)