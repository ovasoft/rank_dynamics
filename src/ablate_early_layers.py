"""
ablate_early_layers.py — Freeze-and-finetune ablation for early-layer bottleneck hypothesis.

Protocol:
  1. Load trained LR final checkpoint.
  2. Re-initialise blocks 0..k-1 as full-rank (fresh weights).
  3. Freeze blocks k..end, head, and embeddings.
  4. Fine-tune for FINETUNE_EPOCHS.
  5. Record best val metric.
  6. Sweep k in {0, 1, 2, 3, 4, n_layers} where k=0 is LR fine-tune baseline.

Supports both vision (ViT/CIFAR) and language (GPT-2/BabyLM) via task_registry.

Usage:
    python src/ablate_early_layers.py --config configs/cifar10.yaml
    python src/ablate_early_layers.py --config configs/babylm_strict_small.yaml \
        --lr_run_dir outputs/babylm_strict_small/LR_r8 \
        --out_dir outputs/babylm_strict_small/ablation
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, math, argparse, csv, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from task_registry import (
    load_config, get_task,
    replace_with_low_rank, LowRankLinear
)

# ── Hyperparameters ────────────────────────────────────────────────────────
FINETUNE_LR     = 1e-4
FINETUNE_MIN_LR = 1e-6
WARMUP_EPOCHS   = 1
GRAD_CLIP       = 1.0

# ── Conv1D shape fix ──────────────────────────────────────────────────────

_GPT2_CONV1D_SUFFIXES = (
    'attn.c_attn.weight', 'attn.c_proj.weight',
    'mlp.c_fc.weight',    'mlp.c_proj.weight',
)

def _fix_conv1d_shapes(sd, model):
    if not hasattr(model, 'hf_model'):
        return sd
    hf = model.hf_model
    fixed = {}
    for k, v in sd.items():
        if any(k.endswith(sfx) for sfx in _GPT2_CONV1D_SUFFIXES):
            try:
                parts = k.split('.')
                t = hf
                for p in parts: t = getattr(t, p)
                if v.shape != t.shape and v.shape == t.shape[::-1]:
                    v = v.T
            except AttributeError:
                pass
        fixed[k] = v
    return fixed


# ── Re-initialise a single Block as full-rank ────────────────────────────

def reinit_block_full_rank(block):
    """
    Replace any LowRankLinear in this block with fresh full-rank nn.Linear.
    For HF GPT-2 blocks, also replaces Conv1D with nn.Linear.
    New layers are created on the same device as the existing block weights.
    """
    # Determine the device of this block
    try:
        block_device = next(block.parameters()).device
    except StopIteration:
        block_device = torch.device('cpu')

    for name, module in list(block.named_modules()):
        cls_name = type(module).__name__
        if cls_name not in ('LowRankLinear', 'Conv1D'):
            continue
        parts  = name.split('.')
        parent = block
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child = parts[-1]
        if cls_name == 'LowRankLinear':
            in_f  = module.B.shape[0]
            out_f = module.A.shape[0]
        else:  # Conv1D: weight (in, out)
            in_f, out_f = module.weight.shape
        new_linear = nn.Linear(in_f, out_f, bias=False).to(block_device)
        nn.init.kaiming_uniform_(new_linear.weight, nonlinearity='linear')
        setattr(parent, child, new_linear)
    return block


# ── LR schedule ───────────────────────────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * t))


# ── Evaluate ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device, loss_fn='cross_entropy_cls'):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            if loss_fn == 'cross_entropy_cls':
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)
            else:
                B, T, V    = logits.shape
                total_loss += F.cross_entropy(
                    logits.reshape(B*T, V), y.reshape(B*T),
                    reduction='sum').item()
                total      += B * T
    model.train()
    if loss_fn == 'cross_entropy_cls':
        return correct / max(1, total)
    return math.exp(min(total_loss / max(1, total), 20))


# ── Fine-tune ─────────────────────────────────────────────────────────────

def finetune(model, train_loader, val_loader, device,
             n_epochs, peak_lr, min_lr, warmup_epochs,
             loss_fn='cross_entropy_cls', label=''):
    steps_per_epoch = len(train_loader)
    total_steps     = n_epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        metric = evaluate(model, val_loader, device, loss_fn)
        print(f"  {label}: no trainable params, metric={metric:.4f}")
        return metric

    optimizer    = AdamW(params, lr=peak_lr, weight_decay=0.05)
    higher_better = loss_fn == 'cross_entropy_cls'
    best_metric  = 0.0 if higher_better else float('inf')
    step         = 0

    for epoch in range(n_epochs):
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            lr   = get_lr(step, total_steps, warmup_steps, peak_lr, min_lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                if loss_fn == 'cross_entropy_cls':
                    loss = F.cross_entropy(logits, y)
                else:
                    B, T, V = logits.shape
                    loss    = F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            step += 1

        metric = evaluate(model, val_loader, device, loss_fn)
        if higher_better:
            best_metric = max(best_metric, metric)
        else:
            best_metric = min(best_metric, metric)
        print(f"  {label} epoch {epoch:2d} | metric={metric:.4f} "
              f"(best={best_metric:.4f})")

    return best_metric


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Config and task ───────────────────────────────────────────────
    cfg = {}
    if getattr(args, 'config', None):
        try:
            cfg = load_config(args.config)
        except Exception as e:
            print(f'Warning: could not load config: {e}')

    task_name     = args.dataset or cfg.get('task', 'cifar10')
    task          = get_task(task_name)
    rank          = args.rank or cfg.get('primary_rank', 8)
    finetune_epochs = args.finetune_epochs
    sweep_k       = cfg.get('ablation_k_values', [0, 1, 2, 3, 4, 9])

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn      = meta['loss_fn']
    higher_better = loss_fn == 'cross_entropy_cls'
    metric_col   = 'best_val_acc' if higher_better else 'best_val_ppl'

    # ── LR checkpoint ─────────────────────────────────────────────────
    lr_ckpt = os.path.join(args.lr_run_dir, 'final', 'model.pt')
    if not os.path.exists(lr_ckpt):
        lr_ckpt = os.path.join(args.lr_run_dir, 'best', 'model.pt')
    print(f'Loading LR checkpoint: {lr_ckpt}')

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'ablation_results.csv')
    results_f    = open(results_path, 'w', newline='')
    writer       = csv.writer(results_f)
    writer.writerow(['condition', 'k_reinit', metric_col])

    # ── Fresh model loader ─────────────────────────────────────────────
    def fresh_lr_model():
        # Build and load entirely on CPU, move to device last
        m  = task['make_model'](dict(cfg, _role='probe'))
        sd = torch.load(lr_ckpt, map_location='cpu', weights_only=True)
        # Always apply LR structure so block shapes match checkpoint
        replace_with_low_rank(m, rank)  # in-place
        print(f'  Params after replace_with_low_rank: '
              f'{sum(p.numel() for p in m.parameters()):,}')
        sd = _fix_conv1d_shapes(sd, m)
        target = m.hf_model if hasattr(m, 'hf_model') else m
        target.load_state_dict(sd, strict=False)
        # Move everything to device in one shot after all CPU operations
        return m.to(device)

    # ── Baseline ──────────────────────────────────────────────────────
    base_model  = fresh_lr_model()
    base_metric = evaluate(base_model, val_loader, device, loss_fn)
    print(f'\nLR baseline (no finetune): metric={base_metric:.4f}')
    writer.writerow(['lr_baseline', 0, f'{base_metric:.4f}'])
    results_f.flush()
    del base_model

    # ── Full fine-tune control ─────────────────────────────────────────
    print('\n── Full fine-tune control (all layers, LR weights) ──────────')
    ft_model = fresh_lr_model()
    for p in ft_model.parameters():
        p.requires_grad = True
    ft_metric = finetune(ft_model, train_loader, val_loader, device,
                         finetune_epochs, FINETUNE_LR, FINETUNE_MIN_LR,
                         WARMUP_EPOCHS, loss_fn=loss_fn, label='full_finetune')
    writer.writerow(['full_finetune', 'all', f'{ft_metric:.4f}'])
    results_f.flush()
    del ft_model

    if args.finetune_only:
        results_f.close()
        print(f'\nResults written to: {results_path}')
        return

    # ── k-sweep ───────────────────────────────────────────────────────
    for k in sweep_k:
        n_blocks = len(task['make_model'](dict(cfg, _quiet=True)).blocks)
        print(f'\n── k={k}: re-init blocks 0..{k-1}, '
              f'freeze blocks {k}..{n_blocks-1} ──')
        model = fresh_lr_model()

        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # Re-initialise and unfreeze blocks 0..k-1
        blocks = model.blocks
        for i in range(k):
            reinit_block_full_rank(blocks[i])
            for p in blocks[i].parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in model.parameters())
        print(f'  Trainable params: {n_trainable:,} / {n_total:,}')

        metric = finetune(model, train_loader, val_loader, device,
                          finetune_epochs, FINETUNE_LR, FINETUNE_MIN_LR,
                          WARMUP_EPOCHS, loss_fn=loss_fn, label=f'k{k}')

        writer.writerow([f'k{k}', k, f'{metric:.4f}'])
        results_f.flush()
        print(f'  k={k} best metric: {metric:.4f}')
        del model

    results_f.close()
    print(f'\nResults written to: {results_path}')
    print('\n── Summary ──────────────────────────────────────────────────')
    with open(results_path) as f:
        for line in f:
            print(' ', line.strip())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',          default=None)
    p.add_argument('--dataset',         default=None)
    p.add_argument('--rank',            type=int, default=None)
    p.add_argument('--lr_run_dir',      default=None)
    p.add_argument('--out_dir',         default=None)
    p.add_argument('--finetune_epochs', type=int, default=None)
    p.add_argument('--finetune_only',   action='store_true')
    args = p.parse_args()

    cfg = {}
    if args.config:
        try:
            import yaml
            with open(args.config) as f: cfg = yaml.safe_load(f)
            out_root     = cfg.get('output_root', 'outputs')
            primary_rank = cfg.get('primary_rank', 8)
            if args.dataset        is None: args.dataset        = cfg.get('task', 'cifar10')
            if args.rank           is None: args.rank           = primary_rank
            if args.lr_run_dir     is None:
                args.lr_run_dir = os.path.join(out_root, f'LR_r{primary_rank}')
            if args.out_dir        is None:
                args.out_dir = os.path.join(out_root, 'ablation')
            if args.finetune_epochs is None:
                args.finetune_epochs = cfg.get('ablation_finetune_epochs', 10)
        except Exception as e:
            print(f'Warning: could not load config: {e}')

    if args.dataset         is None: args.dataset         = 'cifar10'
    if args.rank            is None: args.rank            = 8
    if args.lr_run_dir      is None: p.error('--lr_run_dir required')
    if args.out_dir         is None: p.error('--out_dir required')
    if args.finetune_epochs is None: args.finetune_epochs = 10

    main(args)