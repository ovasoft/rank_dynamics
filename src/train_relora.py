"""
train_relora.py — ReLoRA comparison for R2-W1 (Reviewer 2, Weakness 1).

R2-W1's second suggested discriminating test: "use methods that
periodically expand the update subspace, or use ReLoRA." ReLoRA (Lialin
et al. 2024) trains a low-rank adapter on top of a base weight,
periodically MERGES the adapter into the base and reinitialises a fresh
adapter — recovering full-rank expressivity across many merge cycles
while each individual gradient step is low-rank. If periodically
expanding the update subspace measurably narrows the FR-LR CKA gap or
slows exhaustion-type stabilization relative to static LR_r8, that is a
second, independent line of evidence for gradient-subspace restriction
as the operative mechanism (rather than the static weight factorization
per se) — and, since it's an intervention rather than a passive
correlation, it also speaks to R1-W2's causality concern.

Uses HuggingFace `peft` (pip install peft) for the LoRA layer
implementation and merge_and_unload() — the adapter math and merge
logic (W_eff = W_base + B@A, reconstructed and written back into a plain
Linear/Conv1D) are NOT reimplemented here. This file only adds the
ReLoRA-specific orchestration peft doesn't provide: periodic
merge-and-reinit, optimizer-state reset, and LR re-warmup per cycle.

Reuses:
  - load_config, get_task                        (task_registry.py)
  - get_lr_by_tokens, evaluate                    (train.py)
    (save_checkpoint is NOT reused as-is — see save_relora_checkpoint
    below for why a thin variant was necessary)

Usage:
    pip install peft
    python train_relora.py --config configs/babylm_strict_small.yaml \\
        --rank 8 --merge_every_tokens 5000000
"""

import os, copy, time, csv, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_registry import load_config, get_task
from train import get_lr_by_tokens, evaluate

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError("peft is required for this script: pip install peft") from e

# Same target-module family task_registry._is_replaceable selects for LR
# training (attention Q/K/V/O + both feed-forward projections), matched
# by suffix so the comparison targets the same matrices in both
# conditions. peft's LoraLayer handles HF Conv1D (GPT-2) transparently.
TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]


# ── Thin interface adapter ──────────────────────────────────────────────────
# train.py's evaluate()/loss code expects model(x) to return a logits
# tensor directly (this is how BabyLMWrapper.forward is written). A
# peft-wrapped GPT2LMHeadModel returns a CausalLMOutput object instead,
# so a one-line forward() adapter is needed to reuse evaluate() unchanged
# rather than duplicating its loss/perplexity logic here.

class HFLogitsAdapter(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, idx):
        return self.hf_model(idx).logits

    def train(self, mode=True):
        self.hf_model.train(mode)
        return self

    def eval(self):
        self.hf_model.eval()
        return self


# ── LoRA wrap / merge-and-reinit ─────────────────────────────────────────────

def wrap_with_lora(hf_model, rank, alpha=None, dropout=0.0):
    """Wrap hf_model's target layers with a fresh LoRA adapter via peft."""
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank,   # alpha=rank -> scale factor 1.0, matching
                                    # the paper's LowRankLinear (no extra scaling)
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    return get_peft_model(hf_model, cfg)


def merge_and_reinit(peft_model, rank, alpha=None, dropout=0.0):
    """
    ReLoRA's core operation, built entirely on peft primitives: merge the
    current adapter into the base weights (peft's merge_and_unload — no
    manual A@B.T reconstruction here), then wrap the merged model with a
    freshly initialised adapter of the same rank.
    """
    merged = peft_model.merge_and_unload()
    return wrap_with_lora(merged, rank, alpha=alpha, dropout=dropout)


def save_relora_checkpoint(peft_model, path):
    """
    Save a checkpoint compatible with the existing probing pipeline
    (probe_dynamics.py / probe_gradient_erank.py), which expects a plain
    GPT2LMHeadModel state dict with no LoRA-specific keys.

    Merges a DEEP COPY of the current adapter into the base weights (via
    peft's merge_and_unload — reused, not reimplemented) so the live
    training adapter is undisturbed, then saves the merged copy's state
    dict directly. train.py's save_checkpoint() is not reused here
    because it calls task_registry.materialize_low_rank(), which looks
    for LowRankLinear instances specifically and would silently no-op on
    a peft-wrapped model, saving LoRA adapter weights instead of a merged
    full-rank checkpoint.
    """
    os.makedirs(path, exist_ok=True)
    merged_copy = copy.deepcopy(peft_model).merge_and_unload().cpu()
    torch.save(merged_copy.state_dict(), os.path.join(path, 'model.pt'))


# ── Training loop ────────────────────────────────────────────────────────────

def train_relora_token_budget(peft_model, train_loader, val_loader, device, loss_fn,
                              cfg, run_dir, log, rank, merge_every_tokens,
                              alpha, dropout):
    token_budget  = int(cfg.get('token_budget', 10_000_000))
    warmup_tokens = int(cfg.get('warmup_tokens', 500_000))
    peak_lr       = cfg.get('lr', 3e-4)
    min_lr        = cfg.get('lr_min', 1e-6)
    weight_decay  = cfg.get('weight_decay', 0.1)
    grad_clip     = cfg.get('grad_clip', 1.0)

    ckpt_tokens = sorted(int(t) for t in cfg.get('checkpoint_tokens', [token_budget]))
    next_ckpt = 0

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=peak_lr,
                                  weight_decay=weight_decay)

    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    tokens_seen, step, t0 = 0, 0, time.time()
    tokens_since_merge = 0
    n_merges = 0
    peft_model.train()

    print(f"[ReLoRA] token_budget={token_budget:,} rank={rank} "
          f"merge_every_tokens={merge_every_tokens:,}")

    while tokens_seen < token_budget:
        for batch in train_loader:
            if tokens_seen >= token_budget:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            cur_tokens = x.numel()

            # LR schedule restarts warmup relative to the CURRENT merge
            # cycle, per the ReLoRA paper's prescription — a freshly
            # reinitialised adapter needs its own warmup, not a
            # continuation of the global schedule.
            cur_lr = get_lr_by_tokens(tokens_since_merge, merge_every_tokens,
                                      min(warmup_tokens, merge_every_tokens // 4),
                                      peak_lr, min_lr)
            for g in optimizer.param_groups:
                g['lr'] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = peft_model(x).logits
                if loss_fn == 'cross_entropy_cls':
                    loss = F.cross_entropy(logits, y)
                else:
                    B, T, V = logits.shape
                    loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(peft_model.parameters(), grad_clip)
            optimizer.step()

            tokens_seen += cur_tokens
            tokens_since_merge += cur_tokens
            step += 1

            if step % 100 == 0:
                pct = tokens_seen / token_budget * 100
                print(f'tokens {tokens_seen:>10,} ({pct:5.1f}%) | '
                      f'loss {loss.item():.4f} | lr {cur_lr:.2e} | '
                      f'merges={n_merges} | {time.time()-t0:.0f}s')

            # ── Merge-and-reinit cycle ──
            if tokens_since_merge >= merge_every_tokens and tokens_seen < token_budget:
                peft_model = merge_and_reinit(peft_model, rank, alpha=alpha,
                                              dropout=dropout).to(device)
                # Optimizer state reset — stale Adam moments from the old
                # adapter would otherwise bias early post-merge steps
                # toward the previous subspace (ReLoRA paper, Sec. on
                # optimizer reset).
                optimizer = torch.optim.AdamW(peft_model.parameters(), lr=peak_lr,
                                              weight_decay=weight_decay)
                tokens_since_merge = 0
                n_merges += 1
                print(f'  [merge-and-reinit #{n_merges}] @ {tokens_seen:,} tokens')

            # ── Checkpoint schedule (unchanged from FR/LR/GaLore runs) ──
            if next_ckpt < len(ckpt_tokens) and tokens_seen >= ckpt_tokens[next_ckpt]:
                t_label = ckpt_tokens[next_ckpt]
                # Evaluate via a thin logits-adapter so evaluate() (which
                # expects model(x) -> logits tensor) can be reused as-is.
                # This does NOT merge the adapter — merge_and_unload on
                # the live model would remove it from ongoing training;
                # HFLogitsAdapter just exposes .logits from the peft
                # model's normal forward pass.
                eval_wrapper = HFLogitsAdapter(peft_model)
                val_loss, val_metric = evaluate(eval_wrapper, val_loader, device, loss_fn)
                is_best = (val_metric < best_metric if loss_fn != 'cross_entropy_cls'
                          else val_metric > best_metric)
                if is_best:
                    best_metric = val_metric
                    save_relora_checkpoint(peft_model, os.path.join(run_dir, 'best'))
                ckpt_name = f'tokens_{t_label:010d}'
                save_relora_checkpoint(peft_model, os.path.join(run_dir, 'checkpoints', ckpt_name))
                print(f'  \u2713 checkpoint @ {tokens_seen:,} tokens | '
                      f'val_loss={val_loss:.4f} | {metric_name}={val_metric:.4f} | '
                      f'best={best_metric:.4f}')
                log.writerow([next_ckpt, tokens_seen, f'{tokens_seen:,}',
                              f'{val_loss:.4f}', f'{val_metric:.4f}',
                              f'{best_metric:.4f}', f'{time.time()-t0:.0f}'])
                next_ckpt += 1

    return best_metric, peft_model


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    cfg  = load_config(args.config)
    task = get_task(cfg['task'])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Task: {cfg['task']}  |  ReLoRA rank={args.rank}")

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    train_cfg = dict(cfg); train_cfg['_role'] = 'train'
    model_wrapper = task['make_model'](train_cfg)   # BabyLMWrapper(GPT2LMHeadModel)
    hf_model = model_wrapper.hf_model if hasattr(model_wrapper, 'hf_model') else model_wrapper

    peft_model = wrap_with_lora(hf_model, args.rank, alpha=args.alpha,
                               dropout=args.dropout).to(device)
    peft_model.print_trainable_parameters()

    merge_every_tokens = args.merge_every_tokens or (
        int(cfg.get('token_budget', 10_000_000)) // 10)  # default: 10 merge cycles

    run_dir = args.run_dir or os.path.join(
        cfg['output_root'], f'ReLoRA_r{args.rank}')
    # Note: unlike GaLore, ReLoRA's checkpoints ARE genuinely low-rank-
    # structured during training (merged back to full-rank weights only
    # at merge points and at checkpoint-save time via
    # save_relora_checkpoint's merge-on-copy), so a '_r{rank}' name here
    # does not create the same ambiguity flagged for GaLore: saved
    # checkpoints are plain full-rank state dicts either way (post-merge),
    # so probe_gradient_erank.py's name-based rank detection would try to
    # call replace_with_low_rank on an already-full-rank checkpoint if
    # this ever gets pattern-matched as 'LR_r{N}'. Recommend passing
    # --run_dir explicitly with a name that avoids the 'LR_r' prefix
    # pattern, e.g. outputs/.../ReLoRA_r8, which starts with 'ReLoRA' not
    # 'LR_r' and is therefore safe under both probe_dynamics.py (state-
    # dict-based, unaffected) and probe_gradient_erank.py (name-based,
    # checks for 'LR_r' prefix specifically, not just '_r').
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}  |  merge_every_tokens={merge_every_tokens:,}")

    log_path = os.path.join(run_dir, 'log.csv')
    log_file = open(log_path, 'w', newline='')
    log = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    log.writerow(['checkpoint_idx', 'tokens_seen', 'tokens_label',
                  'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])

    best, final_peft_model = train_relora_token_budget(
        peft_model, train_loader, val_loader, device, loss_fn, cfg, run_dir, log,
        rank=args.rank, merge_every_tokens=merge_every_tokens,
        alpha=args.alpha, dropout=args.dropout)

    save_relora_checkpoint(final_peft_model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--rank', type=int, default=8,
                   help='LoRA adapter rank — match LR_r8 for direct comparison')
    p.add_argument('--merge_every_tokens', type=int, default=None,
                   help='Tokens between merge-and-reinit cycles '
                        '(default: token_budget // 10, i.e. 10 cycles)')
    p.add_argument('--alpha', type=float, default=None,
                   help='LoRA alpha (default: = rank, i.e. scale factor 1.0)')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run_dir', default=None)
    main(p.parse_args())