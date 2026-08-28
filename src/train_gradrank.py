"""
train_gradrank.py — Direct gradient effective-rank reduction for R2-W1.

Neither reviewer specifically asked for GaLore; R2-W1's actual request was
to "intervene on gradient dimensionality directly." GaLore does this
indirectly, via low-rank projection inside the optimizer, periodic subspace
rotation, and full-rank decoupled weight decay -- and the latter two turned
out to confound the eRank measurement in this project's own GaLore ablation
(cumulative grad_erank_approx stayed high even with rotation disabled,
traced to decoupled weight decay acting on the full-rank weight rather than
anything the projector does).

This script instead directly targets what the paper's own metric measures:
effective rank (Roy and Vetterli, 2007), the exponential Shannon entropy of
the gradient matrix G's normalised singular value spectrum. At every step,
for each target weight matrix, the TRUE gradient is projected onto a FIXED
rank-r subspace before the optimizer step -- directly bounding the
gradient's rank to at most r.

DESIGN NOTE -- fixed projection, not oracle-adaptive SVD truncation: an
earlier version of this script recomputed a fresh top-r SVD of the
gradient every step. That is not a fair analogue of LR_r8's constraint:
top-r SVD truncation is the mathematically OPTIMAL rank-r approximation to
whatever the true gradient wants to be that step (Eckart-Young), which
effectively hands the model oracle knowledge of the best possible r
directions at every step -- a much more favorable condition than LR_r8's
actual constraint, which is a structural byproduct of the W=B@A
parameterization (confined to the tangent space of the rank-<=r manifold
at the CURRENT A, B, not adaptively re-optimized from scratch). This
script instead projects onto a FIXED random orthonormal subspace chosen
once at initialization and never updated -- removing the "always pick the
best directions" advantage while still hard-constraining every step's
gradient to rank <= r.

Target parameters are also excluded from decoupled weight decay: applying
decay to the full weight would inject a full-rank drift the projection
never sees, reintroducing the same confound diagnosed in the GaLore
ablation. LR_r8 avoids this because decay there acts on A/B directly,
which structurally keeps the effective weight rank-bounded; excluding
target params from decay here is the closest practical analogue.

Evaluation should be via probe_gradient_erank.py's own eRank formula on the
TRUE gradient -- exactly what this script manipulates, sidestepping the
weight-diff proxy (grad_erank_approx) confound entirely.

Reuses:
  - load_config, get_task, _is_replaceable       (task_registry.py)
  - get_lr_by_tokens, save_checkpoint, evaluate  (train.py)

New in this file:
  - make_fixed_projection(): builds a fixed random orthonormal projection
    basis once per target parameter (via QR of a random Gaussian matrix)
  - project_gradient_(): in-place projection of a gradient onto that fixed
    subspace
  - select_target_params(): same eligibility rule as LR_r8's factorization,
    so this experiment touches exactly the same matrices
  - a token-budget training loop with a post-backward projection hook and
    a zero-weight-decay parameter group for target params (train.py's loop
    hardcodes plain AdamW with a single weight-decay value and no hook)

Usage:
    python train_gradrank.py --config configs/babylm_strict_small.yaml \\
        --grad_rank 8
"""

import os, time, csv, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_registry import load_config, get_task, _is_replaceable
from train import get_lr_by_tokens, save_checkpoint, evaluate


# ── Fixed-subspace gradient projection ──────────────────────────────────────
#
# IMPORTANT DESIGN NOTE: earlier versions of this script recomputed a fresh
# top-r SVD of the gradient every step. That is NOT a fair analogue of
# LR_r8's constraint: top-r SVD truncation is the mathematically OPTIMAL
# rank-r approximation to whatever the true gradient wants to be that step
# (Eckart-Young), effectively giving the model oracle knowledge of the best
# possible r directions at every single step. LR_r8's low-rank-ness is a
# structural byproduct of the W=B@A parameterization instead -- the update
# is confined to the tangent space of the rank-<=r manifold at the CURRENT
# (A, B), which is neither adaptively optimal nor fixed in the way an SVD
# recomputed from scratch each step would be.
#
# The fix used here: project onto a FIXED, non-adaptive random orthonormal
# subspace, chosen once at initialization and never updated. This removes
# the "always pick the best r directions" advantage while still hard-
# constraining every step's gradient to rank <= r, and is a closer (if
# still imperfect) analogue of a genuine structural rank constraint.

def make_fixed_projection(grad_shape, rank, device, generator=None):
    """
    Build a fixed (m, r) or (r, n) random orthonormal projection matrix for
    a gradient of shape (m, n), chosen once via QR of a random Gaussian
    matrix and never updated afterward.
    """
    m, n = grad_shape
    # Project the smaller dimension's side for efficiency, matching
    # GaLore's own convention of projecting whichever side is larger.
    if m >= n:
        rand = torch.randn(n, rank, device=device, generator=generator)
        V, _ = torch.linalg.qr(rand)   # (n, rank), orthonormal columns
        return V
    else:
        rand = torch.randn(m, rank, device=device, generator=generator)
        U, _ = torch.linalg.qr(rand)   # (m, rank), orthonormal columns
        return U


def project_gradient_(grad, proj):
    """
    In-place projection of `grad` (m, n) onto the fixed subspace spanned by
    `proj`'s columns, applied to whichever side matches proj's row count.
    Bounds grad's rank to at most proj.shape[1] without ever recomputing
    the subspace from the current gradient's own content.
    """
    if grad.ndim != 2:
        return grad
    m, n = grad.shape
    if proj.shape[0] == n:
        # proj is (n, rank): project columns -> grad @ proj @ proj.T
        result = (grad @ proj) @ proj.t()
    else:
        # proj is (m, rank): project rows -> proj @ proj.T @ grad
        result = proj @ (proj.t() @ grad)
    grad.copy_(result.to(grad.dtype))
    return grad


def select_target_params(model):
    """
    Same eligibility rule as task_registry._is_replaceable, so this
    experiment truncates gradients for exactly the matrices the paper's
    LR condition factorizes (attention Q/K/V/O + both feed-forward
    projections) -- keeping the comparison apples-to-apples with LR_r8.
    """
    root = model.hf_model if hasattr(model, 'hf_model') else model
    targets = []
    for name, module in root.named_modules():
        if not _is_replaceable(name, module):
            continue
        for pname, p in module.named_parameters(recurse=False):
            if pname == 'weight':
                targets.append(p)
    return targets


# ── Training loop ────────────────────────────────────────────────────────────
# Mirrors train.train_token_budget_mode's checkpoint/LR schedule exactly
# (reusing get_lr_by_tokens, save_checkpoint, evaluate directly). The only
# structural addition is the post-backward, pre-step gradient truncation
# hook -- train.py's loop has no such hook and hardcodes plain AdamW, so
# this variant is needed rather than reusable as-is.

def train_gradrank_token_budget(model, train_loader, val_loader, device, loss_fn,
                                cfg, run_dir, log, grad_rank, seed=0):
    token_budget  = int(cfg.get('token_budget', 10_000_000))
    warmup_tokens = int(cfg.get('warmup_tokens', 500_000))
    peak_lr       = cfg.get('lr', 3e-4)
    min_lr        = cfg.get('lr_min', 1e-6)
    weight_decay  = cfg.get('weight_decay', 0.1)
    grad_clip     = cfg.get('grad_clip', 1.0)

    ckpt_tokens = sorted(int(t) for t in cfg.get('checkpoint_tokens', [token_budget]))
    next_ckpt = 0

    target_params = select_target_params(model)
    print(f"GradRank targets: {len(target_params)} weight matrices "
         f"(rank={grad_rank})")

    # Fixed projection basis per target param -- built once, never updated.
    # Using a dedicated generator so the projection choice is reproducible
    # but independent of the model-init/data-shuffling RNG stream.
    proj_generator = torch.Generator(device=device)
    proj_generator.manual_seed(seed + 12345)
    projections = {
        id(p): make_fixed_projection(tuple(p.shape), grad_rank, device,
                                     generator=proj_generator)
        for p in target_params
    }

    # Target params get NO weight decay: decoupled decay applied to the full
    # weight would inject a full-rank drift the gradient projection never
    # sees, reintroducing exactly the confound diagnosed in the GaLore
    # ablation. LR_r8 avoids this because decay there acts on A/B directly,
    # which structurally keeps the effective weight rank-bounded; excluding
    # target params from decay here is the closest practical analogue.
    target_ids = set(id(p) for p in target_params)
    other_params = [p for p in model.parameters() if id(p) not in target_ids]
    optimizer = torch.optim.AdamW([
        {'params': target_params, 'weight_decay': 0.0},
        {'params': other_params, 'weight_decay': weight_decay},
    ], lr=peak_lr)

    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    best_metric = 0.0 if loss_fn == 'cross_entropy_cls' else float('inf')
    tokens_seen, step, t0 = 0, 0, time.time()
    model.train()

    print(f"[GradRank] token_budget={token_budget:,} rank={grad_rank} "
         f"(fixed projection subspace, target params excluded from weight decay)")

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

            # Direct gradient effective-rank reduction via a FIXED
            # projection basis (see module docstring for why fixed, not
            # oracle-adaptive SVD).
            with torch.no_grad():
                for p in target_params:
                    if p.grad is not None:
                        project_gradient_(p.grad, projections[id(p)])

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
    print(f"Device: {device}  |  Task: {cfg['task']}  |  GradRank rank={args.grad_rank}")

    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn = meta['loss_fn']

    train_cfg = dict(cfg); train_cfg['_role'] = 'train'
    model = task['make_model'](train_cfg)   # FULL-RANK -- no replace_with_low_rank
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters (full-rank, gradient truncated only): {n_params:,}")

    # Naming: avoid the 'LR_r' prefix and the bare '_r<N>' substring the
    # existing probe scripts' name-based rank detection matches (which
    # would otherwise call replace_with_low_rank on a checkpoint that has
    # no .A/.B keys -- the same bug class flagged for GaLore/ReLoRA).
    run_dir = args.run_dir or os.path.join(
        cfg['output_root'], f'GradTrunc{args.grad_rank}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    log_path = os.path.join(run_dir, 'log.csv')
    log_file = open(log_path, 'w', newline='')
    log = csv.writer(log_file)
    metric_name = 'val_acc' if loss_fn == 'cross_entropy_cls' else 'val_ppl'
    log.writerow(['checkpoint_idx', 'tokens_seen', 'tokens_label',
                 'val_loss', metric_name, f'best_{metric_name}', 'elapsed_s'])

    best = train_gradrank_token_budget(
        model, train_loader, val_loader, device, loss_fn, cfg, run_dir, log,
        grad_rank=args.grad_rank, seed=args.seed)

    save_checkpoint(model, os.path.join(run_dir, 'final'))
    log_file.close()
    print(f'\nDone. Best {metric_name}={best:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--grad_rank', type=int, default=8,
                  help='Rank to truncate the true gradient to at every step '
                      '(match LR_r8 for direct comparison)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run_dir', default=None)
    main(p.parse_args())