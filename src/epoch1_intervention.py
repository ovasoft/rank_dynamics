"""
epoch1_intervention.py — Static initialisation interventions for epoch-1 divergence test.

Tests four LR variants after exactly one training epoch, measuring CKA against
the FR model's epoch-1 representations:

  baseline     : standard LR (rank 8, default init)
  drift_scaled : LR with learning rate scaled by DRIFT_SCALE (matches FR parameter drift)
  alpha_flat   : LR with singular value spectrum flattened to match FR's target alpha
  combined     : both drift scaling and alpha flattening

All four variants train for exactly one epoch from the same random seed.
FR model is loaded from its epoch-1 checkpoint for CKA reference.

Usage:
    python epoch1_intervention.py \
        --dataset cifar10 \
        --fr_ckpt  outputs_cifar/cifar10/FR/checkpoints/epoch_00/model.pt \
        --out_dir  outputs_cifar/cifar10/interventions
"""

import os, math, argparse, csv, copy, importlib.util, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from task_registry import load_config, get_task, replace_with_low_rank, LowRankLinear

# ── Constants ─────────────────────────────────────────────────────────────
RANK        = 8
DRIFT_SCALE = 2.3       # LR/FR first-epoch relative drift ratio from probes
TARGET_ALPHA = 0.43     # FR alpha at epoch 1 (from spectral summary)
LR_BASE     = 1e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP   = 1.0
SEED        = 0
N_PROBE     = 512       # number of images for CKA probe set

# ── Load train_cifar helpers ──────────────────────────────────────────────

def load_tc():
    spec = importlib.util.spec_from_file_location(
        "train_cifar",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_cifar.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Spectral flattening ───────────────────────────────────────────────────

def alpha_of(S):
    """Fit power-law exponent to singular values S (descending)."""
    n = len(S)
    if n < 5:
        return float('nan')
    lo, hi = max(0, int(0.05*n)), max(2, int(0.99*n))
    ranks  = torch.arange(lo+1, hi+1, dtype=torch.float32)
    log_s  = torch.log(S[lo:hi].clamp(min=1e-10))
    log_r  = torch.log(ranks)
    A      = torch.stack([log_r, torch.ones_like(log_r)], dim=1)
    coeffs, _, _, _ = torch.linalg.lstsq(A, log_s.unsqueeze(1))
    return float(-coeffs[0])


def flatten_spectrum(W, target_alpha):
    """
    Rescale the singular values of W so they follow a power law with
    target_alpha, preserving total Frobenius norm. Returns reshaped W.
    """
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = len(S)
    # Target: S_i ∝ i^{-target_alpha}
    ranks      = torch.arange(1, r+1, dtype=torch.float32, device=S.device)
    target_S   = ranks.pow(-target_alpha)
    target_S   = target_S * (S.norm() / target_S.norm())   # preserve Frobenius norm
    W_new      = (U * target_S.unsqueeze(0)) @ Vh
    return W_new.to(W.dtype)


def apply_alpha_flattening(model, target_alpha):
    """
    For every LowRankLinear in model, materialise W = A@B.T,
    flatten its spectrum to target_alpha, then re-factorise back
    into A, B via thin SVD so rank is preserved.
    """
    for module in model.modules():
        if type(module).__name__ == 'LowRankLinear':
            with torch.no_grad():
                W   = module.A @ module.B.T          # (out, in)
                W_f = flatten_spectrum(W, target_alpha)
                # Re-factorise: W_f = U * S * Vh, take rank-r truncation
                r   = module.A.shape[1]
                U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
                U_r  = U[:, :r]                      # (out, r)
                S_r  = S[:r]                          # (r,)
                Vh_r = Vh[:r, :]                      # (r, in)
                # A = U_r * sqrt(S_r),  B = Vh_r.T * sqrt(S_r)
                sqrt_S = S_r.sqrt()
                module.A.copy_((U_r  * sqrt_S.unsqueeze(0)))
                module.B.copy_((Vh_r * sqrt_S.unsqueeze(1)).T)
    return model

# ── CKA ───────────────────────────────────────────────────────────────────

def linear_cka(H_a, H_b):
    H_a = H_a.float() - H_a.float().mean(0, keepdim=True)
    H_b = H_b.float() - H_b.float().mean(0, keepdim=True)
    n   = H_a.shape[0]
    hsic_ab = (H_a.T @ H_b).norm(p='fro').pow(2).item() / (n-1)**2
    hsic_aa = (H_a.T @ H_a).norm(p='fro').pow(2).item() / (n-1)**2
    hsic_bb = (H_b.T @ H_b).norm(p='fro').pow(2).item() / (n-1)**2
    return hsic_ab / math.sqrt(max(hsic_aa * hsic_bb, 1e-10))


def get_representations(model, loader, device, n_samples,
                        probe_hook='cls_token'):
    """Extract token representation from every Block."""
    hidden = {}
    hooks  = []
    def make_hook(idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if probe_hook == 'last_token':
                vec = h[:, -1, :].detach().cpu()
            elif probe_hook == 'mean_token':
                vec = h.mean(dim=1).detach().cpu()
            else:
                vec = h[:, 0, :].detach().cpu()
            hidden.setdefault(idx, []).append(vec)
        return hook
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))
    model.eval()
    collected = 0
    with torch.no_grad():
        for batch in loader:
            if collected >= n_samples: break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))
            collected += x.size(0)
    for h in hooks:
        h.remove()
    model.train()
    return {i: torch.cat(t, 0)[:n_samples] for i, t in hidden.items()}

# ── One-epoch training ────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, device, lr, loss_fn='cross_entropy_cls'):
    total_steps  = len(train_loader)
    warmup_steps = max(1, int(total_steps * 0.05))
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()
    for step, batch in enumerate(train_loader):
        x = batch[0].to(device)
        y = batch[1].to(device)
        if step < warmup_steps:
            cur_lr = lr * step / max(1, warmup_steps)
        else:
            t      = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cur_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + math.cos(math.pi * t))
        for g in optimizer.param_groups:
            g['lr'] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
            if loss_fn == 'cross_entropy_cls':
                loss = F.cross_entropy(logits, y)
            else:
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    return model

# ── Evaluate accuracy ─────────────────────────────────────────────────────

def evaluate(model, loader, device, loss_fn='cross_entropy_cls'):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            if loss_fn == 'cross_entropy_cls':
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
            else:
                B, T, V = logits.shape
                total_loss += F.cross_entropy(
                    logits.reshape(B*T, V), y.reshape(B*T),
                    reduction='sum').item()
                total += B * T
    model.train()
    if loss_fn == 'cross_entropy_cls':
        return correct / max(1, total)
    return math.exp(min(total_loss / max(1, total), 20))

# ── Relative drift ────────────────────────────────────────────────────────

def relative_drift(model_init_sd, model_trained):
    """Mean relative Frobenius drift across all Linear/LowRankLinear weights."""
    drifts = []
    sd0    = model_init_sd
    sd1    = {k: v for k, v in model_trained.state_dict().items()}
    for k in sd0:
        if 'weight' not in k and '.A' not in k and '.B' not in k:
            continue
        w1    = sd1[k].float()
        w0    = sd0[k].float().to(w1.device)   # match device
        norm0 = w0.norm(p='fro').item()
        if norm0 > 1e-8:
            drifts.append((w1 - w0).norm(p='fro').item() / norm0)
    return sum(drifts) / len(drifts) if drifts else float('nan')

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
                parts  = k.split('.')
                t = hf
                for p in parts: t = getattr(t, p)
                if v.shape != t.shape and v.shape == t.shape[::-1]:
                    v = v.T
            except AttributeError:
                pass
        fixed[k] = v
    return fixed


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    torch.manual_seed(SEED)
    random.seed(SEED)

    # ── Load config and task ──────────────────────────────────────────
    cfg = {}
    if getattr(args, 'config', None):
        try:
            cfg = load_config(args.config)
        except Exception as e:
            print(f"Warning: could not load config: {e}")

    task_name    = args.dataset or cfg.get('task', 'cifar10')
    task         = get_task(task_name)
    rank         = cfg.get('primary_rank', RANK)
    drift_scale  = cfg.get('intervention_drift_scale',  DRIFT_SCALE)
    target_alpha = cfg.get('intervention_target_alpha', TARGET_ALPHA)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, meta = task['make_loaders'](cfg)
    loss_fn    = meta['loss_fn']
    probe_hook = meta.get('probe_hook', 'cls_token')

    # ── Probe set ─────────────────────────────────────────────────────
    from probe_dynamics import make_probe_loader
    probe_loader, _ = make_probe_loader(task_name, cfg=cfg, n_samples=N_PROBE)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load FR t* checkpoint for CKA reference ───────────────────────
    print(f'\nLoading FR t* checkpoint: {args.fr_ckpt}')
    fr_model = task['make_model'](dict(cfg, _role='ref'))
    fr_sd    = torch.load(args.fr_ckpt, map_location='cpu', weights_only=True)
    fr_sd    = _fix_conv1d_shapes(fr_sd, fr_model)
    target   = fr_model.hf_model if hasattr(fr_model, 'hf_model') else fr_model
    target.load_state_dict(fr_sd, strict=False)
    print(f"  [verify] FR params: {sum(p.numel() for p in fr_model.parameters()):,}")
    fr_model  = fr_model.to(device)
    fr_hidden = get_representations(fr_model, probe_loader, device, N_PROBE,
                                    probe_hook=probe_hook)
    fr_metric = evaluate(fr_model, val_loader, device, loss_fn)
    print(f'FR t* metric: {fr_metric:.4f}')

    # ── Define LR variants ────────────────────────────────────────────
    def fresh_lr():
        torch.manual_seed(SEED)
        m = task['make_model'](dict(cfg, _role='probe'))
        replace_with_low_rank(m, rank)  # in-place
        print(f"  Params after replace_with_low_rank: "
              f"{sum(p.numel() for p in m.parameters()):,}")
        return m

    variants = {
        'baseline':     (fresh_lr(), LR_BASE,               False),
        'drift_scaled': (fresh_lr(), LR_BASE * drift_scale,  False),
        'alpha_flat':   (fresh_lr(), LR_BASE,               True),
        'combined':     (fresh_lr(), LR_BASE * drift_scale,  True),
    }

    # ── Results storage ───────────────────────────────────────────────
    results_path = os.path.join(args.out_dir, 'epoch1_intervention.csv')
    f_out        = open(results_path, 'w', newline='')
    writer       = csv.writer(f_out)
    writer.writerow(['variant', 'val_acc', 'rel_drift',
                     'cka_L0', 'cka_L1', 'cka_L2', 'cka_L3',
                     'cka_L4', 'cka_L5', 'cka_L6', 'cka_L7', 'cka_L8',
                     'mean_cka'])

    print(f'\n{"variant":<16} {"val_acc":>8} {"rel_drift":>10} '
          f'{"CKA_L0":>8} {"CKA_L4":>8} {"CKA_L8":>8} {"mean_CKA":>10}')
    print('-' * 72)

    for name, (model, lr, do_alpha) in variants.items():
        # Save init state for drift computation
        init_sd = copy.deepcopy(model.state_dict())

        # Apply alpha flattening before training if requested
        if do_alpha:
            apply_alpha_flattening(model, TARGET_ALPHA)
            # Re-save init after flattening so drift is measured from flattened init
            init_sd = copy.deepcopy(model.state_dict())

        model = model.to(device)

        # Train one epoch
        model = train_one_epoch(model, train_loader, device, lr, loss_fn)

        # Metrics
        val_metric = evaluate(model, val_loader, device, loss_fn)
        drift     = relative_drift(init_sd, model)
        hidden    = get_representations(model, probe_loader, device, N_PROBE,
                                        probe_hook=probe_hook)
        cka_vals  = [linear_cka(hidden[l], fr_hidden[l]) for l in range(9)]
        mean_cka  = sum(cka_vals) / 9

        row = [name, f'{val_metric:.4f}', f'{drift:.4f}'] + \
              [f'{v:.4f}' for v in cka_vals] + [f'{mean_cka:.4f}']
        writer.writerow(row)
        f_out.flush()

        print(f'{name:<16} {val_metric:>8.4f} {drift:>10.4f} '
              f'{cka_vals[0]:>8.4f} {cka_vals[4]:>8.4f} '
              f'{cka_vals[8]:>8.4f} {mean_cka:>10.4f}')

    f_out.close()
    print(f'\nFR reference:')
    fr_cka_self = [1.0] * 9   # FR vs itself is 1.0 by definition
    print(f'  FR t* metric={fr_metric:.4f}  (checkpoint: {args.fr_ckpt})')
    print(f'\nResults: {results_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',   default=None,
                   help='YAML config path')
    p.add_argument('--dataset',  default=None)
    p.add_argument('--fr_ckpt',  default=None,
                   help='Path to FR epoch-1 checkpoint')
    p.add_argument('--out_dir',  default=None)
    args = p.parse_args()
    # Apply config defaults
    if args.config:
        try:
            import yaml
            with open(args.config) as f: cfg = yaml.safe_load(f)
            out_root = cfg.get('output_root', 'outputs')
            if args.dataset is None: args.dataset = cfg.get('task', 'cifar10')
            if args.fr_ckpt is None:
                args.fr_ckpt = os.path.join(
                    out_root, 'FR', 'checkpoints', 'epoch_00', 'model.pt')
            if args.out_dir is None:
                args.out_dir = os.path.join(out_root, 'interventions')
            # Override intervention constants from config
            import sys
            sys.modules[__name__]  # keep module ref
            iv_scale = cfg.get('intervention_drift_scale', None)
            iv_alpha = cfg.get('intervention_target_alpha', None)
            if iv_scale: globals()['DRIFT_SCALE']    = float(iv_scale)
            if iv_alpha: globals()['TARGET_ALPHA']   = float(iv_alpha)
        except Exception as e:
            print(f"Warning: could not load config: {e}")
    if args.dataset is None: args.dataset = 'cifar10'
    if args.fr_ckpt is None: p.error("--fr_ckpt required")
    if args.out_dir is None: args.out_dir = 'outputs/interventions'
    main(args)