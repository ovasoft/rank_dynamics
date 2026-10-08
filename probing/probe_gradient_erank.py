"""
probe_gradient_erank.py — Compute TRUE gradient effective rank from actual
gradients (not weight-diff approximations) for existing checkpoints.

Used for gradient eRank under DSN and for SVD distribution shapes of FR
vs LR gradients.

For each checkpoint in a run:
  1. Load model weights.
  2. Run a forward-backward pass on the fixed probe batch.
  3. Compute eRank of each weight matrix's gradient.
  4. Save the full gradient singular value distribution.

Outputs:
  <run_dir>/probes/grad_erank.csv         — scalar eRank per checkpoint per layer
  <run_dir>/probes/grad_svd/              — full gradient SVD spectra (.pt files)

Usage:
    # FR run
    python probing/probe_gradient_erank.py \\
        --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/FR

    # LR run
    python probing/probe_gradient_erank.py \\
        --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/LR_r8

    # DSN run (same command — works on any checkpoint directory)
    python probing/probe_gradient_erank.py \\
        --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/DSN_static_r8

    # Vision
    python probing/probe_gradient_erank.py \\
        --config configs/cifar10.yaml \\
        --run_dir outputs/cifar10/FR

After running on FR, LR, and all DSN variants, collate with:
    python analysis/collate_grad_erank.py --config configs/babylm_strict_small.yaml
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, math, argparse, csv
import torch
import torch.nn.functional as F
import numpy as np

try:
    from task_registry import load_config, get_task, replace_with_low_rank
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False
    raise ImportError("task_registry.py must be on the Python path")

EPS = 1e-10


# ── Effective rank ────────────────────────────────────────────────────────

def erank(S: torch.Tensor) -> float:
    """Exponential Shannon entropy of normalised singular values."""
    S = S.float().clamp(min=EPS)
    p = S / S.sum()
    return math.exp(-(p * p.log()).sum().item())


# ── Gradient extraction ───────────────────────────────────────────────────

def compute_gradient_erank(model, probe_loader, device, loss_fn, probe_hook,
                           n_batches=4, save_svd_dir=None, ckpt_label=None):
    """
    Run forward-backward on the probe batch and compute eRank of each
    weight matrix's gradient.

    Returns:
        records: list of dicts with keys
            {weight_name, grad_erank, grad_sigma1, grad_n_sv, grad_frobenius}
        Also saves full gradient SVD spectra to save_svd_dir if provided.
    """
    model.train()
    # Accumulate gradients over n_batches for stability
    model.zero_grad()
    total_tokens = 0

    for i, batch in enumerate(probe_loader):
        if i >= n_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        if loss_fn == 'cross_entropy_cls':
            loss = F.cross_entropy(logits, y)
        else:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        loss.backward()
        total_tokens += x.numel()

    records = []
    root = model.hf_model if hasattr(model, 'hf_model') else model

    # Collect all parameters once for LowRankLinear A/B reconstruction
    param_dict = dict(root.named_parameters())
    processed_lr_layers = set()

    for name, param in param_dict.items():
        if param.grad is None:
            continue
        if param.grad.ndim < 2:
            continue

        # LowRankLinear: reconstruct effective weight gradient
        # For W = A @ B.T: dL/dW_eff = dL/dA @ B.T + A @ dL/dB.T
        if name.endswith('.A'):
            layer_prefix = name[:-2]
            if layer_prefix in processed_lr_layers:
                continue
            b_name = layer_prefix + '.B'
            if b_name not in param_dict or param_dict[b_name].grad is None:
                continue
            A      = param.float().detach()
            B      = param_dict[b_name].float().detach()
            grad_A = param.grad.float().detach()
            grad_B = param_dict[b_name].grad.float().detach()
            G = grad_A @ B.T + A @ grad_B.T   # (out, in)
            processed_lr_layers.add(layer_prefix)
            weight_name = layer_prefix + '.weight'

        elif name.endswith('.weight'):
            if name[:-7] in processed_lr_layers:
                continue
            weight_name = name
            G = param.grad.float().detach()
            if G.ndim > 2:
                G = G.flatten(1)

        elif name.endswith('.B'):
            continue
        else:
            continue

        # Economy SVD
        try:
            S = torch.linalg.svdvals(G)
        except Exception as e:
            print(f"  SVD failed for {weight_name}: {e}")
            continue

        er = erank(S)
        records.append({
            'weight_name':    weight_name,
            'grad_erank':     er,
            'grad_sigma1':    S[0].item(),
            'grad_n_sv':      len(S),
            'grad_frobenius': G.norm(p='fro').item(),
        })

        if save_svd_dir and ckpt_label:
            svd_path = os.path.join(
                save_svd_dir,
                f"{ckpt_label}_{weight_name.replace('.', '_')}_grad.pt"
            )
            torch.save(S.cpu(), svd_path)

    model.zero_grad()
    model.eval()
    return records


# ── Checkpoint discovery ──────────────────────────────────────────────────

def discover_checkpoints(run_dir):
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints directory found at {ckpt_dir}")

    def sort_key(name):
        if name.startswith('epoch_'):
            return (0, int(name.split('_')[1]))
        elif name.startswith('tokens_'):
            return (1, int(name.split('_')[1]))
        return (2, 0)

    entries = sorted([
        d for d in os.listdir(ckpt_dir)
        if (d.startswith('epoch_') or d.startswith('tokens_'))
        and os.path.isfile(os.path.join(ckpt_dir, d, 'model.pt'))
    ], key=sort_key)

    return [(e, os.path.join(ckpt_dir, e, 'model.pt')) for e in entries]


# ── Model loading ─────────────────────────────────────────────────────────

def load_model_for_checkpoint(ckpt_path, task, cfg, device):
    """
    Load checkpoint weights into a model shell.
    Reuses a single architecture across checkpoints to avoid repeated
    instantiation overhead.

    Rank is detected from the CHECKPOINT'S OWN state dict (presence of
    '.A'/'.B' keys), not from the run directory name. train.py's
    save_checkpoint always calls materialize_low_rank before saving, so a
    saved checkpoint should never actually contain '.A'/'.B' keys
    regardless of what rank the run used -- detecting from the directory
    name instead risked building a LowRankLinear-replaced model whose
    '.A'/'.B' parameters a materialized checkpoint's plain '.weight' keys
    could never populate (strict=False then silently drops those keys
    rather than erroring, leaving the model at its random initialization
    instead of the trained weights).
    """
    from task_registry import replace_with_low_rank

    # _fix_conv1d_shapes imported from probe_dynamics if available
    try:
        from probe_dynamics import _fix_conv1d_shapes
    except ImportError:
        def _fix_conv1d_shapes(sd, model):
            return sd

    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    run_rank = None
    for k, v in sd.items():
        if k.endswith('.A'):
            run_rank = v.shape[1]
            break

    model = task['make_model'](dict(cfg, _role='probe'))
    if run_rank is not None:
        replace_with_low_rank(model, run_rank, verbose=False)
    model = model.to(device)

    sd = _fix_conv1d_shapes(sd, model)
    target = model.hf_model if hasattr(model, 'hf_model') else model
    target.load_state_dict(sd, strict=False)
    return model


# ── Probe loader ──────────────────────────────────────────────────────────

def make_probe_loader(task, cfg, n_samples=512):
    """Build a small fixed probe loader from the validation set."""
    probe_cfg = dict(cfg)
    probe_cfg['batch_size'] = 64
    _, val_loader, meta = task['make_loaders'](probe_cfg)

    class _FixedLoader:
        def __init__(self, loader, n):
            self.loader = loader
            self.n = n
            self.batch_size = loader.batch_size
        def __iter__(self):
            seen = 0
            for batch in self.loader:
                if seen >= self.n:
                    break
                yield batch
                seen += batch[0].size(0)

    return _FixedLoader(val_loader, n_samples), meta


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    task_name = cfg.get('task', 'cifar10')
    task = get_task(task_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Task: {task_name}  |  Run: {args.run_dir}")

    # Rank is detected per-checkpoint from each checkpoint's own state dict
    # (see load_model_for_checkpoint) rather than once from the run
    # directory name -- a materialized checkpoint never contains '.A'/'.B'
    # keys regardless of what rank the run used, so name-based detection
    # here would be unreliable.

    # Probe loader — fixed across all checkpoints
    print("Building probe loader...")
    probe_loader, meta = make_probe_loader(task, cfg, n_samples=args.n_samples)
    loss_fn    = meta.get('loss_fn', 'cross_entropy_lm')
    probe_hook = meta.get('probe_hook', 'last_token')
    print(f"Probe: loss_fn={loss_fn}, probe_hook={probe_hook}, "
          f"n_samples={args.n_samples}")

    # Output dirs
    probe_dir   = os.path.join(args.run_dir, 'probes')
    grad_svd_dir = os.path.join(probe_dir, 'grad_svd')
    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(grad_svd_dir, exist_ok=True)

    out_csv = os.path.join(probe_dir, 'grad_erank.csv')
    csv_file = open(out_csv, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'checkpoint_name', 'tokens_or_epoch',
        'weight_name', 'grad_erank', 'grad_sigma1',
        'grad_n_sv', 'grad_frobenius',
    ])

    checkpoints = discover_checkpoints(args.run_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    for ckpt_name, ckpt_path in checkpoints:
        # Parse token count or epoch index
        if ckpt_name.startswith('tokens_'):
            tok_or_ep = int(ckpt_name.split('_')[1])
        else:
            tok_or_ep = int(ckpt_name.split('_')[1])

        print(f"\nCheckpoint: {ckpt_name} ({tok_or_ep:,})", flush=True)

        model = load_model_for_checkpoint(
            ckpt_path, task, cfg, device)

        records = compute_gradient_erank(
            model, probe_loader, device,
            loss_fn=loss_fn,
            probe_hook=probe_hook,
            n_batches=args.n_grad_batches,
            save_svd_dir=grad_svd_dir if args.save_svd else None,
            ckpt_label=ckpt_name,
        )

        for r in records:
            writer.writerow([
                ckpt_name, tok_or_ep,
                r['weight_name'], r['grad_erank'],
                r['grad_sigma1'], r['grad_n_sv'],
                r['grad_frobenius'],
            ])
        csv_file.flush()

        n_layers_repr = len(set(
            '.'.join(r['weight_name'].split('.')[:3]) for r in records))
        mean_erank = sum(r['grad_erank'] for r in records) / max(1, len(records))
        print(f"  → {len(records)} weight matrices | "
              f"mean grad_erank={mean_erank:.1f}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nDone. Gradient eRank written to: {out_csv}")
    if args.save_svd:
        print(f"Gradient SVD spectra written to: {grad_svd_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compute true gradient eRank from actual gradients on '
                    'existing checkpoints (FR, LR, and DSN runs).')
    p.add_argument('--config',         required=True,
                   help='YAML config path')
    p.add_argument('--run_dir',        required=True,
                   help='Run directory, e.g. outputs/babylm_strict_small/FR')
    p.add_argument('--n_samples',      type=int, default=512,
                   help='Number of probe samples for gradient computation')
    p.add_argument('--n_grad_batches', type=int, default=4,
                   help='Number of probe batches to accumulate gradients over')
    p.add_argument('--save_svd',       action='store_true',
                   help='Save full gradient SVD spectra to probes/grad_svd/')
    main(p.parse_args())