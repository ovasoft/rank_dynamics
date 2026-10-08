"""
probe_dynamics.py — Compute rank dynamics probes from saved epoch checkpoints.

Loads each epoch checkpoint and computes:
  Axis 2 (Spectral):     eRank, stable rank, nuclear norm ratio, full SVD spectrum
  Axis 3 (Optimization): spectral norm of weight update, Frobenius norm of update,
                         gradient effective rank (approximated from weight diff)
  Axis 1 (Parameter):    drift from initialization (Frobenius distance from epoch 0)
  Axis 4 (Functional):   CKA between layer representations across epochs,
                         and optionally between two runs (FR vs LR)

Results written to:
  <run_dir>/probes/probes.csv       — scalar metrics per epoch per layer per weight
  <run_dir>/probes/cka_self.csv     — CKA between consecutive epochs (within run)
  <run_dir>/probes/cka_cross.csv    — CKA between this run and a reference run (FR vs LR)
  <run_dir>/probes/svd/             — full singular value vectors (.pt files)

Usage:
    # Single run (within-run CKA only)
    python probing/probe_dynamics.py --run_dir outputs_cifar/cifar10/LR_r8 --dataset cifar10

    # Cross-run CKA (FR vs LR)
    python probing/probe_dynamics.py --run_dir outputs_cifar/cifar10/LR_r8 \\
                             --ref_run_dir outputs_cifar/cifar10/FR \\
                             --dataset cifar10
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, math, argparse, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from task_registry import load_config, get_task, LowRankLinear
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

EPS = 1e-10

# ── Spectral probes ───────────────────────────────────────────────────────────

def svd(W):
    return torch.linalg.svdvals(W.float())


def erank(S):
    S = S.clamp(min=EPS)
    p = S / S.sum()
    return math.exp(-(p * p.log()).sum().item())


def stable_rank(S):
    return (S.pow(2).sum() / S[0].pow(2).clamp(min=EPS)).item()


def nuclear_norm_ratio(S):
    return (S.sum() / S.pow(2).sum().sqrt().clamp(min=EPS)).item()


def alpha_req(S):
    S_np = S.numpy()
    n    = len(S_np)
    if n < 5 or S_np.min() <= 0:
        return float("nan")
    lo, hi  = max(0, int(0.05 * n)), max(2, int(0.99 * n))
    ranks   = np.arange(lo + 1, hi + 1, dtype=float)
    log_s   = np.log(S_np[lo:hi])
    log_r   = np.log(ranks)
    A       = np.column_stack([log_r, np.ones_like(log_r)])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_s, rcond=None)
    return float(-coeffs[0])


def spectral_probes(W):
    S = svd(W)
    return {
        "erank":  erank(S),
        "srank":  stable_rank(S),
        "nnr":    nuclear_norm_ratio(S),
        "alpha":  alpha_req(S),
        "sigma1": S[0].item(),
        "n_sv":   len(S),
    }, S

# ── Optimization probes ───────────────────────────────────────────────────────

def optimizer_probes(W_prev, W_curr):
    if W_prev is None:
        return {
            "spectral_norm_update": float("nan"),
            "frobenius_norm_update": float("nan"),
            "relative_update": float("nan"),
            "grad_erank_approx": float("nan"),
        }
    delta   = W_curr.float() - W_prev.float()
    S_delta = svd(delta)
    return {
        "spectral_norm_update":  S_delta[0].item(),
        "frobenius_norm_update": delta.norm(p="fro").item(),
        "relative_update":       delta.norm(p="fro").item() /
                                 W_prev.float().norm(p="fro").clamp(min=EPS).item(),
        "grad_erank_approx":     erank(S_delta),
    }

# ── Parameter space probes ────────────────────────────────────────────────────

def param_probes(W_init, W_curr):
    if W_init is None:
        return {"drift_from_init": float("nan"), "relative_drift": float("nan")}
    diff = W_curr.float() - W_init.float()
    return {
        "drift_from_init": diff.norm(p="fro").item(),
        "relative_drift":  diff.norm(p="fro").item() /
                           W_init.float().norm(p="fro").clamp(min=EPS).item(),
    }

# ── CKA ───────────────────────────────────────────────────────────────────────

def linear_cka(H_a, H_b):
    """
    Linear CKA between two representation matrices.
    H_a, H_b: (n_samples, d) — same n_samples, any d.
    Returns scalar in [0, 1]. 1 = identical representations up to rotation.
    """
    H_a = H_a.float() - H_a.float().mean(0, keepdim=True)
    H_b = H_b.float() - H_b.float().mean(0, keepdim=True)
    n   = H_a.shape[0]

    # HSIC(K, L) = ||H_a.T @ H_b||_F^2 / (n-1)^2
    hsic_ab = (H_a.T @ H_b).norm(p="fro").pow(2).item() / (n - 1) ** 2
    hsic_aa = (H_a.T @ H_a).norm(p="fro").pow(2).item() / (n - 1) ** 2
    hsic_bb = (H_b.T @ H_b).norm(p="fro").pow(2).item() / (n - 1) ** 2

    denom = math.sqrt(max(hsic_aa * hsic_bb, EPS))
    return hsic_ab / denom

# ── Model loading and forward hooks ──────────────────────────────────────────

def build_vit(n_classes, img_size, in_channels):
    """Re-import ViT from train_cifar without executing the full script."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "train_cifar",
        os.path.join(_ROOT, "src", "vision", "train_cifar.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ViT(img_size=img_size, in_channels=in_channels, n_classes=n_classes)


def get_hidden_states(model, loader, device, n_batches=20,
                      probe_hook='cls_token'):
    """
    Run probe inputs through model, collect output of every Block.
    Returns {block_idx: tensor (n_samples, d_model)} on CPU.

    probe_hook:
      'cls_token'  — ViT: position 0 of (B, N+1, E)
      'last_token' — GPT-2: last position of (B, T, E)
      'mean_token' — mean over sequence length
    """
    hidden = {}
    hooks  = []

    def make_hook(idx):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if probe_hook == 'cls_token':
                vec = h[:, 0, :].detach().cpu()
            elif probe_hook == 'last_token':
                vec = h[:, -1, :].detach().cpu()
            else:
                vec = h.mean(dim=1).detach().cpu()
            hidden.setdefault(idx, []).append(vec)
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    collected = 0
    with torch.no_grad():
        for batch in loader:
            if collected >= n_batches * loader.batch_size:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(device))
            collected += x.size(0)

    for h in hooks:
        h.remove()
    model.train()

    return {idx: torch.cat(tensors, dim=0) for idx, tensors in hidden.items()}


# GPT-2 Conv1D layer names whose weights are stored (in, out) in the HF model
# but may have been saved as (out, in) by materialize_low_rank before the
# was_conv1d fix was applied.
_GPT2_CONV1D_SUFFIXES = (
    'attn.c_attn.weight', 'attn.c_proj.weight',
    'mlp.c_fc.weight',    'mlp.c_proj.weight',
)


def _fix_conv1d_shapes(sd, model):
    """
    For GPT2LMHeadModel: if a checkpoint weight has shape (out, in) but the
    model expects (in, out) for a Conv1D layer, transpose it in-place in sd.
    Safe to call on any checkpoint — skips keys that already match.
    """
    if not hasattr(model, 'hf_model'):
        return sd   # vision model — no Conv1D layers
    hf = model.hf_model
    fixed = {}
    for k, v in sd.items():
        if any(k.endswith(sfx) for sfx in _GPT2_CONV1D_SUFFIXES):
            # Find expected shape from model
            try:
                parts  = k.split('.')
                target = hf
                for p in parts:
                    target = getattr(target, p)
                expected = target.shape
                if v.shape != expected and v.shape == expected[::-1]:
                    v = v.T   # transpose to match Conv1D expectation
            except AttributeError:
                pass
        fixed[k] = v
    return fixed


def load_model_from_checkpoint(ckpt_path, task_meta, cfg=None):
    """
    Load a model from checkpoint using the task registry.
    Handles both vision (ViT) and NLP (GPT-2/BabyLMWrapper).
    Handles full-rank, materialised LR, and un-materialised LR checkpoints.
    Handles Conv1D weight orientation mismatches from checkpoints saved before
    the was_conv1d fix.
    """
    from task_registry import get_task, replace_with_low_rank

    task_name = (cfg or {}).get('task', 'cifar10')
    task      = get_task(task_name)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    has_lr_keys = any(k.endswith(".A") or k.endswith(".B") for k in sd)
    rank = None
    if has_lr_keys:
        for k, v in sd.items():
            if k.endswith(".A"):
                rank = v.shape[1]
                break

    model = task['make_model'](cfg or {})
    if has_lr_keys and rank is not None:
        replace_with_low_rank(model, rank, verbose=False)  # in-place

    # Fix Conv1D weight orientation for checkpoints saved before was_conv1d fix
    sd = _fix_conv1d_shapes(sd, model)

    target = model.hf_model if hasattr(model, 'hf_model') else model
    target.load_state_dict(sd, strict=False)
    return model


def _detect_rank_from_checkpoint(ckpt_path):
    """
    Peek at a checkpoint's state dict for '.A'/'.B' keys (LowRankLinear
    factors) and return the rank if found, else None (materialized /
    full-rank checkpoint). train.py's save_checkpoint always calls
    materialize_low_rank before saving, so a saved checkpoint should never
    actually contain '.A'/'.B' keys regardless of what rank the run used --
    this content-based check replaces the previous run-directory-NAME-based
    detection (matching an 'LR_r' prefix), which risked building a
    LowRankLinear-replaced model shell that a materialized checkpoint's
    plain '.weight' keys could never populate (strict=False then silently
    drops those keys rather than erroring, leaving the shell at its random
    initialization instead of the trained weights).
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    for k, v in sd.items():
        if k.endswith(".A"):
            return v.shape[1]
    return None

# ── Weight extraction ─────────────────────────────────────────────────────────

def extract_weights(state_dict):
    weights = {}
    for k, v in state_dict.items():
        if v.ndim == 2 and "weight" in k:
            weights[k] = v.float().cpu()
        elif v.ndim == 4 and "weight" in k:
            weights[k] = v.float().cpu().flatten(1)
    return weights

# ── Probe set ─────────────────────────────────────────────────────────────────

DATASET_INFO = {
    "cifar10":      (datasets.CIFAR10,  10, 32, 3,
                     (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    "cifar100":     (datasets.CIFAR100,100, 32, 3,
                     (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)),
    "mnist":        (datasets.MNIST,   10, 28, 1, (0.1307,), (0.3081,)),
    "fashion_mnist":(datasets.FashionMNIST,10,28,1,(0.2860,),(0.3530,)),
    "svhn":         (datasets.SVHN,    10, 32, 3,
                     (0.4377,0.4438,0.4728), (0.1980,0.2010,0.1970)),
}


def make_probe_loader(dataset_name, cfg=None, n_samples=512, seed=0):
    """
    Build a fixed probe DataLoader.
    Vision tasks: fixed random subset of val split.
    NLP tasks: use task registry val loader, limited to n_samples sequences.
    Returns (loader, task_meta).
    """
    # NLP path
    if dataset_name not in DATASET_INFO and HAS_REGISTRY:
        from task_registry import get_task
        probe_cfg = dict(cfg or {})
        probe_cfg['batch_size'] = 64
        task = get_task(dataset_name)
        _, val_loader, meta = task['make_loaders'](probe_cfg)
        class _FixedLoader:
            def __init__(self, loader, n):
                self.loader     = loader
                self.n          = n
                self.batch_size = loader.batch_size
                self.dataset    = loader.dataset
            def __iter__(self):
                seen = 0
                for batch in self.loader:
                    if seen >= self.n: break
                    yield batch
                    seen += batch[0].size(0)
        return _FixedLoader(val_loader, n_samples), meta

    # Vision path
    cls, n_classes, img_size, n_ch, mean, std = DATASET_INFO[dataset_name]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset_name == "svhn":
        ds = cls("data", split="test", download=True, transform=tf)
    else:
        ds = cls("data", train=False, download=True, transform=tf)
    g       = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
    subset  = torch.utils.data.Subset(ds, indices)
    loader  = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
    meta    = {
        'n_classes': n_classes, 'img_size': img_size, 'in_channels': n_ch,
        'loss_fn': 'cross_entropy_cls', 'probe_hook': 'cls_token',
    }
    return loader, meta

# ── Main ──────────────────────────────────────────────────────────────────────

def probe_run(args):
    ckpt_dir  = os.path.join(args.run_dir, "checkpoints")
    probe_dir = os.path.join(args.run_dir, "probes")
    svd_dir   = os.path.join(probe_dir, "svd")
    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(svd_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {}
    if getattr(args, 'config', None) and HAS_REGISTRY:
        try:
            cfg = load_config(args.config)
        except Exception as e:
            print(f"Warning: could not load config: {e}")

    probe_loader, task_meta = make_probe_loader(
        args.dataset, cfg=cfg,
        n_samples=cfg.get('probe_n_samples', 512))
    probe_hook = task_meta.get('probe_hook', 'cls_token')
    print(f"Probe set: {args.dataset} | probe_hook={probe_hook}")

    # Find all checkpoints — supports both epoch_ and tokens_ naming
    def _ckpt_sort_key(name):
        # epoch_NN -> (0, NN), tokens_NNNNNNNNNN -> (1, tokens)
        if name.startswith("epoch_"):
            return (0, int(name.split("_")[1]))
        elif name.startswith("tokens_"):
            return (1, int(name.split("_")[1]))
        return (2, 0)

    epochs = sorted([
        d for d in os.listdir(ckpt_dir)
        if (d.startswith("epoch_") or d.startswith("tokens_")) and
           os.path.isfile(os.path.join(ckpt_dir, d, "model.pt"))
    ], key=_ckpt_sort_key)
    print(f"Found {len(epochs)} checkpoints in {ckpt_dir}")

    # Load reference run checkpoints if provided (for cross-run CKA)
    ref_ckpt_dir = (
        os.path.join(args.ref_run_dir, "checkpoints")
        if args.ref_run_dir else None
    )

    # ── CSV writers ──
    # Written as *.csv.partial and renamed on completion, so an interrupted
    # run never leaves a file that run_pipeline.sh would mistake for done.
    def _partial(name):
        return os.path.join(probe_dir, name + ".partial")

    # Weight probes
    wp_file   = open(_partial("probes.csv"), "w", newline="")
    wp_writer = csv.writer(wp_file)
    wp_writer.writerow([
        "epoch", "weight_name",
        "erank", "srank", "nnr", "alpha", "sigma1", "n_sv",
        "spectral_norm_update", "frobenius_norm_update",
        "relative_update", "grad_erank_approx",
        "drift_from_init", "relative_drift",
        "checkpoint_name",   # epoch_NN or tokens_NNNNNNNNNN
    ])

    # Within-run CKA (consecutive epochs)
    cka_self_file   = open(_partial("cka_self.csv"), "w", newline="")
    cka_self_writer = csv.writer(cka_self_file)
    cka_self_writer.writerow(["epoch", "layer", "cka_vs_prev", "cka_vs_epoch0", "checkpoint_name"])

    # Cross-run CKA (this run vs reference)
    cka_cross_file, cka_cross_writer = None, None
    if ref_ckpt_dir:
        cka_cross_file   = open(_partial("cka_cross.csv"), "w", newline="")
        cka_cross_writer = csv.writer(cka_cross_file)
        cka_cross_writer.writerow(["epoch", "layer", "cka_vs_ref", "checkpoint_name"])

    prev_weights    = None
    init_weights    = None
    prev_hidden     = None   # hidden states from previous epoch (for self-CKA)
    init_hidden     = None   # hidden states from epoch 0 (for drift-from-init CKA)

    # ── Build model architecture ONCE — reuse across all checkpoints ──────
    # Detect rank from the checkpoint's OWN state dict (presence of '.A'/'.B'
    # keys), not from the run directory name -- see _detect_rank_from_checkpoint
    # docstring for why the previous name-based approach was unsafe.
    from task_registry import get_task, replace_with_low_rank
    task_name  = (cfg or {}).get('task', 'cifar10')
    task       = get_task(task_name)

    first_ckpt_path = os.path.join(ckpt_dir, epochs[0], "model.pt") if epochs else None
    run_rank = _detect_rank_from_checkpoint(first_ckpt_path) if first_ckpt_path else None
    if run_rank is not None:
        print(f"  [detected] {args.run_dir}: checkpoint contains low-rank "
              f"factors (.A/.B), rank={run_rank}")
    else:
        print(f"  [detected] {args.run_dir}: materialized checkpoint "
              f"(plain .weight keys) -- building a full-rank model shell")

    probe_cfg_main = dict(cfg or {}); probe_cfg_main['_role'] = 'probe'
    base_model = task['make_model'](probe_cfg_main)
    if run_rank is not None:
        replace_with_low_rank(base_model, run_rank, verbose=True)  # in-place
        print(f"  [verify] Params after replace_with_low_rank: {sum(p.numel() for p in base_model.parameters()):,}")
    base_model = base_model.to(device)

    # Build reference model architecture once (if cross-run CKA requested)
    ref_base_model = None
    if ref_ckpt_dir:
        first_ref_ckpt_path = os.path.join(ref_ckpt_dir, epochs[0], "model.pt") if epochs else None
        ref_rank = _detect_rank_from_checkpoint(first_ref_ckpt_path) if first_ref_ckpt_path else None
        ref_run_name = os.path.basename(os.path.normpath(args.ref_run_dir))
        if ref_rank is not None:
            print(f"  [detected] {args.ref_run_dir}: checkpoint contains "
                  f"low-rank factors (.A/.B), rank={ref_rank}")
        else:
            print(f"  [detected] {args.ref_run_dir}: materialized checkpoint "
                  f"(plain .weight keys) -- building a full-rank model shell")
        # Build reference model shell — weights loaded from checkpoint each epoch
        ref_cfg = dict(cfg or {}); ref_cfg['_role'] = 'ref'
        ref_base_model = task['make_model'](ref_cfg)
        if ref_rank is not None:
            replace_with_low_rank(ref_base_model, ref_rank, verbose=False)  # in-place
            print(f"  [verify ref] Params after replace_with_low_rank: {sum(p.numel() for p in ref_base_model.parameters()):,}")
        ref_base_model = ref_base_model.to(device)
        print(f"Reference model ({ref_run_name}) architecture ready — "
              f"weights will be loaded from checkpoint each epoch")

    def _load_weights_into(model, ckpt_path):
        """Load checkpoint weights into existing model architecture."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = _fix_conv1d_shapes(sd, model)
        target = model.hf_model if hasattr(model, 'hf_model') else model
        target.load_state_dict(sd, strict=False)
        return model, sd

    for epoch_name in epochs:
        if epoch_name.startswith("epoch_"):
            epoch_idx = int(epoch_name.split("_")[1])
        else:
            epoch_idx = int(epoch_name.split("_")[1])
        ckpt_path = os.path.join(ckpt_dir, epoch_name, "model.pt")

        # ── Load checkpoint weights into pre-built model ──
        model, state_dict = _load_weights_into(base_model, ckpt_path)
        print(f"  [run] Loaded weights from {ckpt_path}", flush=True)
        curr_weights = extract_weights(state_dict)

        if init_weights is None:
            init_weights = curr_weights

        print(f"Epoch {epoch_idx:02d} | weights: {len(curr_weights)}", end="")

        # ── Weight probes (Axes 1, 2, 3) ──
        for name, W in curr_weights.items():
            spec, S = spectral_probes(W)
            W_prev  = prev_weights.get(name) if prev_weights else None
            opt     = optimizer_probes(W_prev, W)
            par     = param_probes(init_weights.get(name), W)

            wp_writer.writerow([
                epoch_idx, name,
                spec["erank"], spec["srank"], spec["nnr"],
                spec["alpha"], spec["sigma1"], spec["n_sv"],
                opt["spectral_norm_update"], opt["frobenius_norm_update"],
                opt["relative_update"], opt["grad_erank_approx"],
                par["drift_from_init"], par["relative_drift"],
                epoch_name,
            ])

            svd_path = os.path.join(
                svd_dir,
                f"{epoch_name}_{name.replace('.','_')}.pt"
            )
            torch.save(S.float(), svd_path)

        # ── CKA: get hidden states for this epoch ──
        curr_hidden = get_hidden_states(model, probe_loader, device)
        print(f" | layers: {len(curr_hidden)}", end="")

        # Within-run: vs previous epoch and vs epoch 0
        for layer_idx, H_curr in curr_hidden.items():
            cka_prev  = (linear_cka(prev_hidden[layer_idx], H_curr)
                         if prev_hidden else float("nan"))
            cka_init  = (linear_cka(init_hidden[layer_idx], H_curr)
                         if init_hidden else float("nan"))
            cka_self_writer.writerow([epoch_idx, layer_idx, cka_prev, cka_init, epoch_name])

        # Cross-run: vs reference run at same epoch
        if ref_ckpt_dir and cka_cross_writer:
            ref_ckpt_path = os.path.join(ref_ckpt_dir, epoch_name, "model.pt")
            if os.path.exists(ref_ckpt_path):
                ref_base_model, _ = _load_weights_into(ref_base_model, ref_ckpt_path)
                print(f"  [ref] Loaded weights from {ref_ckpt_path}", flush=True)
                ref_hidden = get_hidden_states(ref_base_model, probe_loader, device,
                                               probe_hook=probe_hook)
                for layer_idx, H_curr in curr_hidden.items():
                    H_ref = ref_hidden.get(layer_idx)
                    if H_ref is not None:
                        cka_val = linear_cka(H_curr, H_ref)
                        cka_cross_writer.writerow([epoch_idx, layer_idx, cka_val, epoch_name])
            else:
                print(f" | ref epoch {epoch_idx} not found", end="")

        print()  # newline after per-epoch status

        # Roll buffers
        prev_weights = curr_weights
        if init_hidden is None:
            init_hidden = curr_hidden
        prev_hidden  = curr_hidden

        for f in [wp_file, cka_self_file]:
            f.flush()
        if cka_cross_file:
            cka_cross_file.flush()

        del model

    wp_file.close()
    cka_self_file.close()
    if cka_cross_file:
        cka_cross_file.close()
    for f in [wp_file, cka_self_file, cka_cross_file]:
        if f:
            os.replace(f.name, f.name[:-len(".partial")])

    print(f"\nDone.")
    print(f"  Weight probes : {probe_dir}/probes.csv")
    print(f"  Self CKA      : {probe_dir}/cka_self.csv")
    if ref_ckpt_dir:
        print(f"  Cross CKA     : {probe_dir}/cka_cross.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default=None,
                   help="YAML config path (sets task/output_root defaults)")
    p.add_argument("--run_dir",     default=None,
                   help="Run directory to probe, e.g. outputs/cifar10/FR")
    p.add_argument("--dataset",     default=None,
                   choices=list(DATASET_INFO) + [None],
                   help="Dataset/task name (overrides config)")
    p.add_argument("--ref_run_dir", default=None,
                   help="Reference run for cross-run CKA")
    args = p.parse_args()
    # Apply config defaults
    if args.config:
        try:
            import yaml
            with open(args.config) as f_cfg:
                cfg = yaml.safe_load(f_cfg)
            if args.dataset is None:
                args.dataset = cfg.get("task", "cifar10")
            if args.run_dir is None:
                args.run_dir = cfg.get("output_root", "outputs")
        except Exception as e:
            print(f"Warning: could not load config: {e}")
    print('#'*10,args)
    if args.dataset is None:
        args.dataset = "cifar10"
    if args.run_dir is None:
        p.error("--run_dir is required (or provide --config)")
    probe_run(args)