"""
probe_mutual_knn.py — Mutual k-nearest-neighbor representational alignment
(Huh et al. 2024, "The Platonic Representation Hypothesis"); paper Appendix J.1.

Checks whether the CKA findings hold under mutual kNN alignment, a different notion of representational similarity than
CKA: where CKA measures global linear-subspace correspondence between
two representation spaces, mutual kNN measures whether two models agree
on which points are near each other LOCALLY -- for each point, whether
its k nearest neighbors in one model's representation space are also its
k nearest neighbors in the other's. This is less sensitive to global
rotation/scaling than CKA and is the primary alignment metric used in
the Platonic Representation Hypothesis's cross-model/cross-modality
convergence claims.

Reuses:
  - get_hidden_states, make_probe_loader, load_model_from_checkpoint,
    load_config, HAS_REGISTRY                       (probe_dynamics.py)

New in this file:
  - mutual_knn_alignment(): the actual metric computation
  - discover_checkpoints(): checkpoint discovery (mirrors
    probe_gradient_erank.py's version, needed since probe_dynamics.py's
    own discovery logic is embedded inside probe_run() rather than
    exposed as a standalone function)
  - a cross-run probing loop mirroring probe_dynamics.py's cka_cross
    computation, writing mutual_knn_cross.csv instead

Usage:
    python probing/probe_mutual_knn.py --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/LR_r8 \\
        --ref_run_dir outputs/babylm_strict_small/FR \\
        --k 10
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, argparse, csv
import torch
import torch.nn.functional as F

from probe_dynamics import (
    get_hidden_states, make_probe_loader, load_model_from_checkpoint,
    load_config,
)


# ── Mutual kNN alignment ─────────────────────────────────────────────────

def mutual_knn_alignment(H_a, H_b, k=10):
    """
    Mutual k-NN alignment (Huh et al. 2024) between two representation
    matrices H_a, H_b of shape (n_samples, d_a) and (n_samples, d_b) --
    same n_samples (same probe inputs), arbitrary/differing feature dims.

    For each sample i, finds its k nearest neighbors (by cosine
    similarity, excluding itself) in H_a and separately in H_b, then
    computes the fraction of those neighbor sets that overlap. Returns
    the mean overlap fraction across all samples, in [0, 1]. 1.0 = the
    two representation spaces agree perfectly on local neighborhood
    structure; k/(n-1) is the chance level for unrelated spaces.
    """
    n = H_a.shape[0]
    k = min(k, n - 1)

    H_a = F.normalize(H_a.float(), dim=1)
    H_b = F.normalize(H_b.float(), dim=1)

    sim_a = H_a @ H_a.T
    sim_b = H_b @ H_b.T

    diag_mask = torch.eye(n, dtype=torch.bool, device=sim_a.device)
    sim_a = sim_a.masked_fill(diag_mask, float('-inf'))
    sim_b = sim_b.masked_fill(diag_mask, float('-inf'))

    topk_a = sim_a.topk(k, dim=1).indices  # (n, k)
    topk_b = sim_b.topk(k, dim=1).indices  # (n, k)

    overlaps = []
    for i in range(n):
        set_a = set(topk_a[i].tolist())
        set_b = set(topk_b[i].tolist())
        overlaps.append(len(set_a & set_b) / k)

    return sum(overlaps) / n


# ── Checkpoint discovery ──────────────────────────────────────────────────
# Mirrors probe_gradient_erank.py's discover_checkpoints -- probe_dynamics.py
# has equivalent logic but embedded inside probe_run() rather than exposed
# as a standalone, reusable function.

def _ckpt_sort_key(name):
    if name.startswith("epoch_"):
        return (0, int(name.split("_")[1]))
    elif name.startswith("tokens_"):
        return (1, int(name.split("_")[1]))
    return (2, 0)


def discover_checkpoints(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    entries = sorted([
        d for d in os.listdir(ckpt_dir)
        if (d.startswith("epoch_") or d.startswith("tokens_"))
        and os.path.isfile(os.path.join(ckpt_dir, d, "model.pt"))
    ], key=_ckpt_sort_key)
    return [(e, os.path.join(ckpt_dir, e, "model.pt")) for e in entries]


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_loader, task_meta = make_probe_loader(
        args.dataset or cfg.get('task'), cfg=cfg,
        n_samples=cfg.get('probe_n_samples', 512))
    probe_hook = task_meta.get('probe_hook', 'last_token')
    print(f"Probe set: {args.dataset or cfg.get('task')} | probe_hook={probe_hook} | k={args.k}")

    run_checkpoints = discover_checkpoints(args.run_dir)
    print(f"Found {len(run_checkpoints)} checkpoints in {args.run_dir}")

    probe_dir = os.path.join(args.run_dir, "probes")
    os.makedirs(probe_dir, exist_ok=True)
    out_path = os.path.join(probe_dir, "mutual_knn_cross.csv")
    csv_file = open(out_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["checkpoint_name", "layer", "mutual_knn_k", "mutual_knn_alignment"])

    for ckpt_name, ckpt_path in run_checkpoints:
        ref_ckpt_path = os.path.join(args.ref_run_dir, "checkpoints", ckpt_name, "model.pt")
        if not os.path.exists(ref_ckpt_path):
            print(f"  [skip] {ckpt_name}: no matching reference checkpoint at {ref_ckpt_path}")
            continue

        model = load_model_from_checkpoint(ckpt_path, task_meta=None, cfg=cfg).to(device)
        ref_model = load_model_from_checkpoint(ref_ckpt_path, task_meta=None, cfg=cfg).to(device)

        hidden = get_hidden_states(model, probe_loader, device, probe_hook=probe_hook)
        ref_hidden = get_hidden_states(ref_model, probe_loader, device, probe_hook=probe_hook)

        print(f"{ckpt_name}: ", end="")
        layer_scores = []
        for layer_idx, H in hidden.items():
            H_ref = ref_hidden.get(layer_idx)
            if H_ref is None:
                continue
            score = mutual_knn_alignment(H, H_ref, k=args.k)
            layer_scores.append(score)
            writer.writerow([ckpt_name, layer_idx, args.k, score])
        csv_file.flush()
        mean_score = sum(layer_scores) / len(layer_scores) if layer_scores else float('nan')
        print(f"mean mutual-kNN alignment = {mean_score:.4f}")

        del model, ref_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nDone. Written: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ref_run_dir", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--k", type=int, default=10,
                  help="Number of nearest neighbors (default 10, matching "
                      "Huh et al. 2024's default)")
    main(p.parse_args())