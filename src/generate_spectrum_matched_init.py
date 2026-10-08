"""
generate_spectrum_matched_init.py — Spectrum-matched low-rank initialization
control (paper §3.2, §4.3 O4.3).

Tests whether the early representational split is an initialization or
spectral-scaling artifact of the low-rank factorization.

task_registry.LowRankLinear's default init (independent nn.init.normal_ on
A and B, std=1/sqrt(rank)) produces a very different singular-value
distribution from a full-rank random matrix's own top-r singular
directions -- typically flatter and smaller in magnitude. This confounds
"is the early FR-LR split caused by the rank constraint" with "is it
caused by this particular initializer's atypical spectral scale."

This script builds an alternative initialization: for each target weight
matrix, load FR's own actual initial weight (from its true step-0 /
first-checkpoint) and set A, B to reconstruct EXACTLY that matrix's best
rank-r approximation (top-r truncated SVD, Eckart-Young optimal). The
resulting model's initial effective weight is, by construction, the
closest possible rank-r matrix to what FR itself started from -- as
spectrum-matched as a rank-r factorization can get. If the paper's
reported early representational split (Aim 1) still appears at step 0
despite this, that rules out "atypical init scale/shape from the default
LowRankLinear initializer" as an explanation and isolates the rank
constraint itself as the cause.

Reuses:
  - load_config, get_task, LowRankLinear, replace_with_low_rank
                                                    (task_registry.py)
  - save_checkpoint                                (train.py)

New in this file:
  - spectrum_matched_init_(): reconstructs A, B in place from the top-r
    SVD of a reference weight matrix
  - build_spectrum_matched_model(): builds a full LR model whose every
    target layer's initial effective weight is the rank-r truncation of
    the corresponding FR reference weight
  - verify_spectrum_match(): a standalone, direct verification of the
    construction -- see naming note below for why this does NOT rely on
    round-tripping through the existing probe scripts

NAMING / LOADING NOTE: train.py's save_checkpoint always calls
materialize_low_rank before saving, so checkpoints are stored in plain
nn.Linear/Conv1D ('.weight') shape regardless of rank -- there is no
'.A'/'.B' key in any saved checkpoint, low-rank or not. probe_dynamics.py
and probe_gradient_erank.py both detect rank from the run DIRECTORY NAME
(matching 'LR_r' prefix or a bare '_r<N>' substring) and call
replace_with_low_rank BEFORE loading the checkpoint; if the checkpoint
was saved in materialized ('.weight') form, this sequence creates a model
expecting '.A'/'.B' parameters that the checkpoint doesn't have, and
strict=False silently drops the mismatched keys rather than erroring --
worth verifying directly (e.g., checking whether probed values change
checkpoint-to-checkpoint at all, since a silently-failed load would leave
weights frozen at their initial random values) before trusting results
from any run whose directory name matches those patterns. This script's
default output directory name deliberately avoids both patterns, and
verify_spectrum_match() checks the construction directly in-memory rather
than depending on round-tripping through those scripts.

Usage:
    python src/generate_spectrum_matched_init.py \\
        --config configs/babylm_strict_small.yaml \\
        --fr_reference_ckpt outputs/babylm_strict_small/FR/checkpoints/tokens_0000000000/model.pt \\
        --rank 8 \\
        --run_dir outputs/babylm_strict_small/LR_specmatch8
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, argparse
import torch

from task_registry import load_config, get_task, LowRankLinear, replace_with_low_rank
from train import save_checkpoint


# ── Spectrum-matched initialization ─────────────────────────────────────────

def spectrum_matched_init_(lr_module, ref_weight):
    """
    In-place: set lr_module.A, lr_module.B such that A @ B.T equals the
    best rank-`rank` approximation (top-r truncated SVD) of `ref_weight`,
    where `ref_weight` is oriented as (out, in) -- the same orientation
    LowRankLinear's A @ B.T reconstructs.
    """
    rank = lr_module.A.shape[1]
    U, S, Vh = torch.linalg.svd(ref_weight.float(), full_matrices=False)
    sqrt_S = S[:rank].clamp(min=0).sqrt()
    A_new = U[:, :rank] * sqrt_S.unsqueeze(0)       # (out, rank)
    B_new = Vh[:rank, :].t() * sqrt_S.unsqueeze(0)  # (in, rank)
    lr_module.A.data.copy_(A_new.to(lr_module.A.dtype))
    lr_module.B.data.copy_(B_new.to(lr_module.B.dtype))


def _ref_weight_for(module, ref_W):
    """
    Orient a reference weight to match LowRankLinear's (out, in)
    reconstruction convention. nn.Linear stores (out, in) already;
    Conv1D stores (in, out) and needs transposing.
    """
    if module.was_conv1d and ref_W.shape == (module.B.shape[0], module.A.shape[0]):
        return ref_W.t()
    return ref_W


def build_spectrum_matched_model(task, cfg, rank, fr_reference_state_dict, device):
    """
    Build an LR model whose every target layer (same eligibility rule
    replace_with_low_rank itself uses) is initialized to the rank-`rank`
    truncated SVD of the corresponding FR reference weight, rather than
    LowRankLinear's default independent-normal init.
    """
    train_cfg = dict(cfg); train_cfg['_role'] = 'train'
    model = task['make_model'](train_cfg)
    replace_with_low_rank(model, rank, verbose=True)  # in-place
    model = model.to(device)

    root = model.hf_model if hasattr(model, 'hf_model') else model
    n_matched = 0
    for name, module in list(root.named_modules()):
        if not isinstance(module, LowRankLinear):
            continue
        weight_key = f"{name}.weight"
        if weight_key not in fr_reference_state_dict:
            print(f"  [skip] {name}: no matching key '{weight_key}' in FR reference")
            continue
        ref_W = _ref_weight_for(module, fr_reference_state_dict[weight_key].to(device))
        spectrum_matched_init_(module, ref_W)
        n_matched += 1

    print(f"Spectrum-matched init applied to {n_matched} layers (rank={rank})")
    return model


def verify_spectrum_match(model, fr_reference_state_dict, rank, device, atol=1e-3):
    """
    Direct, standalone verification that spectrum_matched_init_ produced
    what it was supposed to -- independent of any downstream probing
    script's checkpoint-loading logic (see module docstring). Recomputes
    each target layer's effective weight directly from the in-memory
    model and compares its top-`rank` singular values against the FR
    reference's own top-`rank` singular values.
    """
    root = model.hf_model if hasattr(model, 'hf_model') else model
    all_ok = True
    for name, module in list(root.named_modules()):
        if not isinstance(module, LowRankLinear):
            continue
        weight_key = f"{name}.weight"
        if weight_key not in fr_reference_state_dict:
            continue
        ref_W = _ref_weight_for(module, fr_reference_state_dict[weight_key].to(device))
        ref_S = torch.linalg.svdvals(ref_W.float())[:rank]
        eff_W = (module.A @ module.B.T).detach()
        eff_S = torch.linalg.svdvals(eff_W.float())[:rank]
        match = torch.allclose(eff_S, ref_S, atol=atol, rtol=1e-2)
        all_ok = all_ok and match
        status = "OK" if match else "MISMATCH"
        print(f"  [{status}] {name}: top-{rank} singular values "
             f"ref={ref_S.cpu().numpy().round(3)} "
             f"eff={eff_S.cpu().numpy().round(3)}")
    print(f"\nVerification {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)
    task = get_task(cfg['task'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading FR reference checkpoint: {args.fr_reference_ckpt}")
    fr_reference_state_dict = torch.load(args.fr_reference_ckpt, map_location=device,
                                        weights_only=True)

    model = build_spectrum_matched_model(task, cfg, args.rank, fr_reference_state_dict, device)

    print("\nVerifying spectrum match...")
    ok = verify_spectrum_match(model, fr_reference_state_dict, args.rank, device)
    if not ok:
        print("WARNING: verification failed for at least one layer -- do not "
             "proceed to training/probing until this is resolved.")

    # Naming deliberately avoids the 'LR_r' prefix / bare '_r<N>' substring
    # (see module docstring).
    run_dir = args.run_dir or os.path.join(
        cfg['output_root'], f'LR_specmatch{args.rank}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun dir: {run_dir}")

    ckpt_name = 'tokens_' + '0' * 10  # step-0, matches existing tokens_NNNNNNNNNN convention
    save_checkpoint(model, os.path.join(run_dir, 'checkpoints', ckpt_name))
    print(f"Saved step-0 spectrum-matched checkpoint to "
         f"{run_dir}/checkpoints/{ckpt_name}")
    print("\nNext steps:")
    print(f"  1. Probe this checkpoint's CKA against FR at step 0:")
    print(f"     python probing/probe_dynamics.py --config {args.config} \\")
    print(f"         --run_dir {run_dir} --ref_run_dir <FR run dir>")
    print(f"  2. If a full trajectory (not just step 0) is needed, continue "
         f"training this checkpoint under the normal LR schedule.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--fr_reference_ckpt', required=True,
                  help="Path to FR model.pt at its earliest checkpoint "
                      "(true step-0 or first-batch), used as the reference "
                      "spectrum to match")
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--run_dir', default=None)
    main(p.parse_args())