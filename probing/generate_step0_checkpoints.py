"""
generate_step0_checkpoints.py — Generate step-0 (pre-training) checkpoints
for FR and all LR ranks WITHOUT re-running training.

Used to report language FR-LR CKA at a zero-update checkpoint (before
any gradient step).

The key insight: at step 0, the model weights are determined entirely by
the random seed and the initialization scheme (nn.init.normal_ for
LowRankLinear A and B). We can reproduce this exactly by:
  1. Setting the same random seed used in training.
  2. Instantiating the model (FR or LR) — this triggers __init__ which runs
     the same initialization as during training.
  3. Saving the checkpoint immediately, before any optimizer.step().

This is mathematically identical to saving at step 0 during training.

Usage:
    python probing/generate_step0_checkpoints.py \\
        --config configs/babylm_strict_small.yaml \\
        --seeds 0 1

Outputs:
    outputs/babylm_strict_small/FR_seed0/checkpoints/tokens_0000000000/model.pt
    outputs/babylm_strict_small/LR_r8_seed0/checkpoints/tokens_0000000000/model.pt
    ... etc for all ranks and seeds

Then run probe_dynamics.py as normal — it will pick up tokens_0000000000
as the first checkpoint in the sort order.

To get CKA at step 0, run:
    python probing/probe_dynamics.py \\
        --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/FR_seed0 \\
        --ref_run_dir outputs/babylm_strict_small/LR_r8_seed0

Or use probe_step0_cka.py (simpler, single-purpose) which this script
also supports via --probe flag.
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_os.path.join(_ROOT, d) for d in ('src', 'src/vision', 'probing', 'analysis')]

import os, argparse, copy, math, torch, random
import torch.nn as nn

from task_registry import (
    load_config, get_task,
    replace_with_low_rank, materialize_low_rank, LowRankLinear
)


def save_checkpoint(model, path):
    os.makedirs(path, exist_ok=True)
    m2 = copy.deepcopy(model).cpu()
    materialize_low_rank(m2)
    torch.save(m2.state_dict(), os.path.join(path, 'model.pt'))
    print(f"    saved → {path}/model.pt")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def build_step0_model(cfg, run_type, rank, seed):
    """
    Instantiate a model with the same seed used during training.
    Returns the model at initialization (step 0).
    """
    set_seed(seed)
    task  = get_task(cfg['task'])
    model = task['make_model'](cfg)
    if run_type == 'LR':
        replace_with_low_rank(model, rank, verbose=False)
    return model


def get_run_dir(cfg, run_type, rank, seed):
    output_root = cfg.get('output_root', 'outputs/babylm_strict_small')
    if run_type == 'FR':
        run_name = 'FR' if seed == 0 else f'FR_seed{seed}'
    else:
        run_name = f'LR_r{rank}' if seed == 0 else f'LR_r{rank}_seed{seed}'
    return os.path.join(output_root, run_name)


def main(args):
    cfg   = load_config(args.config)
    ranks = args.ranks or cfg.get('ranks', [8, 16, 32, 64, 128, 256])
    seeds = args.seeds

    print(f"Generating step-0 checkpoints for seeds={seeds}, ranks={ranks}")
    print(f"Config: {args.config}")
    print()

    generated = []

    for seed in seeds:
        # FR
        print(f"[FR seed={seed}]")
        model   = build_step0_model(cfg, 'FR', None, seed)
        run_dir = get_run_dir(cfg, 'FR', None, seed)
        ckpt_path = os.path.join(run_dir, 'checkpoints', 'tokens_0000000000')
        if os.path.exists(os.path.join(ckpt_path, 'model.pt')) and not args.overwrite:
            print(f"    already exists, skipping (use --overwrite to regenerate)")
        else:
            save_checkpoint(model, ckpt_path)
            generated.append(('FR', seed, ckpt_path))
        del model

        # LR ranks
        for r in ranks:
            print(f"[LR_r{r} seed={seed}]")
            model   = build_step0_model(cfg, 'LR', r, seed)
            run_dir = get_run_dir(cfg, 'LR', r, seed)
            ckpt_path = os.path.join(run_dir, 'checkpoints', 'tokens_0000000000')
            if os.path.exists(os.path.join(ckpt_path, 'model.pt')) and not args.overwrite:
                print(f"    already exists, skipping (use --overwrite to regenerate)")
            else:
                save_checkpoint(model, ckpt_path)
                generated.append((f'LR_r{r}', seed, ckpt_path))
            del model

    print(f"\nGenerated {len(generated)} step-0 checkpoints.")

    if args.probe:
        print("\nRunning step-0 CKA probe...")
        _run_step0_cka(cfg, ranks, seeds)


def _run_step0_cka(cfg, ranks, seeds):
    """
    Compute FR-LR CKA at step 0 across all ranks and seeds.
    Prints a summary table directly.
    """
    from probe_dynamics import make_probe_loader, get_hidden_states, linear_cka

    output_root = cfg.get('output_root', 'outputs/babylm_strict_small')
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task        = get_task(cfg['task'])

    print(f"\nStep-0 CKA (before any gradient update)")
    print(f"{'Run':<20} {'Seed':<6} {'Mean FR-LR CKA':>16}  (FR-FR baseline for ref)")
    print("-" * 55)

    probe_loader, meta = make_probe_loader(
        cfg['task'], cfg=cfg, n_samples=cfg.get('probe_n_samples', 512))

    for seed in seeds:
        # Load FR step-0
        fr_run_dir = get_run_dir(cfg, 'FR', None, seed)
        fr_ckpt    = os.path.join(fr_run_dir, 'checkpoints',
                                  'tokens_0000000000', 'model.pt')
        if not os.path.exists(fr_ckpt):
            print(f"  FR seed={seed}: step-0 checkpoint not found, run without --probe first")
            continue

        fr_model = task['make_model'](dict(cfg, _role='probe'))
        sd = torch.load(fr_ckpt, map_location='cpu', weights_only=True)
        target = fr_model.hf_model if hasattr(fr_model, 'hf_model') else fr_model
        target.load_state_dict(sd, strict=False)
        fr_model = fr_model.to(device)
        fr_hidden = get_hidden_states(fr_model, probe_loader, device,
                                      probe_hook=meta.get('probe_hook', 'last_token'))
        del fr_model

        # FR-FR baseline: second FR seed
        fr_fr_cka = float('nan')
        for other_seed in seeds:
            if other_seed == seed:
                continue
            other_fr_dir  = get_run_dir(cfg, 'FR', None, other_seed)
            other_fr_ckpt = os.path.join(other_fr_dir, 'checkpoints',
                                         'tokens_0000000000', 'model.pt')
            if not os.path.exists(other_fr_ckpt):
                continue
            other_fr_model = task['make_model'](dict(cfg, _role='probe'))
            sd2 = torch.load(other_fr_ckpt, map_location='cpu', weights_only=True)
            t2  = other_fr_model.hf_model if hasattr(other_fr_model, 'hf_model') \
                  else other_fr_model
            t2.load_state_dict(sd2, strict=False)
            other_fr_model = other_fr_model.to(device)
            other_hidden = get_hidden_states(
                other_fr_model, probe_loader, device,
                probe_hook=meta.get('probe_hook', 'last_token'))
            del other_fr_model
            layer_ckas = [linear_cka(fr_hidden[l], other_hidden[l])
                          for l in fr_hidden if l in other_hidden]
            fr_fr_cka = sum(layer_ckas) / len(layer_ckas) if layer_ckas else float('nan')
            break

        # FR-LR CKA for each rank
        for r in ranks:
            lr_run_dir = get_run_dir(cfg, 'LR', r, seed)
            lr_ckpt    = os.path.join(lr_run_dir, 'checkpoints',
                                      'tokens_0000000000', 'model.pt')
            if not os.path.exists(lr_ckpt):
                print(f"  LR_r{r} seed={seed}: step-0 checkpoint not found")
                continue

            lr_model = task['make_model'](dict(cfg, _role='probe'))
            replace_with_low_rank(lr_model, r, verbose=False)
            sd_lr = torch.load(lr_ckpt, map_location='cpu', weights_only=True)
            t_lr  = lr_model.hf_model if hasattr(lr_model, 'hf_model') else lr_model
            t_lr.load_state_dict(sd_lr, strict=False)
            lr_model = lr_model.to(device)
            lr_hidden = get_hidden_states(
                lr_model, probe_loader, device,
                probe_hook=meta.get('probe_hook', 'last_token'))
            del lr_model

            layer_ckas = [linear_cka(fr_hidden[l], lr_hidden[l])
                          for l in fr_hidden if l in lr_hidden]
            mean_cka = sum(layer_ckas) / len(layer_ckas) if layer_ckas else float('nan')
            print(f"  LR_r{r:<5} seed={seed}  FR-LR CKA = {mean_cka:.4f}"
                  f"  (FR-FR = {fr_fr_cka:.4f})")

    print("\nDone.")
    print("Key question: is FR-LR CKA at step 0 already below FR-FR baseline?")
    print("If yes → split is due to initialization statistics, not gradient dynamics.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True)
    p.add_argument('--seeds',     type=int, nargs='+', default=[0, 1])
    p.add_argument('--ranks',     type=int, nargs='+', default=None,
                   help='Ranks to generate (default: from config)')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing step-0 checkpoints')
    p.add_argument('--probe',     action='store_true',
                   help='After generating, immediately compute and print step-0 CKA')
    main(p.parse_args())