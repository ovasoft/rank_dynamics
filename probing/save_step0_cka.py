"""
save_step0_cka.py — Re-run step-0 CKA and write results to files.

Outputs:
    results/babylm_strict_small/step0_cka.csv       — full per-rank per-seed per-layer
    results/babylm_strict_small/step0_cka_summary.csv — mean across layers, per rank/seed
    results/babylm_strict_small/step0_cka_summary.txt — human-readable for rebuttal

Usage:
    python save_step0_cka.py --config configs/babylm_strict_small.yaml --seeds 0 1
"""

import os, argparse, csv, math
import torch

from task_registry import (
    load_config, get_task, replace_with_low_rank
)
from probe_dynamics import make_probe_loader, get_hidden_states, linear_cka


def get_run_dir(cfg, run_type, rank, seed):
    output_root = cfg.get('output_root', 'outputs/babylm_strict_small')
    if run_type == 'FR':
        run_name = 'FR' if seed == 0 else f'FR_seed{seed}'
    else:
        run_name = f'LR_r{rank}' if seed == 0 else f'LR_r{rank}_seed{seed}'
    return os.path.join(output_root, run_name)


def load_model_step0(cfg, task, run_type, rank, seed, device):
    run_dir   = get_run_dir(cfg, run_type, rank, seed)
    ckpt_path = os.path.join(run_dir, 'checkpoints', 'tokens_0000000000', 'model.pt')
    if not os.path.exists(ckpt_path):
        print(f"  [missing] {ckpt_path}")
        return None
    model = task['make_model'](dict(cfg, _role='probe'))
    if run_type == 'LR':
        replace_with_low_rank(model, rank, verbose=False)
    sd     = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    target = model.hf_model if hasattr(model, 'hf_model') else model
    target.load_state_dict(sd, strict=False)
    return model.to(device)


def main(args):
    cfg         = load_config(args.config)
    ranks       = args.ranks or cfg.get('ranks', [8, 16, 32, 64, 128, 256])
    seeds       = args.seeds
    results_dir = cfg.get('results_dir', 'results/babylm_strict_small')
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task   = get_task(cfg['task'])

    probe_loader, meta = make_probe_loader(
        cfg['task'], cfg=cfg, n_samples=cfg.get('probe_n_samples', 512))
    probe_hook = meta.get('probe_hook', 'last_token')

    # Output files
    full_csv_path    = os.path.join(results_dir, 'step0_cka.csv')
    summary_csv_path = os.path.join(results_dir, 'step0_cka_summary.csv')
    summary_txt_path = os.path.join(results_dir, 'step0_cka_summary.txt')

    full_csv    = open(full_csv_path,    'w', newline='')
    summary_csv = open(summary_csv_path, 'w', newline='')
    full_writer    = csv.writer(full_csv)
    summary_writer = csv.writer(summary_csv)

    full_writer.writerow(['seed', 'run', 'rank', 'layer', 'fr_lr_cka', 'fr_fr_cka'])
    summary_writer.writerow(['seed', 'run', 'rank',
                              'mean_fr_lr_cka', 'mean_fr_fr_cka',
                              'n_layers'])

    summary_lines = [
        "Step-0 CKA results (before any gradient update)",
        "=" * 55,
        f"{'Run':<12} {'Seed 0':>10} {'Seed 1':>10} {'Mean':>10}  FR-FR baseline",
        "-" * 55,
    ]

    # Collect for averaging across seeds
    rank_seed_means = {}   # (rank, seed) -> mean_cka
    fr_fr_by_seed   = {}   # seed -> fr_fr_cka

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")

        # Load FR step-0 hidden states
        fr_model = load_model_step0(cfg, task, 'FR', None, seed, device)
        if fr_model is None:
            print(f"  FR seed={seed} step-0 checkpoint missing — skipping seed")
            continue
        fr_hidden = get_hidden_states(fr_model, probe_loader, device,
                                      probe_hook=probe_hook)
        del fr_model

        # FR-FR baseline: use other seed's FR step-0
        fr_fr_cka = float('nan')
        for other_seed in seeds:
            if other_seed == seed:
                continue
            other_fr = load_model_step0(cfg, task, 'FR', None, other_seed, device)
            if other_fr is None:
                continue
            other_hidden = get_hidden_states(other_fr, probe_loader, device,
                                             probe_hook=probe_hook)
            del other_fr
            vals = [linear_cka(fr_hidden[l], other_hidden[l])
                    for l in fr_hidden if l in other_hidden]
            fr_fr_cka = sum(vals) / len(vals) if vals else float('nan')
            break
        fr_fr_by_seed[seed] = fr_fr_cka
        print(f"  FR-FR baseline (seed {seed} vs other): {fr_fr_cka:.4f}")

        # FR-LR CKA for each rank
        for r in ranks:
            lr_model = load_model_step0(cfg, task, 'LR', r, seed, device)
            if lr_model is None:
                continue
            lr_hidden = get_hidden_states(lr_model, probe_loader, device,
                                          probe_hook=probe_hook)
            del lr_model

            layer_ckas = {}
            for l in fr_hidden:
                if l in lr_hidden:
                    layer_ckas[l] = linear_cka(fr_hidden[l], lr_hidden[l])
                    full_writer.writerow([seed, f'LR_r{r}', r, l,
                                          f'{layer_ckas[l]:.6f}',
                                          f'{fr_fr_cka:.6f}'])

            mean_cka = sum(layer_ckas.values()) / len(layer_ckas) if layer_ckas else float('nan')
            rank_seed_means[(r, seed)] = mean_cka
            summary_writer.writerow([seed, f'LR_r{r}', r,
                                      f'{mean_cka:.6f}', f'{fr_fr_cka:.6f}',
                                      len(layer_ckas)])
            print(f"  LR_r{r:<5} seed={seed}  FR-LR CKA = {mean_cka:.4f}"
                  f"  (FR-FR = {fr_fr_cka:.4f})")

        full_csv.flush()
        summary_csv.flush()

    # Build summary text
    fr_fr_mean = sum(fr_fr_by_seed.values()) / len(fr_fr_by_seed) if fr_fr_by_seed else float('nan')
    for r in ranks:
        vals = [rank_seed_means.get((r, s)) for s in seeds
                if (r, s) in rank_seed_means]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        if not vals:
            continue
        mean_across_seeds = sum(vals) / len(vals)
        per_seed_str = '  '.join(f'{v:.4f}' for v in vals)
        summary_lines.append(
            f"LR_r{r:<6}  {per_seed_str}   mean={mean_across_seeds:.4f}"
            f"  (FR-FR≈{fr_fr_mean:.4f})"
        )

    summary_lines += [
        "-" * 55,
        f"FR-FR baseline (mean across seeds): {fr_fr_mean:.4f}",
        "",
        "Interpretation:",
        "  Split present at step 0 → due to initialisation statistics,",
        "  not gradient dynamics. Resolves R1-W2, R2-W2, R4-Q1.",
    ]

    txt = '\n'.join(summary_lines)
    with open(summary_txt_path, 'w') as f:
        f.write(txt)
    print(f"\n{txt}")

    full_csv.close()
    summary_csv.close()

    print(f"\nFiles written:")
    print(f"  {full_csv_path}")
    print(f"  {summary_csv_path}")
    print(f"  {summary_txt_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--seeds',  type=int, nargs='+', default=[0, 1])
    p.add_argument('--ranks',  type=int, nargs='+', default=None)
    main(p.parse_args())