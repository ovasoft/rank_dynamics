"""
patch_step0.py — Patch train.py to save a step-0 checkpoint before any
gradient update in token-budget mode.

This addresses reviewer requests (R1-W2, R2-W2, R4-Q1) for a zero-update
language CKA baseline. The patch inserts a single save_checkpoint call at
the very start of train_token_budget_mode, before the training loop begins.

Usage:
    python patch_step0.py          # patches train.py in-place, backs up original

After patching, re-run training with --run_type FR and each LR rank.
The step-0 checkpoint will appear as:
    outputs/babylm_strict_small/<run>/checkpoints/tokens_0000000000/model.pt

Then probe it like any other checkpoint:
    python probe_dynamics.py \
        --config configs/babylm_strict_small.yaml \
        --run_dir outputs/babylm_strict_small/FR \
        --ref_run_dir outputs/babylm_strict_small/LR_r8
"""

import re, shutil, sys, os

TARGET = "train.py"
BACKUP = "train.py.bak"

SEARCH = '''    print(f"Token budget: {token_budget:,}  |  "
          f"Checkpoints at: {[f'{t:,}' for t in ckpt_tokens]}")

    pass_num = 0'''

REPLACE = '''    print(f"Token budget: {token_budget:,}  |  "
          f"Checkpoints at: {[f\\'{t:,}\\' for t in ckpt_tokens]}")

    # ── Step-0 checkpoint: before any gradient update ─────────────────────
    # Addresses reviewer request for zero-update CKA baseline.
    # Saved as tokens_0000000000 so probe_dynamics.py sorts it first.
    step0_ckpt = os.path.join(run_dir, 'checkpoints', 'tokens_0000000000')
    if not os.path.exists(os.path.join(step0_ckpt, 'model.pt')):
        print("Saving step-0 checkpoint (before any gradient update)...")
        save_checkpoint(model, step0_ckpt)
        print(f"  ✓ step-0 checkpoint saved: {step0_ckpt}")
    else:
        print(f"  step-0 checkpoint already exists: {step0_ckpt}")

    pass_num = 0'''

def patch():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found in current directory.")
        sys.exit(1)

    with open(TARGET) as f:
        src = f.read()

    if 'tokens_0000000000' in src:
        print("train.py already patched — step-0 checkpoint save already present.")
        return

    if SEARCH not in src:
        print("ERROR: Could not find insertion point in train.py.")
        print("Expected text around line containing 'Token budget:'")
        print("The file may have been modified. Please apply the patch manually.")
        print("\nInsert this block after the 'Checkpoints at:' print in")
        print("train_token_budget_mode(), before 'pass_num = 0':\n")
        print(REPLACE)
        sys.exit(1)

    shutil.copy(TARGET, BACKUP)
    print(f"Backed up {TARGET} → {BACKUP}")

    patched = src.replace(SEARCH, REPLACE, 1)
    with open(TARGET, 'w') as f:
        f.write(patched)
    print(f"✓ Patched {TARGET} — step-0 checkpoint will be saved before training begins.")
    print(f"\nNow re-run training for FR and all LR ranks:")
    print(f"  python train.py --config configs/babylm_strict_small.yaml --run_type FR --seed 0")
    print(f"  python train.py --config configs/babylm_strict_small.yaml --run_type LR --rank 8 --seed 0")
    print(f"  # ... etc for other ranks and seeds")
    print(f"\nThe step-0 checkpoint will appear at:")
    print(f"  outputs/babylm_strict_small/FR/checkpoints/tokens_0000000000/model.pt")
    print(f"\nThen probe as normal — probe_dynamics.py will pick it up automatically.")

if __name__ == '__main__':
    patch()