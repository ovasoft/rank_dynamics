"""
probe_transfer.py — BLiMP minimal-pair evaluation for downstream/functional
comparison across FR, untreated LR, and DSN-treated LR checkpoints.

Addresses R2-W3 and R3-W1: the paper's CKA-based "representational cost"
claim needs a functional/behavioural check, not just a geometry metric.
BLiMP (Warstadt et al. 2020) gives 67 grammatical-phenomenon paradigms of
minimal sentence pairs (sentence_good vs sentence_bad); a competent LM
should assign higher log-likelihood to sentence_good.

The key comparison this script enables: take FR, untreated LR_r8, and each
DSN-treated LR_r8 variant (all roughly matched or improved in perplexity
relative to untreated LR), and check whether BLiMP accuracy tracks CKA
(representational similarity to FR) or is dissociated from it — exactly
the question R2/R3 are asking.

Reuses:
  - load_model_from_checkpoint, _fix_conv1d_shapes  (probe_dynamics.py)
  - _load_babylm_config_and_tokenizer                (task_registry.py)
  - load_config, get_task                            (task_registry.py)

No new model-loading or checkpoint-detection logic is implemented here —
this script is evaluation-only.

Usage:
    # Single run
    python probe_transfer.py --config configs/babylm_strict_small.yaml \\
        --run_dir outputs/babylm_strict_small/FR --checkpoint final

    # Compare a set of runs in one pass (writes one combined CSV)
    python probe_transfer.py --config configs/babylm_strict_small.yaml \\
        --run_dirs outputs/babylm_strict_small/FR \\
                   outputs/babylm_strict_small/LR_r8 \\
                   outputs/babylm_strict_small/LR_r8_dsn_static \\
                   outputs/babylm_strict_small/LR_r8_dsn_dynamic \\
                   outputs/babylm_strict_small/LR_r8_dsn_flat \\
        --checkpoint final

Output:
    <results_dir>/blimp_results.csv       — per-run, per-phenomenon accuracy
    <results_dir>/blimp_summary.csv       — per-run overall accuracy
"""

import os, math, argparse, csv
import torch
import torch.nn.functional as F

from task_registry import load_config, get_task, _load_babylm_config_and_tokenizer
from probe_dynamics import load_model_from_checkpoint

# A representative, phenomenon-diverse subset of BLiMP's 67 paradigms.
# Kept small deliberately: each paradigm is ~1000 pairs and rebuttal time
# is limited. Pass --phenomena to override with a custom list, or
# --all_phenomena to run the full BLiMP suite.
DEFAULT_PHENOMENA = [
    "anaphor_gender_agreement",
    "determiner_noun_agreement_1",
    "irregular_plural_subject_verb_agreement_1",
    "distractor_agreement_relational_noun",
    "wh_questions_object_gap",
    "npi_present_1",
    "existential_there_subject_raising",
    "regular_plural_subject_verb_agreement_1",
]

ALL_BLIMP_PHENOMENA = DEFAULT_PHENOMENA + [
    # Extend here or pass --all_phenomena to fetch the full config list
    # from the dataset repo directly at runtime.
]


# ── Sequence log-likelihood ────────────────────────────────────────────────

@torch.no_grad()
def sequence_logprob(model, tokenizer, text, device):
    """
    Teacher-forced total log-likelihood of `text` under the model.
    Returns (total_logprob, n_scored_tokens), or (None, None) if the sentence
    is too short to score. n_scored_tokens is needed to turn accumulated
    log-likelihood into a perplexity for the training-log cross-check below.
    """
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        return None, None  # degenerate — cannot compute next-token likelihood
    logits = model(ids)                       # (1, T, V) via BabyLMWrapper.forward
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = ids[:, 1:]
    token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum().item(), targets.numel()


# ── BLiMP evaluation for one checkpoint ────────────────────────────────────

def evaluate_blimp(model, tokenizer, device, phenomena, max_pairs_per_phenomenon=None):
    """
    Returns:
      per_phenomenon: dict {phenomenon: (accuracy, n_pairs)}
      overall_accuracy: float across all evaluated pairs
      total_pairs: int
      blimp_ppl: perplexity computed from sentence_good log-likelihoods only,
                 over this same BLiMP sample. NOT directly comparable to the
                 training-log val_ppl (different data distribution — BLiMP
                 sentences vs BabyLM validation text) but useful as a sanity
                 check: if this checkpoint's weights are corrupted or mid-
                 correction (e.g. a DSN rescaling step caught mid-application),
                 blimp_ppl will typically be wildly higher or NaN/Inf even
                 though the *comparison* between good/bad could still look
                 like reasonable accuracy by coincidence.
    """
    from datasets import load_dataset

    model.eval()
    per_phenomenon = {}
    total_correct = 0
    total_pairs = 0
    good_logp_sum = 0.0
    good_token_sum = 0

    for phenomenon in phenomena:
        try:
            ds = load_dataset("blimp", phenomenon, split="train")
        except Exception as e:
            print(f"  [skip] {phenomenon}: could not load ({e})")
            continue

        n = len(ds) if max_pairs_per_phenomenon is None else min(
            len(ds), max_pairs_per_phenomenon)
        correct = 0
        evaluated = 0

        for i in range(n):
            ex = ds[i]
            lp_good, ntok_good = sequence_logprob(model, tokenizer, ex["sentence_good"], device)
            lp_bad, ntok_bad = sequence_logprob(model, tokenizer, ex["sentence_bad"], device)
            if lp_good is None or lp_bad is None:
                continue
            evaluated += 1
            if lp_good > lp_bad:
                correct += 1
            good_logp_sum += lp_good
            good_token_sum += ntok_good

        if evaluated == 0:
            print(f"  [skip] {phenomenon}: no evaluable pairs")
            continue

        acc = correct / evaluated
        per_phenomenon[phenomenon] = (acc, evaluated)
        total_correct += correct
        total_pairs += evaluated
        print(f"  {phenomenon:<50} acc={acc:.3f}  n={evaluated}")

    overall = total_correct / total_pairs if total_pairs else None
    if good_token_sum > 0:
        mean_nll = -good_logp_sum / good_token_sum
        blimp_ppl = math.exp(min(mean_nll, 50))  # cap to avoid overflow on garbage checkpoints
    else:
        blimp_ppl = None
    return per_phenomenon, overall, total_pairs, blimp_ppl


# ── Training-log cross-check ────────────────────────────────────────────────

def _sf(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_training_log_metric(run_dir, checkpoint):
    """
    Cross-check: read log.csv and return the val_ppl/val_acc that was
    recorded DURING TRAINING for the checkpoint being BLiMP-evaluated.

    This exists to catch checkpoint/eval mismatches — e.g. a DSN run whose
    'final' checkpoint was saved at or near a spectral-rescaling step rather
    than a settled post-training state. If the training-log perplexity here
    doesn't match what the paper reports for that run (e.g. DSN-static ~117,
    untreated LR ~457), the checkpoint being BLiMP-evaluated is not the one
    the paper's numbers describe, and the BLiMP result is not yet trustworthy.

    Returns dict {metric_name, metric_value, tokens_seen} or None if log.csv
    is missing or empty.
    """
    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(log_path):
        return None
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    metric_name = "val_ppl" if "val_ppl" in rows[0] else (
        "val_acc" if "val_acc" in rows[0] else None)
    if metric_name is None:
        return None
    higher_better = metric_name == "val_acc"

    row = None
    if checkpoint == "final":
        row = rows[-1]
    elif checkpoint == "best":
        scored = [(_sf(r.get(metric_name)), r) for r in rows]
        scored = [(v, r) for v, r in scored if v is not None]
        if scored:
            row = (max(scored, key=lambda vr: vr[0]) if higher_better
                   else min(scored, key=lambda vr: vr[0]))[1]
    elif checkpoint.startswith("tokens_"):
        try:
            target_tokens = int(checkpoint.split("_")[1])
        except (IndexError, ValueError):
            target_tokens = None
        if target_tokens is not None:
            for r in rows:
                ts = _sf(r.get("tokens_seen"))
                if ts is not None and int(ts) == target_tokens:
                    row = r
                    break

    if row is None:
        row = rows[-1]  # fallback: best available reference point

    return {
        "metric_name":  metric_name,
        "metric_value": _sf(row.get(metric_name)),
        "tokens_seen":  _sf(row.get("tokens_seen")),
    }


def cross_check_checkpoint(run_dir, checkpoint, blimp_ppl):
    """
    Print and return a cross-check dict comparing training-log perplexity
    against the BLiMP-side perplexity for the checkpoint just evaluated.
    Flags (does not fail on) suspicious mismatches — final judgment on
    whether a checkpoint is trustworthy is left to the person reading it.
    """
    log_metric = load_training_log_metric(run_dir, checkpoint)

    if log_metric is None:
        print("  [cross-check] no log.csv found — cannot verify checkpoint "
              "against training-time metrics")
        return {"training_log_metric_name": None, "training_log_metric_value": None,
                "training_log_tokens_seen": None, "blimp_ppl": blimp_ppl,
                "cross_check_flag": "no_log"}

    name, value, tokens_seen = (log_metric["metric_name"],
                                 log_metric["metric_value"],
                                 log_metric["tokens_seen"])
    flag = "ok"
    print(f"  [cross-check] training-log {name}={value} at tokens_seen={tokens_seen}"
          + (f"  |  BLiMP-sample ppl={blimp_ppl:.1f}" if blimp_ppl is not None else ""))

    if value is None:
        flag = "training_log_metric_missing"
        print("  [cross-check] WARNING: training-log metric missing/NaN for "
              "this checkpoint — treat BLiMP result as unverified")
    elif name == "val_ppl" and (value != value or value in (float("inf"), float("-inf"))):
        flag = "training_log_nan_inf"
        print("  [cross-check] WARNING: training-log val_ppl is NaN/Inf — "
              "this checkpoint may be corrupted")
    elif blimp_ppl is not None and name == "val_ppl":
        # Different data distributions (BLiMP vs BabyLM val), so exact match
        # isn't expected — but an order-of-magnitude blowup is a red flag,
        # e.g. a checkpoint caught mid-DSN-rescaling producing incoherent output.
        ratio = blimp_ppl / max(value, 1e-6)
        if ratio > 10 or ratio < 0.1:
            flag = "order_of_magnitude_mismatch"
            print(f"  [cross-check] WARNING: BLiMP-sample ppl is {ratio:.1f}x "
                  f"the training-log val_ppl — check that '{checkpoint}' is a "
                  f"settled checkpoint, not a mid-correction snapshot")

    return {"training_log_metric_name": name, "training_log_metric_value": value,
            "training_log_tokens_seen": tokens_seen, "blimp_ppl": blimp_ppl,
            "cross_check_flag": flag}


# ── Main ────────────────────────────────────────────────────────────────────

def resolve_checkpoint_path(run_dir, checkpoint):
    """
    checkpoint: 'final' | 'best' | a literal checkpoint dir name
    (e.g. 'tokens_0050000000')
    """
    if checkpoint in ("final", "best"):
        return os.path.join(run_dir, checkpoint, "model.pt")
    return os.path.join(run_dir, "checkpoints", checkpoint, "model.pt")


def main(args):
    cfg = load_config(args.config)
    task_name = cfg.get("task", "babylm_strict_small")
    task = get_task(task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, tokenizer, _ = _load_babylm_config_and_tokenizer(
        block_size_override=cfg.get("model", {}).get("block_size"))

    phenomena = args.phenomena if args.phenomena else DEFAULT_PHENOMENA
    if args.all_phenomena:
        phenomena = ALL_BLIMP_PHENOMENA

    run_dirs = args.run_dirs if args.run_dirs else [args.run_dir]
    results_dir = cfg.get("results_dir", "results/babylm_strict_small")
    os.makedirs(results_dir, exist_ok=True)

    per_row_records = []
    summary_records = []

    for run_dir in run_dirs:
        run_name = os.path.basename(os.path.normpath(run_dir))
        ckpt_path = resolve_checkpoint_path(run_dir, args.checkpoint)
        if not os.path.exists(ckpt_path):
            print(f"[MISSING] {ckpt_path} — skipping {run_name}")
            continue

        print(f"\n=== {run_name} ({ckpt_path}) ===")
        model = load_model_from_checkpoint(ckpt_path, task_meta=None, cfg=cfg)
        model = model.to(device)

        per_phenomenon, overall, n_total, blimp_ppl = evaluate_blimp(
            model, tokenizer, device, phenomena,
            max_pairs_per_phenomenon=args.max_pairs_per_phenomenon)

        cross_check = cross_check_checkpoint(run_dir, args.checkpoint, blimp_ppl)

        for phenomenon, (acc, n) in per_phenomenon.items():
            per_row_records.append({
                "run": run_name, "phenomenon": phenomenon,
                "accuracy": acc, "n_pairs": n,
            })
        summary_records.append({
            "run": run_name, "overall_accuracy": overall, "n_total_pairs": n_total,
            "blimp_ppl": blimp_ppl,
            "training_log_metric_name": cross_check["training_log_metric_name"],
            "training_log_metric_value": cross_check["training_log_metric_value"],
            "training_log_tokens_seen": cross_check["training_log_tokens_seen"],
            "cross_check_flag": cross_check["cross_check_flag"],
        })
        print(f"  -> overall BLiMP accuracy: {overall:.4f}  (n={n_total})"
              if overall is not None else "  -> no evaluable pairs")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_row_path = os.path.join(results_dir, "blimp_results.csv")
    with open(per_row_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "phenomenon", "accuracy", "n_pairs"])
        w.writeheader()
        w.writerows(per_row_records)
    print(f"\n✓ Per-phenomenon results: {per_row_path}")

    summary_fields = ["run", "overall_accuracy", "n_total_pairs", "blimp_ppl",
                       "training_log_metric_name", "training_log_metric_value",
                       "training_log_tokens_seen", "cross_check_flag"]
    summary_path = os.path.join(results_dir, "blimp_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_records)
    print(f"✓ Summary: {summary_path}")

    print("\nSummary:")
    for r in summary_records:
        acc_str = f"{r['overall_accuracy']:.4f}" if r["overall_accuracy"] is not None else "NA"
        flag = r["cross_check_flag"]
        flag_str = f"  [{flag}]" if flag != "ok" else ""
        print(f"  {r['run']:<30} {acc_str}  (n={r['n_total_pairs']}){flag_str}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None,
                   help="Single run directory (use --run_dirs for multiple)")
    p.add_argument("--run_dirs", nargs="+", default=None,
                   help="Multiple run directories to evaluate in one pass")
    p.add_argument("--checkpoint", default="final",
                   help="'final' | 'best' | literal checkpoint dir name")
    p.add_argument("--phenomena", nargs="+", default=None,
                   help="BLiMP phenomenon names to evaluate (default: DEFAULT_PHENOMENA)")
    p.add_argument("--all_phenomena", action="store_true",
                   help="Evaluate the full BLiMP suite instead of the default subset")
    p.add_argument("--max_pairs_per_phenomenon", type=int, default=None,
                   help="Cap pairs per phenomenon for a quick pass (e.g. 200)")
    args = p.parse_args()
    if not args.run_dir and not args.run_dirs:
        p.error("--run_dir or --run_dirs is required")
    main(args)