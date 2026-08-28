"""
collate_results.py — Aim-driven collation of rank dynamics experiment results.

Produces one CSV per aim, each directly answering the stated objectives.
All outputs are domain-aware (vision / language) and rank-aware (r=8,16,32).

Outputs
-------
aim1_divergence.csv       — Aim 1: when and how sharply does FR-LR split occur?
aim2_stabilisation.csv    — Aim 2: when does each run stabilise, and by what mechanism?
aim3_grad_erank.csv       — Aim 3: gradient eRank gap magnitude and constancy
aim4_dsn.csv              — Aim 4: spectral correction impact at init and during training
aim5_performance_gap.csv  — Aim 5: normalised performance gap vs rank
aim6_recovery.csv         — Aim 6: how much of the gap is recoverable via fine-tuning?
rank_trajectory.csv       — Supporting: all key metrics at final checkpoint vs rank
report.txt                — Human-readable summary answering each objective

Usage
-----
    # Single domain
    python collate_results.py --config configs/cifar10.yaml

    # Cross-domain (pass both configs; outputs written to --out_dir)
    python collate_results.py \\
        --config configs/cifar10.yaml \\
        --config2 configs/babylm_strict_small.yaml \\
        --out_dir results/cross_domain
"""

import os, csv, math, argparse, collections
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def sf(x):
    """safe float — returns None on failure or NaN"""
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def write_csv(path, rows, fields):
    if not rows:
        print(f"  [EMPTY]   {path}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written:  {path}  ({len(rows)} rows)")

def fmt(v, decimals=4):
    if v is None: return "NA"
    return f"{v:.{decimals}f}"


# ── Run discovery ─────────────────────────────────────────────────────────

def discover_runs(base_dir):
    """
    Return a dict mapping run_label -> run_dir for all runs found under base_dir.
    Discovers FR, LR_rN, LR_rN_seedM, LR_rN_dsn_*, FR_seedN automatically.
    """
    runs = {}
    if not os.path.isdir(base_dir):
        return runs
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        # Accept FR, FR_seedN, LR_rN, LR_rN_seedM, LR_rN_dsn_*
        if (name == "FR" or
                name.startswith("FR_seed") or
                name.startswith("LR_r")):
            runs[name] = full
    return runs


def extract_rank(run_name):
    """Return integer rank from run name, or None for FR."""
    if run_name.startswith("LR_r"):
        parts = run_name.split("_")
        try:
            return int(parts[1][1:])
        except (IndexError, ValueError):
            return None
    return None


def is_fr(run_name):
    return run_name == "FR" or run_name.startswith("FR_seed")


def is_lr_primary(run_name):
    """LR_rN with no seed and no dsn suffix."""
    if not run_name.startswith("LR_r"):
        return False
    return "_seed" not in run_name and "_dsn_" not in run_name


def is_dsn(run_name):
    return "_dsn_" in run_name


def checkpoint_label(ckpt_name):
    """
    Convert checkpoint directory name to a sortable numeric label.
    epoch_NN       -> integer N
    tokens_NNNNN   -> integer N
    """
    if ckpt_name.startswith("epoch_"):
        return int(ckpt_name.split("_")[1])
    elif ckpt_name.startswith("tokens_"):
        return int(ckpt_name.split("_")[1])
    return 0


def load_probes(run_dir):
    """Load probes.csv, return rows grouped by (checkpoint_idx, weight_name)."""
    path = os.path.join(run_dir, "probes", "probes.csv")
    rows = read_csv(path)
    if not rows:
        return {}, []
    # Determine checkpoint column name
    ckpt_col = "checkpoint_name" if "checkpoint_name" in rows[0] else None
    by_ckpt  = collections.defaultdict(list)
    for r in rows:
        ep  = sf(r.get("epoch", 0)) or 0
        ck  = r.get(ckpt_col, f"epoch_{int(ep):02d}") if ckpt_col else f"epoch_{int(ep):02d}"
        by_ckpt[ck].append(r)
    return by_ckpt, sorted(by_ckpt.keys(), key=checkpoint_label)


def load_cka_cross(run_dir):
    """Load cka_cross.csv, return {ckpt_label: {layer: cka}}."""
    path = os.path.join(run_dir, "probes", "cka_cross.csv")
    rows = read_csv(path)
    if not rows:
        return {}
    ckpt_col = "checkpoint_name" if "checkpoint_name" in rows[0] else None
    out = collections.defaultdict(dict)
    for r in rows:
        ep  = sf(r.get("epoch", 0)) or 0
        ck  = r.get(ckpt_col, f"epoch_{int(ep):02d}") if ckpt_col else f"epoch_{int(ep):02d}"
        layer = int(r.get("layer", 0))
        cka   = sf(r.get("cka_vs_ref") or r.get("cka"))
        out[ck][layer] = cka
    return dict(out)


def load_cka_self(run_dir):
    """Load cka_self.csv, return {ckpt_label: {layer: (cka_vs_prev, cka_vs_init)}}."""
    path = os.path.join(run_dir, "probes", "cka_self.csv")
    rows = read_csv(path)
    if not rows:
        return {}
    ckpt_col = "checkpoint_name" if "checkpoint_name" in rows[0] else None
    out = collections.defaultdict(dict)
    for r in rows:
        ep  = sf(r.get("epoch", 0)) or 0
        ck  = r.get(ckpt_col, f"epoch_{int(ep):02d}") if ckpt_col else f"epoch_{int(ep):02d}"
        layer = int(r.get("layer", 0))
        out[ck][layer] = (
            sf(r.get("cka_vs_prev")),
            sf(r.get("cka_vs_epoch0"))
        )
    return dict(out)


def load_training_log(run_dir):
    """
    Load log.csv. Returns list of dicts with keys:
      checkpoint, metric_value, metric_name, loss
    Handles both epoch-mode (val_acc) and token-budget mode (val_ppl).
    """
    path = os.path.join(run_dir, "log.csv")
    rows = read_csv(path)
    if not rows:
        return []
    out = []
    for r in rows:
        # Determine metric — acc for vision, ppl for language
        metric_val  = sf(r.get("val_acc")) or sf(r.get("val_ppl"))
        metric_name = "val_acc" if sf(r.get("val_acc")) is not None else "val_ppl"
        tokens      = sf(r.get("tokens_seen")) or sf(r.get("epoch"))
        loss        = sf(r.get("val_loss"))
        if metric_val is not None:
            out.append({
                "checkpoint":   tokens,
                "metric_value": metric_val,
                "metric_name":  metric_name,
                "val_loss":     loss,
            })
    return out


# ── AIM 1 — Divergence ────────────────────────────────────────────────────

def aim1_divergence(base_dir, out_dir, domain, runs):
    """
    Aim 1: When and how sharply does FR-LR split occur?

    Objectives addressed:
      1.1 — Is the divergence gradual or abrupt? (mean CKA drop at t=1 vs t=0)
      1.2 — Does the divergence point differ across ranks?
      1.3/1.4 — Answered cross-domain when called for both domains.

    One row per (domain, rank, checkpoint) with mean CKA and L0 CKA.
    Also includes FR_vs_FR baseline for reference.
    """
    rows_out = []
    fr_run   = "FR"

    # FR-FR baseline
    ff_dir = os.path.join(base_dir, "FR_seed1")
    ff_cka = load_cka_cross(ff_dir)
    if ff_cka:
        for ck in sorted(ff_cka.keys(), key=checkpoint_label):
            layer_vals = [v for v in ff_cka[ck].values() if v is not None]
            rows_out.append({
                "domain":        domain,
                "comparison":    "FR_vs_FR",
                "rank":          "FR",
                "checkpoint":    checkpoint_label(ck),
                "checkpoint_name": ck,
                "mean_cka":      mean(layer_vals),
                "cka_L0":        ff_cka[ck].get(0),
                "cka_Lmid":      ff_cka[ck].get(len(ff_cka[ck])//2),
                "cka_Llast":     ff_cka[ck].get(max(ff_cka[ck].keys())),
                "n_layers":      len(layer_vals),
            })

    # FR vs LR_rN for each rank
    for run_name, run_dir in runs.items():
        if not is_lr_primary(run_name):
            continue
        rank = extract_rank(run_name)
        cka  = load_cka_cross(run_dir)
        if not cka:
            continue
        for ck in sorted(cka.keys(), key=checkpoint_label):
            layer_vals = [v for v in cka[ck].values() if v is not None]
            rows_out.append({
                "domain":        domain,
                "comparison":    f"FR_vs_LR_r{rank}",
                "rank":          rank,
                "checkpoint":    checkpoint_label(ck),
                "checkpoint_name": ck,
                "mean_cka":      mean(layer_vals),
                "cka_L0":        cka[ck].get(0),
                "cka_Lmid":      cka[ck].get(len(cka[ck])//2),
                "cka_Llast":     cka[ck].get(max(cka[ck].keys())),
                "n_layers":      len(layer_vals),
            })

    fields = ["domain","comparison","rank","checkpoint","checkpoint_name",
              "mean_cka","cka_L0","cka_Lmid","cka_Llast","n_layers"]
    return rows_out, fields


# ── AIM 2 — Stabilisation ─────────────────────────────────────────────────

def aim2_stabilisation(base_dir, out_dir, domain, runs):
    """
    Aim 2: When does each run stabilise, and by what mechanism?

    Objectives:
      2.1 — Do FR and LR stabilise at the same checkpoint?
      2.2 — Is LR stabilisation associated with capacity exhaustion?
            (cka_vs_prev high early, cka_vs_epoch0 low — moved far but stopped)
      2.3 — How does stabilisation checkpoint change with rank?

    One row per (domain, run, layer) with:
      stab_checkpoint — first checkpoint where cka_vs_prev > STAB_THRESHOLD
      final_cka_vs_init — cka_vs_epoch0 at the stabilisation point
      mechanism — 'exhaustion' if cka_vs_prev high while cka_vs_init low, else 'convergence'
    """
    # Stabilisation detection for sparse token-based checkpoints:
    # The naive "first checkpoint above threshold" catches pre-learning
    # similarity (model hasn't moved yet). We instead find the LAST
    # contiguous run above threshold — i.e. the genuine convergence
    # after active learning has finished.
    # We also require that at least one checkpoint was BELOW the threshold
    # before this stable run (to exclude trivial pre-learning stability).
    STAB_THRESHOLD = 0.95
    rows_out = []

    target_runs = {k: v for k, v in runs.items()
                   if is_fr(k) and "_seed" not in k}
    target_runs.update({k: v for k, v in runs.items() if is_lr_primary(k)})

    for run_name, run_dir in target_runs.items():
        rank     = extract_rank(run_name) if not is_fr(run_name) else "FR"
        cka_self = load_cka_self(run_dir)
        if not cka_self:
            continue
        ckpts      = sorted(cka_self.keys(), key=checkpoint_label)
        all_layers = sorted(set(
            l for ck in cka_self.values() for l in ck.keys()
        ))
        for layer in all_layers:
            stab_ck      = None
            init_at_stab = None

            # Collect all (ck, prev, init) tuples with non-None prev
            traj = []
            for ck in ckpts:
                prev, init = cka_self[ck].get(layer, (None, None))
                if prev is not None:
                    traj.append((checkpoint_label(ck), prev, init))

            # Find whether the trajectory dips below threshold at any point
            has_dip = any(prev < STAB_THRESHOLD for _, prev, _ in traj)

            if has_dip:
                # Find the first checkpoint after the last dip that stays
                # above threshold — genuine post-learning stabilisation
                last_dip_idx = max(i for i, (_, prev, _) in enumerate(traj)
                                   if prev < STAB_THRESHOLD)
                # First checkpoint after last dip that is above threshold
                for ck_label, prev, init in traj[last_dip_idx + 1:]:
                    if prev > STAB_THRESHOLD:
                        stab_ck      = ck_label
                        init_at_stab = init
                        break

            last_ck = ckpts[-1] if ckpts else None
            final_prev, final_init = cka_self[last_ck].get(layer, (None, None)) \
                                     if last_ck else (None, None)

            # Mechanism — relative rather than absolute threshold:
            # exhaustion:   stabilised (cka_vs_prev high) but cka_vs_init
            #               continues to decline after stabilisation point,
            #               indicating representations keep drifting from init
            #               even after step-to-step changes appear small.
            #               Also flag if cka_vs_init < 0.65 (absolute, for vision).
            # convergence:  stabilised and cka_vs_init is stable or rising.
            # never_stable: never reached threshold for CONSEC checkpoints.
            if stab_ck is None:
                mechanism = "never_stable"
            else:
                init_drop = None
                if init_at_stab is not None and final_init is not None:
                    init_drop = init_at_stab - final_init  # positive = kept drifting
                absolute_exhaustion = (init_at_stab is not None and
                                       init_at_stab < 0.65)
                relative_exhaustion = (init_drop is not None and
                                       init_drop > 0.05)   # 5pp further drift after stab
                if absolute_exhaustion or relative_exhaustion:
                    mechanism = "exhaustion"
                else:
                    mechanism = "convergence"

            rows_out.append({
                "domain":              domain,
                "run":                 run_name,
                "rank":                rank,
                "layer":               layer,
                "stab_checkpoint":     stab_ck,
                "cka_vs_init_at_stab": init_at_stab,
                "final_cka_vs_prev":   final_prev,
                "final_cka_vs_init":   final_init,
                "mechanism":           mechanism,
            })

    fields = ["domain","run","rank","layer","stab_checkpoint",
              "cka_vs_init_at_stab","final_cka_vs_prev",
              "final_cka_vs_init","mechanism"]
    return rows_out, fields


# ── AIM 3 — Gradient eRank gap ────────────────────────────────────────────

def aim3_grad_erank(base_dir, out_dir, domain, runs):
    """
    Aim 3: Is the gradient eRank gap constant throughout training?

    Objectives:
      3.1 — Constant or changing over training?
      3.2 — How does it scale with rank?

    One row per (domain, checkpoint) with FR grad_erank, LR_rN grad_erank,
    and the ratio FR/LR for each rank.
    """
    rows_out = []

    # Load FR probes
    fr_dir    = os.path.join(base_dir, "FR")
    fr_by_ck, fr_ckpts = load_probes(fr_dir)

    # Load each LR rank
    lr_data = {}
    for run_name, run_dir in runs.items():
        if not is_lr_primary(run_name):
            continue
        rank = extract_rank(run_name)
        by_ck, ckpts = load_probes(run_dir)
        lr_data[rank] = (by_ck, ckpts)

    if not fr_by_ck:
        return rows_out, []

    for ck in fr_ckpts:
        fr_rows    = fr_by_ck.get(ck, [])
        # Filter to attention weights; fall back to all weights if none found
        attn_rows  = [r for r in fr_rows if "attn" in r.get("weight_name","")]
        use_rows   = attn_rows if attn_rows else fr_rows
        fr_ge      = mean([sf(r.get("grad_erank_approx")) for r in use_rows])
        fr_erank   = mean([sf(r.get("erank")) for r in use_rows])
        row = {
            "domain":          domain,
            "checkpoint":      checkpoint_label(ck),
            "checkpoint_name": ck,
            "fr_grad_erank":   fr_ge,
            "fr_erank":        fr_erank,
        }
        for rank, (lr_by_ck, _) in lr_data.items():
            lr_rows  = lr_by_ck.get(ck, [])
            attn_lr  = [r for r in lr_rows if "attn" in r.get("weight_name","")]
            use_lr   = attn_lr if attn_lr else lr_rows
            lr_ge    = mean([sf(r.get("grad_erank_approx")) for r in use_lr])
            lr_erank = mean([sf(r.get("erank")) for r in use_lr])
            ratio    = (fr_ge / lr_ge) if (fr_ge and lr_ge) else None
            row[f"lr_r{rank}_grad_erank"]   = lr_ge
            row[f"lr_r{rank}_erank"]        = lr_erank
            row[f"grad_erank_ratio_r{rank}"] = ratio
        rows_out.append(row)

    # Build field list dynamically
    fields = ["domain","checkpoint","checkpoint_name","fr_grad_erank","fr_erank"]
    for rank in sorted(lr_data.keys()):
        fields += [f"lr_r{rank}_grad_erank", f"lr_r{rank}_erank",
                   f"grad_erank_ratio_r{rank}"]
    return rows_out, fields


# ── AIM 4 — DSN ───────────────────────────────────────────────────────────

def aim4_dsn(base_dir, out_dir, domain, runs):
    """
    Aim 4: Does spectral correction at init or during training reduce divergence?

    Objectives:
      4.1 — Does alpha-flattening at init reduce CKA divergence at t*?
            (from epoch1_intervention.csv)
      4.2 — Does continuous DSN sustain early alignment gains?
            (from DSN run CKA cross trajectories and training logs)

    Two output tables:
      - Intervention results (static, one checkpoint)
      - DSN trajectory (across all checkpoints, all modes)
    """
    rows_out = []

    # 4.1 — Epoch-1 / t* intervention
    interv_path = os.path.join(base_dir, "interventions",
                               "epoch1_intervention.csv")
    interv_rows = read_csv(interv_path)
    for r in interv_rows:
        rows_out.append({
            "domain":      domain,
            "source":      "intervention",
            "variant":     r.get("variant",""),
            "checkpoint":  "t_star",
            "val_metric":  sf(r.get("val_acc")) or sf(r.get("val_ppl")),
            "rel_drift":   sf(r.get("rel_drift")),
            "mean_cka":    sf(r.get("mean_cka")),
            "cka_L0":      sf(r.get("cka_L0")),
            "dsn_mode":    None,
        })

    # 4.2 — DSN training trajectories
    dsn_modes = ["static", "dynamic", "flat"]
    # Pick the lowest primary rank (r=8 by convention) — DSN runs are
    # named LR_r{primary_rank}_dsn_{mode}. Sort numerically to avoid
    # alphabetical ordering picking r128 before r8.
    primary_rank = None
    lr_primaries = sorted(
        [run_name for run_name in runs if is_lr_primary(run_name)],
        key=lambda n: extract_rank(n) or 9999
    )
    if lr_primaries:
        primary_rank = extract_rank(lr_primaries[0])

    print(f"  [aim4] primary_rank={primary_rank}")
    for mode in dsn_modes:
        run_name = f"LR_r{primary_rank}_dsn_{mode}" if primary_rank else None
        if not run_name:
            continue
        run_dir = os.path.join(base_dir, run_name)
        exists  = os.path.isdir(run_dir)
        print(f"  [aim4] DSN {mode}: {run_dir}  exists={exists}")
        cka     = load_cka_cross(run_dir)
        tr      = load_training_log(run_dir)
        print(f"  [aim4]   cka checkpoints={len(cka)}  log_rows={len(tr)}")
        # Best metric
        best_metric = None
        if tr:
            metrics = [r["metric_value"] for r in tr if r["metric_value"]]
            if metrics:
                best_metric = (max(metrics) if tr[0]["metric_name"] == "val_acc"
                               else min(metrics))
        # Per-checkpoint CKA
        for ck, layer_ckas in cka.items():
            layer_vals = [v for v in layer_ckas.values() if v is not None]
            rows_out.append({
                "domain":      domain,
                "source":      "dsn_trajectory",
                "variant":     mode,
                "checkpoint":  checkpoint_label(ck),
                "val_metric":  None,
                "rel_drift":   None,
                "mean_cka":    mean(layer_vals),
                "cka_L0":      layer_ckas.get(0),
                "dsn_mode":    mode,
            })
        # Add best metric row
        if best_metric is not None:
            rows_out.append({
                "domain":      domain,
                "source":      "dsn_best_metric",
                "variant":     mode,
                "checkpoint":  "final",
                "val_metric":  best_metric,
                "rel_drift":   None,
                "mean_cka":    None,
                "cka_L0":      None,
                "dsn_mode":    mode,
            })

    fields = ["domain","source","variant","checkpoint","val_metric",
              "rel_drift","mean_cka","cka_L0","dsn_mode"]
    return rows_out, fields


# ── AIM 5 — Performance gap vs rank ──────────────────────────────────────

def aim5_performance_gap(base_dir, out_dir, domain, runs):
    """
    Aim 5: How does the performance cost of rank constraint scale with rank?

    Objectives:
      5.1 — Relationship between rank and final task performance (per domain)
      5.2/5.3 — Normalised gap vs rank
      5.4 — Cross-domain comparison when both domains present

    One row per (domain, rank) with:
      fr_metric, lr_metric, absolute_gap, normalised_gap
    """
    rows_out = []

    # FR best metric
    fr_tr = load_training_log(os.path.join(base_dir, "FR"))
    if not fr_tr:
        return rows_out, []
    fr_metrics  = [r["metric_value"] for r in fr_tr if r["metric_value"]]
    metric_name = fr_tr[0]["metric_name"] if fr_tr else "val_acc"
    higher_better = metric_name == "val_acc"
    fr_best     = max(fr_metrics) if higher_better else min(fr_metrics)
    fr_worst    = min(fr_metrics) if higher_better else max(fr_metrics)

    for run_name, run_dir in runs.items():
        if not is_lr_primary(run_name):
            continue
        rank = extract_rank(run_name)
        tr   = load_training_log(run_dir)
        if not tr:
            continue
        lr_metrics = [r["metric_value"] for r in tr if r["metric_value"]]
        if not lr_metrics:
            continue
        lr_best = max(lr_metrics) if higher_better else min(lr_metrics)
        lr_final = lr_metrics[-1]

        abs_gap  = (fr_best - lr_best) if higher_better else (lr_best - fr_best)
        # Normalised gap: gap / fr_best  (fraction of FR performance lost)
        norm_gap = abs_gap / abs(fr_best) if fr_best else None

        rows_out.append({
            "domain":          domain,
            "rank":            rank,
            "metric_name":     metric_name,
            "fr_best":         fr_best,
            "lr_best":         lr_best,
            "lr_final":        lr_final,
            "absolute_gap":    abs_gap,
            "normalised_gap":  norm_gap,
        })

    fields = ["domain","rank","metric_name","fr_best","lr_best",
              "lr_final","absolute_gap","normalised_gap"]
    return rows_out, fields


# ── AIM 6 — Recovery ─────────────────────────────────────────────────────

def aim6_recovery(base_dir, out_dir, domain, runs):
    """
    Aim 6: How much of the performance gap is recoverable via fine-tuning?

    Objectives:
      6.1 — Gap closed by full fine-tuning
      6.2 — Effect of which layers are unfrozen (k-sweep)
      6.3 — Residual gap that persists

    Reads ablation_results.csv for both 10ep and convergence budgets.
    """
    rows_out = []
    sources  = {
        "short":       os.path.join(base_dir, "ablation",
                                    "ablation_results.csv"),
        "convergence": os.path.join(base_dir, "ablation_convergence",
                                    "ablation_results.csv"),
    }

    # FR ceiling and LR baseline from training logs
    fr_tr = load_training_log(os.path.join(base_dir, "FR"))
    fr_metrics = [r["metric_value"] for r in fr_tr if r["metric_value"]] if fr_tr else []
    metric_name = fr_tr[0]["metric_name"] if fr_tr else "val_acc"
    higher_better = metric_name == "val_acc"
    fr_best = (max(fr_metrics) if higher_better else min(fr_metrics)) if fr_metrics else None

    lr_tr = load_training_log(os.path.join(base_dir, "LR_r8"))
    lr_metrics = [r["metric_value"] for r in lr_tr if r["metric_value"]] if lr_tr else []
    lr_best = (max(lr_metrics) if higher_better else min(lr_metrics)) if lr_metrics else None

    total_gap = abs(fr_best - lr_best) if (fr_best and lr_best) else None

    for budget, path in sources.items():
        for r in read_csv(path):
            metric = sf(r.get("best_val_acc") or r.get("best_val_ppl"))
            if metric is None:
                continue
            gap_closed = None
            residual   = None
            if total_gap and lr_best is not None and fr_best is not None:
                improvement = abs(metric - lr_best)
                gap_closed  = improvement / total_gap if total_gap else None
                residual    = abs(fr_best - metric)
            rows_out.append({
                "domain":        domain,
                "budget":        budget,
                "condition":     r.get("condition",""),
                "k_reinit":      r.get("k_reinit",""),
                "metric":        metric,
                "metric_name":   metric_name,
                "fr_ceiling":    fr_best,
                "lr_baseline":   lr_best,
                "total_gap":     total_gap,
                "gap_closed":    gap_closed,
                "residual_gap":  residual,
            })

    fields = ["domain","budget","condition","k_reinit","metric","metric_name",
              "fr_ceiling","lr_baseline","total_gap","gap_closed","residual_gap"]
    return rows_out, fields


# ── RANK TRAJECTORY ────────────────────────────────────────────────────────

def rank_trajectory(base_dir, out_dir, domain, runs):
    """
    Supporting analysis: key metrics at the FINAL checkpoint vs rank.
    Feeds the scaling law figures for Aims 1, 2, 3.
    """
    rows_out = []

    fr_by_ck, fr_ckpts = load_probes(os.path.join(base_dir, "FR"))
    fr_self   = load_cka_self(os.path.join(base_dir, "FR"))
    fr_tr     = load_training_log(os.path.join(base_dir, "FR"))
    fr_metrics = [r["metric_value"] for r in fr_tr if r["metric_value"]] if fr_tr else []
    metric_name = fr_tr[0]["metric_name"] if fr_tr else "val_acc"
    higher_better = metric_name == "val_acc"
    fr_best = (max(fr_metrics) if higher_better else min(fr_metrics)) if fr_metrics else None

    # FR row — initialise to None so LR loop always has a defined reference
    fr_ge = fr_erank = fr_alpha = fr_srank = None
    if fr_ckpts:
        last_ck   = fr_ckpts[-1]
        fr_rows   = fr_by_ck.get(last_ck, [])
        attn_fr   = [r for r in fr_rows if "attn" in r.get("weight_name","")]
        use_fr    = attn_fr if attn_fr else fr_rows
        fr_ge     = mean([sf(r.get("grad_erank_approx")) for r in use_fr])
        fr_erank  = mean([sf(r.get("erank")) for r in use_fr])
        fr_alpha  = mean([sf(r.get("alpha")) for r in use_fr])
        fr_srank  = mean([sf(r.get("srank")) for r in use_fr])
        # FR stabilisation
        all_layers = sorted(set(l for ck in fr_self.values() for l in ck))
        stab_epochs = []
        for layer in all_layers:
            for ck in sorted(fr_self.keys(), key=checkpoint_label):
                prev, _ = fr_self[ck].get(layer, (None, None))
                if prev is not None and prev > 0.95:
                    stab_epochs.append(checkpoint_label(ck))
                    break
        mean_stab = mean(stab_epochs)
        rows_out.append({
            "domain": domain, "run": "FR", "rank": "FR",
            "final_checkpoint": checkpoint_label(last_ck),
            "best_metric": fr_best, "metric_name": metric_name,
            "grad_erank": fr_ge, "erank": fr_erank,
            "alpha": fr_alpha, "srank": fr_srank,
            "mean_stab_checkpoint": mean_stab,
            "grad_erank_ratio_vs_fr": 1.0,
            "normalised_gap": 0.0,
        })

    for run_name, run_dir in runs.items():
        if not is_lr_primary(run_name):
            continue
        rank     = extract_rank(run_name)
        by_ck, ckpts = load_probes(run_dir)
        lr_self  = load_cka_self(run_dir)
        lr_tr    = load_training_log(run_dir)
        lr_cka   = load_cka_cross(run_dir)

        if not ckpts:
            continue
        last_ck  = ckpts[-1]
        lr_rows  = by_ck.get(last_ck, [])

        attn_lr2  = [r for r in lr_rows if "attn" in r.get("weight_name","")]
        use_lr2   = attn_lr2 if attn_lr2 else lr_rows
        lr_ge    = mean([sf(r.get("grad_erank_approx")) for r in use_lr2])
        lr_erank = mean([sf(r.get("erank")) for r in use_lr2])
        lr_alpha = mean([sf(r.get("alpha")) for r in use_lr2])
        lr_srank = mean([sf(r.get("srank")) for r in use_lr2])

        lr_metrics_all = [r["metric_value"] for r in lr_tr
                          if r.get("metric_value")] if lr_tr else []
        lr_best  = (max(lr_metrics_all) if higher_better
                    else min(lr_metrics_all)) if lr_metrics_all else None

        # Stabilisation
        all_layers = sorted(set(l for ck in lr_self.values() for l in ck))
        stab_ckpts = []
        for layer in all_layers:
            for ck in sorted(lr_self.keys(), key=checkpoint_label):
                prev, _ = lr_self[ck].get(layer, (None, None))
                if prev is not None and prev > 0.95:
                    stab_ckpts.append(checkpoint_label(ck))
                    break
        mean_stab = mean(stab_ckpts)

        # Final CKA mean
        final_cka_mean = None
        if lr_cka:
            last_cka_ck = sorted(lr_cka.keys(), key=checkpoint_label)[-1]
            vals = [v for v in lr_cka[last_cka_ck].values() if v is not None]
            final_cka_mean = mean(vals)

        ge_ratio  = (fr_ge / lr_ge) if (fr_ge and lr_ge) else None
        abs_gap   = abs(fr_best - lr_best) if (fr_best and lr_best) else None
        norm_gap  = (abs_gap / abs(fr_best)) if (abs_gap and fr_best) else None

        rows_out.append({
            "domain": domain, "run": run_name, "rank": rank,
            "final_checkpoint": checkpoint_label(last_ck),
            "best_metric": lr_best, "metric_name": metric_name,
            "grad_erank": lr_ge, "erank": lr_erank,
            "alpha": lr_alpha, "srank": lr_srank,
            "mean_stab_checkpoint": mean_stab,
            "grad_erank_ratio_vs_fr": ge_ratio,
            "normalised_gap": norm_gap,
            "final_cka_mean_vs_fr": final_cka_mean,
        })

    fields = ["domain","run","rank","final_checkpoint","best_metric",
              "metric_name","grad_erank","erank","alpha","srank",
              "mean_stab_checkpoint","grad_erank_ratio_vs_fr",
              "normalised_gap","final_cka_mean_vs_fr"]
    return rows_out, fields


# ── REPORT ────────────────────────────────────────────────────────────────

def write_report(results, out_dir):
    lines = []
    W = lambda s: lines.append(s)

    W("=" * 72)
    W("RANK DYNAMICS — AIM-DRIVEN RESULTS REPORT")
    W("=" * 72)

    for domain, data in results.items():
        W(f"\n{'━'*72}")
        W(f"  DOMAIN: {domain.upper()}")
        W(f"{'━'*72}")

        # Aim 5 — performance gap (quick orientation)
        a5 = data.get("aim5", [])
        if a5:
            W("\nAim 5 — Performance gap vs rank")
            W(f"  {'rank':>6}  {'FR best':>10}  {'LR best':>10}  "
              f"{'abs gap':>9}  {'norm gap':>10}")
            W("  " + "-"*52)
            for r in sorted(a5, key=lambda x: (str(x["rank"]))):
                W(f"  {str(r['rank']):>6}  {fmt(r['fr_best']):>10}  "
                  f"{fmt(r['lr_best']):>10}  {fmt(r['absolute_gap']):>9}  "
                  f"{fmt(r['normalised_gap']):>10}")

        # Aim 1 — divergence point
        a1 = data.get("aim1", [])
        if a1:
            W("\nAim 1 — Divergence: mean CKA at first and final checkpoint")
            by_comp = collections.defaultdict(list)
            for r in a1:
                by_comp[r["comparison"]].append(r)
            for comp, rows in sorted(by_comp.items()):
                rows_s = sorted(rows, key=lambda x: x["checkpoint"])
                first, last = rows_s[0], rows_s[-1]
                W(f"  {comp:<25}  "
                  f"ck0={fmt(first['mean_cka'],3)}  "
                  f"ck{last['checkpoint']}={fmt(last['mean_cka'],3)}  "
                  f"L0_ck0={fmt(first['cka_L0'],3)}  "
                  f"L0_final={fmt(last['cka_L0'],3)}")

        # Aim 2 — stabilisation
        a2 = data.get("aim2", [])
        if a2:
            W("\nAim 2 — Stabilisation (mean across layers)")
            by_run = collections.defaultdict(list)
            for r in a2:
                by_run[r["run"]].append(r)
            for run, rows in sorted(by_run.items()):
                stab_vals = [r["stab_checkpoint"] for r in rows
                             if r["stab_checkpoint"] is not None]
                mech_vals = [r["mechanism"] for r in rows
                             if r["mechanism"] is not None]
                exhaustion_pct   = 100*mech_vals.count("exhaustion")/len(mech_vals) if mech_vals else 0
                convergence_pct  = 100*mech_vals.count("convergence")/len(mech_vals) if mech_vals else 0
                never_stable_pct = 100*mech_vals.count("never_stable")/len(mech_vals) if mech_vals else 0
                stab_vals_valid  = [v for v in stab_vals if v is not None]
                W(f"  {run:<25}  "
                  f"mean_stab_ck={fmt(mean(stab_vals_valid),1)}  "
                  f"exhaustion={exhaustion_pct:.0f}%  "
                  f"convergence={convergence_pct:.0f}%  "
                  f"never_stable={never_stable_pct:.0f}%")

        # Aim 3 — gradient eRank
        a3 = data.get("aim3", [])
        if a3:
            W("\nAim 3 — Gradient eRank ratio FR/LR (first and final checkpoint)")
            ratio_cols = [k for k in a3[0] if k.startswith("grad_erank_ratio")]
            rows_s = sorted(a3, key=lambda x: x["checkpoint"])
            if rows_s:
                first, last = rows_s[0], rows_s[-1]
                for col in ratio_cols:
                    rank_label = col.replace("grad_erank_ratio_","")
                    W(f"  {rank_label:<10}  "
                      f"ck0={fmt(first.get(col),2)}  "
                      f"final={fmt(last.get(col),2)}")

        # Aim 6 — recovery
        a6 = data.get("aim6", [])
        if a6:
            W("\nAim 6 — Gap recovery via fine-tuning")
            for r in a6:
                if r["gap_closed"] is not None:
                    W(f"  [{r['budget']:>12}] {r['condition']:<22}  "
                      f"k={str(r['k_reinit']):>4}  "
                      f"gap_closed={fmt(r['gap_closed'],3)}  "
                      f"residual={fmt(r['residual_gap'],4)}")

    W("\n" + "=" * 72)
    path = os.path.join(out_dir, "report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written:  {path}")
    # Also print to stdout
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────

def process_domain(base_dir, out_dir, domain):
    """Run all aims for a single domain, return dict of results."""
    runs = discover_runs(base_dir)
    print(f"\n  Discovered runs for {domain}: {list(runs.keys())}")

    data = {}

    r, f = aim1_divergence(base_dir, out_dir, domain, runs)
    data["aim1"] = r
    write_csv(os.path.join(out_dir, f"aim1_divergence_{domain}.csv"), r, f)

    r, f = aim2_stabilisation(base_dir, out_dir, domain, runs)
    data["aim2"] = r
    write_csv(os.path.join(out_dir, f"aim2_stabilisation_{domain}.csv"), r, f)

    r, f = aim3_grad_erank(base_dir, out_dir, domain, runs)
    data["aim3"] = r
    write_csv(os.path.join(out_dir, f"aim3_grad_erank_{domain}.csv"), r, f)

    r, f = aim4_dsn(base_dir, out_dir, domain, runs)
    data["aim4"] = r
    write_csv(os.path.join(out_dir, f"aim4_dsn_{domain}.csv"), r, f)

    r, f = aim5_performance_gap(base_dir, out_dir, domain, runs)
    data["aim5"] = r
    write_csv(os.path.join(out_dir, f"aim5_performance_gap_{domain}.csv"), r, f)

    r, f = aim6_recovery(base_dir, out_dir, domain, runs)
    data["aim6"] = r
    write_csv(os.path.join(out_dir, f"aim6_recovery_{domain}.csv"), r, f)

    r, f = rank_trajectory(base_dir, out_dir, domain, runs)
    data["rank_trajectory"] = r
    write_csv(os.path.join(out_dir, f"rank_trajectory_{domain}.csv"), r, f)

    return data


def merge_cross_domain(all_results, out_dir):
    """
    Merge outputs across domains into combined CSVs for Objectives 1.3, 1.4,
    2.4, 3.3, 4.3, 5.4 — the cross-domain comparisons.
    """
    for aim_key, fname in [
        ("aim1",           "aim1_divergence_combined.csv"),
        ("aim5",           "aim5_performance_gap_combined.csv"),
        ("rank_trajectory","rank_trajectory_combined.csv"),
    ]:
        combined = []
        fields   = []
        for domain, data in all_results.items():
            rows = data.get(aim_key, [])
            combined.extend(rows)
            if rows and not fields:
                fields = list(rows[0].keys())
        write_csv(os.path.join(out_dir, fname), combined, fields)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nOutput dir: {args.out_dir}")

    all_results = {}

    # Primary domain
    if args.base_dir:
        domain = args.domain or os.path.basename(args.base_dir.rstrip("/"))
        print(f"\n{'═'*60}")
        print(f"  Processing domain: {domain}")
        print(f"  Base dir:          {args.base_dir}")
        all_results[domain] = process_domain(args.base_dir, args.out_dir, domain)

    # Optional second domain (cross-domain analysis)
    if args.base_dir2:
        domain2 = args.domain2 or os.path.basename(args.base_dir2.rstrip("/"))
        print(f"\n{'═'*60}")
        print(f"  Processing domain: {domain2}")
        print(f"  Base dir:          {args.base_dir2}")
        all_results[domain2] = process_domain(args.base_dir2, args.out_dir, domain2)

    if len(all_results) > 1:
        print(f"\n{'═'*60}")
        print("  Merging cross-domain outputs...")
        merge_cross_domain(all_results, args.out_dir)

    write_report(all_results, args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default=None,
                   help="YAML config for primary domain")
    p.add_argument("--config2",   default=None,
                   help="YAML config for second domain (cross-domain analysis)")
    p.add_argument("--base_dir",  default=None,
                   help="Output root for primary domain")
    p.add_argument("--base_dir2", default=None,
                   help="Output root for second domain")
    p.add_argument("--out_dir",   default=None,
                   help="Where to write collated results")
    p.add_argument("--domain",    default=None,
                   help="Domain label for primary (inferred from base_dir if omitted)")
    p.add_argument("--domain2",   default=None,
                   help="Domain label for second domain")
    args = p.parse_args()

    # Apply config defaults
    for cfg_attr, base_attr, res_attr, dom_attr in [
        ("config",  "base_dir",  None,       "domain"),
        ("config2", "base_dir2", None,       "domain2"),
    ]:
        cfg_path = getattr(args, cfg_attr, None)
        if cfg_path:
            try:
                import yaml
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                if not getattr(args, base_attr):
                    setattr(args, base_attr, cfg.get("output_root"))
                if args.out_dir is None:
                    args.out_dir = cfg.get("results_dir", "results")
                if not getattr(args, dom_attr):
                    setattr(args, dom_attr, cfg.get("task"))
            except Exception as e:
                print(f"Warning: could not load {cfg_attr}: {e}")

    if args.base_dir is None:
        p.error("--base_dir or --config required")
    if args.out_dir is None:
        args.out_dir = "results"

    main(args)