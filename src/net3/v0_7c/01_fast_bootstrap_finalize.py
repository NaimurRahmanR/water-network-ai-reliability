
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    confusion_matrix,
)

BASELINE_THRESHOLD = 0.464201702923
GRAPH_THRESHOLD = 0.376129878873
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1618033
ECE_BINS = 15


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ece_score(y, p, bins=ECE_BINS):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def compute_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.c_[1-p, p], labels=[0,1])),
        "ece15": float(ece_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def metric_delta(y, pm, pg, key):
    if key == "balanced_accuracy":
        return (
            balanced_accuracy_score(y, pm >= BASELINE_THRESHOLD)
            - balanced_accuracy_score(y, pg >= GRAPH_THRESHOLD)
        )
    if key == "f1":
        return (
            f1_score(y, pm >= BASELINE_THRESHOLD, zero_division=0)
            - f1_score(y, pg >= GRAPH_THRESHOLD, zero_division=0)
        )
    if key == "auprc":
        return average_precision_score(y, pm) - average_precision_score(y, pg)
    if key == "auroc":
        return roc_auc_score(y, pm) - roc_auc_score(y, pg)
    if key == "brier_advantage":
        # Positive means MLP better (lower Brier).
        return (-brier_score_loss(y, pm)) - (-brier_score_loss(y, pg))
    raise ValueError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise SystemExit(
            "heldout_test_predictions.csv not found.\n"
            "The v0.7b run must have reached [BOOTSTRAP] first."
        )

    df = pd.read_csv(pred_path)

    required = {
        "context_id",
        "target_active_leak",
        "mlp_probability",
        "gcn_probability",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Prediction file missing columns: {sorted(missing)}")

    if len(df) != 2700:
        raise SystemExit(f"Expected 2700 rows, got {len(df)}")
    if df["context_id"].nunique() != 30:
        raise SystemExit(
            f"Expected 30 contexts, got {df['context_id'].nunique()}"
        )

    y = df["target_active_leak"].to_numpy(dtype=int)
    pm = df["mlp_probability"].to_numpy(dtype=float)
    pg = df["gcn_probability"].to_numpy(dtype=float)

    if not np.isfinite(pm).all() or not np.isfinite(pg).all():
        raise SystemExit("Non-finite probabilities found.")

    mlp_metrics = compute_metrics(y, pm, BASELINE_THRESHOLD)
    gcn_metrics = compute_metrics(y, pg, GRAPH_THRESHOLD)

    # Precompute integer row indices per context ONCE.
    contexts = np.array(sorted(df["context_id"].unique()))
    context_indices = {
        cid: np.flatnonzero(df["context_id"].to_numpy() == cid)
        for cid in contexts
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_keys = [
        "balanced_accuracy",
        "f1",
        "auprc",
        "auroc",
        "brier_advantage",
    ]
    samples = {k: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
               for k in metric_keys}

    print("FAST PAIRED CONTEXT-CLUSTER BOOTSTRAP")
    print("=" * 72)
    print(f"Prediction SHA-256: {sha256(pred_path)}")
    print(f"Rows: {len(df)}")
    print(f"Contexts: {len(contexts)}")
    print(f"Resamples: {BOOTSTRAP_RESAMPLES}")
    print(f"Seed: {BOOTSTRAP_SEED}")
    print()

    for b in range(BOOTSTRAP_RESAMPLES):
        sampled_contexts = rng.choice(
            contexts,
            size=len(contexts),
            replace=True,
        )
        idx = np.concatenate(
            [context_indices[cid] for cid in sampled_contexts]
        )

        by = y[idx]
        bpm = pm[idx]
        bpg = pg[idx]

        for key in metric_keys:
            samples[key][b] = metric_delta(by, bpm, bpg, key)

        if (b + 1) % 250 == 0:
            print(f"[OK] {b+1}/{BOOTSTRAP_RESAMPLES}")

    bootstrap = {}
    for key in metric_keys:
        arr = samples[key]
        bootstrap[key] = {
            "mean_delta": float(arr.mean()),
            "ci95": [
                float(np.quantile(arr, 0.025)),
                float(np.quantile(arr, 0.975)),
            ],
            "delta_definition": "MLP minus GCN-GRU",
        }

    record = {
        "status": "PASS",
        "source": "heldout_test_predictions.csv generated by v0.7b before slow bootstrap",
        "prediction_sha256": sha256(pred_path),
        "models_rerun": False,
        "predictions_regenerated": False,
        "mlp": mlp_metrics,
        "gcn_gru": gcn_metrics,
        "bootstrap_mlp_minus_gcn": bootstrap,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    metrics_path = out / "heldout_test_metrics_v0_7c.json"
    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    pretty = {
        "balanced_accuracy": "balanced accuracy",
        "f1": "F1",
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "brier_advantage": "Brier advantage",
    }

    lines = [
        "AQUA-VERA HELD-OUT TEST FINALIZER v0.7c",
        "=" * 78,
        "STATUS: PASS",
        "",
        "SOURCE",
        "  Frozen predictions from v0.7b: YES",
        "  Models rerun: NO",
        "  Predictions regenerated: NO",
        f"  prediction SHA-256: {sha256(pred_path)}",
        "",
        "TEST SET",
        f"  contexts: {len(contexts)}",
        f"  windows: {len(df):,}",
        f"  positives: {int(y.sum()):,}",
        f"  negatives: {int((1-y).sum()):,}",
        "",
        "NON-GRAPH MLP",
        f"  balanced accuracy: {mlp_metrics['balanced_accuracy']:.6f}",
        f"  accuracy: {mlp_metrics['accuracy']:.6f}",
        f"  F1: {mlp_metrics['f1']:.6f}",
        f"  AUPRC: {mlp_metrics['auprc']:.6f}",
        f"  AUROC: {mlp_metrics['auroc']:.6f}",
        f"  Brier: {mlp_metrics['brier']:.6f}",
        f"  ECE15: {mlp_metrics['ece15']:.6f}",
        f"  TN/FP/FN/TP: {mlp_metrics['tn']}/{mlp_metrics['fp']}/{mlp_metrics['fn']}/{mlp_metrics['tp']}",
        "",
        "GCN-GRU",
        f"  balanced accuracy: {gcn_metrics['balanced_accuracy']:.6f}",
        f"  accuracy: {gcn_metrics['accuracy']:.6f}",
        f"  F1: {gcn_metrics['f1']:.6f}",
        f"  AUPRC: {gcn_metrics['auprc']:.6f}",
        f"  AUROC: {gcn_metrics['auroc']:.6f}",
        f"  Brier: {gcn_metrics['brier']:.6f}",
        f"  ECE15: {gcn_metrics['ece15']:.6f}",
        f"  TN/FP/FN/TP: {gcn_metrics['tn']}/{gcn_metrics['fp']}/{gcn_metrics['fn']}/{gcn_metrics['tp']}",
        "",
        "PAIRED CONTEXT-CLUSTER BOOTSTRAP: MLP - GCN-GRU",
    ]

    for key in metric_keys:
        b = bootstrap[key]
        lines.append(
            f"  {pretty[key]}: {b['mean_delta']:.6f} "
            f"95% CI [{b['ci95'][0]:.6f}, {b['ci95'][1]:.6f}]"
        )

    lines += [
        f"  resamples: {BOOTSTRAP_RESAMPLES}",
        f"  seed: {BOOTSTRAP_SEED}",
        "",
        "INTEGRITY",
        "  v0.7b predictions reused byte-for-byte: YES",
        "  no model retraining: YES",
        "  no threshold changes: YES",
        "",
        "NEXT",
        "  Retain heldout_test_summary_v0_7c.txt with the project record.",
    ]

    summary_path = out / "heldout_test_summary_v0_7c.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("COMPLETE")
    print(summary_path)


if __name__ == "__main__":
    main()
