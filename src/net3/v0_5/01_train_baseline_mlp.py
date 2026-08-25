
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
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
except ImportError:
    print("ERROR: required packages missing.")
    print("Run: py -m pip install torch scikit-learn pandas numpy")
    sys.exit(1)

EXPECTED = {
    "window_manifest.csv": "ee67fa17f169b9ac3d8b48efa49a6e634dc5ca405dca657ea5f82043c05a301d",
    "normalization_train_only.json": "a6a4bf24c805e9e02a635cd363b006aab0830ab925fe70b7c9dbe0b0f5e3e1b8",
    "window_config.json": "d6ff1aa93bf5d607b648d1e6c62c6a6a2d2728204161b372e3ea5d56974cb1a1",
}

SEED = 314159
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
HIDDEN1 = 256
HIDDEN2 = 64
ECE_BINS = 15


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str):
    if not path.exists():
        raise SystemExit(f"Missing frozen artifact: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH: {path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {path.name}")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(manifest, split, generated_dir, mean, scale):
    if split == "test":
        raise RuntimeError(
            "HELD-OUT TEST ACCESS BLOCKED in baseline-training script."
        )

    rows = manifest[manifest["split"] == split].copy()
    if rows.empty:
        raise RuntimeError(f"No rows for split={split}")

    # Group by context so each NPZ is opened once.
    xs, ys, meta_rows = [], [], []
    for cid, grp in rows.groupby("context_id", sort=True):
        p = generated_dir / "contexts" / f"{cid}.npz"
        if not p.exists():
            raise RuntimeError(f"Missing context file: {p}")

        with np.load(p, allow_pickle=False) as z:
            clean = np.asarray(z["clean_sensor"], dtype=np.float32)
            leak = np.asarray(z["leak_sensor"], dtype=np.float32)

            for _, r in grp.sort_values(["variant", "final_idx"]).iterrows():
                arr = clean if r["variant"] == "CLEAN" else leak
                start_idx = int(r["start_idx"])
                final_idx = int(r["final_idx"])
                window = arr[start_idx:final_idx+1]
                if window.shape != (24, 33):
                    raise RuntimeError(
                        f"{r['window_id']}: expected (24,33), got {window.shape}"
                    )
                # Training-only frozen normalization.
                window = (window - mean) / scale
                xs.append(window.reshape(-1).astype(np.float32))
                ys.append(int(r["target_active_leak"]))
                meta_rows.append({
                    "window_id": r["window_id"],
                    "pair_id": r["pair_id"],
                    "context_id": r["context_id"],
                    "variant": r["variant"],
                    "split": r["split"],
                    "final_idx": int(r["final_idx"]),
                    "target_active_leak": int(r["target_active_leak"]),
                })

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    expected = {"train": 13500, "validation": 2700, "calibration": 2700}[split]
    if len(y) != expected:
        raise RuntimeError(f"{split}: expected {expected} windows, got {len(y)}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"{split}: non-finite normalized feature values")

    return X, y, meta


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    # Stable sigmoid.
    out = np.empty_like(logits)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    expx = np.exp(logits[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def ece_score(y, p, bins=ECE_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": i, "lo": lo, "hi": hi, "n": 0,
                "mean_confidence": None, "fraction_positive": None
            })
            continue
        conf = float(p[mask].mean())
        frac = float(y[mask].mean())
        ece += (n / total) * abs(conf - frac)
        rows.append({
            "bin": i, "lo": lo, "hi": hi, "n": n,
            "mean_confidence": conf, "fraction_positive": frac
        })
    return float(ece), rows


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0,1]).ravel()
    ece, _ = ece_score(y, p)
    return {
        "n": int(len(y)),
        "positive_fraction": float(np.mean(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.c_[1-p, p], labels=[0,1])),
        "ece_15bin": ece,
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def logits_in_batches(model, X, batch_size=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start+batch_size])
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def choose_calibration_threshold(y, p):
    # Frozen threshold objective:
    # maximize balanced accuracy on calibration.
    # Candidate thresholds are midpoints between unique sorted probabilities
    # plus 0 and 1. Ties: choose threshold closest to 0.5, then lower threshold.
    unique = np.unique(np.asarray(p, dtype=np.float64))
    if len(unique) == 1:
        candidates = np.array([0.5], dtype=np.float64)
    else:
        mids = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.concatenate(([0.0], mids, [1.0]))

    best = None
    for t in candidates:
        pred = (p >= t).astype(int)
        score = balanced_accuracy_score(y, pred)
        key = (float(score), -abs(float(t) - 0.5), -float(t))
        if best is None or key > best[0]:
            best = (key, float(t), float(score))
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    windows_dir = Path(args.windows)
    generated_dir = Path(args.generated)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA NON-GRAPH TEMPORAL MLP BASELINE v0.5")
    print("=" * 78)

    for name, h in EXPECTED.items():
        verify(windows_dir / name, h)

    seed_everything(SEED)

    manifest = pd.read_csv(windows_dir / "window_manifest.csv")
    # Guardrail: we may inspect only split labels in the manifest, never load test features.
    split_counts = manifest["split"].value_counts().to_dict()
    expected_counts = {
        "train":13500, "validation":2700, "calibration":2700, "test":2700
    }
    if split_counts != expected_counts:
        raise SystemExit(f"Window split count mismatch: {split_counts}")

    norm = json.loads(
        (windows_dir / "normalization_train_only.json").read_text(encoding="utf-8")
    )
    if norm.get("source_split") != "train only":
        raise SystemExit("Normalization is not marked training-only.")
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["scale_used"], dtype=np.float32)
    if mean.shape != (33,) or scale.shape != (33,):
        raise SystemExit(f"Normalization shape mismatch: {mean.shape}, {scale.shape}")

    print("[LOAD] train")
    X_train, y_train, meta_train = load_split(
        manifest, "train", generated_dir, mean, scale
    )
    print("[LOAD] validation")
    X_val, y_val, meta_val = load_split(
        manifest, "validation", generated_dir, mean, scale
    )
    print("[LOAD] calibration")
    X_cal, y_cal, meta_cal = load_split(
        manifest, "calibration", generated_dir, mean, scale
    )
    print("[GUARD] test feature files not loaded")

    input_dim = X_train.shape[1]
    if input_dim != 24 * 33:
        raise SystemExit(f"Expected input dim 792, got {input_dim}")

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    if pos <= 0:
        raise SystemExit("No positive training samples.")
    pos_weight_value = neg / pos
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)

    model = MLP(input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(SEED + 1)
    best_state = None
    best_epoch = None
    best_val_auprc = -math.inf
    stale = 0
    history = []

    print("[TRAIN]")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(X_train))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start+BATCH_SIZE]
            xb = torch.from_numpy(X_train[idx])
            yb = torch.from_numpy(y_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)

        val_logits = logits_in_batches(model, X_val)
        val_prob = sigmoid_np(val_logits)
        val_auprc = float(average_precision_score(y_val, val_prob))
        val_auroc = float(roc_auc_score(y_val, val_prob))
        train_loss = total_loss / seen

        history.append({
            "epoch": epoch,
            "train_weighted_bce": train_loss,
            "validation_auprc": val_auprc,
            "validation_auroc": val_auroc,
        })
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_AUPRC={val_auprc:.6f} val_AUROC={val_auroc:.6f}"
        )

        if val_auprc > best_val_auprc + 1e-12:
            best_val_auprc = val_auprc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[EARLY STOP] no validation AUPRC improvement for {PATIENCE} epochs")
                break

    if best_state is None:
        raise SystemExit("No best model state recorded.")
    model.load_state_dict(best_state)

    # Validation metrics at conventional 0.5 are descriptive only.
    val_logits = logits_in_batches(model, X_val)
    val_prob = sigmoid_np(val_logits)
    val_metrics = metrics(y_val, val_prob, threshold=0.5)

    # Calibration-only threshold selection.
    cal_logits = logits_in_batches(model, X_cal)
    cal_prob = sigmoid_np(cal_logits)
    threshold, threshold_balacc = choose_calibration_threshold(y_cal, cal_prob)
    cal_metrics = metrics(y_cal, cal_prob, threshold=threshold)

    # Save predictions for audit. No test predictions exist.
    val_pred = meta_val.copy()
    val_pred["logit"] = val_logits
    val_pred["probability"] = val_prob
    val_pred["prediction_at_0_5"] = (val_prob >= 0.5).astype(int)
    val_pred.to_csv(out / "validation_predictions.csv", index=False)

    cal_pred = meta_cal.copy()
    cal_pred["logit"] = cal_logits
    cal_pred["probability"] = cal_prob
    cal_pred["prediction_at_frozen_threshold"] = (cal_prob >= threshold).astype(int)
    cal_pred.to_csv(out / "calibration_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    # Save exact checkpoint.
    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "hidden1": HIDDEN1,
        "hidden2": HIDDEN2,
        "dropout": DROPOUT,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_val_auprc,
        "pos_weight": pos_weight_value,
        "frozen_calibration_threshold": threshold,
    }
    model_path = out / "baseline_mlp_v0_5.pt"
    torch.save(checkpoint, model_path)

    config = {
        "status": "BASELINE_FROZEN_PRETEST",
        "model": "Flattened 2h temporal MLP",
        "architecture": [input_dim, HIDDEN1, HIDDEN2, 1],
        "activations": ["ReLU", "ReLU"],
        "dropout": DROPOUT,
        "loss": "BCEWithLogitsLoss",
        "train_pos_weight": pos_weight_value,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "early_stopping_metric": "validation AUPRC",
        "seed": SEED,
        "best_epoch": best_epoch,
        "threshold_source": "calibration only",
        "threshold_objective": "maximize balanced accuracy",
        "frozen_threshold": threshold,
        "test_split_accessed_for_features_or_predictions": False,
        "window_hashes_verified": EXPECTED,
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    config_path = out / "baseline_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics_record = {
        "validation_at_0_5": val_metrics,
        "calibration_at_frozen_threshold": cal_metrics,
        "frozen_threshold": threshold,
        "calibration_threshold_balanced_accuracy_objective": threshold_balacc,
        "test_metrics": None,
        "test_status": "NOT EVALUATED",
    }
    metrics_path = out / "pretest_metrics.json"
    metrics_path.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")

    core = [
        model_path,
        config_path,
        metrics_path,
        out / "training_history.csv",
        out / "validation_predictions.csv",
        out / "calibration_predictions.csv",
    ]
    hashes = {p.name: sha256(p) for p in core}
    freeze = {
        "status": "BASELINE_FROZEN_PRETEST",
        "artifact_sha256": hashes,
        "test_evaluated": False,
        "note": "Baseline frozen before graph-model training and before held-out test evaluation."
    }
    freeze_path = out / "FROZEN_BASELINE_RECORD.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA NON-GRAPH TEMPORAL MLP BASELINE v0.5",
        "=" * 78,
        "STATUS: PASS / BASELINE FROZEN PRE-TEST",
        "",
        "MODEL",
        f"  architecture: 792 -> {HIDDEN1} -> {HIDDEN2} -> 1",
        f"  dropout: {DROPOUT}",
        f"  weighted BCE pos_weight: {pos_weight_value:.8f}",
        f"  optimizer: AdamW lr={LEARNING_RATE} weight_decay={WEIGHT_DECAY}",
        f"  seed: {SEED}",
        f"  best epoch: {best_epoch}",
        "",
        "VALIDATION (threshold=0.5; descriptive)",
        f"  AUPRC: {val_metrics['auprc']:.6f}",
        f"  AUROC: {val_metrics['auroc']:.6f}",
        f"  balanced accuracy: {val_metrics['balanced_accuracy']:.6f}",
        f"  F1: {val_metrics['f1']:.6f}",
        f"  Brier: {val_metrics['brier']:.6f}",
        f"  ECE(15): {val_metrics['ece_15bin']:.6f}",
        "",
        "CALIBRATION",
        f"  frozen threshold: {threshold:.12f}",
        f"  balanced accuracy: {cal_metrics['balanced_accuracy']:.6f}",
        f"  F1: {cal_metrics['f1']:.6f}",
        f"  AUPRC: {cal_metrics['auprc']:.6f}",
        f"  AUROC: {cal_metrics['auroc']:.6f}",
        "",
        "HELD-OUT TEST",
        "  feature files loaded: NO",
        "  predictions generated: NO",
        "  metrics inspected: NO",
        "",
        "ARTIFACT HASHES",
    ]
    for k, v in hashes.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        "NEXT",
        "  Retain baseline_summary.txt with the project record.",
        "  Next step is to build/freeze the graph model using train/validation/calibration only.",
        "  Held-out test remains sealed."
    ]
    summary_path = out / "baseline_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nBASELINE TRAINING COMPLETE / PRE-TEST FROZEN")
    print(f"Summary: {summary_path}")
    print(f"Freeze:  {freeze_path}")


if __name__ == "__main__":
    main()
