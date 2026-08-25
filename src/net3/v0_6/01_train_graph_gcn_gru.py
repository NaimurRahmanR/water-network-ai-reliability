
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
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        brier_score_loss,
        log_loss,
        confusion_matrix,
    )
except ImportError:
    print("ERROR: required packages missing.")
    print("Run: py -m pip install torch scikit-learn pandas numpy")
    sys.exit(1)

WINDOW_HASHES = {
    "window_manifest.csv": "ee67fa17f169b9ac3d8b48efa49a6e634dc5ca405dca657ea5f82043c05a301d",
    "normalization_train_only.json": "a6a4bf24c805e9e02a635cd363b006aab0830ab925fe70b7c9dbe0b0f5e3e1b8",
    "window_config.json": "d6ff1aa93bf5d607b648d1e6c62c6a6a2d2728204161b372e3ea5d56974cb1a1",
}
GENERATED_HASHES = {
    "generation_index.csv": "bc33af176da705ed87850e6032dc313761e2c839faf04aa03c59bd12deb5848a",
    "tensor_layout.json": "33220279d7e1ff6c92caacd54785040764168c576d5eb218401515be206a9261",
    "graph_nodes.csv": "dfb90c5fa1b7c80935edcca5294ec0b87b22cb3788732f4432d58135c8e6f295",
    "graph_links.csv": "d92ab147d4678c8aff999a86f761f15deeb843a738b3021acc3062b824860c1e",
}

SEED = 2718281
BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
GCN_HIDDEN = 32
GRU_HIDDEN = 64
HEAD_HIDDEN = 32
ECE_BINS = 15

WINDOW_STEPS = 24
N_SENSOR_FEATURES = 33
NODE_FEATURES = 10


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def build_topology(generated_dir, tensor_layout):
    nodes_df = pd.read_csv(generated_dir / "graph_nodes.csv", dtype={"node_id": str})
    links_df = pd.read_csv(
        generated_dir / "graph_links.csv",
        dtype={"link_id": str, "start_node": str, "end_node": str}
    )

    node_ids = [str(x) for x in tensor_layout["original_node_ids"]]
    node_index = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    if n != 97:
        raise RuntimeError(f"Expected 97 Net3 nodes, got {n}")

    A = np.zeros((n, n), dtype=np.float32)
    link_endpoints = {}
    for _, r in links_df.iterrows():
        lid = str(r["link_id"])
        u, v = str(r["start_node"]), str(r["end_node"])
        if u not in node_index or v not in node_index:
            raise RuntimeError(f"Link {lid} references missing node.")
        i, j = node_index[u], node_index[v]
        A[i, j] = 1.0
        A[j, i] = 1.0
        link_endpoints[lid] = (i, j)

    # Add self loops and symmetric GCN normalization D^-1/2 A D^-1/2.
    A_hat = A + np.eye(n, dtype=np.float32)
    deg = A_hat.sum(axis=1)
    inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    A_norm = inv_sqrt[:, None] * A_hat * inv_sqrt[None, :]

    # Static node features.
    raw_deg = A.sum(axis=1)
    degree_norm = raw_deg / max(float(raw_deg.max()), 1.0)

    node_type = {}
    for _, r in nodes_df.iterrows():
        node_type[str(r["node_id"])] = str(r["node_type"]).lower()

    static = np.zeros((n, 4), dtype=np.float32)
    static[:, 0] = degree_norm
    for nid, i in node_index.items():
        t = node_type.get(nid, "")
        if "junction" in t:
            static[i, 1] = 1.0
        elif "tank" in t:
            static[i, 2] = 1.0
        elif "reservoir" in t:
            static[i, 3] = 1.0

    return node_ids, node_index, link_endpoints, A_norm, static


def build_sensor_mapping(tensor_layout, node_index, link_endpoints):
    feature_order = tensor_layout["sensor_feature_order"]
    if len(feature_order) != N_SENSOR_FEATURES:
        raise RuntimeError(f"Expected 33 sensor features, got {len(feature_order)}")

    pressure = []
    flow = []
    tank = []

    for fi, name in enumerate(feature_order):
        if name.startswith("pressure::"):
            nid = name.split("::", 1)[1]
            pressure.append((fi, node_index[nid]))
        elif name.startswith("flow::"):
            lid = name.split("::", 1)[1]
            if lid not in link_endpoints:
                raise RuntimeError(f"Flow sensor link {lid} not in graph.")
            flow.append((fi, *link_endpoints[lid]))
        elif name.startswith("tank_level::"):
            nid = name.split("::", 1)[1]
            tank.append((fi, node_index[nid]))
        else:
            raise RuntimeError(f"Unknown sensor feature: {name}")

    if (len(pressure), len(flow), len(tank)) != (24, 6, 3):
        raise RuntimeError(
            f"Expected pressure/flow/tank = 24/6/3, got "
            f"{len(pressure)}/{len(flow)}/{len(tank)}"
        )
    return pressure, flow, tank


def sensor_to_graph_sequence(sensor, mean, scale, n_nodes, static,
                             pressure_map, flow_map, tank_map):
    """
    sensor: [T,33] raw frozen model-visible sensors
    output: [T,N,10]

    Dynamic channels:
      0 normalized pressure value
      1 pressure observation mask
      2 normalized tank-level value
      3 tank observation mask
      4 signed normalized flow aggregate at incident observed-flow edges
      5 flow observation count/mask

    Static channels:
      6 normalized graph degree
      7 junction indicator
      8 tank indicator
      9 reservoir indicator
    """
    x = (sensor.astype(np.float32) - mean[None, :]) / scale[None, :]
    T = x.shape[0]
    out = np.zeros((T, n_nodes, NODE_FEATURES), dtype=np.float32)

    for fi, ni in pressure_map:
        out[:, ni, 0] = x[:, fi]
        out[:, ni, 1] = 1.0

    for fi, ni in tank_map:
        out[:, ni, 2] = x[:, fi]
        out[:, ni, 3] = 1.0

    # Frozen direction convention: negative at link start node, positive at end node.
    for fi, u, v in flow_map:
        val = x[:, fi]
        out[:, u, 4] -= val
        out[:, v, 4] += val
        out[:, u, 5] += 1.0
        out[:, v, 5] += 1.0

    out[:, :, 6:10] = static[None, :, :]
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite graph sequence.")
    return out


def preload_split(manifest, split, generated_dir, mean, scale, n_nodes, static,
                  pressure_map, flow_map, tank_map):
    if split == "test":
        raise RuntimeError("HELD-OUT TEST ACCESS BLOCKED in graph-training script.")

    rows = manifest[manifest["split"] == split].copy()
    context_graph = {}
    print(f"[PRELOAD] {split} contexts")

    for cid in sorted(rows["context_id"].unique()):
        p = generated_dir / "contexts" / f"{cid}.npz"
        if not p.exists():
            raise RuntimeError(f"Missing context: {p}")
        with np.load(p, allow_pickle=False) as z:
            clean_sensor = np.asarray(z["clean_sensor"], dtype=np.float32)
            leak_sensor = np.asarray(z["leak_sensor"], dtype=np.float32)

        context_graph[(cid, "CLEAN")] = sensor_to_graph_sequence(
            clean_sensor, mean, scale, n_nodes, static,
            pressure_map, flow_map, tank_map
        )
        context_graph[(cid, "LEAK")] = sensor_to_graph_sequence(
            leak_sensor, mean, scale, n_nodes, static,
            pressure_map, flow_map, tank_map
        )
    return rows.reset_index(drop=True), context_graph


class WindowDataset(Dataset):
    def __init__(self, rows, cache):
        self.rows = rows.reset_index(drop=True)
        self.cache = cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        seq = self.cache[(str(r["context_id"]), str(r["variant"]))]
        start = int(r["start_idx"])
        final = int(r["final_idx"])
        x = seq[start:final+1]
        if x.shape[0] != WINDOW_STEPS:
            raise RuntimeError(f"{r['window_id']}: bad slice {x.shape}")
        y = float(r["target_active_leak"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), idx


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A):
        # x [B,T,N,F], A [N,N]
        agg = torch.einsum("ij,btjf->btif", A, x)
        return self.linear(agg)


class GraphTemporalModel(nn.Module):
    def __init__(self, A_norm):
        super().__init__()
        self.register_buffer("A_norm", torch.tensor(A_norm, dtype=torch.float32))
        self.gcn1 = GCNLayer(NODE_FEATURES, GCN_HIDDEN)
        self.gcn2 = GCNLayer(GCN_HIDDEN, GCN_HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(
            input_size=GCN_HIDDEN * 2,
            hidden_size=GRU_HIDDEN,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, x):
        h = torch.relu(self.gcn1(x, self.A_norm))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(h, self.A_norm))
        h = self.dropout(h)

        mean_pool = h.mean(dim=2)
        max_pool = h.max(dim=2).values
        temporal = torch.cat([mean_pool, max_pool], dim=-1)

        out, _ = self.gru(temporal)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(logits)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    e = np.exp(logits[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def ece_score(y, p, bins=ECE_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & ((p <= hi) if i == bins - 1 else (p < hi))
        n = int(mask.sum())
        if n:
            ece += (n / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0,1]).ravel()
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
        "ece_15bin": ece_score(y, p),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def predict(model, loader):
    model.eval()
    logits_all, y_all, idx_all = [], [], []
    with torch.no_grad():
        for xb, yb, idx in loader:
            logits = model(xb)
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.cpu().numpy())
            idx_all.append(idx.cpu().numpy())
    return (
        np.concatenate(logits_all),
        np.concatenate(y_all),
        np.concatenate(idx_all),
    )


def choose_threshold(y, p):
    unique = np.unique(np.asarray(p, dtype=np.float64))
    if len(unique) == 1:
        candidates = np.array([0.5])
    else:
        candidates = np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2.0, [1.0]))
    best = None
    for t in candidates:
        pred = (p >= t).astype(int)
        score = float(balanced_accuracy_score(y, pred))
        key = (score, -abs(float(t)-0.5), -float(t))
        if best is None or key > best[0]:
            best = (key, float(t), score)
    return best[1], best[2]


def prediction_frame(rows, idxs, logits, probs, threshold, column):
    meta = rows.iloc[idxs].copy().reset_index(drop=True)
    meta["logit"] = logits
    meta["probability"] = probs
    meta[column] = (probs >= threshold).astype(int)
    return meta


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

    print("AQUA-VERA GRAPH GCN-GRU v0.6")
    print("=" * 78)

    for name, h in WINDOW_HASHES.items():
        verify(windows_dir / name, h)
    for name, h in GENERATED_HASHES.items():
        verify(generated_dir / name, h)

    seed_everything(SEED)

    manifest = pd.read_csv(windows_dir / "window_manifest.csv")
    expected_counts = {
        "train":13500, "validation":2700, "calibration":2700, "test":2700
    }
    if manifest["split"].value_counts().to_dict() != expected_counts:
        raise SystemExit("Window split counts differ from frozen v0.4.")

    norm = json.loads(
        (windows_dir / "normalization_train_only.json").read_text(encoding="utf-8")
    )
    if norm.get("source_split") != "train only":
        raise SystemExit("Normalization is not training-only.")
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["scale_used"], dtype=np.float32)

    tensor_layout = json.loads(
        (generated_dir / "tensor_layout.json").read_text(encoding="utf-8")
    )
    node_ids, node_index, link_endpoints, A_norm, static = build_topology(
        generated_dir, tensor_layout
    )
    pressure_map, flow_map, tank_map = build_sensor_mapping(
        tensor_layout, node_index, link_endpoints
    )

    topology_record = {
        "node_count": len(node_ids),
        "node_feature_channels": [
            "normalized_pressure",
            "pressure_mask",
            "normalized_tank_level",
            "tank_mask",
            "signed_normalized_observed_flow_aggregate",
            "observed_flow_incidence_count",
            "normalized_degree",
            "junction_indicator",
            "tank_indicator",
            "reservoir_indicator",
        ],
        "pressure_sensor_count": len(pressure_map),
        "flow_sensor_count": len(flow_map),
        "tank_sensor_count": len(tank_map),
        "gcn_adjacency": "undirected original Net3 links + self loops; symmetric D^-1/2 A D^-1/2",
        "flow_direction_convention": "negative at original start node, positive at original end node",
        "normalization": "frozen train-only per original sensor feature",
        "test_features_loaded": False,
    }
    (out / "graph_input_definition.json").write_text(
        json.dumps(topology_record, indent=2), encoding="utf-8"
    )

    train_rows, train_cache = preload_split(
        manifest, "train", generated_dir, mean, scale, len(node_ids), static,
        pressure_map, flow_map, tank_map
    )
    val_rows, val_cache = preload_split(
        manifest, "validation", generated_dir, mean, scale, len(node_ids), static,
        pressure_map, flow_map, tank_map
    )
    print("[GUARD] test context files not loaded")

    train_ds = WindowDataset(train_rows, train_cache)
    val_ds = WindowDataset(val_rows, val_cache)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    pos = float(train_rows["target_active_leak"].sum())
    neg = float(len(train_rows) - pos)
    pos_weight_value = neg / pos
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)

    model = GraphTemporalModel(A_norm)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_state = None
    best_epoch = None
    best_val_auprc = -math.inf
    stale = 0
    history = []

    print("[TRAIN]")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for xb, yb, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        val_logits, val_y, val_idxs = predict(model, val_loader)
        val_prob = sigmoid_np(val_logits)
        val_auprc = float(average_precision_score(val_y, val_prob))
        val_auroc = float(roc_auc_score(val_y, val_prob))
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
                print(f"[EARLY STOP] patience={PATIENCE}")
                break

    if best_state is None:
        raise SystemExit("No best graph-model state recorded.")
    model.load_state_dict(best_state)

    val_logits, val_y, val_idxs = predict(model, val_loader)
    val_prob = sigmoid_np(val_logits)
    val_metrics = metrics(val_y, val_prob, 0.5)
    val_pred = prediction_frame(
        val_rows, val_idxs, val_logits, val_prob, 0.5, "prediction_at_0_5"
    )
    val_pred.to_csv(out / "validation_predictions.csv", index=False)

    # Drop train cache before calibration to reduce memory.
    del train_cache, train_ds, train_loader

    cal_rows, cal_cache = preload_split(
        manifest, "calibration", generated_dir, mean, scale, len(node_ids), static,
        pressure_map, flow_map, tank_map
    )
    cal_ds = WindowDataset(cal_rows, cal_cache)
    cal_loader = DataLoader(cal_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cal_logits, cal_y, cal_idxs = predict(model, cal_loader)
    cal_prob = sigmoid_np(cal_logits)
    threshold, threshold_balacc = choose_threshold(cal_y, cal_prob)
    cal_metrics = metrics(cal_y, cal_prob, threshold)
    cal_pred = prediction_frame(
        cal_rows, cal_idxs, cal_logits, cal_prob, threshold,
        "prediction_at_frozen_threshold"
    )
    cal_pred.to_csv(out / "calibration_predictions.csv", index=False)

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    checkpoint = {
        "model_state_dict": best_state,
        "seed": SEED,
        "node_feature_count": NODE_FEATURES,
        "gcn_hidden": GCN_HIDDEN,
        "gru_hidden": GRU_HIDDEN,
        "head_hidden": HEAD_HIDDEN,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_val_auprc,
        "train_pos_weight": pos_weight_value,
        "frozen_calibration_threshold": threshold,
        "A_norm": torch.tensor(A_norm, dtype=torch.float32),
    }
    model_path = out / "graph_gcn_gru_v0_6.pt"
    torch.save(checkpoint, model_path)

    config = {
        "status": "GRAPH_MODEL_FROZEN_PRETEST",
        "model": "2-layer GCN per timestep -> mean+max graph pooling -> GRU -> binary head",
        "architecture": {
            "node_features": NODE_FEATURES,
            "gcn_hidden": GCN_HIDDEN,
            "gcn_layers": 2,
            "graph_pooling": "mean + max",
            "gru_hidden": GRU_HIDDEN,
            "head_hidden": HEAD_HIDDEN,
            "dropout": DROPOUT,
        },
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
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        }
    }
    config_path = out / "graph_model_config.json"
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
        out / "graph_input_definition.json",
        out / "training_history.csv",
        out / "validation_predictions.csv",
        out / "calibration_predictions.csv",
    ]
    hashes = {p.name: sha256(p) for p in core}
    freeze = {
        "status": "GRAPH_MODEL_FROZEN_PRETEST",
        "artifact_sha256": hashes,
        "test_evaluated": False,
        "note": "Graph model frozen before any held-out test feature loading."
    }
    freeze_path = out / "FROZEN_GRAPH_MODEL_RECORD.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA GRAPH GCN-GRU v0.6",
        "=" * 78,
        "STATUS: PASS / GRAPH MODEL FROZEN PRE-TEST",
        "",
        "MODEL",
        "  dynamic graph: 97 Net3 nodes",
        f"  node features: {NODE_FEATURES}",
        f"  GCN: {NODE_FEATURES} -> {GCN_HIDDEN} -> {GCN_HIDDEN}",
        f"  pooling: mean + max -> {GCN_HIDDEN*2}",
        f"  GRU hidden: {GRU_HIDDEN}",
        f"  head: {GRU_HIDDEN} -> {HEAD_HIDDEN} -> 1",
        f"  dropout: {DROPOUT}",
        f"  weighted BCE pos_weight: {pos_weight_value:.8f}",
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
        "  context files loaded: NO",
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
        "  Retain graph_summary.txt with the project record.",
        "  If clean, both model families are frozen and the controlled held-out test may be opened once."
    ]
    summary_path = out / "graph_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nGRAPH TRAINING COMPLETE / PRE-TEST FROZEN")
    print(f"Summary: {summary_path}")
    print(f"Freeze:  {freeze_path}")


if __name__ == "__main__":
    main()
