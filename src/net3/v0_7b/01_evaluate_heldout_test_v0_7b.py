
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# ---------------------------------------------------------------------
# Frozen pre-test artifacts / thresholds
# ---------------------------------------------------------------------

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

BASELINE_HASHES = {
    "baseline_mlp_v0_5.pt": "3e83192de687c42db771964aa82715b6221ad8dc89423484115a9b6aed23a040",
    "baseline_config.json": "d11c031777c4881d3b312d2f8f2c61957b96a32699bf12b6cf53a619e6a5a5c6",
    "pretest_metrics.json": "159faae09c73be7fed12be44e6c012b18a2fd1c794334de3010139c033df3b38",
}

GRAPH_HASHES = {
    "graph_gcn_gru_v0_6.pt": "c8d104c5512556dbbcb459eb63ccef8ba8e71cb04fc9b22a5b16c23eee51e0af",
    "graph_model_config.json": "73f0b36ab7e4b75afa3593950e1b382c3e8b86ace314095cd18df07c50822651",
    "pretest_metrics.json": "0d97c90c5ef88d6e1e41229983d864e6f3821d93a6602f7a95745e1b87448f97",
    "graph_input_definition.json": "7d3db649ecf035fbca8ff744dcbaca6ed3cb85523b3123557fb4eff94ebc4726",
}

BASELINE_THRESHOLD = 0.464201702923
GRAPH_THRESHOLD = 0.376129878873

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1618033
ECE_BINS = 15

# Exact frozen model architecture constants
MLP_HIDDEN1 = 256
MLP_HIDDEN2 = 64
MLP_DROPOUT = 0.20

NODE_FEATURES = 10
GCN_HIDDEN = 32
GRU_HIDDEN = 64
HEAD_HIDDEN = 32
GRAPH_DROPOUT = 0.20


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str):
    if not path.exists():
        raise SystemExit(f"Missing artifact: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH: {path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {path.name}")


# ---------------------------------------------------------------------
# EXACT frozen model class names from training scripts.
# These names matter because state_dict keys include module attribute names.
# ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim=792):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN1),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN1, MLP_HIDDEN2),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A):
        agg = torch.einsum("ij,btjf->btif", A, x)
        return self.linear(agg)


class GraphTemporalModel(nn.Module):
    def __init__(self, A_norm):
        super().__init__()
        self.register_buffer(
            "A_norm",
            torch.tensor(A_norm, dtype=torch.float32),
        )
        # IMPORTANT: exact training-time attribute names.
        self.gcn1 = GCNLayer(NODE_FEATURES, GCN_HIDDEN)
        self.gcn2 = GCNLayer(GCN_HIDDEN, GCN_HIDDEN)
        self.dropout = nn.Dropout(GRAPH_DROPOUT)
        self.gru = nn.GRU(
            input_size=GCN_HIDDEN * 2,
            hidden_size=GRU_HIDDEN,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(GRAPH_DROPOUT),
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
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------
# Graph reconstruction identical to training semantics.
# ---------------------------------------------------------------------

def build_topology(generated_dir: Path, tensor_layout: dict):
    nodes_df = pd.read_csv(
        generated_dir / "graph_nodes.csv",
        dtype={"node_id": str},
    )
    links_df = pd.read_csv(
        generated_dir / "graph_links.csv",
        dtype={"link_id": str, "start_node": str, "end_node": str},
    )

    node_ids = [str(x) for x in tensor_layout["original_node_ids"]]
    node_index = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    A = np.zeros((n, n), dtype=np.float32)
    link_endpoints = {}

    for _, r in links_df.iterrows():
        lid = str(r["link_id"])
        u = str(r["start_node"])
        v = str(r["end_node"])
        i, j = node_index[u], node_index[v]
        A[i, j] = 1.0
        A[j, i] = 1.0
        link_endpoints[lid] = (i, j)

    A_hat = A + np.eye(n, dtype=np.float32)
    degree = A_hat.sum(axis=1)

    # FIX v0.7 warning: initialize output explicitly rather than using
    # np.power(where=...) without `out`.
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    valid = degree > 0
    inv_sqrt[valid] = np.power(degree[valid], -0.5).astype(np.float32)

    A_norm = inv_sqrt[:, None] * A_hat * inv_sqrt[None, :]

    raw_degree = A.sum(axis=1)
    degree_norm = raw_degree / max(float(raw_degree.max()), 1.0)

    node_type = {
        str(r["node_id"]): str(r["node_type"]).lower()
        for _, r in nodes_df.iterrows()
    }

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

    return node_index, link_endpoints, A_norm, static


def build_sensor_mapping(tensor_layout, node_index, link_endpoints):
    pressure, flow, tank = [], [], []

    for fi, name in enumerate(tensor_layout["sensor_feature_order"]):
        if name.startswith("pressure::"):
            nid = name.split("::", 1)[1]
            pressure.append((fi, node_index[nid]))
        elif name.startswith("flow::"):
            lid = name.split("::", 1)[1]
            flow.append((fi, *link_endpoints[lid]))
        elif name.startswith("tank_level::"):
            nid = name.split("::", 1)[1]
            tank.append((fi, node_index[nid]))
        else:
            raise RuntimeError(f"Unknown feature: {name}")

    if (len(pressure), len(flow), len(tank)) != (24, 6, 3):
        raise RuntimeError(
            f"Sensor mapping mismatch: {len(pressure)}/{len(flow)}/{len(tank)}"
        )

    return pressure, flow, tank


def sensor_to_graph_sequence(
    sensor,
    mean,
    scale,
    n_nodes,
    static,
    pressure_map,
    flow_map,
    tank_map,
):
    x = (sensor.astype(np.float32) - mean[None, :]) / scale[None, :]
    T = x.shape[0]

    out = np.zeros((T, n_nodes, NODE_FEATURES), dtype=np.float32)

    for fi, ni in pressure_map:
        out[:, ni, 0] = x[:, fi]
        out[:, ni, 1] = 1.0

    for fi, ni in tank_map:
        out[:, ni, 2] = x[:, fi]
        out[:, ni, 3] = 1.0

    for fi, u, v in flow_map:
        val = x[:, fi]
        out[:, u, 4] -= val
        out[:, v, 4] += val
        out[:, u, 5] += 1.0
        out[:, v, 5] += 1.0

    out[:, :, 6:10] = static[None, :, :]

    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite graph input.")

    return out


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def ece_score(y, p, bins=ECE_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ece = 0.0

    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += (
                mask.mean()
                * abs(float(p[mask].mean()) - float(y[mask].mean()))
            )

    return float(ece)


def compute_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "n": int(len(y)),
        "positive_fraction": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.c_[1 - p, p], labels=[0, 1])),
        "ece15": float(ece_score(y, p)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_in_batches(model, X, batch_size):
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size])
            outputs.append(model(xb).cpu().numpy())

    return np.concatenate(outputs)


def bootstrap_context_delta(df, metric_key):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contexts = np.array(sorted(df["context_id"].unique()))
    deltas = []

    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.choice(
            contexts,
            size=len(contexts),
            replace=True,
        )
        chunks = [
            df[df["context_id"] == cid]
            for cid in sampled
        ]
        b = pd.concat(chunks, ignore_index=True)

        y = b["target_active_leak"].to_numpy(dtype=int)
        pm = b["mlp_probability"].to_numpy(dtype=float)
        pg = b["gcn_probability"].to_numpy(dtype=float)

        if metric_key == "balanced_accuracy":
            vm = balanced_accuracy_score(
                y, (pm >= BASELINE_THRESHOLD).astype(int)
            )
            vg = balanced_accuracy_score(
                y, (pg >= GRAPH_THRESHOLD).astype(int)
            )
        elif metric_key == "f1":
            vm = f1_score(
                y, (pm >= BASELINE_THRESHOLD).astype(int), zero_division=0
            )
            vg = f1_score(
                y, (pg >= GRAPH_THRESHOLD).astype(int), zero_division=0
            )
        elif metric_key == "auprc":
            vm = average_precision_score(y, pm)
            vg = average_precision_score(y, pg)
        elif metric_key == "auroc":
            vm = roc_auc_score(y, pm)
            vg = roc_auc_score(y, pg)
        elif metric_key == "brier_advantage":
            # Lower Brier is better; negate both so positive MLP-GCN
            # still means MLP advantage.
            vm = -brier_score_loss(y, pm)
            vg = -brier_score_loss(y, pg)
        else:
            raise ValueError(metric_key)

        deltas.append(float(vm - vg))

    arr = np.asarray(deltas, dtype=np.float64)

    return {
        "metric": metric_key,
        "delta_definition": "MLP minus GCN-GRU",
        "mean_delta": float(arr.mean()),
        "ci95": [
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        ],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    windows = Path(args.windows)
    generated = Path(args.generated)
    baseline_dir = Path(args.baseline)
    graph_dir = Path(args.graph)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA SINGLE HELD-OUT TEST EVALUATION v0.7b")
    print("=" * 78)
    print("Attempt note: v0.7 failed before prediction because evaluator")
    print("module names did not match the frozen GCN-GRU checkpoint.")
    print("v0.7b changes evaluator reconstruction only; models stay frozen.")
    print()

    for name, h in WINDOW_HASHES.items():
        verify(windows / name, h)

    for name, h in GENERATED_HASHES.items():
        verify(generated / name, h)

    for name, h in BASELINE_HASHES.items():
        verify(baseline_dir / name, h)

    for name, h in GRAPH_HASHES.items():
        verify(graph_dir / name, h)

    baseline_config = json.loads(
        (baseline_dir / "baseline_config.json").read_text(encoding="utf-8")
    )
    graph_config = json.loads(
        (graph_dir / "graph_model_config.json").read_text(encoding="utf-8")
    )

    if abs(float(baseline_config["frozen_threshold"]) - BASELINE_THRESHOLD) > 1e-12:
        raise SystemExit("Baseline threshold mismatch.")

    if abs(float(graph_config["frozen_threshold"]) - GRAPH_THRESHOLD) > 1e-12:
        raise SystemExit("Graph threshold mismatch.")

    # -----------------------------------------------------------------
    # IMPORTANT: checkpoint compatibility is verified BEFORE test data is
    # loaded. This prevents another evaluator-class failure after unsealing.
    # -----------------------------------------------------------------

    print("[PRE-UNSEAL] verifying checkpoint/model-class compatibility...")

    baseline_ckpt = torch.load(
        baseline_dir / "baseline_mlp_v0_5.pt",
        map_location="cpu",
        weights_only=False,
    )
    mlp = MLP(input_dim=792)
    mlp.load_state_dict(
        baseline_ckpt["model_state_dict"],
        strict=True,
    )
    mlp.eval()
    print("[OK] baseline checkpoint loads strictly")

    tensor_layout = json.loads(
        (generated / "tensor_layout.json").read_text(encoding="utf-8")
    )
    node_index, link_endpoints, A_norm, static = build_topology(
        generated, tensor_layout
    )

    graph_ckpt = torch.load(
        graph_dir / "graph_gcn_gru_v0_6.pt",
        map_location="cpu",
        weights_only=False,
    )
    graph_model = GraphTemporalModel(A_norm)
    graph_model.load_state_dict(
        graph_ckpt["model_state_dict"],
        strict=True,
    )
    graph_model.eval()
    print("[OK] graph checkpoint loads strictly")

    # Check stored A_norm buffer agrees exactly with our safe reconstruction.
    stored_A = graph_ckpt.get("A_norm")
    if stored_A is not None:
        stored_A_np = stored_A.detach().cpu().numpy()
        max_diff = float(np.max(np.abs(stored_A_np - A_norm)))
        if max_diff > 1e-7:
            raise SystemExit(
                f"A_norm reconstruction mismatch: max_diff={max_diff}"
            )
        print(f"[OK] A_norm reconstruction max_diff={max_diff:.3g}")

    # Only now load test.
    print("[UNSEAL] frozen model classes/checkpoints verified; loading test")

    manifest = pd.read_csv(windows / "window_manifest.csv")
    test_rows = (
        manifest[manifest["split"] == "test"]
        .copy()
        .reset_index(drop=True)
    )

    if len(test_rows) != 2700:
        raise SystemExit(
            f"Expected 2700 held-out windows, got {len(test_rows)}"
        )

    if test_rows["context_id"].nunique() != 30:
        raise SystemExit("Expected 30 held-out contexts.")

    norm = json.loads(
        (windows / "normalization_train_only.json").read_text(
            encoding="utf-8"
        )
    )
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["scale_used"], dtype=np.float32)

    pressure_map, flow_map, tank_map = build_sensor_mapping(
        tensor_layout,
        node_index,
        link_endpoints,
    )

    raw_cache = {}
    graph_cache = {}

    for cid in sorted(test_rows["context_id"].unique()):
        path = generated / "contexts" / f"{cid}.npz"

        with np.load(path, allow_pickle=False) as z:
            clean = np.asarray(
                z["clean_sensor"], dtype=np.float32
            )
            leak = np.asarray(
                z["leak_sensor"], dtype=np.float32
            )

        for variant, sensor in (
            ("CLEAN", clean),
            ("LEAK", leak),
        ):
            raw_cache[(cid, variant)] = sensor
            graph_cache[(cid, variant)] = sensor_to_graph_sequence(
                sensor,
                mean,
                scale,
                len(node_index),
                static,
                pressure_map,
                flow_map,
                tank_map,
            )

    X_mlp = []
    X_graph = []
    y = []
    metadata = []

    for _, row in test_rows.iterrows():
        cid = str(row["context_id"])
        variant = str(row["variant"])
        start_idx = int(row["start_idx"])
        final_idx = int(row["final_idx"])

        raw_window = raw_cache[(cid, variant)][
            start_idx:final_idx + 1
        ]
        raw_window = (
            raw_window - mean[None, :]
        ) / scale[None, :]

        graph_window = graph_cache[(cid, variant)][
            start_idx:final_idx + 1
        ]

        if raw_window.shape != (24, 33):
            raise RuntimeError(
                f"Bad MLP window shape: {raw_window.shape}"
            )

        if graph_window.shape != (24, 97, 10):
            raise RuntimeError(
                f"Bad graph window shape: {graph_window.shape}"
            )

        X_mlp.append(
            raw_window.reshape(-1).astype(np.float32)
        )
        X_graph.append(
            graph_window.astype(np.float32)
        )
        y.append(
            int(row["target_active_leak"])
        )

        metadata.append({
            "window_id": str(row["window_id"]),
            "pair_id": str(row["pair_id"]),
            "context_id": cid,
            "variant": variant,
            "final_idx": final_idx,
            "target_active_leak": int(row["target_active_leak"]),
        })

    X_mlp = np.stack(X_mlp).astype(np.float32)
    X_graph = np.stack(X_graph).astype(np.float32)
    y = np.asarray(y, dtype=int)

    print("[PREDICT] frozen MLP")
    mlp_logits = predict_in_batches(
        mlp, X_mlp, batch_size=1024
    )

    print("[PREDICT] frozen GCN-GRU")
    graph_logits = predict_in_batches(
        graph_model, X_graph, batch_size=128
    )

    mlp_prob = sigmoid_np(mlp_logits)
    graph_prob = sigmoid_np(graph_logits)

    mlp_metrics = compute_metrics(
        y, mlp_prob, BASELINE_THRESHOLD
    )
    graph_metrics = compute_metrics(
        y, graph_prob, GRAPH_THRESHOLD
    )

    pred = pd.DataFrame(metadata)
    pred["mlp_logit"] = mlp_logits
    pred["mlp_probability"] = mlp_prob
    pred["mlp_prediction"] = (
        mlp_prob >= BASELINE_THRESHOLD
    ).astype(int)
    pred["gcn_logit"] = graph_logits
    pred["gcn_probability"] = graph_prob
    pred["gcn_prediction"] = (
        graph_prob >= GRAPH_THRESHOLD
    ).astype(int)

    pred_path = out / "heldout_test_predictions.csv"
    pred.to_csv(pred_path, index=False)

    print("[BOOTSTRAP] paired 30-context cluster bootstrap")

    bootstrap = {}
    for metric_key in (
        "balanced_accuracy",
        "f1",
        "auprc",
        "auroc",
        "brier_advantage",
    ):
        bootstrap[metric_key] = bootstrap_context_delta(
            pred.rename(
                columns={
                    "target_active_leak": "target_active_leak",
                }
            ),
            metric_key,
        )

    metrics_record = {
        "status": "HELDOUT_TEST_EVALUATED",
        "attempt_history": [
            {
                "attempt": "v0.7",
                "result": "FAILED_BEFORE_PREDICTION",
                "reason": "Evaluator module attribute names did not match frozen GCN-GRU state_dict keys.",
                "predictions_generated": False,
                "metrics_generated": False,
            },
            {
                "attempt": "v0.7b",
                "result": "PASS",
                "models_changed": False,
                "thresholds_changed": False,
            },
        ],
        "test_contexts": 30,
        "test_windows": 2700,
        "mlp": mlp_metrics,
        "gcn_gru": graph_metrics,
        "bootstrap_mlp_minus_gcn": bootstrap,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "frozen_thresholds": {
            "mlp": BASELINE_THRESHOLD,
            "gcn_gru": GRAPH_THRESHOLD,
        },
    }

    metrics_path = out / "heldout_test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_record, indent=2),
        encoding="utf-8",
    )

    output_hashes = {
        pred_path.name: sha256(pred_path),
        metrics_path.name: sha256(metrics_path),
    }

    run_record = {
        "status": "HELDOUT_TEST_EVALUATED",
        "evaluator_version": "0.7b",
        "v0_7_failed_before_predictions": True,
        "models_or_thresholds_modified_after_v0_7_failure": False,
        "output_sha256": output_hashes,
    }

    (out / "HELDOUT_TEST_RECORD.json").write_text(
        json.dumps(run_record, indent=2),
        encoding="utf-8",
    )

    lines = [
        "AQUA-VERA SINGLE HELD-OUT TEST EVALUATION v0.7b",
        "=" * 78,
        "STATUS: PASS / HELD-OUT TEST EVALUATED",
        "",
        "ATTEMPT HISTORY",
        "  v0.7: failed before prediction due evaluator/checkpoint module-name mismatch",
        "  v0.7 predictions generated: NO",
        "  v0.7 metrics generated: NO",
        "  models changed before v0.7b: NO",
        "  thresholds changed before v0.7b: NO",
        "",
        "TEST SET",
        "  contexts: 30",
        "  windows: 2,700",
        f"  positives: {int(y.sum()):,}",
        f"  negatives: {int((1 - y).sum()):,}",
        "",
        "NON-GRAPH MLP",
        f"  threshold: {BASELINE_THRESHOLD:.12f}",
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
        f"  threshold: {GRAPH_THRESHOLD:.12f}",
        f"  balanced accuracy: {graph_metrics['balanced_accuracy']:.6f}",
        f"  accuracy: {graph_metrics['accuracy']:.6f}",
        f"  F1: {graph_metrics['f1']:.6f}",
        f"  AUPRC: {graph_metrics['auprc']:.6f}",
        f"  AUROC: {graph_metrics['auroc']:.6f}",
        f"  Brier: {graph_metrics['brier']:.6f}",
        f"  ECE15: {graph_metrics['ece15']:.6f}",
        f"  TN/FP/FN/TP: {graph_metrics['tn']}/{graph_metrics['fp']}/{graph_metrics['fn']}/{graph_metrics['tp']}",
        "",
        "PAIRED CONTEXT-CLUSTER BOOTSTRAP: MLP - GCN-GRU",
    ]

    pretty = {
        "balanced_accuracy": "balanced accuracy",
        "f1": "F1",
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "brier_advantage": "Brier advantage",
    }

    for key in (
        "balanced_accuracy",
        "f1",
        "auprc",
        "auroc",
        "brier_advantage",
    ):
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
        "  frozen MLP checkpoint hash verified: PASS",
        "  frozen GCN-GRU checkpoint hash verified: PASS",
        "  checkpoint classes loaded strictly before v0.7b test access: PASS",
        "  thresholds remained frozen: PASS",
        "  post-test clean-model redesign: NO",
        "",
        "OUTPUT HASHES",
    ]

    for name, h in output_hashes.items():
        lines.append(f"  {name}: {h}")

    lines += [
        "",
        "NEXT",
        "  Retain heldout_test_summary_v0_7b.txt with the project record.",
        "  Do not retune either clean model.",
    ]

    summary_path = out / "heldout_test_summary_v0_7b.txt"
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("HELD-OUT TEST EVALUATION COMPLETE")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
