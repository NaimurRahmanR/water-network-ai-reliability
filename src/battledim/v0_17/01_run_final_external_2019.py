import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from frozen_v16b_model_code import (
    MLP,
    SparseGraphTemporalModel,
    build_graph_metadata,
    input_x_gradient,
    js_divergence,
    parse_inp,
    predict_batches,
    sigmoid,
    entropy01,
)

# -----------------------------------------------------------------------------
# Frozen artifact hashes
# -----------------------------------------------------------------------------
H19_SCADA = "a021ade90c3cfa17362520f0fd7e9794a6c816e4787d7e49a3e52fdd6aa0d15d"
H19_LEAKS = "76fa8c50f19b6dffc2010b4413681179c43da5aa984374b9062897b6fe45c27f"
H_L_TOWN = "a7551b86745f4cc3433c78e60023077fa1386947d4d35372ffd3a0405622b436"

H_NORM = "44cab3f6ef97752245bdce42d1cce70db929b6c1325b6e979650067da3f1ee0f"
H_GRAPH_INPUT = "89bd64f6758b46af9ba39ea575ab78ee0da042c94a6b706855c89e69141411ff"
H_MLP = "d3a11328c1884481d10177bf624b542e2be0882760e74bb137d6ca28ac385f08"
H_GCN = "da1b7d0365d2b2a97a7b9b1b9795767163d7a90daaa979e973aa67e907522358"
H_THRESHOLDS = "243ae4956479e7cb244b3a0f877b8826263ee7cd54cda517b20478e6dc6050b2"
H_CENTROIDS = "c017d1e933b4378b576d02f13a3a2f4c98c7275fb99ae28e895bdd9e92f6df46"
H_RECON = "49b2983f040c3e282cb6feddbb4f1b3c2ce326b29d9c4d0da11ae7d460cf2ce7"
H_FINAL_IMPL = "b4099d9dee3a118c8b51588d8d7495f290444a9ade57aa96a2320002789c15ef"

PRED_THRESHOLD = 0.5
WINDOW_STEPS = 24
STRIDE = 6
HORIZON_STEPS = 24 * 60 // 5  # 24 h at 5-min resolution
ATTR_EPS = 1e-12

FAMILIES_NUMERIC = [
    "MISSING_SENSOR",
    "SENSOR_NOISE",
    "SENSOR_BIAS",
    "STALE_SENSOR",
]
SEVERITIES = ["low", "medium", "high"]
NOISE_FACTOR = {"low": 0.25, "medium": 0.50, "high": 1.00}
BIAS_FACTOR = {"low": 0.50, "medium": 1.00, "high": 2.00}
STALE_STEPS = {"low": 6, "medium": 12, "high": 24}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected, label):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"{label} HASH MISMATCH\nexpected={expected}\ngot={got}\npath={path}"
        )
    print(f"[OK] {label}", flush=True)


def find_unique(root, filename):
    root = Path(root)
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one {filename} under {root}; found {len(matches)}"
        )
    return matches[0]


def observed_transitions(mask):
    """Observed within-year 0->1 onsets and 1->0 ends.

    If a leakage trace is already active at the first 2019 sample, it is treated
    as pre-existing rather than as an artificial onset at the year boundary.
    Likewise, activity that remains positive at the final sample is not assigned
    an unobserved repair/end transition beyond the dataset.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.flatnonzero(mask[1:] & ~mask[:-1]) + 1
    ends = np.flatnonzero(mask[:-1] & ~mask[1:])
    return starts.astype(int), ends.astype(int)


def build_2019_manifest(scada_ts, leak_values, leak_cols):
    active = leak_values > 0.0
    onset_indices = []
    end_indices = []
    event_rows = []

    for j, col in enumerate(leak_cols):
        starts, ends = observed_transitions(active[:, j])
        onset_indices.extend(starts.tolist())
        end_indices.extend(ends.tolist())
        for k, s in enumerate(starts, start=1):
            event_rows.append({
                "leak_column": col,
                "transition_type": "ONSET",
                "transition_index_within_column": k,
                "row_index": int(s),
                "timestamp": str(scada_ts.iloc[int(s)]),
            })
        for k, e in enumerate(ends, start=1):
            event_rows.append({
                "leak_column": col,
                "transition_type": "END",
                "transition_index_within_column": k,
                "row_index": int(e),
                "timestamp": str(scada_ts.iloc[int(e)]),
            })

    onset_indices = np.asarray(sorted(onset_indices), dtype=int)
    end_indices = np.asarray(sorted(end_indices), dtype=int)

    rows = []
    candidate_index = 0
    for final_idx in range(WINDOW_STEPS - 1, len(scada_ts), STRIDE):
        start_idx = final_idx - WINDOW_STEPS + 1
        recent_onset = bool(
            ((onset_indices <= final_idx) & (onset_indices > final_idx - HORIZON_STEPS)).any()
        )
        repair_guard = bool(
            ((end_indices <= final_idx) & (end_indices > final_idx - HORIZON_STEPS)).any()
        )
        rows.append({
            "window_index": candidate_index,
            "start_idx": int(start_idx),
            "final_idx": int(final_idx),
            "start_timestamp": str(scada_ts.iloc[start_idx]),
            "final_timestamp": str(scada_ts.iloc[final_idx]),
            "target_recent_leak_onset_24h": int(recent_onset),
            "repair_guard_excluded": int(repair_guard),
            "eligible_external_2019": int(not repair_guard),
            "attribution_subset": int(candidate_index % 10 == 0),
        })
        candidate_index += 1

    manifest = pd.DataFrame(rows)
    events = pd.DataFrame(event_rows)
    return manifest, events


def materialize_windows(series, eligible_manifest):
    X = np.empty((len(eligible_manifest), 24, series.shape[1]), dtype=np.float32)
    for i, (_, r) in enumerate(eligible_manifest.iterrows()):
        s, f = int(r.start_idx), int(r.final_idx)
        X[i] = series[s:f + 1]
    return X


def classification_metrics(y, p):
    q = (p >= PRED_THRESHOLD).astype(int)
    out = {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "predicted_positive": int(q.sum()),
        "accuracy": float((q == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, q)),
        "f1": float(f1_score(y, q, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) == 2:
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auprc"] = None
        out["auroc"] = None
    return out


def make_graph_meta_with_removed(inp_path, feature_cols, removed_ids):
    graph = parse_inp(inp_path)
    node_ids = graph["nodes"]
    ix = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    endpoints_full = {
        r["link_id"]: (ix[r["start"]], ix[r["end"]])
        for r in graph["links"]
    }
    active_links = [r for r in graph["links"] if r["link_id"] not in set(removed_ids)]

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from((r["start"], r["end"]) for r in active_links)
    if not nx.is_connected(G):
        raise RuntimeError("Frozen topology corruption disconnected L-Town")

    A = np.zeros((n, n), dtype=np.uint8)
    for r in active_links:
        u, v = ix[r["start"]], ix[r["end"]]
        A[u, v] = 1
        A[v, u] = 1

    Ah = A.astype(np.float32) + np.eye(n, dtype=np.float32)
    deg_h = Ah.sum(axis=1)
    rows, cols = np.nonzero(Ah)
    weights = 1.0 / np.sqrt(deg_h[rows] * deg_h[cols])

    degree = A.sum(axis=1).astype(np.float32)
    static = np.zeros((n, 4), dtype=np.float32)
    static[:, 0] = degree / max(float(degree.max()), 1.0)

    jset = set(graph["junctions"])
    tset = set(graph["tanks"])
    rset = set(graph["reservoirs"])
    for nid, ni in ix.items():
        static[ni, 1] = 1.0 if nid in jset else 0.0
        static[ni, 2] = 1.0 if nid in tset else 0.0
        static[ni, 3] = 1.0 if nid in rset else 0.0

    pressure, flow, tank = [], [], []
    for fi, name in enumerate(feature_cols):
        if name.startswith("pressure__"):
            nid = name.split("__", 1)[1]
            pressure.append((fi, nid, ix[nid]))
        elif name.startswith("flow__"):
            lid = name.split("__", 1)[1]
            u, v = endpoints_full[lid]
            flow.append((fi, lid, u, v))
        elif name.startswith("level__"):
            nid = name.split("__", 1)[1]
            tank.append((fi, nid, ix[nid]))

    return {
        "node_ids": node_ids,
        "edge_target": rows.astype(np.int64),
        "edge_source": cols.astype(np.int64),
        "edge_weight": weights.astype(np.float32),
        "static": static,
        "pressure": pressure,
        "flow": flow,
        "tank": tank,
        "graph": graph,
    }


class RawToGraphWrapperMissing(torch.nn.Module):
    def __init__(self, graph_model, graph_meta, missing_feature_indices=None):
        super().__init__()
        missing = set(int(x) for x in (missing_feature_indices or []))
        n = len(graph_meta["node_ids"])
        F = 37
        P = np.zeros((F, n), dtype=np.float32)
        T = np.zeros((F, n), dtype=np.float32)
        Flow = np.zeros((F, n), dtype=np.float32)
        pmask = np.zeros(n, dtype=np.float32)
        tmask = np.zeros(n, dtype=np.float32)
        fcount = np.zeros(n, dtype=np.float32)

        for fi, sid, ni in graph_meta["pressure"]:
            P[fi, ni] = 1.0
            pmask[ni] = 0.0 if fi in missing else 1.0
        for fi, sid, ni in graph_meta["tank"]:
            T[fi, ni] = 1.0
            tmask[ni] = 1.0
        for fi, lid, u, v in graph_meta["flow"]:
            Flow[fi, u] -= 1.0
            Flow[fi, v] += 1.0
            fcount[u] += 1.0
            fcount[v] += 1.0

        self.graph_model = graph_model
        self.register_buffer("P", torch.tensor(P))
        self.register_buffer("T", torch.tensor(T))
        self.register_buffer("Flow", torch.tensor(Flow))
        self.register_buffer("pmask", torch.tensor(pmask))
        self.register_buffer("tmask", torch.tensor(tmask))
        self.register_buffer("fcount", torch.tensor(fcount))
        self.register_buffer("static", torch.tensor(graph_meta["static"], dtype=torch.float32))

    def forward(self, x):
        pv = torch.einsum("btf,fn->btn", x, self.P)
        tv = torch.einsum("btf,fn->btn", x, self.T)
        fv = torch.einsum("btf,fn->btn", x, self.Flow)
        B, T, _ = pv.shape
        dyn = torch.stack([
            pv,
            self.pmask[None, None, :].expand(B, T, -1),
            tv,
            self.tmask[None, None, :].expand(B, T, -1),
            fv,
            self.fcount[None, None, :].expand(B, T, -1),
        ], dim=-1)
        static = self.static[None, None, :, :].expand(B, T, -1, -1)
        return self.graph_model(torch.cat([dyn, static], dim=-1))


def load_core(ckpt_path, graph_meta):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    core = SparseGraphTemporalModel(
        graph_meta["edge_target"],
        graph_meta["edge_source"],
        graph_meta["edge_weight"],
    )
    state = ck["model_state_dict"]
    learned = {
        k: v for k, v in state.items()
        if k not in {"edge_target", "edge_source", "edge_weight"}
    }
    missing, unexpected = core.load_state_dict(learned, strict=False)
    expected_missing = {"edge_target", "edge_source", "edge_weight"}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"GCN state load mismatch missing={missing}, unexpected={unexpected}"
        )
    core.eval()
    return core


def predict_with_progress(model, X, batch, label):
    model.eval()
    out = []
    total = math.ceil(len(X) / batch)
    with torch.no_grad():
        for b, i in enumerate(range(0, len(X), batch), start=1):
            xb = torch.from_numpy(X[i:i + batch])
            out.append(model(xb).cpu().numpy())
            if b == 1 or b % 50 == 0 or b == total:
                print(f"[{label}] batch {b}/{total}", flush=True)
    return np.concatenate(out)


def input_x_gradient_progress(model, X, batch, label):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    profiles, logits = [], []
    total = math.ceil(len(X) / batch)
    for b, i in enumerate(range(0, len(X), batch), start=1):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32, requires_grad=True)
        z = model(xb)
        g = torch.autograd.grad(z.sum(), xb)[0]
        a = torch.abs(xb * g).sum(dim=1) + ATTR_EPS
        a = a / a.sum(dim=1, keepdim=True)
        profiles.append(a.detach().cpu().numpy().astype(np.float32))
        logits.append(z.detach().cpu().numpy().astype(np.float64))
        if b == 1 or b % 25 == 0 or b == total:
            print(f"[{label}] gradient batch {b}/{total}", flush=True)
    return np.concatenate(profiles), np.concatenate(logits)


def corrupt_series(base_raw, family, severity, feature_cols, norm, impl):
    out = base_raw.copy()
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    fmap = {name.split("__", 1)[1]: i for i, name in enumerate(feature_cols) if name.startswith("pressure__")}
    selected_ids = impl["pressure_sensor_selection"][family][severity]
    fis = [fmap[s] for s in selected_ids]

    if family == "MISSING_SENSOR":
        for fi in fis:
            out[:, fi] = mean[fi]
    elif family == "SENSOR_NOISE":
        seed = int(impl["noise_rng_seed_by_severity"][severity])
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 1.0, size=(len(out), len(fis)))
        for j, fi in enumerate(fis):
            out[:, fi] += z[:, j].astype(np.float32) * NOISE_FACTOR[severity] * std[fi]
    elif family == "SENSOR_BIAS":
        signs = impl["bias_sign_by_selected_sensor"]
        for sid, fi in zip(selected_ids, fis):
            out[:, fi] += int(signs[sid]) * BIAS_FACTOR[severity] * std[fi]
    elif family == "STALE_SENSOR":
        lag = STALE_STEPS[severity]
        factual = base_raw
        for fi in fis:
            out[:lag, fi] = factual[0, fi]
            out[lag:, fi] = factual[:-lag, fi]
    else:
        raise ValueError(family)
    return out.astype(np.float32), fis


def normalize(raw, norm):
    mean = np.asarray(norm["mean"], dtype=np.float64)
    scale = np.asarray(norm["scale_used"], dtype=np.float64)
    return ((raw.astype(np.float64) - mean[None, :]) / scale[None, :]).astype(np.float32)


def reconstruct_normalized(norm_series, missing_fis, recon):
    base = norm_series.copy()
    out = norm_series.copy()
    models = recon["models"]
    for fi in missing_fis:
        r = models[str(int(fi))]
        keep = np.asarray(r["keep_indices"], dtype=int)
        coef = np.asarray(r["coef"], dtype=np.float64)
        intercept = float(r["intercept"])
        out[:, fi] = base[:, keep] @ coef + intercept
    return out.astype(np.float32)


def divergence_against_frozen_centroid(model_name, profiles, probs, centroids):
    preds = (probs >= PRED_THRESHOLD).astype(int)
    d = np.zeros(len(profiles), dtype=np.float64)
    for cls in [0, 1]:
        mask = preds == cls
        if not mask.any():
            continue
        c = np.asarray(centroids[model_name][str(cls)]["profile"], dtype=np.float64)
        d[mask] = js_divergence(profiles[mask], c[None, :])
    return preds, d


def action(ent, div, ent_thr, div_thr):
    eh = ent > ent_thr
    dh = div > div_thr
    if not eh and not dh:
        return "PROCEED"
    if eh and not dh:
        return "VERIFY"
    if not eh and dh:
        return "REQUEST_SENSOR"
    return "ESCALATE"


def safe_auroc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def safe_auprc(y, score):
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None


def checkpoint_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ready", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-freeze", required=True)
    ap.add_argument("--eval-freeze", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mr = Path(args.model_ready)
    data = Path(args.data)
    tf = Path(args.train_freeze)
    ef = Path(args.eval_freeze)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckdir = out / "_checkpoints"
    ckdir.mkdir(exist_ok=True)

    source_hash = sha256(Path(__file__))
    print("AQUA-VERA BATTLEDIM/L-TOWN FINAL EXTERNAL 2019 EVALUATION v0.17")
    print("=" * 78)
    print(f"Evaluator source SHA-256: {source_hash}")
    print()

    # Frozen pre-2019 artifacts first.
    verify(tf / "FROZEN_2018_NORMALIZATION.json", H_NORM, "2018 normalization")
    verify(tf / "FROZEN_LTown_GRAPH_INPUT.json", H_GRAPH_INPUT, "L-Town graph input")
    verify(tf / "BattLeDIM_MLP_v0_16b.pt", H_MLP, "frozen MLP")
    verify(tf / "BattLeDIM_GCN_GRU_v0_16b.pt", H_GCN, "frozen GCN-GRU")
    verify(tf / "FROZEN_2018_RELIABILITY_THRESHOLDS.json", H_THRESHOLDS, "reliability thresholds")
    verify(tf / "FROZEN_2018_ATTRIBUTION_CENTROIDS.json", H_CENTROIDS, "attribution centroids")
    verify(tf / "FROZEN_2018_PRESSURE_RECONSTRUCTION.json", H_RECON, "pressure reconstruction")
    verify(ef / "FROZEN_FINAL_EVAL_IMPLEMENTATION_v0_16c.json", H_FINAL_IMPL, "final-eval implementation freeze")
    inp = find_unique(data, "L-TOWN.inp")
    verify(inp, H_L_TOWN, "L-TOWN.inp")

    # Freeze checkpoint compatibility before unsealing.
    ckmeta_path = ckdir / "checkpoint_meta.json"
    current_ckmeta = {
        "evaluator_source_sha256": source_hash,
        "final_eval_implementation_sha256": H_FINAL_IMPL,
        "mlp_sha256": H_MLP,
        "gcn_sha256": H_GCN,
    }
    if ckmeta_path.exists():
        prior = json.loads(ckmeta_path.read_text(encoding="utf-8"))
        if prior != current_ckmeta:
            raise SystemExit("Checkpoint metadata differs from this evaluator/freeze. Use a fresh output folder.")
    else:
        ckmeta_path.write_text(json.dumps(current_ckmeta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # UNSEAL 2019
    # ------------------------------------------------------------------
    print()
    print("[UNSEAL 2019] verifying reserved 2019 model-ready artifacts", flush=True)
    scada_path = find_unique(mr, "2019_scada_model_ready.csv")
    leaks_path = find_unique(mr, "2019_leakages_model_ready.csv")
    verify(scada_path, H19_SCADA, "2019 SCADA")
    verify(leaks_path, H19_LEAKS, "2019 leakage ground truth")

    ledger_path = out / "2019_evaluation_attempt_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = {"attempts_started": 0, "attempts_completed": 0}
    ledger["attempts_started"] += 1
    ledger["last_evaluator_source_sha256"] = source_hash
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    scada = pd.read_csv(scada_path)
    leaks = pd.read_csv(leaks_path)
    scada["Timestamp"] = pd.to_datetime(scada["Timestamp"], errors="raise")
    leaks["Timestamp"] = pd.to_datetime(leaks["Timestamp"], errors="raise")

    if len(scada) != 105120 or len(leaks) != 105120:
        raise SystemExit(f"2019 row mismatch SCADA={len(scada)}, leaks={len(leaks)}")
    if not np.array_equal(scada.Timestamp.to_numpy(), leaks.Timestamp.to_numpy()):
        raise SystemExit("2019 SCADA/leak timestamps differ")
    dt = scada.Timestamp.diff().dropna()
    if not (dt == pd.Timedelta(minutes=5)).all():
        raise SystemExit("2019 timestamps are not an exact 5-minute series")

    norm = json.loads((tf / "FROZEN_2018_NORMALIZATION.json").read_text(encoding="utf-8"))
    feature_cols = [c for c in scada.columns if c != "Timestamp"]
    if feature_cols != norm["feature_order"]:
        raise SystemExit("2019 SCADA feature order differs from frozen 2018 feature order")
    leak_cols = [c for c in leaks.columns if c != "Timestamp"]
    if len(feature_cols) != 37 or len(leak_cols) != 23:
        raise SystemExit(f"Unexpected 2019 schema features={len(feature_cols)}, leak_cols={len(leak_cols)}")

    leak_values = leaks[leak_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(leak_values).all():
        raise SystemExit("Non-finite 2019 leakage values")

    manifest, events = build_2019_manifest(scada.Timestamp, leak_values, leak_cols)
    eligible = manifest[manifest.eligible_external_2019 == 1].copy().reset_index(drop=True)
    y = eligible.target_recent_leak_onset_24h.to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise SystemExit("Frozen 2019 external target has fewer than two classes; stop without modifying protocol")

    manifest_path = out / "2019_external_window_manifest.csv"
    events_path = out / "2019_external_event_transitions.csv"
    manifest.to_csv(manifest_path, index=False)
    events.to_csv(events_path, index=False)

    attr_mask = eligible.attribution_subset.to_numpy(dtype=int) == 1
    attr_meta = eligible[attr_mask].copy().reset_index(drop=True)

    print(
        f"[2019 TARGET] candidate={len(manifest):,}, eligible={len(eligible):,}, "
        f"positive={int(y.sum()):,}, negative={int((1-y).sum()):,}, "
        f"attr_subset={len(attr_meta):,}",
        flush=True,
    )

    raw = scada[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise SystemExit("Non-finite 2019 SCADA values")
    base_norm = normalize(raw, norm)
    X_clean = materialize_windows(base_norm, eligible)
    X_attr_clean = X_clean[attr_mask]

    base_graph_meta = build_graph_metadata(inp, feature_cols)

    # Load frozen models.
    mlp_ck = torch.load(tf / "BattLeDIM_MLP_v0_16b.pt", map_location="cpu", weights_only=False)
    mlp = MLP(input_dim=24 * 37)
    mlp.load_state_dict(mlp_ck["model_state_dict"], strict=True)
    mlp.eval()

    base_core = load_core(tf / "BattLeDIM_GCN_GRU_v0_16b.pt", base_graph_meta)
    base_gcn = RawToGraphWrapperMissing(base_core, base_graph_meta)
    base_gcn.eval()

    impl = json.loads((ef / "FROZEN_FINAL_EVAL_IMPLEMENTATION_v0_16c.json").read_text(encoding="utf-8"))
    thresholds = json.loads((tf / "FROZEN_2018_RELIABILITY_THRESHOLDS.json").read_text(encoding="utf-8"))["thresholds"]
    centroids = json.loads((tf / "FROZEN_2018_ATTRIBUTION_CENTROIDS.json").read_text(encoding="utf-8"))
    recon = json.loads((tf / "FROZEN_2018_PRESSURE_RECONSTRUCTION.json").read_text(encoding="utf-8"))

    # Map pressure sensor IDs to feature indices from frozen graph input.
    graph_input = json.loads((tf / "FROZEN_LTown_GRAPH_INPUT.json").read_text(encoding="utf-8"))
    pressure_id_to_fi = {row["sensor_id"]: int(row["feature_index"]) for row in graph_input["pressure_mapping"]}

    pred_blocks = []
    attr_blocks = []
    profile_blocks = []
    recovery_blocks = []
    recovered_attr_blocks = []

    def get_pred(model_name, family, severity, X, model, batch):
        key = f"pred_{model_name}_{family}_{severity}.npz"
        path = ckdir / key
        if path.exists():
            z = np.load(path)
            p = z["probability"]
            if len(p) != len(eligible):
                raise RuntimeError(f"Checkpoint length mismatch {key}")
            print(f"[REUSE] {key}", flush=True)
            return p
        logits = predict_with_progress(model, X, batch, f"{model_name} {family}/{severity}")
        p = sigmoid(logits)
        checkpoint_npz(path, probability=p.astype(np.float64))
        return p

    def get_attr(model_name, family, severity, X, model, batch, probs_full):
        key = f"attr_{model_name}_{family}_{severity}.npz"
        path = ckdir / key
        if path.exists():
            z = np.load(path)
            profiles = z["profiles"]
            probs = z["probability"]
            if profiles.shape != (len(attr_meta), 37) or len(probs) != len(attr_meta):
                raise RuntimeError(f"Attribution checkpoint shape mismatch {key}")
            print(f"[REUSE] {key}", flush=True)
            return profiles, probs
        profiles, logits = input_x_gradient_progress(model, X, batch, f"{model_name} {family}/{severity}")
        probs = sigmoid(logits)
        expected = probs_full[attr_mask]
        err = float(np.max(np.abs(probs - expected)))
        if err > 2e-6:
            raise RuntimeError(f"Attribution forward mismatch {key}: {err}")
        checkpoint_npz(path, profiles=profiles.astype(np.float32), probability=probs.astype(np.float64))
        return profiles, probs

    def append_prediction_rows(model_name, family, severity, probs):
        pred = (probs >= PRED_THRESHOLD).astype(int)
        ent = entropy01(probs)
        block = eligible[[
            "window_index", "start_idx", "final_idx", "final_timestamp",
            "target_recent_leak_onset_24h", "attribution_subset"
        ]].copy()
        block.insert(0, "severity", severity)
        block.insert(0, "family", family)
        block.insert(0, "model", model_name)
        block["probability"] = probs
        block["prediction"] = pred
        block["correct"] = (pred == y).astype(int)
        block["predictive_entropy"] = ent
        pred_blocks.append(block)

    def append_attr_rows(model_name, family, severity, profiles, probs):
        preds, div = divergence_against_frozen_centroid(model_name, profiles, probs, centroids)
        ent = entropy01(probs)
        block = attr_meta[[
            "window_index", "final_idx", "final_timestamp",
            "target_recent_leak_onset_24h"
        ]].copy()
        start_profile = sum(x.shape[0] for x in profile_blocks)
        block.insert(0, "severity", severity)
        block.insert(0, "family", family)
        block.insert(0, "model", model_name)
        block["profile_row"] = np.arange(start_profile, start_profile + len(block))
        block["probability"] = probs
        block["prediction"] = preds
        block["correct"] = (preds == block.target_recent_leak_onset_24h.to_numpy(dtype=int)).astype(int)
        block["predictive_entropy"] = ent
        block["attribution_profile_divergence"] = div
        profile_blocks.append(profiles.astype(np.float32))
        attr_blocks.append(block)

    # CLEAN predictions / attribution.
    print("\n=== CLEAN 2019 ===", flush=True)
    p_mlp_clean = get_pred("MLP", "CLEAN", "clean", X_clean, mlp, 1024)
    p_gcn_clean = get_pred("GCN_GRU", "CLEAN", "clean", X_clean, base_gcn, 16)
    append_prediction_rows("MLP", "CLEAN", "clean", p_mlp_clean)
    append_prediction_rows("GCN_GRU", "CLEAN", "clean", p_gcn_clean)

    a_mlp_clean, ap_mlp_clean = get_attr("MLP", "CLEAN", "clean", X_attr_clean, mlp, 128, p_mlp_clean)
    a_gcn_clean, ap_gcn_clean = get_attr("GCN_GRU", "CLEAN", "clean", X_attr_clean, base_gcn, 8, p_gcn_clean)
    append_attr_rows("MLP", "CLEAN", "clean", a_mlp_clean, ap_mlp_clean)
    append_attr_rows("GCN_GRU", "CLEAN", "clean", a_gcn_clean, ap_gcn_clean)

    # Numeric degradations.
    missing_norm_by_sev = {}
    missing_fis_by_sev = {}
    for family in FAMILIES_NUMERIC:
        for severity in SEVERITIES:
            print(f"\n=== {family}/{severity} ===", flush=True)
            corrupted_raw, fis = corrupt_series(raw, family, severity, feature_cols, norm, impl)
            corr_norm = normalize(corrupted_raw, norm)
            X = materialize_windows(corr_norm, eligible)
            X_attr = X[attr_mask]
            if family == "MISSING_SENSOR":
                missing_norm_by_sev[severity] = corr_norm
                missing_fis_by_sev[severity] = fis

            p_mlp = get_pred("MLP", family, severity, X, mlp, 1024)
            gcn_core = base_core
            gcn_wrapper = RawToGraphWrapperMissing(
                gcn_core,
                base_graph_meta,
                missing_feature_indices=fis if family == "MISSING_SENSOR" else None,
            )
            p_gcn = get_pred("GCN_GRU", family, severity, X, gcn_wrapper, 16)
            append_prediction_rows("MLP", family, severity, p_mlp)
            append_prediction_rows("GCN_GRU", family, severity, p_gcn)

            prof_mlp, pp_mlp = get_attr("MLP", family, severity, X_attr, mlp, 128, p_mlp)
            prof_gcn, pp_gcn = get_attr("GCN_GRU", family, severity, X_attr, gcn_wrapper, 8, p_gcn)
            append_attr_rows("MLP", family, severity, prof_mlp, pp_mlp)
            append_attr_rows("GCN_GRU", family, severity, prof_gcn, pp_gcn)

    # Topology mismatch: GCN only.
    for severity in SEVERITIES:
        family = "TOPOLOGY_MISMATCH"
        print(f"\n=== {family}/{severity} ===", flush=True)
        removed = impl["topology_edge_selection"][severity]
        meta = make_graph_meta_with_removed(inp, feature_cols, removed)
        core = load_core(tf / "BattLeDIM_GCN_GRU_v0_16b.pt", meta)
        wrapper = RawToGraphWrapperMissing(core, meta)
        p = get_pred("GCN_GRU", family, severity, X_clean, wrapper, 16)
        append_prediction_rows("GCN_GRU", family, severity, p)
        prof, pp = get_attr("GCN_GRU", family, severity, X_attr_clean, wrapper, 8, p)
        append_attr_rows("GCN_GRU", family, severity, prof, pp)

    pred_df = pd.concat(pred_blocks, ignore_index=True)
    attr_df = pd.concat(attr_blocks, ignore_index=True)
    profiles_all = np.concatenate(profile_blocks, axis=0)

    pred_path = out / "2019_external_predictions.csv"
    attr_index_path = out / "2019_external_attribution_index.csv"
    attr_profiles_path = out / "2019_external_attribution_profiles.npz"
    pred_df.to_csv(pred_path, index=False)
    attr_df.to_csv(attr_index_path, index=False)
    np.savez_compressed(attr_profiles_path, profiles=profiles_all, feature_order=np.asarray(feature_cols, dtype=str))

    # ------------------------------------------------------------------
    # Missing-sensor recovery, full-set and attribution subset.
    # ------------------------------------------------------------------
    for severity in SEVERITIES:
        print(f"\n=== MISSING RECONSTRUCTION/{severity} ===", flush=True)
        fis = missing_fis_by_sev[severity]
        rec_norm = reconstruct_normalized(missing_norm_by_sev[severity], fis, recon)
        X_rec = materialize_windows(rec_norm, eligible)
        X_rec_attr = X_rec[attr_mask]

        for model_name in ["MLP", "GCN_GRU"]:
            key = f"recovery_{model_name}_{severity}.npz"
            path = ckdir / key
            if path.exists():
                z = np.load(path)
                probs = z["probability"]
                print(f"[REUSE] {key}", flush=True)
            else:
                if model_name == "MLP":
                    logits = predict_with_progress(mlp, X_rec, 1024, f"MLP RECOVERY/{severity}")
                else:
                    wrapper = RawToGraphWrapperMissing(base_core, base_graph_meta, missing_feature_indices=fis)
                    logits = predict_with_progress(wrapper, X_rec, 16, f"GCN RECOVERY/{severity}")
                probs = sigmoid(logits)
                checkpoint_npz(path, probability=probs)

            b = eligible[["window_index", "final_timestamp", "target_recent_leak_onset_24h"]].copy()
            b.insert(0, "severity", severity)
            b.insert(0, "model", model_name)
            b["reconstructed_probability"] = probs
            b["reconstructed_prediction"] = (probs >= PRED_THRESHOLD).astype(int)
            b["reconstructed_correct"] = (
                b.reconstructed_prediction.to_numpy(dtype=int) == y
            ).astype(int)
            recovery_blocks.append(b)

            akey = f"recovery_attr_{model_name}_{severity}.npz"
            apath = ckdir / akey
            if apath.exists():
                z = np.load(apath)
                prof = z["profiles"]
                aprobs = z["probability"]
                print(f"[REUSE] {akey}", flush=True)
            else:
                if model_name == "MLP":
                    model = mlp
                    batch = 128
                else:
                    model = RawToGraphWrapperMissing(base_core, base_graph_meta, missing_feature_indices=fis)
                    batch = 8
                prof, logits = input_x_gradient_progress(model, X_rec_attr, batch, f"{model_name} RECOVERY/{severity}")
                aprobs = sigmoid(logits)
                err = float(np.max(np.abs(aprobs - probs[attr_mask])))
                if err > 2e-6:
                    raise RuntimeError(f"Recovered attribution probability mismatch {model_name}/{severity}: {err}")
                checkpoint_npz(apath, profiles=prof, probability=aprobs)

            preds, div = divergence_against_frozen_centroid(model_name, prof, aprobs, centroids)
            ab = attr_meta[["window_index", "final_timestamp", "target_recent_leak_onset_24h"]].copy()
            ab.insert(0, "severity", severity)
            ab.insert(0, "model", model_name)
            ab["reconstructed_probability"] = aprobs
            ab["reconstructed_prediction"] = preds
            ab["reconstructed_correct"] = (
                preds == ab.target_recent_leak_onset_24h.to_numpy(dtype=int)
            ).astype(int)
            ab["reconstructed_entropy"] = entropy01(aprobs)
            ab["reconstructed_attribution_divergence"] = div
            recovered_attr_blocks.append(ab)

    recovery_df = pd.concat(recovery_blocks, ignore_index=True)
    recovery_attr_df = pd.concat(recovered_attr_blocks, ignore_index=True)
    recovery_path = out / "2019_missing_reconstruction.csv"
    recovery_attr_path = out / "2019_missing_reconstruction_attribution.csv"
    recovery_df.to_csv(recovery_path, index=False)
    recovery_attr_df.to_csv(recovery_attr_path, index=False)

    # ------------------------------------------------------------------
    # Metrics: predictive, detection, recovery, controller.
    # ------------------------------------------------------------------
    cond_rows = []
    for (model_name, family, severity), d in pred_df.groupby(["model", "family", "severity"], sort=True):
        m = classification_metrics(
            d.target_recent_leak_onset_24h.to_numpy(dtype=int),
            d.probability.to_numpy(dtype=float),
        )
        m.update({"model": model_name, "family": family, "severity": severity})
        cond_rows.append(m)
    cond_df = pd.DataFrame(cond_rows)
    cond_path = out / "2019_external_condition_metrics.csv"
    cond_df.to_csv(cond_path, index=False)

    entropy_rows = []
    for model_name in ["MLP", "GCN_GRU"]:
        clean = pred_df[(pred_df.model == model_name) & (pred_df.family == "CLEAN")]
        deg_groups = pred_df[(pred_df.model == model_name) & (pred_df.family != "CLEAN")].groupby(["family", "severity"], sort=True)
        for (family, severity), deg in deg_groups:
            yd = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(deg), dtype=int)]
            score = np.r_[clean.predictive_entropy.to_numpy(), deg.predictive_entropy.to_numpy()]
            entropy_rows.append({
                "model": model_name,
                "family": family,
                "severity": severity,
                "n_clean": len(clean),
                "n_degraded": len(deg),
                "entropy_auroc": safe_auroc(yd, score),
                "entropy_auprc": safe_auprc(yd, score),
            })
    entropy_df = pd.DataFrame(entropy_rows)
    entropy_path = out / "2019_entropy_detection.csv"
    entropy_df.to_csv(entropy_path, index=False)

    attr_detect_rows = []
    for model_name in ["MLP", "GCN_GRU"]:
        clean = attr_df[(attr_df.model == model_name) & (attr_df.family == "CLEAN")]
        deg_groups = attr_df[(attr_df.model == model_name) & (attr_df.family != "CLEAN")].groupby(["family", "severity"], sort=True)
        for (family, severity), deg in deg_groups:
            yd = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(deg), dtype=int)]
            ad = np.r_[clean.attribution_profile_divergence.to_numpy(), deg.attribution_profile_divergence.to_numpy()]
            en = np.r_[clean.predictive_entropy.to_numpy(), deg.predictive_entropy.to_numpy()]
            aroc = safe_auroc(yd, ad)
            eroc = safe_auroc(yd, en)
            attr_detect_rows.append({
                "model": model_name,
                "family": family,
                "severity": severity,
                "n_clean": len(clean),
                "n_degraded": len(deg),
                "attribution_divergence_auroc": aroc,
                "attribution_divergence_auprc": safe_auprc(yd, ad),
                "entropy_auroc_same_subset": eroc,
                "entropy_auprc_same_subset": safe_auprc(yd, en),
                "auroc_attr_minus_entropy": None if aroc is None or eroc is None else aroc - eroc,
            })
    attr_detect_df = pd.DataFrame(attr_detect_rows)
    attr_detect_path = out / "2019_attribution_detection.csv"
    attr_detect_df.to_csv(attr_detect_path, index=False)

    recovery_metric_rows = []
    for model_name in ["MLP", "GCN_GRU"]:
        for severity in SEVERITIES:
            pre = pred_df[(pred_df.model == model_name) & (pred_df.family == "MISSING_SENSOR") & (pred_df.severity == severity)]
            post = recovery_df[(recovery_df.model == model_name) & (recovery_df.severity == severity)]
            pre_acc = float(pre.correct.mean())
            post_acc = float(post.reconstructed_correct.mean())
            recovery_metric_rows.append({
                "model": model_name,
                "severity": severity,
                "n": len(pre),
                "pre_reconstruction_accuracy": pre_acc,
                "post_reconstruction_accuracy": post_acc,
                "accuracy_change": post_acc - pre_acc,
            })
    recovery_metrics_df = pd.DataFrame(recovery_metric_rows)
    recovery_metrics_path = out / "2019_missing_reconstruction_metrics.csv"
    recovery_metrics_df.to_csv(recovery_metrics_path, index=False)

    # Controller on frozen attribution subset.
    recovery_attr_lookup = {
        (str(r.model), str(r.severity), int(r.window_index)): r
        for _, r in recovery_attr_df.iterrows()
    }
    controller_rows = []
    for _, r in attr_df.iterrows():
        model_name = str(r.model)
        family = str(r.family)
        severity = str(r.severity)
        wid = int(r.window_index)
        ent_thr = float(thresholds[model_name]["entropy_q95"])
        div_thr = float(thresholds[model_name]["attribution_divergence_q95"])

        raw_correct = int(r.correct)
        raw_error = 1 - raw_correct

        if family == "MISSING_SENSOR":
            rr = recovery_attr_lookup[(model_name, severity, wid)]
            initial_action = "RECONSTRUCT"
            final_ent = float(rr.reconstructed_entropy)
            final_div = float(rr.reconstructed_attribution_divergence)
            final_prediction = int(rr.reconstructed_prediction)
            final_correct = int(rr.reconstructed_correct)
            final_action = action(final_ent, final_div, ent_thr, div_thr)
        else:
            initial_action = action(float(r.predictive_entropy), float(r.attribution_profile_divergence), ent_thr, div_thr)
            final_action = initial_action
            final_prediction = int(r.prediction)
            final_correct = int(r.correct)
            final_ent = float(r.predictive_entropy)
            final_div = float(r.attribution_profile_divergence)

        recovery_forced_error = 1 - final_correct
        proceed = int(final_action == "PROCEED")
        unsafe = int(proceed and not final_correct)
        controller_rows.append({
            "model": model_name,
            "family": family,
            "severity": severity,
            "window_index": wid,
            "final_timestamp": str(r.final_timestamp),
            "target_recent_leak_onset_24h": int(r.target_recent_leak_onset_24h),
            "raw_prediction": int(r.prediction),
            "raw_correct": raw_correct,
            "raw_always_act_error": raw_error,
            "initial_action": initial_action,
            "final_action": final_action,
            "final_prediction": final_prediction,
            "final_correct": final_correct,
            "recovery_prescribed_forced_error": recovery_forced_error,
            "final_proceed": proceed,
            "unsafe_proceed": unsafe,
            "final_entropy": final_ent,
            "final_attribution_divergence": final_div,
        })

    controller_df = pd.DataFrame(controller_rows)
    controller_path = out / "2019_controller_outcomes.csv"
    controller_df.to_csv(controller_path, index=False)

    controller_metric_rows = []
    for (model_name, family, severity), d in controller_df.groupby(["model", "family", "severity"], sort=True):
        controller_metric_rows.append({
            "model": model_name,
            "family": family,
            "severity": severity,
            "n": len(d),
            "raw_always_act_error_rate": float(d.raw_always_act_error.mean()),
            "recovery_prescribed_forced_error_rate": float(d.recovery_prescribed_forced_error.mean()),
            "proceed_rate": float(d.final_proceed.mean()),
            "unsafe_proceed_rate": float(d.unsafe_proceed.mean()),
            "unsafe_reduction_vs_raw_always_act": float(d.raw_always_act_error.mean() - d.unsafe_proceed.mean()),
            "unsafe_reduction_vs_recovery_forced": float(d.recovery_prescribed_forced_error.mean() - d.unsafe_proceed.mean()),
            "verify_rate": float((d.final_action == "VERIFY").mean()),
            "request_sensor_rate": float((d.final_action == "REQUEST_SENSOR").mean()),
            "escalate_rate": float((d.final_action == "ESCALATE").mean()),
        })
    controller_metrics_df = pd.DataFrame(controller_metric_rows)
    controller_metrics_path = out / "2019_controller_metrics.csv"
    controller_metrics_df.to_csv(controller_metrics_path, index=False)

    overall_rows = []
    for model_name in ["MLP", "GCN_GRU"]:
        for state in ["clean", "degraded"]:
            if state == "clean":
                d = controller_df[(controller_df.model == model_name) & (controller_df.family == "CLEAN")]
            else:
                d = controller_df[(controller_df.model == model_name) & (controller_df.family != "CLEAN")]
            overall_rows.append({
                "model": model_name,
                "state": state,
                "n": len(d),
                "raw_always_act_error_rate": float(d.raw_always_act_error.mean()),
                "recovery_prescribed_forced_error_rate": float(d.recovery_prescribed_forced_error.mean()),
                "proceed_rate": float(d.final_proceed.mean()),
                "unsafe_proceed_rate": float(d.unsafe_proceed.mean()),
                "unsafe_reduction_vs_raw_always_act": float(d.raw_always_act_error.mean() - d.unsafe_proceed.mean()),
            })
    overall_df = pd.DataFrame(overall_rows)
    overall_path = out / "2019_controller_overall.csv"
    overall_df.to_csv(overall_path, index=False)

    # ------------------------------------------------------------------
    # Final record and human-readable summary.
    # ------------------------------------------------------------------
    artifacts = [
        manifest_path, events_path, pred_path, cond_path, entropy_path,
        attr_profiles_path, attr_index_path, attr_detect_path,
        recovery_path, recovery_attr_path, recovery_metrics_path,
        controller_path, controller_metrics_path, overall_path,
    ]
    hashes = {p.name: sha256(p) for p in artifacts}

    clean_metrics = {
        row.model: {
            k: row[k]
            for k in ["n", "positive", "negative", "accuracy", "balanced_accuracy", "f1", "auprc", "auroc", "brier"]
        }
        for _, row in cond_df[cond_df.family == "CLEAN"].iterrows()
    }

    entropy_vals = entropy_df.entropy_auroc.dropna().to_numpy(dtype=float)
    attr_vals = attr_detect_df.attribution_divergence_auroc.dropna().to_numpy(dtype=float)
    attr_gt = int((attr_detect_df.auroc_attr_minus_entropy.dropna() > 0).sum())

    final_record = {
        "status": "PASS_FINAL_EXTERNAL_2019_EVALUATION_COMPLETE",
        "version": "0.17",
        "evaluator_source_sha256": source_hash,
        "2019_scada_sha256": H19_SCADA,
        "2019_leakages_sha256": H19_LEAKS,
        "frozen_final_eval_implementation_sha256": H_FINAL_IMPL,
        "models_retrained_after_2019": False,
        "thresholds_changed_after_2019": False,
        "controller_changed_after_2019": False,
        "corruptions_changed_after_2019": False,
        "candidate_windows": len(manifest),
        "eligible_windows": len(eligible),
        "eligible_positive_windows": int(y.sum()),
        "eligible_negative_windows": int((1-y).sum()),
        "attribution_subset_windows": len(attr_meta),
        "observed_2019_onset_transitions": int((events.transition_type == "ONSET").sum()) if len(events) else 0,
        "observed_2019_end_transitions": int((events.transition_type == "END").sum()) if len(events) else 0,
        "year_boundary_rule": (
            "leak active at first 2019 sample is treated as pre-existing and not "
            "as an observed onset; activity positive at final sample gets no unobserved end"
        ),
        "clean_external_metrics": clean_metrics,
        "artifact_sha256": hashes,
    }
    record_path = out / "FINAL_EXTERNAL_2019_RECORD_v0_17.json"
    record_path.write_text(json.dumps(final_record, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x)), encoding="utf-8")

    ledger["attempts_completed"] += 1
    ledger["final_record_sha256"] = sha256(record_path)
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA BATTLEDIM/L-TOWN FINAL EXTERNAL 2019 EVALUATION v0.17",
        "=" * 78,
        "STATUS: PASS / FINAL EXTERNAL 2019 EVALUATION COMPLETE",
        "",
        "INTEGRITY",
        f"  evaluator source SHA-256: {source_hash}",
        f"  evaluation attempts started/completed: {ledger['attempts_started']}/{ledger['attempts_completed']}",
        "  frozen MLP/GCN changed after 2019: NO",
        "  frozen thresholds/controller/reconstruction changed after 2019: NO",
        "  post-2019 tuning: NO",
        "",
        "2019 TARGET ACCOUNTING",
        f"  leakage columns: {len(leak_cols)}",
        f"  observed within-2019 onset transitions: {final_record['observed_2019_onset_transitions']}",
        f"  observed within-2019 end transitions: {final_record['observed_2019_end_transitions']}",
        f"  candidate windows: {len(manifest):,}",
        f"  eligible windows: {len(eligible):,}",
        f"  positives: {int(y.sum()):,}",
        f"  negatives: {int((1-y).sum()):,}",
        f"  positive fraction: {y.mean():.4%}",
        f"  attribution subset: {len(attr_meta):,}",
        "",
        "CLEAN EXTERNAL PREDICTIVE METRICS",
    ]
    for model_name in ["MLP", "GCN_GRU"]:
        m = clean_metrics[model_name]
        lines.append(
            f"  {model_name}: balanced_accuracy={m['balanced_accuracy']:.6f}, "
            f"F1={m['f1']:.6f}, AUPRC={m['auprc']:.6f}, "
            f"AUROC={m['auroc']:.6f}, Brier={m['brier']:.6f}"
        )

    lines += [
        "",
        "FULL-SET ENTROPY EVIDENCE-FAILURE DETECTION",
        f"  AUROC min/median/max: {entropy_vals.min():.6f} / {np.median(entropy_vals):.6f} / {entropy_vals.max():.6f}",
        "",
        "ATTRIBUTION DIVERGENCE VS ENTROPY — SAME FROZEN SUBSET",
        f"  attribution AUROC min/median/max: {attr_vals.min():.6f} / {np.median(attr_vals):.6f} / {attr_vals.max():.6f}",
        f"  conditions attribution AUROC > entropy AUROC: {attr_gt}/{len(attr_detect_df)}",
        "",
        "MISSING-SENSOR RECONSTRUCTION — FULL EXTERNAL SET",
    ]
    for _, r in recovery_metrics_df.iterrows():
        lines.append(
            f"  {r.model} {r.severity}: pre={r.pre_reconstruction_accuracy:.6f}, "
            f"post={r.post_reconstruction_accuracy:.6f}, delta={r.accuracy_change:+.6f}"
        )

    lines += ["", "FROZEN CONTROLLER — ATTRIBUTION SUBSET"]
    for _, r in overall_df.iterrows():
        lines.append(
            f"  {r.model} {r.state}: n={int(r.n)}, "
            f"raw_error={r.raw_always_act_error_rate:.6f}, "
            f"recovery_forced_error={r.recovery_prescribed_forced_error_rate:.6f}, "
            f"proceed={r.proceed_rate:.6f}, unsafe_proceed={r.unsafe_proceed_rate:.6f}, "
            f"unsafe_reduction_vs_raw={r.unsafe_reduction_vs_raw_always_act:.6f}"
        )

    lines += ["", "ARTIFACT HASHES"]
    for name, h in hashes.items():
        lines.append(f"  {name}: {h}")
    lines += [
        f"  FINAL_EXTERNAL_2019_RECORD_v0_17.json: {sha256(record_path)}",
        "",
        "NEXT",
        "  Retain BattLeDIM_final_external_2019_summary_v0_17.txt with the project record.",
        "  Do not retrain, retune, or regenerate conditions from 2019 results.",
        "  Next step is Reproducibility external-validation audit and final interpretation.",
    ]

    summary_path = out / "BattLeDIM_final_external_2019_summary_v0_17.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
