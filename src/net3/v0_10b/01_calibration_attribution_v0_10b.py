
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from sklearn.metrics import average_precision_score, roc_auc_score

# ----------------------------------------------------------------------
# Frozen artifact hashes
# ----------------------------------------------------------------------

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

BASELINE_HASH = "3e83192de687c42db771964aa82715b6221ad8dc89423484115a9b6aed23a040"
GRAPH_HASH = "c8d104c5512556dbbcb459eb63ccef8ba8e71cb04fc9b22a5b16c23eee51e0af"

DEGRADATION_SOURCE_HASH = "0203ffcb0a456cbdb09c1186a9d669f0266cc85b200e13dbfcd53a62dd746b2c"
DEGRADATION_WINDOWS_COPY_HASH = "9c378c9396c6432ff555b6274606099ff262bc18fb6ac2e206631ca4b9266815"

CAL_PRED_HASH = "36edd2accc9c990d5806a48079348f18319554bc86fa08717597f29df3a2429e"
CAL_CORRUPTION_MANIFEST_HASH = "df467191e0acc4b257ad6ae8002468d10943352e5b741e9f1d8d34519d0117a8"
CAL_CONDITION_METRICS_HASH = "085a4e1d3f3f3c85f29ad2b5ac153396b55f019a4c18eeb922a4a774b67931f8"
CAL_ENTROPY_HASH = "855317985d3578fc8daf3173d16d9418020a80bdb0c0ebb8c7d5bb0640d38ffe"
CAL_EXECUTION_HASH = "8a4dcbe09472d2c8cf51d326a80c8c2b2f1d1b43947320484066d6d124167b00"

BASELINE_THRESHOLD = 0.464201702923
GRAPH_THRESHOLD = 0.376129878873
ROOT_SEED = 8675309

SEVERITIES = ("low", "medium", "high")
FAMILIES = (
    "MISSING_SENSOR",
    "SENSOR_NOISE",
    "SENSOR_BIAS",
    "STALE_SENSOR",
    "TOPOLOGY_MISMATCH",
    "CROSS_SOURCE_CONFLICT",
)

NOISE_SD = {"low": 0.25, "medium": 0.50, "high": 1.00}
BIAS_SD = {"low": 0.50, "medium": 1.00, "high": 2.00}
STALE_LAG = {"low": 6, "medium": 12, "high": 24}

ATTR_POSITION_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 44]
PROFILE_EPS = 1e-12
PROB_MATCH_TOL = 2e-6

# exact model architecture
MLP_H1, MLP_H2 = 256, 64
NODE_FEATURES, GCN_HIDDEN, GRU_HIDDEN, HEAD_HIDDEN = 10, 32, 64, 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected, label=None):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing artifact: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH {label or path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {label or path.name}")


def stable_digest(*parts):
    return hashlib.sha256(
        " | ".join(str(x) for x in parts).encode("utf-8")
    ).hexdigest()


def stable_seed(*parts):
    return int(stable_digest(*parts)[:16], 16) % (2**63 - 1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(792, MLP_H1),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(MLP_H1, MLP_H2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(MLP_H2, 1),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Linear(i, o)

    def forward(self, x, A):
        return self.linear(torch.einsum("ij,btjf->btif", A, x))


class GraphTemporalModel(nn.Module):
    def __init__(self, A_norm):
        super().__init__()
        self.register_buffer("A_norm", torch.tensor(A_norm, dtype=torch.float32))
        self.gcn1 = GCNLayer(NODE_FEATURES, GCN_HIDDEN)
        self.gcn2 = GCNLayer(GCN_HIDDEN, GCN_HIDDEN)
        self.dropout = nn.Dropout(0.20)
        self.gru = nn.GRU(GCN_HIDDEN * 2, GRU_HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, x):
        h = self.dropout(torch.relu(self.gcn1(x, self.A_norm)))
        h = self.dropout(torch.relu(self.gcn2(h, self.A_norm)))
        z = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=-1)
        o, _ = self.gru(z)
        return self.head(o[:, -1]).squeeze(-1)


def build_topology(gen, layout):
    nodes = pd.read_csv(gen / "graph_nodes.csv", dtype={"node_id": str})
    links = pd.read_csv(
        gen / "graph_links.csv",
        dtype={"link_id": str, "start_node": str, "end_node": str},
    )

    node_ids = [str(x) for x in layout["original_node_ids"]]
    ix = {n: i for i, n in enumerate(node_ids)}
    link_rows = []

    for _, r in links.iterrows():
        lid = str(r["link_id"])
        u, v = str(r["start_node"]), str(r["end_node"])
        link_rows.append({
            "link_id": lid,
            "link_type": str(r["link_type"]),
            "u": u,
            "v": v,
            "ui": ix[u],
            "vi": ix[v],
        })

    node_type = {
        str(r["node_id"]): str(r["node_type"]).lower()
        for _, r in nodes.iterrows()
    }

    return node_ids, ix, link_rows, node_type


def adjacency_from_links(n, link_rows):
    A = np.zeros((n, n), dtype=np.float32)
    for r in link_rows:
        A[r["ui"], r["vi"]] = 1.0
        A[r["vi"], r["ui"]] = 1.0
    return A


def normalized_adjacency(A):
    Ah = A + np.eye(len(A), dtype=np.float32)
    d = Ah.sum(1)
    q = np.zeros_like(d)
    mask = d > 0
    q[mask] = np.power(d[mask], -0.5).astype(np.float32)
    return q[:, None] * Ah * q[None, :]


def static_features(A, node_ids, node_type):
    degree = A.sum(1)
    st = np.zeros((len(node_ids), 4), dtype=np.float32)
    st[:, 0] = degree / max(float(degree.max()), 1.0)
    for i, nid in enumerate(node_ids):
        t = node_type.get(nid, "")
        st[i, 1] = 1.0 if "junction" in t else 0.0
        st[i, 2] = 1.0 if "tank" in t else 0.0
        st[i, 3] = 1.0 if "reservoir" in t else 0.0
    return st


def sensor_maps(layout, ix, link_rows):
    endpoints = {r["link_id"]: (r["ui"], r["vi"]) for r in link_rows}
    pressure, flow, tank = [], [], []

    for fi, name in enumerate(layout["sensor_feature_order"]):
        if name.startswith("pressure::"):
            nid = name.split("::", 1)[1]
            pressure.append((fi, nid, ix[nid]))
        elif name.startswith("flow::"):
            lid = name.split("::", 1)[1]
            flow.append((fi, lid, *endpoints[lid]))
        elif name.startswith("tank_level::"):
            nid = name.split("::", 1)[1]
            tank.append((fi, nid, ix[nid]))

    if (len(pressure), len(flow), len(tank)) != (24, 6, 3):
        raise RuntimeError("sensor-map mismatch")

    return pressure, flow, tank


class GraphRawWrapper(nn.Module):
    """
    Differentiable mapping:
    normalized raw [B,T,33] -> graph [B,T,97,10] -> frozen GCN-GRU.
    """
    def __init__(
        self,
        graph_model,
        n_nodes,
        static,
        pmap,
        fmap,
        tmap,
        missing_feature_indices=None,
    ):
        super().__init__()
        self.graph_model = graph_model
        self.n_nodes = n_nodes

        P = np.zeros((33, n_nodes), dtype=np.float32)
        T = np.zeros((33, n_nodes), dtype=np.float32)
        F = np.zeros((33, n_nodes), dtype=np.float32)

        pressure_mask = np.zeros(n_nodes, dtype=np.float32)
        tank_mask = np.zeros(n_nodes, dtype=np.float32)
        flow_count = np.zeros(n_nodes, dtype=np.float32)

        missing = set(missing_feature_indices or [])

        for fi, sid, ni in pmap:
            P[fi, ni] = 1.0
            pressure_mask[ni] = 0.0 if fi in missing else 1.0

        for fi, sid, ni in tmap:
            T[fi, ni] = 1.0
            tank_mask[ni] = 1.0

        for fi, lid, u, v in fmap:
            F[fi, u] -= 1.0
            F[fi, v] += 1.0
            flow_count[u] += 1.0
            flow_count[v] += 1.0

        self.register_buffer("P", torch.tensor(P))
        self.register_buffer("T", torch.tensor(T))
        self.register_buffer("F", torch.tensor(F))
        self.register_buffer("pressure_mask", torch.tensor(pressure_mask))
        self.register_buffer("tank_mask", torch.tensor(tank_mask))
        self.register_buffer("flow_count", torch.tensor(flow_count))
        self.register_buffer("static", torch.tensor(static, dtype=torch.float32))

    def forward(self, x):
        # x is already frozen-train normalized.
        pressure_value = torch.einsum("btf,fn->btn", x, self.P)
        tank_value = torch.einsum("btf,fn->btn", x, self.T)
        flow_value = torch.einsum("btf,fn->btn", x, self.F)

        B, T, _ = pressure_value.shape

        p_mask = self.pressure_mask[None, None, :].expand(B, T, -1)
        t_mask = self.tank_mask[None, None, :].expand(B, T, -1)
        f_count = self.flow_count[None, None, :].expand(B, T, -1)
        static = self.static[None, None, :, :].expand(B, T, -1, -1)

        dyn = torch.stack(
            [
                pressure_value,
                p_mask,
                tank_value,
                t_mask,
                flow_value,
                f_count,
            ],
            dim=-1,
        )

        graph_x = torch.cat([dyn, static], dim=-1)
        return self.graph_model(graph_x)


def apply_numeric_corruption(
    family,
    severity,
    context_id,
    factual,
    counterfactual,
    selected_ids,
    p_id_to_fi,
    mean,
    scale,
):
    out = factual.copy()
    fis = [p_id_to_fi[sid] for sid in selected_ids]

    if family == "MISSING_SENSOR":
        for fi in fis:
            out[:, fi] = mean[fi]

    elif family == "SENSOR_NOISE":
        rng = np.random.default_rng(
            stable_seed(ROOT_SEED, context_id, family, severity)
        )
        z = rng.normal(0.0, 1.0, size=(len(out), len(fis)))
        for j, fi in enumerate(fis):
            out[:, fi] += (
                z[:, j].astype(np.float32)
                * NOISE_SD[severity]
                * scale[fi]
            )

    elif family == "SENSOR_BIAS":
        for fi, sid in zip(fis, selected_ids):
            sign = (
                1.0
                if int(
                    stable_digest(
                        ROOT_SEED, context_id, family, sid
                    )[-1],
                    16,
                ) % 2 == 0
                else -1.0
            )
            out[:, fi] += sign * BIAS_SD[severity] * scale[fi]

    elif family == "STALE_SENSOR":
        lag = STALE_LAG[severity]
        for fi in fis:
            source = factual[:, fi].copy()
            out[:lag, fi] = source[0]
            out[lag:, fi] = source[:-lag]

    elif family == "CROSS_SOURCE_CONFLICT":
        for fi in fis:
            out[:, fi] = counterfactual[:, fi]

    else:
        raise ValueError(family)

    return out, fis


def attribution_subset(cal):
    chosen = []

    for (cid, variant), group in cal.groupby(
        ["context_id", "variant"], sort=True
    ):
        g = group.sort_values("final_idx").reset_index(drop=True)
        if len(g) != 45:
            raise RuntimeError(
                f"{cid}/{variant}: expected 45 windows, got {len(g)}"
            )
        chosen.append(g.iloc[ATTR_POSITION_INDICES])

    out = pd.concat(chosen, ignore_index=True)

    if len(out) != 600:
        raise RuntimeError(f"Expected 600 attribution windows, got {len(out)}")

    return out


def input_x_gradient_profiles(wrapper, X, batch_size):
    wrapper.eval()
    profiles = []
    logits_all = []

    # Model parameters are frozen; we only need input gradients.
    for p in wrapper.parameters():
        p.requires_grad_(False)

    for start in range(0, len(X), batch_size):
        xb = torch.tensor(
            X[start:start + batch_size],
            dtype=torch.float32,
            requires_grad=True,
        )

        logits = wrapper(xb)
        grads = torch.autograd.grad(
            logits.sum(),
            xb,
            create_graph=False,
            retain_graph=False,
        )[0]

        attr = torch.abs(xb * grads).sum(dim=1)
        attr = attr + PROFILE_EPS
        attr = attr / attr.sum(dim=1, keepdim=True)

        profiles.append(attr.detach().cpu().numpy().astype(np.float32))
        logits_all.append(logits.detach().cpu().numpy().astype(np.float64))

    return np.concatenate(profiles), np.concatenate(logits_all)


def sigmoid(logits):
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, PROFILE_EPS, None)
    q = np.clip(q, PROFILE_EPS, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def parse_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return str(value).split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--degradation-protocol", required=True)
    ap.add_argument("--calibration-degradations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    win = Path(args.windows)
    gen = Path(args.generated)
    baseline_dir = Path(args.baseline)
    graph_dir = Path(args.graph)
    dp = Path(args.degradation_protocol)
    cd = Path(args.calibration_degradations)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA CALIBRATION ATTRIBUTION v0.10b")
    print("=" * 78)

    for n, h in WINDOW_HASHES.items():
        verify(win / n, h)

    for n, h in GENERATED_HASHES.items():
        verify(gen / n, h)

    verify(
        baseline_dir / "baseline_mlp_v0_5.pt",
        BASELINE_HASH,
        "frozen MLP",
    )
    verify(
        graph_dir / "graph_gcn_gru_v0_6.pt",
        GRAPH_HASH,
        "frozen GCN-GRU",
    )

    # Verify frozen protocol source hash from record + known Windows copy hash.
    record = json.loads(
        (dp / "FROZEN_DEGRADATION_PROTOCOL_RECORD.json").read_text(
            encoding="utf-8"
        )
    )
    if record.get("protocol_sha256") != DEGRADATION_SOURCE_HASH:
        raise SystemExit("v0.8 frozen record source hash mismatch.")
    print("[OK] v0.8 frozen source hash")
    verify(
        dp / "degradation_protocol_v0_8.md",
        DEGRADATION_WINDOWS_COPY_HASH,
        "v0.8 Windows protocol copy",
    )

    verify(
        cd / "calibration_degradation_predictions.csv",
        CAL_PRED_HASH,
    )
    verify(
        cd / "calibration_corruption_manifest.csv",
        CAL_CORRUPTION_MANIFEST_HASH,
    )
    verify(
        cd / "calibration_condition_metrics.csv",
        CAL_CONDITION_METRICS_HASH,
    )
    verify(
        cd / "calibration_entropy_detection.csv",
        CAL_ENTROPY_HASH,
    )
    verify(
        cd / "calibration_degradation_execution.json",
        CAL_EXECUTION_HASH,
    )

    manifest = pd.read_csv(win / "window_manifest.csv")
    cal = manifest[manifest["split"] == "calibration"].copy()

    if len(cal) != 2700 or cal["context_id"].nunique() != 30:
        raise SystemExit("Calibration split mismatch.")

    subset = attribution_subset(cal)
    selected_window_ids = set(subset["window_id"])

    norm = json.loads(
        (win / "normalization_train_only.json").read_text(encoding="utf-8")
    )
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["scale_used"], dtype=np.float32)

    layout = json.loads(
        (gen / "tensor_layout.json").read_text(encoding="utf-8")
    )

    node_ids, ix, link_rows, node_type = build_topology(gen, layout)
    base_A = adjacency_from_links(len(node_ids), link_rows)
    base_An = normalized_adjacency(base_A)
    base_static = static_features(base_A, node_ids, node_type)
    pmap, fmap, tmap = sensor_maps(layout, ix, link_rows)

    pressure_ids = [sid for _, sid, _ in pmap]
    p_id_to_fi = {sid: fi for fi, sid, _ in pmap}

    frozen_pred = pd.read_csv(
        cd / "calibration_degradation_predictions.csv"
    )
    frozen_pred = frozen_pred[
        frozen_pred["window_id"].isin(selected_window_ids)
    ].copy()

    corr_manifest = pd.read_csv(
        cd / "calibration_corruption_manifest.csv"
    )

    # Load calibration raw contexts only.
    raw = {}
    for cid in sorted(subset["context_id"].unique()):
        with np.load(
            gen / "contexts" / f"{cid}.npz",
            allow_pickle=False,
        ) as z:
            raw[(cid, "CLEAN")] = z["clean_sensor"].astype(np.float32)
            raw[(cid, "LEAK")] = z["leak_sensor"].astype(np.float32)

    print("[BOUNDARY] calibration contexts loaded:", len(set(c for c, _ in raw)))
    print("[BOUNDARY] degraded test loaded: NO")

    # Strictly load frozen models.
    bc = torch.load(
        baseline_dir / "baseline_mlp_v0_5.pt",
        map_location="cpu",
        weights_only=False,
    )
    mlp = MLP()
    # MLP training checkpoint uses same module names.
    mlp.load_state_dict(bc["model_state_dict"], strict=True)
    mlp.eval()

    gc = torch.load(
        graph_dir / "graph_gcn_gru_v0_6.pt",
        map_location="cpu",
        weights_only=False,
    )
    graph_model = GraphTemporalModel(base_An)
    graph_model.load_state_dict(gc["model_state_dict"], strict=True)
    graph_model.eval()

    profile_blocks = []
    index_rows = []
    profile_offset = 0
    max_probability_reproduction_error = 0.0

    def frozen_condition_rows(model_name, family, severity):
        d = frozen_pred[
            (frozen_pred["model"] == model_name)
            & (frozen_pred["family"] == family)
            & (frozen_pred["severity"] == severity)
        ].copy()
        return d

    def expected_lookup(model_name, family, severity):
        d = frozen_condition_rows(model_name, family, severity)
        return {
            str(r["window_id"]): r
            for _, r in d.iterrows()
        }

    def record_block(
        model_name,
        family,
        severity,
        rows,
        profiles,
        logits,
    ):
        nonlocal profile_offset, max_probability_reproduction_error

        probs = sigmoid(logits)
        expected_all = expected_lookup(model_name, family, severity)

        # v0.10b fix:
        # GCN attribution is intentionally processed context-by-context for
        # context-specific missing-sensor masks and topology. Therefore a
        # record_block call may contain only 20 rows even though the frozen
        # condition contains 600 attribution-subset rows overall.
        #
        # Audit against the exact window IDs in THIS batch rather than requiring
        # the batch length to equal the full-condition lookup length.
        batch_window_ids = [str(x) for x in rows["window_id"].tolist()]
        if len(batch_window_ids) != len(set(batch_window_ids)):
            raise RuntimeError(
                f"{model_name}/{family}/{severity}: duplicate window IDs "
                "inside attribution audit batch"
            )

        missing_expected = [
            wid for wid in batch_window_ids
            if wid not in expected_all
        ]
        if missing_expected:
            raise RuntimeError(
                f"{model_name}/{family}/{severity}: "
                f"{len(missing_expected)} attribution window IDs absent from "
                f"frozen v0.9b predictions; first={missing_expected[:5]}"
            )

        for j, (_, r) in enumerate(rows.iterrows()):
            wid = str(r["window_id"])
            er = expected_all[wid]
            err = abs(float(probs[j]) - float(er["probability"]))
            max_probability_reproduction_error = max(
                max_probability_reproduction_error,
                err,
            )
            if err > PROB_MATCH_TOL:
                raise RuntimeError(
                    f"Probability reproduction mismatch "
                    f"{model_name}/{family}/{severity}/{wid}: "
                    f"new={probs[j]} frozen={er['probability']} err={err}"
                )

            index_rows.append({
                "profile_row": profile_offset + j,
                "model": model_name,
                "family": family,
                "severity": severity,
                "evidence_failure": 0 if family == "CLEAN" else 1,
                "window_id": wid,
                "pair_id": str(r["pair_id"]),
                "context_id": str(r["context_id"]),
                "variant": str(r["variant"]),
                "final_idx": int(r["final_idx"]),
                "target_active_leak": int(r["target_active_leak"]),
                "probability": float(er["probability"]),
                "prediction": int(er["prediction"]),
                "correct": int(er["correct"]),
                "predictive_entropy": float(er["predictive_entropy"]),
            })

        profile_blocks.append(profiles.astype(np.float32))
        profile_offset += len(rows)

    # ------------------------------------------------------------------
    # CLEAN attribution first
    # ------------------------------------------------------------------
    print("[ATTR] CLEAN / MLP")
    X = []
    clean_rows = subset.sort_values(
        ["context_id", "variant", "final_idx"]
    ).reset_index(drop=True)

    for _, r in clean_rows.iterrows():
        arr = raw[(str(r["context_id"]), str(r["variant"]))]
        s, f = int(r["start_idx"]), int(r["final_idx"])
        X.append(((arr[s:f+1] - mean) / scale).astype(np.float32))

    X = np.stack(X)
    profiles, logits = input_x_gradient_profiles(mlp, X, batch_size=128)
    record_block("MLP", "CLEAN", "clean", clean_rows, profiles, logits)

    print("[ATTR] CLEAN / GCN-GRU")
    base_wrapper = GraphRawWrapper(
        graph_model,
        len(node_ids),
        base_static,
        pmap,
        fmap,
        tmap,
    )
    profiles, logits = input_x_gradient_profiles(
        base_wrapper, X, batch_size=32
    )
    record_block(
        "GCN_GRU", "CLEAN", "clean",
        clean_rows, profiles, logits
    )

    # ------------------------------------------------------------------
    # Numeric degradations
    # ------------------------------------------------------------------
    numeric_families = [
        "MISSING_SENSOR",
        "SENSOR_NOISE",
        "SENSOR_BIAS",
        "STALE_SENSOR",
        "CROSS_SOURCE_CONFLICT",
    ]

    for family in numeric_families:
        for severity in SEVERITIES:
            print(f"[ATTR] {family}/{severity}")

            mlp_X = []
            graph_X_by_missing_key = {}
            ordered_rows = clean_rows.copy()

            # For GCN missing-sensor mask varies by context, so process
            # context groups. Other numeric corruptions can also be kept
            # context-grouped for a uniform auditable path.
            context_payload = {}

            for cid, group in ordered_rows.groupby("context_id", sort=True):
                cm = corr_manifest[
                    (corr_manifest["context_id"] == cid)
                    & (corr_manifest["family"] == family)
                    & (corr_manifest["severity"] == severity)
                ]
                if len(cm) != 1:
                    raise RuntimeError(
                        f"Missing corruption manifest row: "
                        f"{cid}/{family}/{severity}"
                    )
                cmr = cm.iloc[0]
                selected_ids = parse_ids(cmr["selected_pressure_ids"])

                group_windows = []
                missing_fis_for_context = []

                for _, r in group.iterrows():
                    variant = str(r["variant"])
                    factual = raw[(cid, variant)]
                    other = raw[
                        (cid, "LEAK" if variant == "CLEAN" else "CLEAN")
                    ]
                    corr, missing_fis = apply_numeric_corruption(
                        family,
                        severity,
                        cid,
                        factual,
                        other,
                        selected_ids,
                        p_id_to_fi,
                        mean,
                        scale,
                    )
                    s, f = int(r["start_idx"]), int(r["final_idx"])
                    group_windows.append(
                        ((corr[s:f+1] - mean) / scale).astype(np.float32)
                    )
                    missing_fis_for_context = missing_fis

                context_payload[cid] = (
                    group.reset_index(drop=True),
                    np.stack(group_windows),
                    missing_fis_for_context,
                )

            # MLP: concatenate in clean_rows ordering by explicit window id.
            mlp_rows = []
            mlp_windows = []
            for cid in sorted(context_payload):
                rows_c, X_c, _ = context_payload[cid]
                mlp_rows.append(rows_c)
                mlp_windows.append(X_c)
            mlp_rows = pd.concat(mlp_rows, ignore_index=True)
            mlp_X = np.concatenate(mlp_windows, axis=0)

            p, l = input_x_gradient_profiles(
                mlp, mlp_X, batch_size=128
            )
            record_block(
                "MLP", family, severity, mlp_rows, p, l
            )

            # GCN: per context because MISSING_SENSOR mask can differ.
            for cid in sorted(context_payload):
                rows_c, X_c, missing_fis = context_payload[cid]
                wrapper = GraphRawWrapper(
                    graph_model,
                    len(node_ids),
                    base_static,
                    pmap,
                    fmap,
                    tmap,
                    missing_feature_indices=(
                        missing_fis if family == "MISSING_SENSOR" else None
                    ),
                )
                p, l = input_x_gradient_profiles(
                    wrapper, X_c, batch_size=32
                )
                record_block(
                    "GCN_GRU", family, severity,
                    rows_c, p, l
                )

    # ------------------------------------------------------------------
    # Topology mismatch: GCN only, context-specific adjacency/static
    # ------------------------------------------------------------------
    for severity in SEVERITIES:
        print(f"[ATTR] TOPOLOGY_MISMATCH/{severity} / GCN-GRU")

        for cid, rows_c in clean_rows.groupby("context_id", sort=True):
            cm = corr_manifest[
                (corr_manifest["context_id"] == cid)
                & (corr_manifest["family"] == "TOPOLOGY_MISMATCH")
                & (corr_manifest["severity"] == severity)
            ]
            if len(cm) != 1:
                raise RuntimeError(
                    f"Missing topology manifest row: {cid}/{severity}"
                )
            removed = set(parse_ids(cm.iloc[0]["selected_edges"]))
            active = [
                r for r in link_rows
                if r["link_id"] not in removed
            ]

            A = adjacency_from_links(len(node_ids), active)
            An = normalized_adjacency(A)
            st = static_features(A, node_ids, node_type)

            # Update the frozen graph model's model-visible adjacency only.
            graph_model.A_norm.copy_(
                torch.tensor(An, dtype=torch.float32)
            )

            X_c = []
            rows_c = rows_c.reset_index(drop=True)
            for _, r in rows_c.iterrows():
                arr = raw[(cid, str(r["variant"]))]
                s, f = int(r["start_idx"]), int(r["final_idx"])
                X_c.append(
                    ((arr[s:f+1] - mean) / scale).astype(np.float32)
                )
            X_c = np.stack(X_c)

            wrapper = GraphRawWrapper(
                graph_model,
                len(node_ids),
                st,
                pmap,
                fmap,
                tmap,
            )
            p, l = input_x_gradient_profiles(
                wrapper, X_c, batch_size=32
            )
            record_block(
                "GCN_GRU",
                "TOPOLOGY_MISMATCH",
                severity,
                rows_c,
                p,
                l,
            )

        # Restore original adjacency.
        graph_model.A_norm.copy_(
            torch.tensor(base_An, dtype=torch.float32)
        )

    if max_probability_reproduction_error > PROB_MATCH_TOL:
        raise RuntimeError("Probability reproduction tolerance exceeded.")

    profiles = np.concatenate(profile_blocks, axis=0)
    index_df = pd.DataFrame(index_rows)

    if len(index_df) != 21000 or profiles.shape != (21000, 33):
        raise RuntimeError(
            f"Unexpected attribution output shape: "
            f"index={len(index_df)} profiles={profiles.shape}"
        )

    # ------------------------------------------------------------------
    # Frozen clean predicted-class centroids.
    # ------------------------------------------------------------------
    centroids = {}

    for model in ["MLP", "GCN_GRU"]:
        clean_idx = index_df[
            (index_df["model"] == model)
            & (index_df["family"] == "CLEAN")
        ]

        centroids[model] = {}

        for pred_class in [0, 1]:
            rows = clean_idx[
                clean_idx["prediction"] == pred_class
            ]
            if len(rows) == 0:
                raise RuntimeError(
                    f"{model}: no clean calibration attribution windows "
                    f"predicted class {pred_class}"
                )

            P = profiles[rows["profile_row"].to_numpy(dtype=int)]
            c = P.mean(axis=0).astype(np.float64)
            c = np.clip(c, PROFILE_EPS, None)
            c = c / c.sum()

            centroids[model][str(pred_class)] = {
                "n": int(len(rows)),
                "profile": [float(x) for x in c],
            }

    centroid_path = out / "calibration_attribution_centroids.json"
    centroid_path.write_text(
        json.dumps(centroids, indent=2),
        encoding="utf-8",
    )

    # JSD for every attribution row using CURRENT predicted class.
    jsd = np.zeros(len(index_df), dtype=np.float64)

    for model in ["MLP", "GCN_GRU"]:
        for pred_class in [0, 1]:
            mask = (
                (index_df["model"] == model)
                & (index_df["prediction"] == pred_class)
            ).to_numpy()

            if not mask.any():
                continue

            c = np.asarray(
                centroids[model][str(pred_class)]["profile"],
                dtype=np.float64,
            )
            P = profiles[
                index_df.loc[mask, "profile_row"].to_numpy(dtype=int)
            ]
            jsd[mask] = js_divergence(P, c[None, :])

    index_df["attribution_profile_divergence"] = jsd

    # Detection comparison on identical attribution subset.
    detection_rows = []

    for model in ["MLP", "GCN_GRU"]:
        clean = index_df[
            (index_df["model"] == model)
            & (index_df["family"] == "CLEAN")
        ]

        for family in FAMILIES:
            if model == "MLP" and family == "TOPOLOGY_MISMATCH":
                continue

            for severity in SEVERITIES:
                deg = index_df[
                    (index_df["model"] == model)
                    & (index_df["family"] == family)
                    & (index_df["severity"] == severity)
                ]

                if len(deg) == 0:
                    continue

                y_det = np.r_[
                    np.zeros(len(clean), dtype=int),
                    np.ones(len(deg), dtype=int),
                ]

                score_attr = np.r_[
                    clean["attribution_profile_divergence"].to_numpy(),
                    deg["attribution_profile_divergence"].to_numpy(),
                ]
                score_ent = np.r_[
                    clean["predictive_entropy"].to_numpy(),
                    deg["predictive_entropy"].to_numpy(),
                ]

                detection_rows.append({
                    "model": model,
                    "family": family,
                    "severity": severity,
                    "n_clean": int(len(clean)),
                    "n_degraded": int(len(deg)),
                    "attribution_divergence_auroc": float(
                        roc_auc_score(y_det, score_attr)
                    ),
                    "attribution_divergence_auprc": float(
                        average_precision_score(y_det, score_attr)
                    ),
                    "entropy_auroc_same_subset": float(
                        roc_auc_score(y_det, score_ent)
                    ),
                    "entropy_auprc_same_subset": float(
                        average_precision_score(y_det, score_ent)
                    ),
                })

    detection = pd.DataFrame(detection_rows)
    detection["auroc_attr_minus_entropy"] = (
        detection["attribution_divergence_auroc"]
        - detection["entropy_auroc_same_subset"]
    )
    detection["auprc_attr_minus_entropy"] = (
        detection["attribution_divergence_auprc"]
        - detection["entropy_auprc_same_subset"]
    )

    # Save profile tensor and audit index.
    profile_path = out / "calibration_attribution_profiles.npz"
    np.savez_compressed(
        profile_path,
        profiles=profiles.astype(np.float32),
        feature_order=np.asarray(
            layout["sensor_feature_order"],
            dtype=str,
        ),
    )

    index_path = out / "calibration_attribution_index.csv"
    index_df.to_csv(index_path, index=False)

    detection_path = out / "calibration_attribution_detection.csv"
    detection.to_csv(detection_path, index=False)

    execution = {
        "status": "CALIBRATION_ATTRIBUTION_COMPLETE",
        "attempt_history": [
            {
                "attempt": "v0.10",
                "result": "FAILED_BEFORE_OUTPUT_FREEZE",
                "reason": "Per-context GCN attribution batch was incorrectly compared to full 600-row frozen-condition lookup.",
                "degraded_test_loaded": False
            },
            {
                "attempt": "v0.10b",
                "result": "PASS",
                "scientific_method_changed": False
            }
        ],
        "method": "Input x Gradient",
        "target": "leak logit",
        "profile": "absolute Input x Gradient summed over 24 time steps and normalized across 33 sensors",
        "reference": "predicted-class clean calibration centroid",
        "divergence": "Jensen-Shannon divergence with natural logarithms",
        "attribution_position_indices": ATTR_POSITION_INDICES,
        "attribution_windows_per_context_variant": 10,
        "attribution_rows_total": int(len(index_df)),
        "profile_shape": list(profiles.shape),
        "maximum_frozen_probability_reproduction_error": float(
            max_probability_reproduction_error
        ),
        "probability_match_tolerance": PROB_MATCH_TOL,
        "split_loaded": "calibration only",
        "degraded_test_loaded": False,
        "models_retrained": False,
        "thresholds_changed": False,
    }

    execution_path = out / "calibration_attribution_execution.json"
    execution_path.write_text(
        json.dumps(execution, indent=2),
        encoding="utf-8",
    )

    artifacts = [
        profile_path,
        index_path,
        detection_path,
        centroid_path,
        execution_path,
    ]
    hashes = {p.name: sha256(p) for p in artifacts}

    freeze = {
        "status": "FROZEN_CALIBRATION_ATTRIBUTION_OUTPUTS",
        "artifact_sha256": hashes,
        "degraded_test_loaded": False,
        "models_retrained": False,
        "thresholds_changed": False,
    }
    (out / "FROZEN_CALIBRATION_ATTRIBUTION_RECORD.json").write_text(
        json.dumps(freeze, indent=2),
        encoding="utf-8",
    )

    attr_aurocs = detection["attribution_divergence_auroc"]
    ent_aurocs = detection["entropy_auroc_same_subset"]
    n_attr_better = int(
        (detection["auroc_attr_minus_entropy"] > 0).sum()
    )

    lines = [
        "AQUA-VERA CALIBRATION ATTRIBUTION v0.10b",
        "=" * 78,
        "STATUS: PASS / CALIBRATION ATTRIBUTION FROZEN",
        "",
        "BOUNDARY",
        "  split loaded: CALIBRATION ONLY",
        "  degraded TEST loaded: NO",
        "  model retraining: NO",
        "  threshold changes: NO",
        "",
        "METHOD",
        "  Input x Gradient target: leak logit",
        "  attribution profile: |input x gradient| summed over time",
        "  profile features: 33 frozen sensor streams",
        "  reference: clean calibration centroid by CURRENT predicted class",
        "  divergence: Jensen-Shannon",
        "  attribution windows/context/variant: 10",
        f"  total attribution rows: {len(index_df):,}",
        "",
        "AUDIT",
        "  every regenerated attribution input was checked against its",
        "  already-frozen v0.9 probability",
        f"  max probability reproduction error: "
        f"{max_probability_reproduction_error:.10g}",
        f"  tolerance: {PROB_MATCH_TOL:.10g}",
        "",
        "EVIDENCE-FAILURE DETECTION ON IDENTICAL ATTRIBUTION SUBSET",
        f"  attribution-divergence AUROC min/median/max: "
        f"{attr_aurocs.min():.6f} / {attr_aurocs.median():.6f} / "
        f"{attr_aurocs.max():.6f}",
        f"  predictive-entropy AUROC min/median/max: "
        f"{ent_aurocs.min():.6f} / {ent_aurocs.median():.6f} / "
        f"{ent_aurocs.max():.6f}",
        f"  conditions where attribution AUROC > entropy AUROC: "
        f"{n_attr_better}/{len(detection)}",
        "",
        "ARTIFACT HASHES",
    ]

    for n, h in hashes.items():
        lines.append(f"  {n}: {h}")

    lines += [
        "",
        "NEXT",
        "  Retain calibration_attribution_summary.txt with the project record.",
        "  Do not generate degraded test yet.",
        "  Next decision: freeze reconstruction and reliability-controller rules",
        "  using calibration evidence only.",
    ]

    summary = out / "calibration_attribution_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
