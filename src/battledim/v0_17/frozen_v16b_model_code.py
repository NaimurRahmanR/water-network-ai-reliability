
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
    r2_score,
    mean_squared_error,
)

# ----------------------------------------------------------------------
# Frozen inputs from v0.15
# ----------------------------------------------------------------------

H18_SCADA = "00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e"
H_EVENTS = "75e10bef517df5d94d0f47fd9d9d35f1a747c0fab939608a5de903b8998939b0"
H_WINDOWS = "cae5f9f968c0b94038a4822db093e51e44e11a0f08061cc332c68a558fe18608"
H_PROTOCOL_RECORD = "cb9ec4d5616da7c2481c815f7b97c60c1f4193cbc9ebf90b750723c1462cda54"
H_PROTOCOL = "058bc380aac09ca5b6874fbd5d58155dfbdebb4e499dd608209358d0ebeb525d"

MLP_EPOCHS = 25
GCN_EPOCHS = 28
MLP_LR = 1e-3
GCN_LR = 8e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
GCN_MICROBATCH = 16
MLP_SEED = 314159
GCN_SEED = 2718281
PRED_THRESHOLD = 0.5
RIDGE_ALPHA = 1.0
ATTR_Q = 0.95
ATTR_EPS = 1e-12

MLP_H1 = 256
MLP_H2 = 64
GCN_H = 32
GRU_H = 64
HEAD_H = 32
NODE_FEATURES = 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected, label):
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"{label} HASH MISMATCH\nexpected={expected}\ngot={got}\npath={path}"
        )
    print(f"[OK] {label}")


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


def parse_inp(path):
    sections = {}
    current = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.upper()
            sections.setdefault(current, [])
            continue
        if current:
            content = line.split(";", 1)[0].strip()
            if content:
                sections[current].append(content)

    def ids(section):
        out = []
        for row in sections.get(section, []):
            parts = row.split()
            if parts:
                out.append(parts[0])
        return out

    junctions = ids("[JUNCTIONS]")
    reservoirs = ids("[RESERVOIRS]")
    tanks = ids("[TANKS]")

    links = []
    for section, typ in [
        ("[PIPES]", "pipe"),
        ("[PUMPS]", "pump"),
        ("[VALVES]", "valve"),
    ]:
        for row in sections.get(section, []):
            parts = row.split()
            if len(parts) >= 3:
                links.append({
                    "link_id": parts[0],
                    "start": parts[1],
                    "end": parts[2],
                    "type": typ,
                })

    return {
        "junctions": junctions,
        "reservoirs": reservoirs,
        "tanks": tanks,
        "nodes": junctions + reservoirs + tanks,
        "links": links,
    }


def build_graph_metadata(inp_path, feature_cols):
    graph = parse_inp(inp_path)
    node_ids = graph["nodes"]
    ix = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    if n != 785 or len(graph["links"]) != 909:
        raise SystemExit(
            f"Unexpected L-Town size nodes={n}, edges={len(graph['links'])}"
        )

    A = np.zeros((n, n), dtype=np.uint8)
    endpoints = {}
    for r in graph["links"]:
        u, v = ix[r["start"]], ix[r["end"]]
        A[u, v] = 1
        A[v, u] = 1
        endpoints[r["link_id"]] = (u, v)

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

    pressure = []
    flow = []
    tank = []

    for fi, name in enumerate(feature_cols):
        if name.startswith("pressure__"):
            nid = name.split("__", 1)[1]
            if nid not in ix:
                raise SystemExit(f"Pressure sensor {nid} not in graph nodes")
            pressure.append((fi, nid, ix[nid]))
        elif name.startswith("flow__"):
            lid = name.split("__", 1)[1]
            if lid not in endpoints:
                raise SystemExit(f"Flow sensor {lid} not in graph links")
            flow.append((fi, lid, *endpoints[lid]))
        elif name.startswith("level__"):
            nid = name.split("__", 1)[1]
            if nid not in ix:
                raise SystemExit(f"Level sensor {nid} not in graph nodes")
            tank.append((fi, nid, ix[nid]))

    if (len(pressure), len(flow), len(tank)) != (33, 3, 1):
        raise SystemExit(
            f"Unexpected sensor map p/f/t="
            f"{len(pressure)}/{len(flow)}/{len(tank)}"
        )

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


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, MLP_H1),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(MLP_H1, MLP_H2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(MLP_H2, 1),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Runtime cache only; not part of the scientific state_dict.
        self._cached_sparse_A = None
        self._cached_sparse_A_device = None

    def forward(self, x, target, source, weight):
        # x: [B,T,N,F]
        #
        # Mathematically identical to:
        #   agg[i] = sum_j A_norm[i,j] * x[j]
        #
        # v0.16 used gather + index_add over every B*T slice.  On the
        # 785-node L-Town graph that is correct but unnecessarily slow on CPU.
        # v0.16b forms the same frozen sparse normalized adjacency once and
        # performs torch.sparse.mm over a flattened B*T*F axis.
        B, T, N, F = x.shape

        if (
            self._cached_sparse_A is None
            or self._cached_sparse_A_device != x.device
        ):
            indices = torch.stack([target, source], dim=0)
            self._cached_sparse_A = torch.sparse_coo_tensor(
                indices,
                weight,
                size=(N, N),
                dtype=x.dtype,
                device=x.device,
            ).coalesce()
            self._cached_sparse_A_device = x.device

        flat = (
            x.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(N, B * T * F)
        )
        agg_flat = torch.sparse.mm(self._cached_sparse_A, flat)
        agg = (
            agg_flat.reshape(N, B, T, F)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        return self.linear(agg)


class SparseGraphTemporalModel(nn.Module):
    def __init__(self, edge_target, edge_source, edge_weight):
        super().__init__()
        self.register_buffer(
            "edge_target", torch.tensor(edge_target, dtype=torch.long)
        )
        self.register_buffer(
            "edge_source", torch.tensor(edge_source, dtype=torch.long)
        )
        self.register_buffer(
            "edge_weight", torch.tensor(edge_weight, dtype=torch.float32)
        )
        self.gcn1 = SparseGCNLayer(NODE_FEATURES, GCN_H)
        self.gcn2 = SparseGCNLayer(GCN_H, GCN_H)
        self.dropout = nn.Dropout(0.20)
        self.gru = nn.GRU(GCN_H * 2, GRU_H, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(GRU_H, HEAD_H),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(HEAD_H, 1),
        )

    def forward(self, x):
        h = self.dropout(torch.relu(
            self.gcn1(
                x, self.edge_target, self.edge_source, self.edge_weight
            )
        ))
        h = self.dropout(torch.relu(
            self.gcn2(
                h, self.edge_target, self.edge_source, self.edge_weight
            )
        ))
        pooled = torch.cat(
            [h.mean(dim=2), h.max(dim=2).values],
            dim=-1,
        )
        out, _ = self.gru(pooled)
        return self.head(out[:, -1]).squeeze(-1)


class RawToGraphWrapper(nn.Module):
    def __init__(self, graph_model, graph_meta):
        super().__init__()
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
            pmask[ni] = 1.0

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
        self.register_buffer(
            "static",
            torch.tensor(graph_meta["static"], dtype=torch.float32),
        )

    def forward(self, x):
        pv = torch.einsum("btf,fn->btn", x, self.P)
        tv = torch.einsum("btf,fn->btn", x, self.T)
        fv = torch.einsum("btf,fn->btn", x, self.Flow)

        B, T, _ = pv.shape
        dyn = torch.stack(
            [
                pv,
                self.pmask[None, None, :].expand(B, T, -1),
                tv,
                self.tmask[None, None, :].expand(B, T, -1),
                fv,
                self.fcount[None, None, :].expand(B, T, -1),
            ],
            dim=-1,
        )
        static = self.static[
            None, None, :, :
        ].expand(B, T, -1, -1)
        graph_x = torch.cat([dyn, static], dim=-1)
        return self.graph_model(graph_x)


def make_windows(scada_values, manifest):
    X = np.empty((len(manifest), 24, scada_values.shape[1]), dtype=np.float32)
    for i, (_, r) in enumerate(manifest.iterrows()):
        s = int(r["start_idx"])
        f = int(r["final_idx"])
        w = scada_values[s:f + 1]
        if w.shape[0] != 24:
            raise RuntimeError(f"Window {i} length {w.shape[0]} != 24")
        X[i] = w
    return X


def metrics(y, p):
    q = (p >= PRED_THRESHOLD).astype(int)
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "predicted_positive": int(q.sum()),
        "accuracy": float((q == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, q)),
        "f1": float(f1_score(y, q, zero_division=0)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def entropy01(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)


def predict_batches(model, X, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def train_mlp(model, X, y, pos_weight, epochs):
    torch.manual_seed(MLP_SEED)
    rng = np.random.default_rng(MLP_SEED)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32),
        reduction="sum",
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(X))
        total_loss = 0.0

        for start in range(0, len(X), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(y[idx].astype(np.float32))

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_sum = loss_fn(logits, yb)
            loss = loss_sum / len(idx)
            loss.backward()
            opt.step()
            total_loss += float(loss_sum.detach())

        avg = total_loss / len(X)
        history.append({"epoch": epoch, "mean_weighted_bce": avg})
        print(f"[MLP] epoch {epoch:02d}/{epochs}: loss={avg:.6f}")

    return pd.DataFrame(history)


def train_gcn(wrapper, X, y, pos_weight, epochs, resume_path=None):
    torch.manual_seed(GCN_SEED)
    rng = np.random.default_rng(GCN_SEED)
    opt = torch.optim.AdamW(
        wrapper.parameters(),
        lr=GCN_LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32),
        reduction="sum",
    )

    history = []
    start_epoch = 1

    # Epoch-boundary recovery only. If Windows/Python is interrupted, rerunning
    # v0.16b resumes from the last COMPLETED GCN epoch with optimizer and RNG
    # states restored. This changes no training rule.
    if resume_path is not None and Path(resume_path).exists():
        ck = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        wrapper.load_state_dict(ck["wrapper_state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer_state_dict"])
        rng.bit_generator.state = ck["numpy_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        history = list(ck["history"])
        start_epoch = int(ck["completed_epoch"]) + 1
        print(
            f"[GCN] resuming after completed epoch "
            f"{int(ck['completed_epoch'])}/{epochs}",
            flush=True,
        )

    total_optimizer_batches = math.ceil(len(X) / BATCH_SIZE)

    for epoch in range(start_epoch, epochs + 1):
        wrapper.train()
        perm = rng.permutation(len(X))
        total_loss = 0.0
        print(
            f"[GCN] starting epoch {epoch:02d}/{epochs} "
            f"({total_optimizer_batches} optimizer batches; "
            f"microbatch={GCN_MICROBATCH})",
            flush=True,
        )

        for batch_no, start in enumerate(
            range(0, len(X), BATCH_SIZE),
            start=1,
        ):
            idx = perm[start:start + BATCH_SIZE]
            yb_full = y[idx].astype(np.float32)
            full_n = len(idx)

            opt.zero_grad(set_to_none=True)
            batch_loss_sum = 0.0

            # Effective optimizer batch is 128. Microbatches are only a
            # memory-management implementation detail.
            for ms in range(0, full_n, GCN_MICROBATCH):
                midx = idx[ms:ms + GCN_MICROBATCH]
                xb = torch.from_numpy(X[midx])
                yb = torch.from_numpy(y[midx].astype(np.float32))

                logits = wrapper(xb)
                loss_sum = loss_fn(logits, yb)
                (loss_sum / full_n).backward()
                batch_loss_sum += float(loss_sum.detach())

            opt.step()
            total_loss += batch_loss_sum

            if (
                batch_no == 1
                or batch_no % 10 == 0
                or batch_no == total_optimizer_batches
            ):
                print(
                    f"[GCN] epoch {epoch:02d}/{epochs} "
                    f"batch {batch_no:03d}/{total_optimizer_batches:03d}",
                    flush=True,
                )

        avg = total_loss / len(X)
        history.append({"epoch": epoch, "mean_weighted_bce": avg})
        print(
            f"[GCN] epoch {epoch:02d}/{epochs}: loss={avg:.6f}",
            flush=True,
        )

        if resume_path is not None:
            torch.save(
                {
                    "completed_epoch": epoch,
                    "wrapper_state_dict": wrapper.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "history": history,
                    "scientific_note": (
                        "Epoch-boundary resume checkpoint only; "
                        "training recipe unchanged."
                    ),
                },
                resume_path,
            )

    return pd.DataFrame(history)


def input_x_gradient(model, X, batch):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    profiles = []
    logits = []

    for i in range(0, len(X), batch):
        xb = torch.tensor(
            X[i:i + batch],
            dtype=torch.float32,
            requires_grad=True,
        )
        z = model(xb)
        g = torch.autograd.grad(z.sum(), xb)[0]
        a = torch.abs(xb * g).sum(dim=1) + ATTR_EPS
        a = a / a.sum(dim=1, keepdim=True)
        profiles.append(a.detach().cpu().numpy().astype(np.float32))
        logits.append(z.detach().cpu().numpy().astype(np.float64))

    return np.concatenate(profiles), np.concatenate(logits)


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, ATTR_EPS, None)
    q = np.clip(q, ATTR_EPS, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    return (
        0.5 * np.sum(p * np.log(p / m), axis=-1)
        + 0.5 * np.sum(q * np.log(q / m), axis=-1)
    )


def fit_reconstructors(X_rows, pressure_indices):
    all_idx = np.arange(X_rows.shape[1])
    models = {}
    quality = []

    for fi in pressure_indices:
        keep = all_idx[all_idx != fi]
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_rows[:, keep], X_rows[:, fi])
        pred = model.predict(X_rows[:, keep])

        models[str(int(fi))] = {
            "keep_indices": [int(x) for x in keep],
            "coef": [float(x) for x in model.coef_],
            "intercept": float(model.intercept_),
        }
        quality.append({
            "feature_index": int(fi),
            "train_rmse_normalized": float(
                mean_squared_error(X_rows[:, fi], pred) ** 0.5
            ),
            "train_r2": float(r2_score(X_rows[:, fi], pred)),
        })

    return models, pd.DataFrame(quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ready", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--protocol-freeze", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mr = Path(args.model_ready)
    data = Path(args.data)
    pf = Path(args.protocol_freeze)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA BATTLEDIM/L-TOWN TRAIN + RELIABILITY FREEZE v0.16b")
    print("=" * 78)
    print("2018 ONLY. 2019 is not opened, checked, parsed, or hashed.")
    print()

    scada_path = find_unique(mr, "2018_scada_model_ready.csv")
    inp_path = find_unique(data, "L-TOWN.inp")

    verify(scada_path, H18_SCADA, "2018 model-ready SCADA")
    verify(
        pf / "2018_frozen_onset_events.csv",
        H_EVENTS,
        "frozen 2018 onset events",
    )
    verify(
        pf / "2018_frozen_window_manifest_v0_15.csv",
        H_WINDOWS,
        "frozen 2018 window manifest",
    )
    verify(
        pf / "FROZEN_BATTLEDIM_PROTOCOL_RECORD_v0_15.json",
        H_PROTOCOL_RECORD,
        "v0.15 protocol record",
    )
    verify(
        pf / "BattLeDIM_external_protocol_v0_15.md",
        H_PROTOCOL,
        "v0.15 protocol",
    )

    print(f"[FREEZE] L-TOWN.inp SHA-256: {sha256(inp_path)}")

    scada = pd.read_csv(scada_path)
    scada["Timestamp"] = pd.to_datetime(scada["Timestamp"])
    feature_cols = [c for c in scada.columns if c != "Timestamp"]

    if len(feature_cols) != 37:
        raise SystemExit(f"Expected 37 SCADA features, got {len(feature_cols)}")

    manifest = pd.read_csv(pf / "2018_frozen_window_manifest_v0_15.csv")
    eligible = manifest[
        manifest["eligible_2018_development"] == 1
    ].copy().reset_index(drop=True)

    if len(eligible) != 17036:
        raise SystemExit(f"Expected 17,036 eligible windows, got {len(eligible)}")

    y = eligible["target_recent_leak_onset_24h"].to_numpy(dtype=np.int64)
    if (int(y.sum()), int((1-y).sum())) != (666, 16370):
        raise SystemExit("Frozen target class counts do not match v0.15")

    # 2018-only normalization.
    raw = scada[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise SystemExit("Non-finite 2018 SCADA values")

    mean = raw.mean(axis=0).astype(np.float64)
    std = raw.std(axis=0, ddof=0).astype(np.float64)
    near_constant = std < 1e-12
    scale = std.copy()
    scale[near_constant] = 1.0

    normalized = (
        (raw.astype(np.float64) - mean[None, :])
        / scale[None, :]
    ).astype(np.float32)

    norm_record = {
        "source": "all 105,120 unique 2018 SCADA rows only",
        "feature_order": feature_cols,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "scale_used": [float(x) for x in scale],
        "near_constant_feature_indices": [
            int(x) for x in np.flatnonzero(near_constant)
        ],
        "2019_used": False,
    }
    norm_path = out / "FROZEN_2018_NORMALIZATION.json"
    norm_path.write_text(json.dumps(norm_record, indent=2), encoding="utf-8")

    print("[WINDOWS] materializing 17,036 x 24 x 37 normalized windows")
    X = make_windows(normalized, eligible)

    pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"[TRAIN] pos_weight={pos_weight:.9f}")

    graph_meta = build_graph_metadata(inp_path, feature_cols)

    graph_meta_record = {
        "ltown_inp_sha256": sha256(inp_path),
        "nodes": len(graph_meta["node_ids"]),
        "links": len(graph_meta["graph"]["links"]),
        "pressure_sensors": len(graph_meta["pressure"]),
        "flow_sensors": len(graph_meta["flow"]),
        "tank_level_sensors": len(graph_meta["tank"]),
        "sparse_normalized_adjacency_entries": len(graph_meta["edge_weight"]),
        "feature_order": feature_cols,
        "node_order": graph_meta["node_ids"],
        "pressure_mapping": [
            {"feature_index": fi, "sensor_id": sid, "node_index": ni}
            for fi, sid, ni in graph_meta["pressure"]
        ],
        "flow_mapping": [
            {
                "feature_index": fi,
                "sensor_id": sid,
                "start_node_index": u,
                "end_node_index": v,
            }
            for fi, sid, u, v in graph_meta["flow"]
        ],
        "tank_mapping": [
            {"feature_index": fi, "sensor_id": sid, "node_index": ni}
            for fi, sid, ni in graph_meta["tank"]
        ],
    }
    graph_meta_path = out / "FROZEN_LTown_GRAPH_INPUT.json"
    graph_meta_path.write_text(
        json.dumps(graph_meta_record, indent=2),
        encoding="utf-8",
    )

    # --------------------------- MLP ----------------------------------
    torch.manual_seed(MLP_SEED)
    mlp = MLP(input_dim=24 * 37)
    mlp_history = train_mlp(mlp, X, y, pos_weight, MLP_EPOCHS)
    mlp_hist_path = out / "2018_mlp_training_history.csv"
    mlp_history.to_csv(mlp_hist_path, index=False)

    mlp_logits = predict_batches(mlp, X, batch=1024)
    mlp_prob = sigmoid(mlp_logits)
    mlp_metrics = metrics(y, mlp_prob)

    mlp_ckpt = out / "BattLeDIM_MLP_v0_16b.pt"
    torch.save(
        {
            "model_state_dict": mlp.state_dict(),
            "architecture": "888 -> 256 -> 64 -> 1",
            "epochs": MLP_EPOCHS,
            "threshold": PRED_THRESHOLD,
            "pos_weight": pos_weight,
            "seed": MLP_SEED,
            "feature_order": feature_cols,
            "normalization_sha256": sha256(norm_path),
        },
        mlp_ckpt,
    )

    # --------------------------- GCN ----------------------------------
    torch.manual_seed(GCN_SEED)
    gcn_core = SparseGraphTemporalModel(
        graph_meta["edge_target"],
        graph_meta["edge_source"],
        graph_meta["edge_weight"],
    )
    gcn = RawToGraphWrapper(gcn_core, graph_meta)

    print(
        "[GCN] entering fixed 28-epoch L-Town training stage. "
        "Progress will be printed within each epoch.",
        flush=True,
    )
    gcn_resume_path = out / "GCN_TRAINING_RESUME_v0_16b.pt"
    gcn_history = train_gcn(
        gcn,
        X,
        y,
        pos_weight,
        GCN_EPOCHS,
        resume_path=gcn_resume_path,
    )
    gcn_hist_path = out / "2018_gcn_training_history.csv"
    gcn_history.to_csv(gcn_hist_path, index=False)

    gcn_logits = predict_batches(gcn, X, batch=GCN_MICROBATCH)
    gcn_prob = sigmoid(gcn_logits)
    gcn_metrics = metrics(y, gcn_prob)

    gcn_ckpt = out / "BattLeDIM_GCN_GRU_v0_16b.pt"
    torch.save(
        {
            "model_state_dict": gcn_core.state_dict(),
            "wrapper_architecture": "37 raw streams -> 785-node sparse graph -> GCN32 -> GCN32 -> mean+max -> GRU64 -> 32 -> 1",
            "epochs": GCN_EPOCHS,
            "threshold": PRED_THRESHOLD,
            "pos_weight": pos_weight,
            "seed": GCN_SEED,
            "feature_order": feature_cols,
            "normalization_sha256": sha256(norm_path),
            "graph_input_sha256": sha256(graph_meta_path),
            "implementation_note": (
                "v0.16b uses torch.sparse.mm for the same symmetric "
                "normalized GCN aggregation frozen in v0.15."
            ),
        },
        gcn_ckpt,
    )
    if gcn_resume_path.exists():
        gcn_resume_path.unlink()

    # Clean 2018 predictions for traceability.
    pred_df = eligible[
        [
            "window_index",
            "start_idx",
            "final_idx",
            "start_timestamp",
            "final_timestamp",
            "target_recent_leak_onset_24h",
            "attribution_subset",
        ]
    ].copy()
    pred_df["mlp_probability"] = mlp_prob
    pred_df["mlp_prediction"] = (mlp_prob >= PRED_THRESHOLD).astype(int)
    pred_df["gcn_probability"] = gcn_prob
    pred_df["gcn_prediction"] = (gcn_prob >= PRED_THRESHOLD).astype(int)
    pred_path = out / "2018_clean_predictions_v0_16b.csv"
    pred_df.to_csv(pred_path, index=False)

    # ---------------- Attribution references / thresholds -------------
    attr_mask = eligible["attribution_subset"].to_numpy(dtype=int) == 1
    X_attr = X[attr_mask]
    attr_meta = eligible[attr_mask].copy().reset_index(drop=True)

    if len(X_attr) != 1702:
        raise SystemExit(f"Expected 1,702 eligible attribution windows, got {len(X_attr)}")

    print("[ATTR] MLP Input x Gradient on 2018 clean subset")
    mlp_profiles, mlp_attr_logits = input_x_gradient(mlp, X_attr, batch=128)
    mlp_attr_prob = sigmoid(mlp_attr_logits)

    print("[ATTR] GCN-GRU Input x Gradient on 2018 clean subset")
    gcn_profiles, gcn_attr_logits = input_x_gradient(gcn, X_attr, batch=8)
    gcn_attr_prob = sigmoid(gcn_attr_logits)

    # Ensure clean forward probabilities reproduce the already-frozen post-training predictions.
    eligible_indices = np.flatnonzero(attr_mask)
    mlp_max_err = float(np.max(np.abs(mlp_attr_prob - mlp_prob[eligible_indices])))
    gcn_max_err = float(np.max(np.abs(gcn_attr_prob - gcn_prob[eligible_indices])))

    if mlp_max_err > 2e-6 or gcn_max_err > 2e-6:
        raise SystemExit(
            f"Attribution forward reproduction failed: "
            f"MLP={mlp_max_err}, GCN={gcn_max_err}"
        )

    centroid_record = {}
    threshold_record = {}
    attr_rows = []
    profile_blocks = []
    offset = 0

    for model_name, profiles, probs in [
        ("MLP", mlp_profiles, mlp_attr_prob),
        ("GCN_GRU", gcn_profiles, gcn_attr_prob),
    ]:
        preds = (probs >= PRED_THRESHOLD).astype(int)

        centroid_record[model_name] = {}
        divergences = np.zeros(len(profiles), dtype=np.float64)

        for cls in [0, 1]:
            mask = preds == cls
            if mask.sum() == 0:
                raise SystemExit(
                    f"{model_name} predicts zero attribution-subset windows as class {cls}; "
                    "predicted-class centroid cannot be frozen without changing protocol."
                )
            c = profiles[mask].mean(axis=0).astype(np.float64)
            c = np.clip(c, ATTR_EPS, None)
            c = c / c.sum()
            centroid_record[model_name][str(cls)] = {
                "n": int(mask.sum()),
                "profile": [float(x) for x in c],
            }
            divergences[mask] = js_divergence(profiles[mask], c[None, :])

        ent = entropy01(probs)
        ent_q = float(np.quantile(ent, ATTR_Q))
        div_q = float(np.quantile(divergences, ATTR_Q))

        threshold_record[model_name] = {
            "entropy_q95": ent_q,
            "attribution_divergence_q95": div_q,
            "source": "clean 2018 attribution subset only",
            "quantile": ATTR_Q,
            "predictive_threshold": PRED_THRESHOLD,
        }

        for i, (_, r) in enumerate(attr_meta.iterrows()):
            attr_rows.append({
                "profile_row": offset + i,
                "model": model_name,
                "window_index": int(r["window_index"]),
                "final_timestamp": str(r["final_timestamp"]),
                "target_recent_leak_onset_24h": int(r["target_recent_leak_onset_24h"]),
                "probability": float(probs[i]),
                "prediction": int(preds[i]),
                "predictive_entropy": float(ent[i]),
                "attribution_profile_divergence": float(divergences[i]),
            })

        profile_blocks.append(profiles.astype(np.float32))
        offset += len(profiles)

    profiles_all = np.concatenate(profile_blocks, axis=0)
    attr_index = pd.DataFrame(attr_rows)

    profiles_path = out / "2018_attribution_profiles_v0_16b.npz"
    np.savez_compressed(
        profiles_path,
        profiles=profiles_all,
        feature_order=np.asarray(feature_cols, dtype=str),
    )

    attr_index_path = out / "2018_attribution_index_v0_16b.csv"
    attr_index.to_csv(attr_index_path, index=False)

    centroid_path = out / "FROZEN_2018_ATTRIBUTION_CENTROIDS.json"
    centroid_path.write_text(
        json.dumps(centroid_record, indent=2),
        encoding="utf-8",
    )

    threshold_path = out / "FROZEN_2018_RELIABILITY_THRESHOLDS.json"
    threshold_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_PRE_2019",
                "selection_rule": "95th percentile of clean 2018 attribution-subset signal",
                "thresholds": threshold_record,
                "degraded_2018_used": False,
                "2019_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------- Sensor reconstruction ----------------------------
    pressure_indices = [fi for fi, _, _ in graph_meta["pressure"]]
    recon_models, recon_quality = fit_reconstructors(
        normalized,
        pressure_indices,
    )

    recon_path = out / "FROZEN_2018_PRESSURE_RECONSTRUCTION.json"
    recon_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_PRE_2019",
                "alpha": RIDGE_ALPHA,
                "source_rows": 105120,
                "source_year": 2018,
                "feature_order": feature_cols,
                "multi_missing_rule": "mean-impute selected pressure streams then simultaneous one-pass reconstruction from that same vector",
                "gcn_reconstructed_pressure_mask": 0,
                "models": recon_models,
                "2019_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    recon_quality_path = out / "2018_pressure_reconstruction_quality.csv"
    recon_quality.to_csv(recon_quality_path, index=False)

    # Development metrics: explicitly NOT held-out.
    metrics_path = out / "2018_development_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "warning": "2018 metrics are in-sample development-year descriptive metrics, not held-out external results",
                "MLP": mlp_metrics,
                "GCN_GRU": gcn_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifacts = [
        norm_path,
        graph_meta_path,
        mlp_ckpt,
        gcn_ckpt,
        mlp_hist_path,
        gcn_hist_path,
        pred_path,
        profiles_path,
        attr_index_path,
        centroid_path,
        threshold_path,
        recon_path,
        recon_quality_path,
        metrics_path,
    ]
    hashes = {p.name: sha256(p) for p in artifacts}

    freeze_record = {
        "status": "PASS_MODELS_AND_RELIABILITY_FROZEN_PRE_2019",
        "version": "0.16",
        "protocol_sha256": H_PROTOCOL,
        "2018_only": True,
        "2019_checked": False,
        "2019_opened": False,
        "2019_hashed": False,
        "eligible_windows": len(eligible),
        "positive_windows": int(y.sum()),
        "negative_windows": int((1-y).sum()),
        "pos_weight": pos_weight,
        "MLP_epochs": MLP_EPOCHS,
        "GCN_GRU_epochs": GCN_EPOCHS,
        "predictive_threshold": PRED_THRESHOLD,
        "attribution_subset_windows": len(X_attr),
        "attribution_probability_reproduction_max_error": {
            "MLP": mlp_max_err,
            "GCN_GRU": gcn_max_err,
        },
        "artifact_sha256": hashes,
    }
    freeze_record_path = out / "FROZEN_BATTLEDIM_MODELS_RECORD_v0_16b.json"
    freeze_record_path.write_text(
        json.dumps(freeze_record, indent=2),
        encoding="utf-8",
    )

    lines = [
        "AQUA-VERA BATTLEDIM/L-TOWN TRAIN + RELIABILITY FREEZE v0.16b",
        "=" * 78,
        "STATUS: PASS / MODELS + RELIABILITY REFERENCES FROZEN PRE-2019",
        "",
        "BOUNDARY",
        "  2018 used: YES",
        "  2019 checked/opened/hashed: NO",
        "  architecture tuning on BattLeDIM: NO",
        "  epoch tuning on BattLeDIM: NO",
        "  predictive threshold tuning: NO",
        "",
        "2018 TRAINING SET",
        f"  eligible windows: {len(eligible):,}",
        f"  positives: {int(y.sum()):,}",
        f"  negatives: {int((1-y).sum()):,}",
        f"  weighted BCE pos_weight: {pos_weight:.9f}",
        "",
        "FROZEN MODELS",
        f"  MLP epochs: {MLP_EPOCHS}",
        f"  GCN-GRU epochs: {GCN_EPOCHS}",
        f"  predictive threshold: {PRED_THRESHOLD:.1f}",
        "",
        "2018 DEVELOPMENT-YEAR METRICS — NOT HELD-OUT",
        f"  MLP balanced_accuracy={mlp_metrics['balanced_accuracy']:.6f}, "
        f"F1={mlp_metrics['f1']:.6f}, "
        f"AUPRC={mlp_metrics['auprc']:.6f}, "
        f"AUROC={mlp_metrics['auroc']:.6f}",
        f"  GCN-GRU balanced_accuracy={gcn_metrics['balanced_accuracy']:.6f}, "
        f"F1={gcn_metrics['f1']:.6f}, "
        f"AUPRC={gcn_metrics['auprc']:.6f}, "
        f"AUROC={gcn_metrics['auroc']:.6f}",
        "",
        "ATTRIBUTION / RELIABILITY REFERENCES",
        f"  clean attribution-subset windows: {len(X_attr):,}",
        f"  MLP entropy q95: "
        f"{threshold_record['MLP']['entropy_q95']:.9f}",
        f"  MLP attribution-divergence q95: "
        f"{threshold_record['MLP']['attribution_divergence_q95']:.9f}",
        f"  GCN-GRU entropy q95: "
        f"{threshold_record['GCN_GRU']['entropy_q95']:.9f}",
        f"  GCN-GRU attribution-divergence q95: "
        f"{threshold_record['GCN_GRU']['attribution_divergence_q95']:.9f}",
        f"  attribution forward max error MLP/GCN: "
        f"{mlp_max_err:.3g} / {gcn_max_err:.3g}",
        "",
        "PRESSURE RECONSTRUCTION",
        "  33 Ridge models",
        "  source: all 105,120 unique normalized 2018 SCADA rows",
        "  alpha: 1.0",
        f"  train R2 min/median/max: "
        f"{recon_quality.train_r2.min():.6f} / "
        f"{recon_quality.train_r2.median():.6f} / "
        f"{recon_quality.train_r2.max():.6f}",
        "",
        "ARTIFACT HASHES",
    ]

    for name, h in hashes.items():
        lines.append(f"  {name}: {h}")

    lines += [
        "",
        "NEXT",
        "  Retain BattLeDIM_train_freeze_summary_v0_16b.txt with the project record.",
        "  If PASS, all model/reliability choices are frozen.",
        "  The next run may open 2019 exactly once for final external evaluation.",
    ]

    summary_path = out / "BattLeDIM_train_freeze_summary_v0_16b.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
