
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
)

# ----------------------- frozen artifact hashes -----------------------

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
DEGRADATION_PROTOCOL_SOURCE_HASH = "0203ffcb0a456cbdb09c1186a9d669f0266cc85b200e13dbfcd53a62dd746b2c"
DEGRADATION_PROTOCOL_WINDOWS_COPY_HASH = "9c378c9396c6432ff555b6274606099ff262bc18fb6ac2e206631ca4b9266815"

BASELINE_THRESHOLD = 0.464201702923
GRAPH_THRESHOLD = 0.376129878873
ROOT_SEED = 8675309

SEVERITIES = ("low", "medium", "high")
MISSING_COUNTS = {"low": 2, "medium": 6, "high": 10}
CONFLICT_COUNTS = {"low": 2, "medium": 6, "high": 10}
NOISE_SD = {"low": 0.25, "medium": 0.50, "high": 1.00}
BIAS_SD = {"low": 0.50, "medium": 1.00, "high": 2.00}
STALE_LAG = {"low": 6, "medium": 12, "high": 24}
TOPOLOGY_EDGES = {"low": 1, "medium": 3, "high": 5}

FAMILIES = (
    "MISSING_SENSOR",
    "SENSOR_NOISE",
    "SENSOR_BIAS",
    "STALE_SENSOR",
    "TOPOLOGY_MISMATCH",
    "CROSS_SOURCE_CONFLICT",
)

# model architecture
MLP_H1, MLP_H2 = 256, 64
NODE_FEATURES, GCN_HIDDEN, GRU_HIDDEN, HEAD_HIDDEN = 10, 32, 64, 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
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
    s = " | ".join(str(x) for x in parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_seed(*parts):
    # 64-bit deterministic seed from SHA-256 prefix.
    return int(stable_digest(*parts)[:16], 16) % (2**63 - 1)


def rank_items(context_id, family, items):
    return sorted(
        items,
        key=lambda item: stable_digest(ROOT_SEED, context_id, family, item)
    )


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
        return self.net(x).squeeze(-1)


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
        self.gru = nn.GRU(GCN_HIDDEN*2, GRU_HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(HEAD_HIDDEN, 1),
        )
    def forward(self, x):
        h = self.dropout(torch.relu(self.gcn1(x, self.A_norm)))
        h = self.dropout(torch.relu(self.gcn2(h, self.A_norm)))
        z = torch.cat([h.mean(2), h.max(2).values], dim=-1)
        o, _ = self.gru(z)
        return self.head(o[:, -1]).squeeze(-1)


def sigmoid(logits):
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def entropy01(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1-1e-12)
    h = -(p*np.log(p) + (1-p)*np.log(1-p))
    return h / np.log(2.0)


def predict(model, X, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i+batch])).cpu().numpy())
    return np.concatenate(out)


def build_base_topology(gen, layout):
    nodes = pd.read_csv(gen/"graph_nodes.csv", dtype={"node_id": str})
    links = pd.read_csv(
        gen/"graph_links.csv",
        dtype={"link_id": str, "start_node": str, "end_node": str}
    )
    node_ids = [str(x) for x in layout["original_node_ids"]]
    ix = {n:i for i,n in enumerate(node_ids)}
    n = len(node_ids)

    # Full link catalog for topology mismatch.
    link_rows = []
    for _, r in links.iterrows():
        lid = str(r["link_id"])
        u, v = str(r["start_node"]), str(r["end_node"])
        link_rows.append({
            "link_id": lid,
            "link_type": str(r["link_type"]),
            "u": u, "v": v,
            "ui": ix[u], "vi": ix[v],
        })

    node_type = {
        str(r["node_id"]): str(r["node_type"]).lower()
        for _, r in nodes.iterrows()
    }

    return node_ids, ix, link_rows, node_type


def adjacency_from_links(n, link_rows):
    A = np.zeros((n,n), dtype=np.float32)
    for r in link_rows:
        A[r["ui"], r["vi"]] = 1.0
        A[r["vi"], r["ui"]] = 1.0
    return A


def normalized_adjacency(A):
    Ah = A + np.eye(len(A), dtype=np.float32)
    d = Ah.sum(1)
    q = np.zeros_like(d)
    m = d > 0
    q[m] = np.power(d[m], -0.5).astype(np.float32)
    return q[:,None] * Ah * q[None,:]


def static_features(A, node_ids, node_type):
    deg = A.sum(1)
    st = np.zeros((len(node_ids), 4), dtype=np.float32)
    st[:,0] = deg / max(float(deg.max()), 1.0)
    for n,i in {n:i for i,n in enumerate(node_ids)}.items():
        t = node_type.get(n,"")
        st[i,1] = 1.0 if "junction" in t else 0.0
        st[i,2] = 1.0 if "tank" in t else 0.0
        st[i,3] = 1.0 if "reservoir" in t else 0.0
    return st


def sensor_maps(layout, ix, link_rows):
    endpoints = {r["link_id"]:(r["ui"],r["vi"]) for r in link_rows}
    pressure, flow, tank = [], [], []
    for fi, name in enumerate(layout["sensor_feature_order"]):
        if name.startswith("pressure::"):
            pressure.append((fi, name.split("::",1)[1], ix[name.split("::",1)[1]]))
        elif name.startswith("flow::"):
            lid = name.split("::",1)[1]
            flow.append((fi, lid, *endpoints[lid]))
        elif name.startswith("tank_level::"):
            nid = name.split("::",1)[1]
            tank.append((fi, nid, ix[nid]))
    if (len(pressure),len(flow),len(tank)) != (24,6,3):
        raise RuntimeError("sensor map mismatch")
    return pressure, flow, tank


def to_graph(sensor, mean, scale, n, static, pmap, fmap, tmap, missing_fi=None):
    missing_fi = set() if missing_fi is None else set(missing_fi)
    x = (sensor.astype(np.float32) - mean[None,:]) / scale[None,:]
    out = np.zeros((len(x), n, 10), dtype=np.float32)

    for fi, nid, ni in pmap:
        out[:,ni,0] = x[:,fi]
        out[:,ni,1] = 0.0 if fi in missing_fi else 1.0

    for fi, nid, ni in tmap:
        out[:,ni,2] = x[:,fi]
        out[:,ni,3] = 1.0

    for fi, lid, u, v in fmap:
        out[:,u,4] -= x[:,fi]
        out[:,v,4] += x[:,fi]
        out[:,u,5] += 1.0
        out[:,v,5] += 1.0

    out[:,:,6:10] = static[None,:,:]
    return out


def select_topology_removals(context_id, link_rows, n_nodes):
    # Candidate ordinary pipes. Rank deterministically, then greedily remove
    # only links whose deletion keeps the binary graph connected AND changes
    # the simple adjacency (i.e. no parallel surviving link with same endpoints).
    pipe_rows = [r for r in link_rows if r["link_type"].lower() == "pipe"]
    ranked_ids = rank_items(context_id, "TOPOLOGY_MISMATCH", [r["link_id"] for r in pipe_rows])
    by_id = {r["link_id"]:r for r in link_rows}

    active = list(link_rows)
    chosen = []

    for lid in ranked_ids:
        if len(chosen) >= 5:
            break
        r = by_id[lid]
        u, v = r["u"], r["v"]

        # If a different active link connects same endpoints, removing this pipe
        # would not alter our frozen simple adjacency representation.
        parallel = [
            x for x in active
            if x["link_id"] != lid
            and {x["u"],x["v"]} == {u,v}
        ]
        if parallel:
            continue

        trial = [x for x in active if x["link_id"] != lid]
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from((x["ui"],x["vi"]) for x in trial)
        if nx.is_connected(G):
            chosen.append(lid)
            active = trial

    if len(chosen) < 5:
        raise RuntimeError(
            f"{context_id}: only {len(chosen)} connectivity-preserving "
            "topology removals found"
        )
    return chosen


def corruption_selection(context_id, pressure_ids, link_rows, n_nodes):
    recs = []

    ranked = {}
    for fam in FAMILIES:
        if fam == "TOPOLOGY_MISMATCH":
            ranked[fam] = select_topology_removals(context_id, link_rows, n_nodes)
        else:
            ranked[fam] = rank_items(context_id, fam, pressure_ids)

    for sev in SEVERITIES:
        recs.append({
            "context_id": context_id,
            "family": "MISSING_SENSOR",
            "severity": sev,
            "selected_pressure_ids": ranked["MISSING_SENSOR"][:MISSING_COUNTS[sev]],
            "selected_edges": [],
            "rng_seed": None,
        })
        recs.append({
            "context_id": context_id,
            "family": "SENSOR_NOISE",
            "severity": sev,
            "selected_pressure_ids": ranked["SENSOR_NOISE"][:6],
            "selected_edges": [],
            "rng_seed": stable_seed(ROOT_SEED, context_id, "SENSOR_NOISE", sev),
        })
        recs.append({
            "context_id": context_id,
            "family": "SENSOR_BIAS",
            "severity": sev,
            "selected_pressure_ids": ranked["SENSOR_BIAS"][:6],
            "selected_edges": [],
            "rng_seed": stable_seed(ROOT_SEED, context_id, "SENSOR_BIAS", sev),
        })
        recs.append({
            "context_id": context_id,
            "family": "STALE_SENSOR",
            "severity": sev,
            "selected_pressure_ids": ranked["STALE_SENSOR"][:6],
            "selected_edges": [],
            "rng_seed": None,
        })
        recs.append({
            "context_id": context_id,
            "family": "TOPOLOGY_MISMATCH",
            "severity": sev,
            "selected_pressure_ids": [],
            "selected_edges": ranked["TOPOLOGY_MISMATCH"][:TOPOLOGY_EDGES[sev]],
            "rng_seed": None,
        })
        recs.append({
            "context_id": context_id,
            "family": "CROSS_SOURCE_CONFLICT",
            "severity": sev,
            "selected_pressure_ids": ranked["CROSS_SOURCE_CONFLICT"][:CONFLICT_COUNTS[sev]],
            "selected_edges": [],
            "rng_seed": None,
        })
    return recs


def apply_numeric_corruption(
    family, severity, context_id, factual, counterfactual,
    p_id_to_fi, selected_ids, mean, scale
):
    out = factual.copy()
    fis = [p_id_to_fi[x] for x in selected_ids]

    if family == "MISSING_SENSOR":
        for fi in fis:
            out[:,fi] = mean[fi]

    elif family == "SENSOR_NOISE":
        rng = np.random.default_rng(stable_seed(ROOT_SEED, context_id, family, severity))
        z = rng.normal(0.0, 1.0, size=(len(out), len(fis)))
        for j,fi in enumerate(fis):
            out[:,fi] += z[:,j].astype(np.float32) * (NOISE_SD[severity] * scale[fi])

    elif family == "SENSOR_BIAS":
        mag = BIAS_SD[severity]
        for fi, sid in zip(fis, selected_ids):
            sign = 1.0 if int(stable_digest(ROOT_SEED, context_id, family, sid)[-1],16) % 2 == 0 else -1.0
            out[:,fi] += sign * mag * scale[fi]

    elif family == "STALE_SENSOR":
        lag = STALE_LAG[severity]
        for fi in fis:
            source = factual[:,fi].copy()
            out[:lag,fi] = source[0]
            out[lag:,fi] = source[:-lag]

    elif family == "CROSS_SOURCE_CONFLICT":
        for fi in fis:
            out[:,fi] = counterfactual[:,fi]

    else:
        raise ValueError(family)

    return out, fis


def windows_from_sequence(seq, rows, flatten=False):
    xs = []
    for _, r in rows.iterrows():
        s, f = int(r["start_idx"]), int(r["final_idx"])
        w = seq[s:f+1]
        xs.append(w.reshape(-1) if flatten else w)
    return np.stack(xs).astype(np.float32)


def evaluate_rows(y, p, threshold):
    q = (p >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y,q)),
        "f1": float(f1_score(y,q,zero_division=0)),
        "auprc": float(average_precision_score(y,p)),
        "auroc": float(roc_auc_score(y,p)),
        "brier": float(brier_score_loss(y,p)),
        "accuracy": float((q==y).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--degradation-protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    win = Path(args.windows)
    gen = Path(args.generated)
    baseline_dir = Path(args.baseline)
    graph_dir = Path(args.graph)
    dp = Path(args.degradation_protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA CALIBRATION DEGRADATIONS v0.9b")
    print("="*78)

    for n,h in WINDOW_HASHES.items(): verify(win/n,h)
    for n,h in GENERATED_HASHES.items(): verify(gen/n,h)
    verify(baseline_dir/"baseline_mlp_v0_5.pt",BASELINE_HASH,"frozen MLP")
    verify(graph_dir/"graph_gcn_gru_v0_6.pt",GRAPH_HASH,"frozen GCN-GRU")
    # v0.8 freeze hashed the packaged source protocol before copying it with
    # Path.write_text() on Windows. Windows newline translation changed LF to
    # CRLF in the copied protocol, so the copy has a different byte hash while
    # preserving identical text content. Verify both artifacts explicitly:
    #   1) frozen record declares the original source hash;
    #   2) Windows copy matches the known CRLF byte hash.
    record_path = dp / "FROZEN_DEGRADATION_PROTOCOL_RECORD.json"
    if not record_path.exists():
        raise SystemExit(f"Missing frozen degradation record: {record_path}")
    frozen_record = json.loads(record_path.read_text(encoding="utf-8"))
    declared = frozen_record.get("protocol_sha256")
    if declared != DEGRADATION_PROTOCOL_SOURCE_HASH:
        raise SystemExit(
            "FROZEN v0.8 RECORD PROTOCOL HASH MISMATCH\n"
            f"expected={DEGRADATION_PROTOCOL_SOURCE_HASH}\n"
            f"got={declared}"
        )
    print(f"[OK] v0.8 frozen record source hash: {declared}")
    verify(
        dp/"degradation_protocol_v0_8.md",
        DEGRADATION_PROTOCOL_WINDOWS_COPY_HASH,
        "v0.8 Windows protocol copy"
    )

    manifest = pd.read_csv(win/"window_manifest.csv")
    cal = manifest[manifest["split"]=="calibration"].copy().reset_index(drop=True)
    if len(cal)!=2700 or cal["context_id"].nunique()!=30:
        raise SystemExit("Calibration split mismatch.")
    # hard boundary
    if set(cal["split"]) != {"calibration"}:
        raise SystemExit("Non-calibration rows present.")

    norm = json.loads((win/"normalization_train_only.json").read_text())
    mean = np.array(norm["mean"],dtype=np.float32)
    scale = np.array(norm["scale_used"],dtype=np.float32)

    layout = json.loads((gen/"tensor_layout.json").read_text())
    node_ids, ix, link_rows, node_type = build_base_topology(gen,layout)
    base_A = adjacency_from_links(len(node_ids),link_rows)
    base_An = normalized_adjacency(base_A)
    base_static = static_features(base_A,node_ids,node_type)
    pmap,fmap,tmap = sensor_maps(layout,ix,link_rows)

    pressure_ids = [sid for _,sid,_ in pmap]
    p_id_to_fi = {sid:fi for fi,sid,_ in pmap}

    # Load frozen models strictly before any corruption inference.
    bc = torch.load(baseline_dir/"baseline_mlp_v0_5.pt",map_location="cpu",weights_only=False)
    mlp = MLP(); mlp.load_state_dict(bc["model_state_dict"],strict=True); mlp.eval()

    gc = torch.load(graph_dir/"graph_gcn_gru_v0_6.pt",map_location="cpu",weights_only=False)
    graph = GraphTemporalModel(base_An); graph.load_state_dict(gc["model_state_dict"],strict=True); graph.eval()

    # Calibration contexts only: load once.
    raw = {}
    for cid in sorted(cal["context_id"].unique()):
        with np.load(gen/"contexts"/f"{cid}.npz",allow_pickle=False) as z:
            raw[(cid,"CLEAN")] = z["clean_sensor"].astype(np.float32)
            raw[(cid,"LEAK")] = z["leak_sensor"].astype(np.float32)
    print("[BOUNDARY] loaded calibration contexts only:",len(set(c for c,_ in raw)))

    # Exact corruption realization manifest.
    realization = []
    topo_selected = {}
    for cid in sorted(cal["context_id"].unique()):
        recs = corruption_selection(cid,pressure_ids,link_rows,len(node_ids))
        for r in recs:
            rr = dict(r)
            rr["selected_pressure_ids"] = ",".join(rr["selected_pressure_ids"])
            rr["selected_edges"] = ",".join(rr["selected_edges"])
            realization.append(rr)
            if r["family"]=="TOPOLOGY_MISMATCH":
                topo_selected[(cid,r["severity"])] = r["selected_edges"]

    realization_df = pd.DataFrame(realization)
    realization_path = out/"calibration_corruption_manifest.csv"
    realization_df.to_csv(realization_path,index=False)

    predictions = []

    def append_predictions(model_name, family, severity, rows, probs, threshold):
        ent = entropy01(probs)
        for j,(_,r) in enumerate(rows.iterrows()):
            predictions.append({
                "model": model_name,
                "family": family,
                "severity": severity,
                "evidence_failure": 0 if family=="CLEAN" else 1,
                "window_id": r["window_id"],
                "pair_id": r["pair_id"],
                "context_id": r["context_id"],
                "variant": r["variant"],
                "final_idx": int(r["final_idx"]),
                "target_active_leak": int(r["target_active_leak"]),
                "probability": float(probs[j]),
                "predictive_entropy": float(ent[j]),
                "threshold": float(threshold),
                "prediction": int(probs[j]>=threshold),
                "correct": int((probs[j]>=threshold)==int(r["target_active_leak"])),
            })

    # -------- CLEAN calibration inference --------
    print("[RUN] CLEAN calibration")
    Xm_all=[]; Xg_all=[]
    for cid,grp in cal.groupby("context_id",sort=True):
        for _,r in grp.iterrows():
            arr=raw[(cid,r["variant"])]
            s,f=int(r["start_idx"]),int(r["final_idx"])
            w=(arr[s:f+1]-mean)/scale
            Xm_all.append(w.reshape(-1))
            gseq=to_graph(arr,mean,scale,len(node_ids),base_static,pmap,fmap,tmap)
            Xg_all.append(gseq[s:f+1])
    Xm_all=np.stack(Xm_all).astype(np.float32)
    Xg_all=np.stack(Xg_all).astype(np.float32)
    pm=sigmoid(predict(mlp,Xm_all,1024))
    pg=sigmoid(predict(graph,Xg_all,128))
    append_predictions("MLP","CLEAN","clean",cal,pm,BASELINE_THRESHOLD)
    append_predictions("GCN_GRU","CLEAN","clean",cal,pg,GRAPH_THRESHOLD)

    # -------- numeric corruption families --------
    numeric_families=[
        "MISSING_SENSOR","SENSOR_NOISE","SENSOR_BIAS",
        "STALE_SENSOR","CROSS_SOURCE_CONFLICT"
    ]

    for fam in numeric_families:
        for sev in SEVERITIES:
            print(f"[RUN] {fam} / {sev}")
            Xm=[];Xg=[]
            # Must preserve global cal row order.
            for _,r in cal.iterrows():
                cid=str(r["context_id"]); variant=str(r["variant"])
                factual=raw[(cid,variant)]
                other=raw[(cid,"LEAK" if variant=="CLEAN" else "CLEAN")]

                rr=realization_df[
                    (realization_df.context_id==cid)&
                    (realization_df.family==fam)&
                    (realization_df.severity==sev)
                ].iloc[0]
                selected=[] if not rr.selected_pressure_ids else str(rr.selected_pressure_ids).split(",")

                corr,missing_fis=apply_numeric_corruption(
                    fam,sev,cid,factual,other,p_id_to_fi,selected,mean,scale
                )
                s,f=int(r["start_idx"]),int(r["final_idx"])
                mw=(corr[s:f+1]-mean)/scale
                Xm.append(mw.reshape(-1))
                gseq=to_graph(
                    corr,mean,scale,len(node_ids),base_static,pmap,fmap,tmap,
                    missing_fi=missing_fis if fam=="MISSING_SENSOR" else None
                )
                Xg.append(gseq[s:f+1])

            Xm=np.stack(Xm).astype(np.float32)
            Xg=np.stack(Xg).astype(np.float32)
            pm=sigmoid(predict(mlp,Xm,1024))
            pg=sigmoid(predict(graph,Xg,128))
            append_predictions("MLP",fam,sev,cal,pm,BASELINE_THRESHOLD)
            append_predictions("GCN_GRU",fam,sev,cal,pg,GRAPH_THRESHOLD)

    # -------- topology mismatch: GCN only --------
    for sev in SEVERITIES:
        print(f"[RUN] TOPOLOGY_MISMATCH / {sev} (GCN only)")
        # Process context-by-context because adjacency differs by context.
        probs_by_index=np.zeros(len(cal),dtype=np.float64)

        for cid,grp in cal.groupby("context_id",sort=True):
            removed=topo_selected[(cid,sev)]
            active=[r for r in link_rows if r["link_id"] not in set(removed)]
            A=adjacency_from_links(len(node_ids),active)
            An=normalized_adjacency(A)
            st=static_features(A,node_ids,node_type)
            graph.A_norm.copy_(torch.tensor(An,dtype=torch.float32))

            X=[];positions=[]
            for idx,r in grp.iterrows():
                arr=raw[(cid,r["variant"])]
                gseq=to_graph(arr,mean,scale,len(node_ids),st,pmap,fmap,tmap)
                s,f=int(r["start_idx"]),int(r["final_idx"])
                X.append(gseq[s:f+1]); positions.append(idx)
            X=np.stack(X).astype(np.float32)
            p=sigmoid(predict(graph,X,128))
            for j,pos in enumerate(positions): probs_by_index[pos]=p[j]

        # restore original adjacency
        graph.A_norm.copy_(torch.tensor(base_An,dtype=torch.float32))
        append_predictions(
            "GCN_GRU","TOPOLOGY_MISMATCH",sev,cal,
            probs_by_index,GRAPH_THRESHOLD
        )

    pred_df=pd.DataFrame(predictions)
    pred_path=out/"calibration_degradation_predictions.csv"
    pred_df.to_csv(pred_path,index=False)

    # -------- condition metrics --------
    metrics=[]
    for (model,fam,sev),d in pred_df.groupby(["model","family","severity"],sort=True):
        y=d.target_active_leak.to_numpy(dtype=int)
        p=d.probability.to_numpy(dtype=float)
        t=BASELINE_THRESHOLD if model=="MLP" else GRAPH_THRESHOLD
        m=evaluate_rows(y,p,t)
        m.update({"model":model,"family":fam,"severity":sev,"n":len(d)})
        metrics.append(m)
    metrics_df=pd.DataFrame(metrics)
    metrics_path=out/"calibration_condition_metrics.csv"
    metrics_df.to_csv(metrics_path,index=False)

    # -------- entropy evidence-failure detection --------
    entropy_results=[]
    for model in ["MLP","GCN_GRU"]:
        clean=pred_df[(pred_df.model==model)&(pred_df.family=="CLEAN")]
        for fam in FAMILIES:
            if model=="MLP" and fam=="TOPOLOGY_MISMATCH":
                continue
            for sev in SEVERITIES:
                deg=pred_df[
                    (pred_df.model==model)&
                    (pred_df.family==fam)&
                    (pred_df.severity==sev)
                ]
                if len(deg)==0: continue
                yy=np.r_[np.zeros(len(clean),dtype=int),np.ones(len(deg),dtype=int)]
                score=np.r_[
                    clean.predictive_entropy.to_numpy(dtype=float),
                    deg.predictive_entropy.to_numpy(dtype=float)
                ]
                entropy_results.append({
                    "model":model,"family":fam,"severity":sev,
                    "n_clean":len(clean),"n_degraded":len(deg),
                    "entropy_evidence_failure_auroc":float(roc_auc_score(yy,score)),
                    "entropy_evidence_failure_auprc":float(average_precision_score(yy,score)),
                })
    entropy_df=pd.DataFrame(entropy_results)
    entropy_path=out/"calibration_entropy_detection.csv"
    entropy_df.to_csv(entropy_path,index=False)

    # -------- artifact freeze --------
    implementation = {
        "status":"CALIBRATION_DEGRADATIONS_COMPLETE",
        "protocol_sha256":DEGRADATION_PROTOCOL_SOURCE_HASH,
        "root_seed":ROOT_SEED,
        "split_loaded":"calibration only",
        "calibration_contexts":30,
        "calibration_windows":2700,
        "models_retrained":False,
        "thresholds_changed":False,
        "topology_mismatch_static_degree_rule":
            "degree feature recomputed from the same model-visible mismatched graph",
        "topology_mismatch_mlp":
            "not applicable; no MLP rows generated for this family",
        "numeric_corruptions":
            "applied once to full 289-step context stream, then frozen 2h windows sliced",
        "next":
            "Input x Gradient attribution pass on frozen deterministic subset",
    }
    impl_path=out/"calibration_degradation_execution.json"
    impl_path.write_text(json.dumps(implementation,indent=2))

    hashes={
        pred_path.name:sha256(pred_path),
        realization_path.name:sha256(realization_path),
        metrics_path.name:sha256(metrics_path),
        entropy_path.name:sha256(entropy_path),
        impl_path.name:sha256(impl_path),
    }
    record={
        "status":"FROZEN_CALIBRATION_DEGRADATION_OUTPUTS",
        "artifact_sha256":hashes,
        "degraded_test_loaded":False,
        "models_retrained":False,
        "thresholds_changed":False,
    }
    (out/"FROZEN_CALIBRATION_DEGRADATION_RECORD.json").write_text(
        json.dumps(record,indent=2)
    )

    lines=[
        "AQUA-VERA CALIBRATION DEGRADATIONS v0.9b",
        "="*78,
        "STATUS: PASS / CALIBRATION DEGRADATIONS FROZEN",
        f"Protocol SHA-256: {DEGRADATION_PROTOCOL_SOURCE_HASH}",
        "",
        "BOUNDARY",
        "  split loaded: CALIBRATION ONLY",
        "  contexts: 30",
        "  windows per condition: 2,700",
        "  degraded TEST loaded: NO",
        "  model retraining: NO",
        "  threshold changes: NO",
        "",
        "PREDICTION ROWS",
        f"  total: {len(pred_df):,}",
        f"  MLP: {int((pred_df.model=='MLP').sum()):,}",
        f"  GCN-GRU: {int((pred_df.model=='GCN_GRU').sum()):,}",
        "",
        "TOPOLOGY MISMATCH",
        "  GCN only: YES",
        "  connectivity-preserving removals: YES",
        "  topology-derived degree feature updated consistently: YES",
        "",
        "ENTROPY DETECTION RANGE (descriptive calibration)",
        f"  minimum AUROC: {entropy_df.entropy_evidence_failure_auroc.min():.6f}",
        f"  median AUROC: {entropy_df.entropy_evidence_failure_auroc.median():.6f}",
        f"  maximum AUROC: {entropy_df.entropy_evidence_failure_auroc.max():.6f}",
        "",
        "ARTIFACT HASHES",
    ]
    for n,h in hashes.items(): lines.append(f"  {n}: {h}")
    lines += [
        "",
        "NEXT",
        "  Retain calibration_degradation_summary.txt with the project record.",
        "  Next pass: Input x Gradient attribution on the pre-specified calibration subset.",
        "  Do not generate degraded test yet.",
    ]
    summary=out/"calibration_degradation_summary.txt"
    summary.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__=="__main__":
    main()
