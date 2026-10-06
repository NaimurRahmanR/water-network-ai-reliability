from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score

ARCHITECTURES = ("central_only", "edge_only", "hybrid_naive", "hybrid_reliability_aware")
CONDITIONS = ("CLEAN", "MISSING_SENSOR", "STALE_SENSOR", "SENSOR_BIAS", "PACKET_LOSS", "CROSS_SOURCE_CONFLICT")


@dataclass
class SensorLayout:
    feature_names: List[str]
    pressure_indices: List[int]
    edge_for_feature: Dict[int, str]


@dataclass
class ReconModel:
    target: int
    features: List[int]
    model: Ridge
    residual_q95: float
    residual_q99: float


@dataclass
class Incident:
    run_id: str
    context_id: str
    variant: str
    timestamp_index: int
    condition: str
    severity: str
    sensor_index: int
    sensor_name: str
    edge_id: str
    architecture: str
    action: str
    proceeded: int
    corrupted: int
    observed_value: float | None
    recovered_value: float | None
    clean_value: float
    abs_error: float | None
    edge_residual: float | None
    central_residual: float | None
    edge_central_disagreement: float
    packet_lost: int
    delay_ms: int
    unsafe_proceed: int


def _load_layout(generated_dir: Path) -> SensorLayout:
    layout = json.loads((generated_dir / "tensor_layout.json").read_text(encoding="utf-8"))
    names = list(layout["sensor_feature_order"])
    pressure = [i for i, n in enumerate(names) if n.startswith("pressure::")]
    edge_for_feature: Dict[int, str] = {}
    # Deterministic three-edge partition. Every pressure feature belongs to exactly one edge.
    for rank, fi in enumerate(pressure):
        edge_for_feature[fi] = f"edge_{rank % 3 + 1}"
    non_pressure = [i for i in range(len(names)) if i not in pressure]
    for rank, fi in enumerate(non_pressure):
        edge_for_feature[fi] = f"edge_{rank % 3 + 1}"
    return SensorLayout(names, pressure, edge_for_feature)


def _context_paths(generated_dir: Path) -> List[Path]:
    paths = sorted((generated_dir / "contexts").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No generated Net3 context NPZ files under {generated_dir / 'contexts'}. "
            "Run the existing src/net3/v0_3a scenario generator first, or use --smoke."
        )
    return paths


def _split_context_ids(generated_dir: Path, paths: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    # Reuse the parent experiment's frozen context assignments rather than creating
    # a new split. New reconstructors fit on TRAIN, thresholds use CALIBRATION,
    # and final architecture metrics use TEST. VALIDATION remains untouched here.
    index_path = generated_dir / "generation_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing frozen split index: {index_path}")
    index = pd.read_csv(index_path, usecols=["context_id", "split"])
    split_map = dict(zip(index["context_id"].astype(str), index["split"].astype(str).str.lower()))
    buckets = {"train": [], "calibration": [], "test": []}
    for path in paths:
        split = split_map.get(path.stem)
        if split in buckets:
            buckets[split].append(path)
    if any(not buckets[k] for k in buckets):
        counts = {k: len(v) for k, v in buckets.items()}
        raise ValueError(f"Frozen split recovery failed: {counts}")
    return buckets["train"], buckets["calibration"], buckets["test"]


def _rows_from_contexts(paths: Iterable[Path], max_rows_per_variant: int = 96) -> pd.DataFrame:
    rows: List[dict] = []
    for p in paths:
        with np.load(p) as z:
            for variant, key in (("CLEAN", "clean_sensor"), ("LEAK", "leak_sensor")):
                x = np.asarray(z[key], dtype=np.float64)
                if x.ndim != 2:
                    raise ValueError(f"Expected 2D sensor matrix in {p}:{key}, got {x.shape}")
                # Uniform deterministic downsampling to bound application-grade runtime.
                take = np.linspace(0, x.shape[0] - 1, min(max_rows_per_variant, x.shape[0]), dtype=int)
                for ti in take:
                    rows.append({
                        "context_id": p.stem,
                        "variant": variant,
                        "timestamp_index": int(ti),
                        "values": x[ti].copy(),
                    })
    return pd.DataFrame(rows)


def _make_smoke(seed: int = 271828) -> Tuple[SensorLayout, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = [f"pressure::P{i:02d}" for i in range(9)] + ["flow::F01", "flow::F02", "flow::F03"]
    pressure = list(range(9))
    edges = {i: f"edge_{i % 3 + 1}" for i in range(len(names))}
    layout = SensorLayout(names, pressure, edges)

    def make(n_ctx: int, prefix: str) -> pd.DataFrame:
        rows = []
        t = np.linspace(0, 2 * np.pi, 72)
        for c in range(n_ctx):
            phase = rng.uniform(-0.3, 0.3)
            base = np.stack([
                40 + 3 * np.sin(t + phase + i * 0.15) + rng.normal(0, 0.15, len(t))
                for i in range(9)
            ], axis=1)
            flows = np.stack([
                7 + 0.8 * np.cos(t + phase + i * 0.25) + rng.normal(0, 0.08, len(t))
                for i in range(3)
            ], axis=1)
            clean = np.concatenate([base, flows], axis=1)
            leak = clean.copy()
            leak[:, :3] -= 1.5 + 0.4 * np.sin(t)[:, None]
            leak[:, 9:] += 0.25
            for variant, mat in (("CLEAN", clean), ("LEAK", leak)):
                for ti in range(0, len(t), 3):
                    rows.append({"context_id": f"{prefix}{c:03d}", "variant": variant, "timestamp_index": ti, "values": mat[ti]})
        return pd.DataFrame(rows)

    return layout, make(18, "TR"), make(6, "CA"), make(6, "TE")


def _matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["values"].to_numpy()).astype(np.float64)


def _fit_one(target: int, features: List[int], train_x: np.ndarray, calib_x: np.ndarray) -> ReconModel:
    model = Ridge(alpha=1.0)
    model.fit(train_x[:, features], train_x[:, target])
    pred = model.predict(calib_x[:, features])
    resid = np.abs(pred - calib_x[:, target])
    return ReconModel(
        target=target,
        features=features,
        model=model,
        residual_q95=float(np.quantile(resid, 0.95)),
        residual_q99=float(np.quantile(resid, 0.99)),
    )


def _fit_reconstructors(layout: SensorLayout, train_x: np.ndarray, calib_x: np.ndarray):
    edge_models: Dict[int, ReconModel] = {}
    central_models: Dict[int, ReconModel] = {}
    hybrid_disagreement_q95: Dict[int, float] = {}

    all_idx = list(range(train_x.shape[1]))
    for target in layout.pressure_indices:
        edge_id = layout.edge_for_feature[target]
        local_features = [i for i in all_idx if i != target and layout.edge_for_feature[i] == edge_id]
        central_features = [i for i in all_idx if i != target]
        if len(local_features) < 2:
            raise ValueError(f"Too few local features for {layout.feature_names[target]}")
        edge = _fit_one(target, local_features, train_x, calib_x)
        central = _fit_one(target, central_features, train_x, calib_x)
        edge_models[target] = edge
        central_models[target] = central
        ep = edge.model.predict(calib_x[:, edge.features])
        cp = central.model.predict(calib_x[:, central.features])
        hybrid_disagreement_q95[target] = float(np.quantile(np.abs(ep - cp), 0.95))
    return edge_models, central_models, hybrid_disagreement_q95


def _corrupt(
    row: np.ndarray,
    opposite: np.ndarray,
    target: int,
    condition: str,
    severity: str,
    std: np.ndarray,
    previous: np.ndarray | None,
    previous_delay_ms: int,
    rng: np.random.Generator,
):
    x = row.copy()
    packet_lost = 0
    delay_ms = 0
    corrupted = int(condition != "CLEAN")
    if condition == "MISSING_SENSOR":
        x[target] = np.nan
    elif condition == "STALE_SENSOR":
        x[target] = (previous[target] if previous is not None else row[target])
        delay_ms = previous_delay_ms if previous is not None else 0
    elif condition == "SENSOR_BIAS":
        mult = {"low": 0.5, "medium": 1.0, "high": 2.0}[severity]
        x[target] = row[target] + mult * max(std[target], 1e-6)
    elif condition == "PACKET_LOSS":
        # Experimental communication intervention: a targeted observation is dropped.
        x[target] = np.nan
        packet_lost = 1
    elif condition == "CROSS_SOURCE_CONFLICT":
        x[target] = opposite[target]
    return x, packet_lost, delay_ms, corrupted


def _predict(model: ReconModel, x: np.ndarray, train_means: np.ndarray) -> float:
    vals = x[model.features].copy()
    bad = ~np.isfinite(vals)
    if bad.any():
        vals[bad] = train_means[np.asarray(model.features)[bad]]
    return float(model.model.predict(vals.reshape(1, -1))[0])


def _decision(
    architecture: str,
    observed: float | None,
    edge_pred: float,
    central_pred: float,
    edge_residual: float | None,
    central_residual: float | None,
    edge_model: ReconModel,
    central_model: ReconModel,
    disagreement_threshold: float,
):
    missing = observed is None or not math.isfinite(observed)
    edge_bad = missing or (edge_residual is not None and edge_residual > edge_model.residual_q95)
    central_bad = missing or (central_residual is not None and central_residual > central_model.residual_q95)
    disagreement = abs(edge_pred - central_pred)

    if architecture == "edge_only":
        return (edge_pred if edge_bad else observed), "EDGE_RECONSTRUCT" if edge_bad else "PROCEED", 1
    if architecture == "central_only":
        return (central_pred if central_bad else observed), "CENTRAL_RECONSTRUCT" if central_bad else "PROCEED", 1
    if architecture == "hybrid_naive":
        candidate = edge_pred if edge_bad else observed
        return candidate, "EDGE_RECONSTRUCT_ACCEPTED" if edge_bad else "PROCEED", 1
    if architecture == "hybrid_reliability_aware":
        if not edge_bad and not central_bad:
            return observed, "PROCEED", 1
        if disagreement <= disagreement_threshold:
            # Both views agree sufficiently to allow a bounded recovery.
            return float((edge_pred + central_pred) / 2.0), "VERIFIED_RECONSTRUCT", 1
        return None, "ESCALATE", 0
    raise ValueError(architecture)


def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS incidents")
    conn.execute(
        """
        CREATE TABLE incidents (
            run_id TEXT, context_id TEXT, variant TEXT, timestamp_index INTEGER,
            condition TEXT, severity TEXT, sensor_index INTEGER, sensor_name TEXT,
            edge_id TEXT, architecture TEXT, action TEXT, proceeded INTEGER,
            corrupted INTEGER, observed_value REAL, recovered_value REAL,
            clean_value REAL, abs_error REAL, edge_residual REAL, central_residual REAL,
            edge_central_disagreement REAL, packet_lost INTEGER, delay_ms INTEGER,
            unsafe_proceed INTEGER
        )
        """
    )
    return conn


def _write_incidents(conn: sqlite3.Connection, incidents: List[Incident]):
    cols = list(asdict(incidents[0]).keys())
    q = f"INSERT INTO incidents ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})"
    conn.executemany(q, [[asdict(x)[c] for c in cols] for x in incidents])
    conn.commit()


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, cond, sev), d in df.groupby(["architecture", "condition", "severity"], dropna=False):
        proceeded = d["proceeded"].astype(bool)
        err = d.loc[proceeded, "abs_error"].dropna()
        rows.append({
            "architecture": arch,
            "condition": cond,
            "severity": sev,
            "n": len(d),
            "coverage": float(proceeded.mean()),
            "escalation_rate": float((d.action == "ESCALATE").mean()),
            "unsafe_proceed_rate_all": float(d.unsafe_proceed.mean()),
            "selective_unsafe_risk": float(d.loc[proceeded, "unsafe_proceed"].mean()) if proceeded.any() else np.nan,
            "mae_when_proceeding": float(err.mean()) if len(err) else np.nan,
            "p95_error_when_proceeding": float(err.quantile(0.95)) if len(err) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["condition", "severity", "architecture"]).reset_index(drop=True)


def _cluster_incidents(df: pd.DataFrame, out_dir: Path, seed: int):
    # Cluster incident behaviour, not the known corruption label.
    work = df[(df.condition != "CLEAN")].copy()
    if len(work) < 20:
        return
    if len(work) > 5000:
        work = work.sample(n=5000, random_state=seed).copy()
    numeric = pd.DataFrame({
        "edge_residual": work.edge_residual.fillna(work.edge_residual.median()).astype(float),
        "central_residual": work.central_residual.fillna(work.central_residual.median()).astype(float),
        "disagreement": work.edge_central_disagreement.astype(float),
        "abs_error": work.abs_error.fillna(work.clean_value.abs() * 0).astype(float),
        "packet_lost": work.packet_lost.astype(float),
        "delay_ms": work.delay_ms.astype(float) / 1000.0,
        "proceeded": work.proceeded.astype(float),
        "unsafe": work.unsafe_proceed.astype(float),
    })
    scale = numeric.std(ddof=0).replace(0, 1.0)
    z = (numeric - numeric.mean()) / scale
    best = None
    for k in range(2, min(6, len(z) - 1)):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        lab = km.fit_predict(z)
        score = silhouette_score(z, lab)
        if best is None or score > best[0]:
            best = (score, k, lab)
    if best is None:
        return
    score, k, labels = best
    work["cluster"] = labels
    work.to_csv(out_dir / "clustered_incidents.csv", index=False)

    summaries = []
    lessons = []
    for c, d in work.groupby("cluster"):
        top_condition = str(d.condition.value_counts().idxmax())
        top_action = str(d.action.value_counts().idxmax())
        unsafe = float(d.unsafe_proceed.mean())
        escalation = float((d.action == "ESCALATE").mean())
        disagreement = float(d.edge_central_disagreement.median())
        summaries.append({
            "cluster": int(c), "n": len(d), "dominant_condition": top_condition,
            "dominant_action": top_action, "unsafe_rate": unsafe,
            "escalation_rate": escalation, "median_edge_central_disagreement": disagreement,
        })
        if unsafe >= 0.10:
            lesson = "Prefer verification/escalation before autonomous reuse when this incident profile recurs."
        elif escalation >= 0.50:
            lesson = "Evidence is frequently insufficient for autonomous recovery; request wider-network evidence."
        else:
            lesson = "Bounded reconstruction is usually adequate for this experimental incident profile."
        lessons.append({
            "cluster": int(c), "support_n": len(d), "dominant_condition": top_condition,
            "lesson": lesson,
            "scope_note": "Experimental Net3/smoke benchmark only; not a utility operating rule.",
        })
    pd.DataFrame(summaries).to_csv(out_dir / "cluster_summary.csv", index=False)
    (out_dir / "lessons.json").write_text(json.dumps({
        "silhouette": float(score), "selected_k": int(k), "lessons": lessons
    }, indent=2), encoding="utf-8")


def run(layout: SensorLayout, train_df: pd.DataFrame, calib_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path, seed: int):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x = _matrix(train_df)
    calib_x = _matrix(calib_df)
    train_means = np.nanmean(train_x, axis=0)
    train_std = np.nanstd(train_x, axis=0)
    edge_models, central_models, disagreement_q95 = _fit_reconstructors(layout, train_x, calib_x)

    run_id = f"aqua_bridge_seed_{seed}"
    incidents: List[Incident] = []
    severities = ("low", "medium", "high")
    test_records = test_df.to_dict("records")
    by_key = {(r["context_id"], r["variant"], r["timestamp_index"]): r["values"] for r in test_records}
    previous_by_variant: Dict[Tuple[str, str], Tuple[np.ndarray, int]] = {}

    for rec in test_records:
        clean = np.asarray(rec["values"], dtype=float)
        opp_variant = "LEAK" if rec["variant"] == "CLEAN" else "CLEAN"
        opposite = np.asarray(by_key.get((rec["context_id"], opp_variant, rec["timestamp_index"]), clean), dtype=float)
        previous_entry = previous_by_variant.get((rec["context_id"], rec["variant"]))
        previous = previous_entry[0] if previous_entry is not None else None
        previous_delay_ms = (
            max(0, int(rec["timestamp_index"]) - int(previous_entry[1])) * 300_000
            if previous_entry is not None else 0
        )
        for target in layout.pressure_indices:
            for condition in CONDITIONS:
                if condition == "STALE_SENSOR" and previous is None:
                    continue
                if condition == "CLEAN":
                    sev_list = ("clean",)
                elif condition == "SENSOR_BIAS":
                    sev_list = severities
                else:
                    sev_list = ("fixed",)
                for severity in sev_list:
                    corrupted_x, packet_lost, delay_ms, corrupted = _corrupt(
                        clean, opposite, target, condition, severity, train_std, previous, previous_delay_ms, rng
                    )
                    edge_m = edge_models[target]
                    central_m = central_models[target]
                    edge_pred = _predict(edge_m, corrupted_x, train_means)
                    central_pred = _predict(central_m, corrupted_x, train_means)
                    obs_raw = corrupted_x[target]
                    observed = float(obs_raw) if np.isfinite(obs_raw) else None
                    edge_resid = abs(observed - edge_pred) if observed is not None else None
                    central_resid = abs(observed - central_pred) if observed is not None else None
                    # Safety threshold is calibration-frozen per sensor; no test tuning.
                    safety_threshold = max(edge_m.residual_q99, central_m.residual_q99) * 2.0
                    safety_threshold = max(safety_threshold, 1e-6)
                    for arch in ARCHITECTURES:
                        recovered, action, proceeded = _decision(
                            arch, observed, edge_pred, central_pred, edge_resid, central_resid,
                            edge_m, central_m, disagreement_q95[target]
                        )
                        abs_error = abs(recovered - clean[target]) if recovered is not None else None
                        unsafe = int(bool(proceeded) and abs_error is not None and abs_error > safety_threshold)
                        incidents.append(Incident(
                            run_id=run_id, context_id=rec["context_id"], variant=rec["variant"],
                            timestamp_index=int(rec["timestamp_index"]), condition=condition, severity=severity,
                            sensor_index=target, sensor_name=layout.feature_names[target],
                            edge_id=layout.edge_for_feature[target], architecture=arch, action=action,
                            proceeded=int(proceeded), corrupted=corrupted, observed_value=observed,
                            recovered_value=float(recovered) if recovered is not None else None,
                            clean_value=float(clean[target]), abs_error=float(abs_error) if abs_error is not None else None,
                            edge_residual=float(edge_resid) if edge_resid is not None else None,
                            central_residual=float(central_resid) if central_resid is not None else None,
                            edge_central_disagreement=float(abs(edge_pred - central_pred)),
                            packet_lost=packet_lost, delay_ms=delay_ms, unsafe_proceed=unsafe,
                        ))
        previous_by_variant[(rec["context_id"], rec["variant"])] = (clean, int(rec["timestamp_index"]))

    conn = _init_db(out_dir / "incident_hub.sqlite")
    _write_incidents(conn, incidents)
    conn.close()
    df = pd.DataFrame([asdict(x) for x in incidents])
    df.to_csv(out_dir / "incidents.csv", index=False)
    metrics = _metrics(df)
    metrics.to_csv(out_dir / "architecture_metrics.csv", index=False)
    _cluster_incidents(df[df.architecture == "hybrid_reliability_aware"].copy(), out_dir, seed)

    model_summary = []
    for t in layout.pressure_indices:
        model_summary.append({
            "sensor": layout.feature_names[t], "edge": layout.edge_for_feature[t],
            "edge_q95": edge_models[t].residual_q95, "central_q95": central_models[t].residual_q95,
            "edge_central_disagreement_q95": disagreement_q95[t]
        })
    pd.DataFrame(model_summary).to_csv(out_dir / "frozen_reconstruction_thresholds.csv", index=False)
    manifest = {
        "run_id": run_id,
        "seed": seed,
        "architectures": list(ARCHITECTURES),
        "conditions": list(CONDITIONS),
        "pressure_sensors": len(layout.pressure_indices),
        "train_rows": len(train_df), "calibration_rows": len(calib_df), "test_rows": len(test_df),
        "claim_boundary": [
            "Experimental benchmark only; not utility deployment.",
            "Containerised/simulated edge nodes are not physical edge hardware.",
            "Controlled interventions support within-benchmark effect comparisons, not general causal claims.",
            "Incident lessons are benchmark-derived and require operator validation before operational use.",
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="AQUA-BRIDGE edge/central WDS reliability experiment")
    ap.add_argument("--generated-dir", type=Path, help="Existing Net3 v0.3a generated directory containing tensor_layout.json + contexts/")
    ap.add_argument("--out", type=Path, default=Path("outputs/aqua_bridge"))
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--smoke", action="store_true", help="Run a self-contained CPU smoke experiment")
    ap.add_argument("--max-rows-per-variant", type=int, default=72)
    args = ap.parse_args()

    if args.smoke:
        layout, train, calib, test = _make_smoke(args.seed)
        # Keep smoke mode intentionally small and CPU-fast.
        test = test.iloc[:48].copy()
    else:
        if args.generated_dir is None:
            ap.error("--generated-dir is required unless --smoke is used")
        layout = _load_layout(args.generated_dir)
        paths = _context_paths(args.generated_dir)
        tr, ca, te = _split_context_ids(args.generated_dir, paths)
        train = _rows_from_contexts(tr, args.max_rows_per_variant)
        calib = _rows_from_contexts(ca, args.max_rows_per_variant)
        test = _rows_from_contexts(te, args.max_rows_per_variant)

    metrics = run(layout, train, calib, test, args.out, args.seed)
    print("AQUA-BRIDGE completed")
    print(f"outputs: {args.out}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
