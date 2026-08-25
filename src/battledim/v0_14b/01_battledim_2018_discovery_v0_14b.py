
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

H18S = "00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e"
H18L = "743ea7b2845b3dced2ada4fba1a54194c54fc9ce18bf7180a1b3afa4b3eaf2f8"

WINDOW_STEPS = 24
STRIDE_STEPS = 6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected, label):
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")
    got = sha256(path)
    if got != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch\nexpected={expected}\nactual={got}\npath={path}"
        )
    print(f"[OK] {label}")
    print(f"     {path}")


def find_unique(root, filename):
    direct = root / filename
    if direct.exists():
        return direct

    matches = list(root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"Could not find {filename} anywhere under:\n{root}")
    raise RuntimeError(
        f"Found multiple copies of {filename} under {root}:\n"
        + "\n".join(str(x) for x in matches)
    )


def intervals(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return list(zip(starts.tolist(), ends.tolist()))


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

        if current is not None:
            content = line.split(";", 1)[0].strip()
            if content:
                sections[current].append(content)

    def ids(section):
        return [
            row.split()[0]
            for row in sections.get(section, [])
            if row.split()
        ]

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
                links.append(
                    {
                        "link_id": parts[0],
                        "start": parts[1],
                        "end": parts[2],
                        "type": typ,
                    }
                )

    return {
        "junctions": junctions,
        "reservoirs": reservoirs,
        "tanks": tanks,
        "nodes": junctions + reservoirs + tanks,
        "links": links,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ready", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mr = Path(args.model_ready).expanduser()
    data = Path(args.data).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA BATTLEDIM/L-TOWN 2018 DISCOVERY v0.14b")
    print("=" * 78)
    print("2018 ONLY. No 2019 file is opened, checked, parsed, or hashed.")
    print("No model training. No split freeze. No threshold selection.")
    print()

    if not mr.exists():
        raise RuntimeError(f"Model-ready folder does not exist:\n{mr}")
    if not data.exists():
        raise RuntimeError(f"Data folder does not exist:\n{data}")

    scada_path = find_unique(mr, "2018_scada_model_ready.csv")
    leaks_path = find_unique(mr, "2018_leakages_model_ready.csv")
    inp_path = find_unique(data, "L-TOWN.inp")

    require_hash(scada_path, H18S, "2018 model-ready SCADA")
    require_hash(leaks_path, H18L, "2018 model-ready leakage data")
    print("[OK] L-TOWN.inp discovered")
    print(f"     {inp_path}")
    print()

    scada = pd.read_csv(scada_path)
    leaks = pd.read_csv(leaks_path)

    print("[SCHEMA] 2018 SCADA columns:", len(scada.columns))
    print("[SCHEMA] 2018 leakage columns:", len(leaks.columns))
    print("[SCHEMA] SCADA first columns:", list(scada.columns[:8]))
    print("[SCHEMA] leakage first columns:", list(leaks.columns[:8]))

    if "Timestamp" not in scada.columns or "Timestamp" not in leaks.columns:
        raise RuntimeError(
            "Timestamp column missing.\n"
            f"SCADA columns={list(scada.columns)}\n"
            f"Leak columns={list(leaks.columns)}"
        )

    scada["Timestamp"] = pd.to_datetime(scada["Timestamp"], errors="raise")
    leaks["Timestamp"] = pd.to_datetime(leaks["Timestamp"], errors="raise")

    if len(scada) != 105120 or len(leaks) != 105120:
        raise RuntimeError(
            f"Expected 105,120 rows in both files; got "
            f"SCADA={len(scada)}, leakage={len(leaks)}"
        )

    if not np.array_equal(
        scada["Timestamp"].to_numpy(),
        leaks["Timestamp"].to_numpy(),
    ):
        raise RuntimeError("SCADA and leakage timestamps are not identical.")

    deltas = scada["Timestamp"].diff().dropna()
    if not (deltas == pd.Timedelta(minutes=5)).all():
        bad = deltas[deltas != pd.Timedelta(minutes=5)].head()
        raise RuntimeError(f"Non-5-minute interval(s) found:\n{bad}")

    feature_cols = [c for c in scada.columns if c != "Timestamp"]
    leak_cols = [c for c in leaks.columns if c != "Timestamp"]

    pressure_cols = [c for c in feature_cols if str(c).startswith("pressure__")]
    flow_cols = [c for c in feature_cols if str(c).startswith("flow__")]
    level_cols = [c for c in feature_cols if str(c).startswith("level__")]

    print()
    print(
        "[FEATURES]",
        f"all={len(feature_cols)}",
        f"pressure={len(pressure_cols)}",
        f"flow={len(flow_cols)}",
        f"level={len(level_cols)}",
        f"leak_columns={len(leak_cols)}",
    )

    expected = (37, 33, 3, 1, 14)
    actual = (
        len(feature_cols),
        len(pressure_cols),
        len(flow_cols),
        len(level_cols),
        len(leak_cols),
    )
    if actual != expected:
        raise RuntimeError(
            "Unexpected model-ready schema.\n"
            f"expected={expected}\nactual={actual}\n"
            f"SCADA columns={feature_cols}\n"
            f"Leak columns={leak_cols}"
        )

    leak_values = leaks[leak_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(leak_values).all():
        bad = np.argwhere(~np.isfinite(leak_values))
        raise RuntimeError(f"Non-finite leak values found; first={bad[:10].tolist()}")

    # Discovery diagnostic only; NOT a frozen label definition.
    active_matrix = leak_values > 0.0
    active_count = active_matrix.sum(axis=1)
    active_any = active_count > 0

    per_leak = []
    per_interval = []

    for j, col in enumerate(leak_cols):
        vals = leak_values[:, j]
        mask = active_matrix[:, j]
        ivs = intervals(mask)

        positive_vals = vals[mask]
        per_leak.append(
            {
                "leak_column": col,
                "positive_rows": int(mask.sum()),
                "positive_fraction": float(mask.mean()),
                "min_value": float(vals.min()),
                "max_value": float(vals.max()),
                "positive_min": float(positive_vals.min()) if positive_vals.size else None,
                "positive_median": float(np.median(positive_vals)) if positive_vals.size else None,
                "positive_max": float(positive_vals.max()) if positive_vals.size else None,
                "contiguous_positive_intervals": len(ivs),
                "active_hours": float(mask.sum() * 5.0 / 60.0),
            }
        )

        for k, (start, end) in enumerate(ivs, 1):
            per_interval.append(
                {
                    "leak_column": col,
                    "interval_index": k,
                    "start_idx": start,
                    "end_idx": end,
                    "start_timestamp": str(leaks["Timestamp"].iloc[start]),
                    "end_timestamp": str(leaks["Timestamp"].iloc[end]),
                    "steps": end - start + 1,
                    "duration_hours": float((end - start + 1) * 5.0 / 60.0),
                    "max_value": float(vals[start:end + 1].max()),
                }
            )

    per_leak_df = pd.DataFrame(per_leak)
    intervals_df = pd.DataFrame(per_interval)

    global_rows = []
    for k, (start, end) in enumerate(intervals(active_any), 1):
        active_union = [
            leak_cols[j]
            for j in range(len(leak_cols))
            if active_matrix[start:end + 1, j].any()
        ]
        global_rows.append(
            {
                "component_index": k,
                "start_idx": start,
                "end_idx": end,
                "start_timestamp": str(leaks["Timestamp"].iloc[start]),
                "end_timestamp": str(leaks["Timestamp"].iloc[end]),
                "steps": end - start + 1,
                "duration_hours": float((end - start + 1) * 5.0 / 60.0),
                "max_simultaneous_active_columns": int(
                    active_count[start:end + 1].max()
                ),
                "active_columns_union": ",".join(active_union),
            }
        )

    global_df = pd.DataFrame(global_rows)

    window_rows = []
    for final_idx in range(WINDOW_STEPS - 1, len(leaks), STRIDE_STEPS):
        start_idx = final_idx - WINDOW_STEPS + 1
        ts = leaks["Timestamp"].iloc[final_idx]

        window_rows.append(
            {
                "window_index": len(window_rows),
                "start_idx": start_idx,
                "final_idx": final_idx,
                "start_timestamp": str(leaks["Timestamp"].iloc[start_idx]),
                "final_timestamp": str(ts),
                "month": str(ts.to_period("M")),
                "candidate_active_leak_final": int(active_any[final_idx]),
                "active_columns_at_final": int(active_count[final_idx]),
                "any_active_within_window": int(
                    active_any[start_idx:final_idx + 1].any()
                ),
                "active_fraction_within_window": float(
                    active_any[start_idx:final_idx + 1].mean()
                ),
            }
        )

    windows_df = pd.DataFrame(window_rows)

    monthly_point = pd.DataFrame(
        {
            "month": leaks["Timestamp"].dt.to_period("M").astype(str),
            "active": active_any.astype(int),
            "active_count": active_count.astype(int),
        }
    ).groupby("month", sort=True).agg(
        rows=("active", "size"),
        active_rows=("active", "sum"),
        active_fraction=("active", "mean"),
        max_simultaneous_active_columns=("active_count", "max"),
    ).reset_index()

    monthly_windows = windows_df.groupby("month", sort=True).agg(
        windows=("candidate_active_leak_final", "size"),
        positive_final_windows=("candidate_active_leak_final", "sum"),
        positive_final_fraction=("candidate_active_leak_final", "mean"),
        windows_touching_any_active_leak=("any_active_within_window", "sum"),
    ).reset_index()

    monthly_df = monthly_point.merge(monthly_windows, on="month", how="outer")

    graph = parse_inp(inp_path)
    node_set = set(graph["nodes"])
    link_set = set(row["link_id"] for row in graph["links"])

    pressure_ids = [c.split("__", 1)[1] for c in pressure_cols]
    flow_ids = [c.split("__", 1)[1] for c in flow_cols]
    level_ids = [c.split("__", 1)[1] for c in level_cols]

    mapping = {
        "network": {
            "junctions": len(graph["junctions"]),
            "reservoirs": len(graph["reservoirs"]),
            "tanks": len(graph["tanks"]),
            "nodes_total": len(graph["nodes"]),
            "pipes": sum(x["type"] == "pipe" for x in graph["links"]),
            "pumps": sum(x["type"] == "pump" for x in graph["links"]),
            "valves": sum(x["type"] == "valve" for x in graph["links"]),
            "edges_total": len(graph["links"]),
        },
        "sensor_mapping": {
            "pressure_ids_not_nodes": sorted(
                [x for x in pressure_ids if x not in node_set]
            ),
            "flow_ids_not_links": sorted(
                [x for x in flow_ids if x not in link_set]
            ),
            "level_ids_not_nodes": sorted(
                [x for x in level_ids if x not in node_set]
            ),
        },
    }

    outputs = {
        "2018_leak_column_audit.csv": per_leak_df,
        "2018_leak_intervals_candidate_positive.csv": intervals_df,
        "2018_global_active_components.csv": global_df,
        "2018_candidate_windows_2h_stride30m.csv": windows_df,
        "2018_monthly_balance.csv": monthly_df,
    }

    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)

    (out / "2018_graph_sensor_mapping.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    discovery = {
        "status": "PASS_2018_DISCOVERY_ONLY",
        "version": "0.14b",
        "2018_only": True,
        "2019_checked": False,
        "2019_opened": False,
        "2019_hashed": False,
        "model_trained": False,
        "split_frozen": False,
        "threshold_selected": False,
        "candidate_positive_rule": "any 2018 leakage column > 0 at timestamp",
        "candidate_positive_rule_frozen": False,
        "rows": len(scada),
        "active_rows": int(active_any.sum()),
        "active_fraction": float(active_any.mean()),
        "rows_with_ge2_active_columns": int((active_count >= 2).sum()),
        "max_simultaneous_active_columns": int(active_count.max()),
        "per_column_intervals": len(intervals_df),
        "global_active_components": len(global_df),
        "candidate_windows": len(windows_df),
        "candidate_positive_windows": int(
            windows_df["candidate_active_leak_final"].sum()
        ),
        "candidate_positive_fraction": float(
            windows_df["candidate_active_leak_final"].mean()
        ),
        "network": mapping["network"],
        "sensor_mapping": mapping["sensor_mapping"],
        "discovered_paths": {
            "2018_scada": str(scada_path),
            "2018_leakages": str(leaks_path),
            "ltown_inp": str(inp_path),
        },
    }

    record_path = out / "DISCOVERY_RECORD_v0_14b.json"
    record_path.write_text(
        json.dumps(discovery, indent=2),
        encoding="utf-8",
    )

    artifact_names = list(outputs) + [
        "2018_graph_sensor_mapping.json",
        "DISCOVERY_RECORD_v0_14b.json",
    ]
    hashes = {name: sha256(out / name) for name in artifact_names}

    lines = [
        "AQUA-VERA BATTLEDIM/L-TOWN EXTERNAL VALIDATION — 2018 DISCOVERY v0.14b",
        "=" * 78,
        "STATUS: PASS / 2018-ONLY FEASIBILITY AUDIT COMPLETE",
        "",
        "BOUNDARY",
        "  2018 parsed: YES",
        "  2019 checked/opened/hashed: NO",
        "  model trained: NO",
        "  split frozen: NO",
        "  threshold selected: NO",
        "",
        "DISCOVERED INPUTS",
        f"  SCADA: {scada_path}",
        f"  leakages: {leaks_path}",
        f"  L-TOWN.inp: {inp_path}",
        "",
        "2018 DATA",
        f"  rows: {len(scada):,}",
        f"  SCADA features: {len(feature_cols)} "
        f"(pressure={len(pressure_cols)}, flow={len(flow_cols)}, level={len(level_cols)})",
        f"  leakage columns: {len(leak_cols)}",
        "",
        "DISCOVERY-ONLY CANDIDATE LEAK ACTIVITY",
        "  diagnostic rule: any leakage column > 0 at timestamp",
        "  FINAL LABEL SEMANTICS FROZEN: NO",
        f"  active rows: {int(active_any.sum()):,}/{len(active_any):,} "
        f"({active_any.mean():.4%})",
        f"  rows with >=2 active columns: {int((active_count >= 2).sum()):,}",
        f"  max simultaneous active columns: {int(active_count.max())}",
        f"  per-column contiguous intervals: {len(intervals_df)}",
        f"  global active components: {len(global_df)}",
        "",
        "2-HOUR / 30-MINUTE CANDIDATE WINDOWS",
        f"  total: {len(windows_df):,}",
        f"  positive-final: "
        f"{int(windows_df['candidate_active_leak_final'].sum()):,}",
        f"  positive fraction: "
        f"{windows_df['candidate_active_leak_final'].mean():.4%}",
        "",
        "L-TOWN GRAPH",
        f"  nodes: {len(graph['nodes']):,}",
        f"  edges: {len(graph['links']):,}",
        f"  pressure sensor IDs not mapped to nodes: "
        f"{len(mapping['sensor_mapping']['pressure_ids_not_nodes'])}",
        f"  flow sensor IDs not mapped to links: "
        f"{len(mapping['sensor_mapping']['flow_ids_not_links'])}",
        f"  level sensor IDs not mapped to nodes: "
        f"{len(mapping['sensor_mapping']['level_ids_not_nodes'])}",
        "",
        "ARTIFACT HASHES",
    ]

    for name, h in hashes.items():
        lines.append(f"  {name}: {h}")

    lines += [
        "",
        "NEXT",
        "  Retain BattLeDIM_2018_discovery_summary.txt with the project record.",
        "  Do not model or inspect 2019.",
    ]

    summary_path = out / "BattLeDIM_2018_discovery_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("DISCOVERY AUDIT FAILED")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("FULL TRACEBACK")
        traceback.print_exc()
        raise
