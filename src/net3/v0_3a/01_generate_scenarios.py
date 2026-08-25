
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import wntr
except ImportError:
    print("ERROR: required packages missing.")
    print("Run: py -m pip install wntr==1.5.0 pandas numpy")
    sys.exit(1)

EXPECTED_WNTR = "1.5.0"
EXPECTED_NET3 = "ea3e825c4fef0b5cba47fb06301bc85253f18b6364dc96c44d9fb492c40faa52"

EXPECTED_FROZEN = {
    "benchmark_protocol_v0_3.md": "cbdc2a46945b2ddb71e586d9507819f76188839a3e93c5f3f580b9a400942632",
    "sensor_layout.json": "cbfe1e308eef967a16bdbec449fa1cdf6b39dd0fa47374a0cb42876a17ed7ac7",
    "leak_pipe_pools.json": "e32f408a1194642413d92e285d94a1a85c94b9587099fe508b5020a2fa6238bc",
    "scenario_manifest.csv": "ee818977fd10cc82e384a12a294389231336d0fedee9807ab26056ca409b5490",
    "corruption_config.json": "37f648b414d12c7a845ecea784376ece85d5a3c9417c4b585e5c96eb287c5fbd",
    "benchmark_design.json": "f2929d390749ef57f7b423d1c0a8cbba19e7ca0cc4c3469341c0b06598e58a5d",
}

EPS = 1e-12

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_file(path: Path, expected: str):
    if not path.exists():
        raise SystemExit(f"Missing frozen artifact: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"FROZEN ARTIFACT HASH MISMATCH: {path.name}\n"
            f"expected={expected}\n"
            f"got={got}"
        )
    print(f"[OK] {path.name}")

def verify_environment(net3: Path, frozen: Path):
    if getattr(wntr, "__version__", "unknown") != EXPECTED_WNTR:
        raise SystemExit(
            f"WNTR VERSION MISMATCH: expected {EXPECTED_WNTR}, "
            f"got {getattr(wntr, '__version__', 'unknown')}"
        )
    if sha256(net3) != EXPECTED_NET3:
        raise SystemExit("Net3 hash mismatch.")
    print(f"[OK] WNTR {EXPECTED_WNTR}")
    print(f"[OK] Net3 {EXPECTED_NET3}")
    for name, h in EXPECTED_FROZEN.items():
        verify_file(frozen / name, h)

def configure_network(wn):
    # Full benchmark generation semantics.
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 300

    # IMPORTANT: preserve Net3's native pattern timestep loaded from the INP.
    # The smoke test changed this only for smoke-test validation; v0.3 did not
    # freeze a replacement pattern timestep.
    native_pattern_timestep = int(wn.options.time.pattern_timestep)

    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.minimum_pressure = 0.0
    wn.options.hydraulic.required_pressure = 15.0
    return native_pattern_timestep

def apply_demand_context(wn, global_mult, per_junction_seed):
    rng = np.random.default_rng(int(per_junction_seed))
    junctions = list(wn.junction_name_list)
    local = rng.lognormal(mean=0.0, sigma=0.05, size=len(junctions))
    local = np.clip(local, 0.85, 1.15)

    factors = {}
    for jname, lm in zip(junctions, local):
        factor = float(global_mult) * float(lm)
        factors[str(jname)] = factor
        junction = wn.get_node(jname)

        # Preserve all original demand patterns; scale only their base values.
        for demand_ts in junction.demand_timeseries_list:
            demand_ts.base_value = float(demand_ts.base_value) * factor
    return factors

def run_clean(net3, row):
    wn = wntr.network.WaterNetworkModel(str(net3))
    native_pattern = configure_network(wn)
    factors = apply_demand_context(
        wn,
        float(row["global_demand_multiplier"]),
        int(row["per_junction_demand_seed"])
    )
    sim = wntr.sim.WNTRSimulator(wn)
    res = sim.run_sim()
    return wn, res, factors, native_pattern

def run_leak(net3, row):
    wn = wntr.network.WaterNetworkModel(str(net3))
    native_pattern = configure_network(wn)
    factors = apply_demand_context(
        wn,
        float(row["global_demand_multiplier"]),
        int(row["per_junction_demand_seed"])
    )

    pipe_name = str(row["paired_leak_pipe"])
    if pipe_name not in wn.pipe_name_list:
        raise ValueError(f"Frozen leak pipe not found: {pipe_name}")

    new_pipe = f"{pipe_name}_{row['context_id']}_B"
    leak_node_name = f"{pipe_name}_{row['context_id']}_LEAKNODE"

    wn = wntr.morph.split_pipe(
        wn,
        pipe_name,
        new_pipe,
        leak_node_name,
        split_at_point=0.5,
        return_copy=True,
    )

    leak_node = wn.get_node(leak_node_name)
    start_s = int(float(row["leak_start_hour"]) * 3600)
    end_s = int(float(row["leak_end_hour"]) * 3600)
    leak_node.add_leak(
        wn,
        area=float(row["leak_area_m2"]),
        discharge_coeff=0.75,
        start_time=start_s,
        end_time=end_s,
    )

    sim = wntr.sim.WNTRSimulator(wn)
    res = sim.run_sim()
    return wn, res, factors, native_pattern, leak_node_name, new_pipe

def finite(name, arr):
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name} contains non-finite values, first={bad[:5].tolist()}")

def extract_original_outputs(res, original_nodes, original_links, sensors):
    pressure = res.node["pressure"]
    flow = res.link["flowrate"]

    missing_nodes = [n for n in original_nodes if n not in pressure.columns]
    missing_links = [e for e in original_links if e not in flow.columns]
    if missing_nodes:
        raise ValueError(f"Missing original pressure nodes: {missing_nodes[:10]}")
    if missing_links:
        raise ValueError(f"Missing original flow links: {missing_links[:10]}")

    times = pressure.index.to_numpy(dtype=np.int64)
    if len(times) != 289:
        raise ValueError(f"Expected 289 reported timestamps, got {len(times)}")
    if not np.array_equal(times, np.arange(0, 24*3600 + 1, 300, dtype=np.int64)):
        raise ValueError("Unexpected report timestamps.")

    full_pressure = pressure[original_nodes].to_numpy(dtype=np.float32)
    full_flow = flow[original_links].to_numpy(dtype=np.float32)

    p_ids = sensors["pressure_sensor_nodes"]
    f_ids = sensors["flow_sensor_pipes"]
    t_ids = sensors["tank_level_nodes"]

    sensor_pressure = pressure[p_ids].to_numpy(dtype=np.float32)
    sensor_flow = flow[f_ids].to_numpy(dtype=np.float32)
    # Tank pressure is water level above tank elevation in EPANET/WNTR.
    tank_level = pressure[t_ids].to_numpy(dtype=np.float32)

    sensor_matrix = np.concatenate(
        [sensor_pressure, sensor_flow, tank_level], axis=1
    ).astype(np.float32)

    finite("full_pressure", full_pressure)
    finite("full_flow", full_flow)
    finite("sensor_matrix", sensor_matrix)

    return times, full_pressure, full_flow, sensor_matrix

def build_active_label(times, start_hour, end_hour):
    start_s = int(float(start_hour) * 3600)
    end_s = int(float(end_hour) * 3600)
    # Frozen convention: active on [start, end), inactive at exact end timestamp.
    return ((times >= start_s) & (times < end_s)).astype(np.uint8)

def save_context(out_dir, row, sensors, original_nodes, original_links,
                 clean_res, leak_res, leak_node_name, new_pipe,
                 native_pattern_timestep):
    context_id = str(row["context_id"])
    npz_path = out_dir / f"{context_id}.npz"
    meta_path = out_dir / f"{context_id}.json"

    ct, cp, cf, cs = extract_original_outputs(
        clean_res, original_nodes, original_links, sensors
    )
    lt, lp, lf, ls = extract_original_outputs(
        leak_res, original_nodes, original_links, sensors
    )
    if not np.array_equal(ct, lt):
        raise ValueError(f"{context_id}: clean/leak timestamp mismatch")

    active_clean = np.zeros_like(ct, dtype=np.uint8)
    active_leak = build_active_label(
        lt, row["leak_start_hour"], row["leak_end_hour"]
    )

    pressure_delta = np.abs(lp.astype(np.float64) - cp.astype(np.float64))
    max_pressure_delta = float(pressure_delta.max())
    mean_pressure_delta = float(pressure_delta.mean())
    if max_pressure_delta <= EPS:
        raise ValueError(f"{context_id}: leak has no measurable pressure effect.")

    leak_demand_max = None
    if "leak_demand" in leak_res.node:
        ld = leak_res.node["leak_demand"]
        if leak_node_name in ld.columns:
            leak_demand_max = float(ld[leak_node_name].max())
            if leak_demand_max <= EPS:
                raise ValueError(f"{context_id}: non-positive max leak demand.")

    np.savez_compressed(
        npz_path,
        time_s=ct,
        clean_sensor=cs,
        leak_sensor=ls,
        clean_full_pressure=cp,
        leak_full_pressure=lp,
        clean_full_flow=cf,
        leak_full_flow=lf,
        clean_active_leak=active_clean,
        leak_active_leak=active_leak,
    )

    meta = {
        "context_id": context_id,
        "split": str(row["split"]),
        "clean_variant_id": str(row["clean_variant_id"]),
        "leak_variant_id": str(row["leak_variant_id"]),
        "context_seed": int(row["context_seed"]),
        "per_junction_demand_seed": int(row["per_junction_demand_seed"]),
        "global_demand_multiplier": float(row["global_demand_multiplier"]),
        "leak_pipe": str(row["paired_leak_pipe"]),
        "leak_pipe_diameter_m": float(row["paired_leak_pipe_diameter_m"]),
        "leak_diameter_fraction": float(row["leak_diameter_fraction"]),
        "leak_area_m2": float(row["leak_area_m2"]),
        "leak_start_hour": int(row["leak_start_hour"]),
        "leak_duration_hour": int(row["leak_duration_hour"]),
        "leak_end_hour": int(row["leak_end_hour"]),
        "leak_node_name": leak_node_name,
        "split_pipe_new_segment": new_pipe,
        "native_pattern_timestep_seconds": int(native_pattern_timestep),
        "report_timestep_seconds": 300,
        "active_label_interval_convention": "[start_time, end_time)",
        "max_abs_pressure_difference": max_pressure_delta,
        "mean_abs_pressure_difference": mean_pressure_delta,
        "max_leak_demand": leak_demand_max,
        "npz_sha256": sha256(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def graph_artifacts(wn, out):
    node_rows = []
    for n in wn.node_name_list:
        node = wn.get_node(n)
        row = {
            "node_id": str(n),
            "node_type": node.node_type,
            "elevation": float(getattr(node, "elevation", np.nan)),
        }
        node_rows.append(row)

    link_rows = []
    for lname in wn.link_name_list:
        link = wn.get_link(lname)
        link_rows.append({
            "link_id": str(lname),
            "link_type": link.link_type,
            "start_node": str(link.start_node_name),
            "end_node": str(link.end_node_name),
            "length": float(getattr(link, "length", np.nan)),
            "diameter": float(getattr(link, "diameter", np.nan)),
            "roughness": float(getattr(link, "roughness", np.nan)),
        })

    pd.DataFrame(node_rows).to_csv(out / "graph_nodes.csv", index=False)
    pd.DataFrame(link_rows).to_csv(out / "graph_links.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net3", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net3 = Path(args.net3)
    frozen = Path(args.frozen)
    out = Path(args.out)
    contexts_dir = out / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA FULL SCENARIO GENERATION v0.3a")
    print("=" * 78)
    verify_environment(net3, frozen)

    manifest = pd.read_csv(frozen / "scenario_manifest.csv")
    if len(manifest) != 240:
        raise SystemExit(f"Expected 240 base contexts, got {len(manifest)}")

    sensors = json.loads((frozen / "sensor_layout.json").read_text(encoding="utf-8"))

    base_wn = wntr.network.WaterNetworkModel(str(net3))
    native_pattern = int(base_wn.options.time.pattern_timestep)
    original_nodes = list(map(str, base_wn.node_name_list))
    original_links = list(map(str, base_wn.link_name_list))
    graph_artifacts(base_wn, out)

    layout = {
        "original_node_ids": original_nodes,
        "original_link_ids": original_links,
        "pressure_sensor_ids": sensors["pressure_sensor_nodes"],
        "flow_sensor_ids": sensors["flow_sensor_pipes"],
        "tank_level_ids": sensors["tank_level_nodes"],
        "sensor_feature_order": (
            [f"pressure::{x}" for x in sensors["pressure_sensor_nodes"]] +
            [f"flow::{x}" for x in sensors["flow_sensor_pipes"]] +
            [f"tank_level::{x}" for x in sensors["tank_level_nodes"]]
        ),
        "native_net3_pattern_timestep_seconds": native_pattern,
        "generation_semantics": {
            "duration_seconds": 86400,
            "hydraulic_timestep_seconds": 300,
            "report_timestep_seconds": 300,
            "pattern_timestep": "preserved from Net3 INP",
            "demand_model": "PDD",
            "minimum_pressure_m": 0.0,
            "required_pressure_m": 15.0,
            "active_leak_interval": "[start,end)",
        },
    }
    (out / "tensor_layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")

    records = []
    start_wall = time.time()

    for idx, row in manifest.iterrows():
        context_id = str(row["context_id"])
        npz_path = contexts_dir / f"{context_id}.npz"
        meta_path = contexts_dir / f"{context_id}.json"

        # Resume-safe skip only if metadata hash matches current NPZ.
        if npz_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("npz_sha256") == sha256(npz_path):
                    print(f"[SKIP {idx+1:03d}/240] {context_id} already verified")
                    records.append(meta)
                    continue
            except Exception:
                pass

        print(f"[RUN  {idx+1:03d}/240] {context_id} split={row['split']} pipe={row['paired_leak_pipe']}")
        clean_wn, clean_res, clean_factors, clean_pattern = run_clean(net3, row)
        leak_wn, leak_res, leak_factors, leak_pattern, leak_node, new_pipe = run_leak(net3, row)

        if clean_pattern != leak_pattern or clean_pattern != native_pattern:
            raise SystemExit(f"{context_id}: pattern timestep mismatch.")
        if clean_factors != leak_factors:
            raise SystemExit(f"{context_id}: clean/leak demand context mismatch.")

        meta = save_context(
            contexts_dir, row, sensors, original_nodes, original_links,
            clean_res, leak_res, leak_node, new_pipe, native_pattern
        )
        records.append(meta)

    if len(records) != 240:
        raise SystemExit(f"Generation record count mismatch: {len(records)}")

    index = pd.DataFrame(records)
    index = index.sort_values("context_id").reset_index(drop=True)
    index_path = out / "generation_index.csv"
    index.to_csv(index_path, index=False)

    # Final integrity checks.
    expected_splits = {"train":150, "validation":30, "calibration":30, "test":30}
    got_splits = index["split"].value_counts().to_dict()
    if got_splits != expected_splits:
        raise SystemExit(f"Generated split count mismatch: {got_splits}")

    if index["npz_sha256"].nunique() != 240:
        # Not strictly impossible for identical byte streams but unexpected enough to stop.
        raise SystemExit("Unexpected duplicate NPZ hashes among contexts.")

    max_diffs = index["max_abs_pressure_difference"].astype(float)
    if (max_diffs <= EPS).any():
        raise SystemExit("One or more leak scenarios has no measurable pressure effect.")

    # Save immutable generation record.
    core_hashes = {
        "generation_index.csv": sha256(index_path),
        "tensor_layout.json": sha256(out / "tensor_layout.json"),
        "graph_nodes.csv": sha256(out / "graph_nodes.csv"),
        "graph_links.csv": sha256(out / "graph_links.csv"),
    }

    record = {
        "status": "GENERATED",
        "generator_version": "0.3a",
        "wntr_version": EXPECTED_WNTR,
        "net3_sha256": EXPECTED_NET3,
        "frozen_benchmark_artifacts": EXPECTED_FROZEN,
        "base_contexts": 240,
        "hydraulic_simulations": 480,
        "native_pattern_timestep_seconds": native_pattern,
        "split_counts": got_splits,
        "core_artifact_sha256": core_hashes,
        "context_npz_count": 240,
        "context_npz_hashes_sha256_of_index": sha256(index_path),
        "wall_seconds": time.time() - start_wall,
    }
    record_path = out / "GENERATION_RECORD.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Compact human summary.
    summary_lines = [
        "AQUA-VERA FULL SCENARIO GENERATION v0.3a",
        "=" * 78,
        "STATUS: PASS / GENERATED",
        f"WNTR version: {EXPECTED_WNTR}",
        f"Net3 SHA-256: {EXPECTED_NET3}",
        f"native Net3 pattern timestep (s): {native_pattern}",
        "",
        "COUNTS",
        "  base contexts: 240",
        "  hydraulic simulations: 480",
        f"  splits: {got_splits}",
        "  context NPZ files: 240",
        "",
        "OUTPUT SHAPES PER CONTEXT",
        f"  timestamps: 289",
        f"  model-visible sensor features: {len(layout['sensor_feature_order'])}",
        f"  full original-node pressure features: {len(original_nodes)}",
        f"  full original-link flow features: {len(original_links)}",
        "",
        "LEAK EFFECT CHECK",
        f"  minimum max-abs pressure effect across contexts: {max_diffs.min():.8g}",
        f"  median max-abs pressure effect: {max_diffs.median():.8g}",
        f"  maximum max-abs pressure effect: {max_diffs.max():.8g}",
        "",
        "CORE ARTIFACT HASHES",
    ]
    for k,v in core_hashes.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines += [
        "",
        "NEXT",
        "  Retain generation_summary.txt with the project record.",
        "  Do not train yet; the next gate constructs fixed 2-hour model windows",
        "  and training-only normalization statistics from these generated contexts."
    ]
    (out / "generation_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nGENERATION COMPLETE")
    print(f"Summary: {out / 'generation_summary.txt'}")
    print(f"Record:  {record_path}")

if __name__ == "__main__":
    main()
