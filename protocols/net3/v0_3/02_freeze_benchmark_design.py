
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import networkx as nx
    import wntr
except ImportError:
    print("ERROR: wntr, numpy, pandas and networkx are required.")
    print("Run: py -m pip install wntr pandas numpy networkx")
    sys.exit(1)

EXPECTED_WNTR = "1.5.0"
EXPECTED_NET3 = "ea3e825c4fef0b5cba47fb06301bc85253f18b6364dc96c44d9fb492c40faa52"

SEED = 424242
N_CONTEXTS = 240
SPLIT_COUNTS = {
    "train": 150,
    "validation": 30,
    "calibration": 30,
    "test": 30,
}
PIPE_POOL_FRACTIONS = {
    "train": 0.60,
    "validation": 0.15,
    "calibration": 0.10,
    # test gets remainder
}

LEAK_DIAMETER_FRACTIONS = [0.10, 0.20, 0.30]
START_HOURS = [4, 6, 8, 10, 12]
DURATIONS = [6, 8, 10, 12]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_environment(net3):
    version = getattr(wntr, "__version__", "unknown")
    if version != EXPECTED_WNTR:
        raise SystemExit(
            f"WNTR VERSION MISMATCH: expected {EXPECTED_WNTR}, got {version}. "
            "Use the same WNTR version as the validated smoke test."
        )
    got = sha256(net3)
    if got != EXPECTED_NET3:
        raise SystemExit(
            f"Net3 HASH MISMATCH\nexpected={EXPECTED_NET3}\ngot={got}"
        )
    print(f"[OK] WNTR {version}")
    print(f"[OK] Net3 SHA-256 {got}")

def build_simple_graph(wn):
    G = nx.Graph()
    for n in wn.node_name_list:
        G.add_node(str(n))
    for lname in wn.link_name_list:
        link = wn.get_link(lname)
        u = str(link.start_node_name)
        v = str(link.end_node_name)
        if u != v:
            G.add_edge(u, v, name=str(lname))
    return G

def farthest_pressure_sensors(wn, k=24):
    G = build_simple_graph(wn)
    junctions = sorted(map(str, wn.junction_name_list))
    if len(junctions) < k:
        raise SystemExit(f"Only {len(junctions)} junctions, cannot select {k} pressure sensors.")

    degrees = {n: G.degree(n) for n in junctions}
    first = sorted(junctions, key=lambda n: (-degrees[n], n))[0]
    selected = [first]

    # Precompute shortest path lengths from candidate junctions.
    lengths = dict(nx.all_pairs_shortest_path_length(G))

    while len(selected) < k:
        best = None
        best_dist = -1
        for n in junctions:
            if n in selected:
                continue
            d = min(lengths[n].get(s, 10**9) for s in selected)
            if d > best_dist or (d == best_dist and (best is None or n < best)):
                best = n
                best_dist = d
        selected.append(best)
    return selected

def flow_sensors(wn, k=6):
    # Use edge betweenness on a MultiGraph keyed by link name when possible.
    MG = nx.MultiGraph()
    for n in wn.node_name_list:
        MG.add_node(str(n))
    for lname in wn.pipe_name_list:
        link = wn.get_link(lname)
        MG.add_edge(
            str(link.start_node_name),
            str(link.end_node_name),
            key=str(lname),
            name=str(lname),
        )
    try:
        bc = nx.edge_betweenness_centrality(MG, normalized=True)
        scored = []
        for edge_key, score in bc.items():
            # MultiGraph key tuple expected (u,v,key)
            if len(edge_key) == 3:
                lname = str(edge_key[2])
                if lname in wn.pipe_name_list:
                    scored.append((float(score), lname))
        scored.sort(key=lambda x: (-x[0], x[1]))
        names = []
        for _, lname in scored:
            if lname not in names:
                names.append(lname)
            if len(names) == k:
                break
        if len(names) == k:
            return names, "edge_betweenness"
    except Exception:
        pass

    # Deterministic fallback: largest diameter, then lexical name.
    vals = []
    for lname in wn.pipe_name_list:
        pipe = wn.get_link(lname)
        vals.append((float(pipe.diameter), str(lname)))
    vals.sort(key=lambda x: (-x[0], x[1]))
    return [x[1] for x in vals[:k]], "diameter_fallback"

def eligible_leak_pipes(wn):
    out = []
    for lname in wn.pipe_name_list:
        p = wn.get_link(lname)
        d = float(p.diameter)
        if d <= 0.9144:
            out.append((str(lname), d))
    out.sort(key=lambda x: x[0])
    if len(out) < 20:
        raise SystemExit(f"Too few eligible leak pipes: {len(out)}")
    return out

def partition_pipes(pipe_names):
    rng = random.Random(SEED + 11)
    names = list(pipe_names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(math.floor(n * PIPE_POOL_FRACTIONS["train"]))
    n_val = int(math.floor(n * PIPE_POOL_FRACTIONS["validation"]))
    n_cal = int(math.floor(n * PIPE_POOL_FRACTIONS["calibration"]))
    pools = {}
    pools["train"] = names[:n_train]
    pools["validation"] = names[n_train:n_train+n_val]
    pools["calibration"] = names[n_train+n_val:n_train+n_val+n_cal]
    pools["test"] = names[n_train+n_val+n_cal:]
    if any(len(v) == 0 for v in pools.values()):
        raise SystemExit(f"Empty pipe pool: { {k: len(v) for k,v in pools.items()} }")
    # Hard disjointness check.
    keys = list(pools)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if set(pools[a]) & set(pools[b]):
                raise SystemExit(f"Pipe pool overlap: {a} vs {b}")
    return pools

def valid_start_duration(rng):
    pairs = [(s,d) for s in START_HOURS for d in DURATIONS if s+d <= 24]
    return rng.choice(pairs)

def build_manifest(wn, pools):
    rng = random.Random(SEED + 101)
    pipe_diam = {str(n): float(wn.get_link(n).diameter) for n in wn.pipe_name_list}

    split_sequence = []
    for split, count in SPLIT_COUNTS.items():
        split_sequence.extend([split] * count)
    rng.shuffle(split_sequence)

    # Maintain split-local cyclic pipe order for better coverage.
    split_pipe_cycles = {}
    split_pipe_pos = {}
    for split, pool in pools.items():
        p = list(pool)
        random.Random(SEED + 1000 + sum(map(ord, split))).shuffle(p)
        split_pipe_cycles[split] = p
        split_pipe_pos[split] = 0

    rows = []
    for i, split in enumerate(split_sequence, start=1):
        scenario_seed = SEED * 100000 + i
        srng = random.Random(scenario_seed)

        cycle = split_pipe_cycles[split]
        pos = split_pipe_pos[split]
        leak_pipe = cycle[pos % len(cycle)]
        split_pipe_pos[split] += 1

        global_mult = srng.uniform(0.85, 1.15)
        per_junction_seed = scenario_seed + 17
        frac = srng.choice(LEAK_DIAMETER_FRACTIONS)
        start, duration = valid_start_duration(srng)
        end = start + duration
        diameter = pipe_diam[leak_pipe]
        leak_diameter = diameter * frac
        leak_area = math.pi * (leak_diameter / 2.0) ** 2

        rows.append({
            "context_id": f"C{i:04d}",
            "split": split,
            "context_seed": scenario_seed,
            "per_junction_demand_seed": per_junction_seed,
            "global_demand_multiplier": global_mult,
            "paired_leak_pipe": leak_pipe,
            "paired_leak_pipe_diameter_m": diameter,
            "leak_diameter_fraction": frac,
            "leak_area_m2": leak_area,
            "leak_start_hour": start,
            "leak_duration_hour": duration,
            "leak_end_hour": end,
            "clean_variant_id": f"C{i:04d}_CLEAN",
            "leak_variant_id": f"C{i:04d}_LEAK",
        })
    df = pd.DataFrame(rows)

    # Exact split counts.
    got = df["split"].value_counts().to_dict()
    if got != SPLIT_COUNTS:
        raise SystemExit(f"Split count mismatch: {got}")

    # Verify each split uses only its pipe pool and pools are disjoint.
    for split, pool in pools.items():
        used = set(df.loc[df["split"] == split, "paired_leak_pipe"])
        if not used.issubset(set(pool)):
            raise SystemExit(f"{split} uses pipe outside its frozen pool.")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net3", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net3 = Path(args.net3)
    protocol = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verify_environment(net3)

    wn = wntr.network.WaterNetworkModel(str(net3))
    pressure = farthest_pressure_sensors(wn, 24)
    flow, flow_method = flow_sensors(wn, 6)
    tanks = sorted(map(str, wn.tank_name_list))

    sensor_layout = {
        "pressure_sensor_nodes": pressure,
        "pressure_sensor_count": len(pressure),
        "pressure_selection": "deterministic farthest-point shortest-path sampling",
        "tank_level_nodes": tanks,
        "tank_sensor_count": len(tanks),
        "flow_sensor_pipes": flow,
        "flow_sensor_count": len(flow),
        "flow_selection": flow_method,
    }

    elig = eligible_leak_pipes(wn)
    eligible_names = [x[0] for x in elig]
    pools = partition_pipes(eligible_names)

    manifest = build_manifest(wn, pools)

    # Save artifacts.
    protocol_copy = out / "benchmark_protocol_v0_3.md"
    protocol_copy.write_text(protocol.read_text(encoding="utf-8"), encoding="utf-8")

    sensor_path = out / "sensor_layout.json"
    sensor_path.write_text(json.dumps(sensor_layout, indent=2), encoding="utf-8")

    pools_path = out / "leak_pipe_pools.json"
    pools_path.write_text(json.dumps({
        "eligibility_rule": "ordinary pipe diameter <= 0.9144 m",
        "eligible_pipe_count": len(eligible_names),
        "pools": pools,
        "pool_counts": {k: len(v) for k,v in pools.items()},
    }, indent=2), encoding="utf-8")

    manifest_path = out / "scenario_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    corruption = {
        "MISSING_SENSOR": {"low": 0.10, "medium": 0.25, "high": 0.40},
        "SENSOR_NOISE_SD": {"low": 0.25, "medium": 0.50, "high": 1.00},
        "SENSOR_BIAS_SD": {"low": 0.50, "medium": 1.00, "high": 2.00},
        "STALE_SENSOR_MINUTES": {"low": 30, "medium": 60, "high": 120},
        "TOPOLOGY_MISMATCH_REMOVED_EDGES": {"low": 1, "medium": 3, "high": 5},
        "CROSS_SOURCE_CONFLICT_PRESSURE_FRACTION": {"low": 0.10, "medium": 0.25, "high": 0.40},
        "threshold_source": "training/calibration only",
        "note": "Degradations alter model-visible evidence, not physical ground truth."
    }
    corruption_path = out / "corruption_config.json"
    corruption_path.write_text(json.dumps(corruption, indent=2), encoding="utf-8")

    design = {
        "status": "FROZEN_BENCHMARK",
        "seed": SEED,
        "wntr_version": EXPECTED_WNTR,
        "net3_sha256": EXPECTED_NET3,
        "base_contexts": N_CONTEXTS,
        "physical_variants_per_context": ["CLEAN", "LEAK"],
        "total_hydraulic_simulations": N_CONTEXTS * 2,
        "split_counts_base_contexts": SPLIT_COUNTS,
        "scenario_hours": 24,
        "timestep_seconds": 300,
        "demand_model": "PDD",
        "required_pressure_m": 15.0,
        "minimum_pressure_m": 0.0,
        "global_demand_multiplier_range": [0.85, 1.15],
        "per_junction_lognormal_sigma": 0.05,
        "per_junction_multiplier_clip": [0.85, 1.15],
        "leak_diameter_fractions": LEAK_DIAMETER_FRACTIONS,
        "leak_start_hours": START_HOURS,
        "leak_durations_hours": DURATIONS,
        "leak_discharge_coefficient": 0.75,
        "pressure_sensors": len(pressure),
        "flow_sensors": len(flow),
        "tank_sensors": len(tanks),
        "eligible_leak_pipes": len(eligible_names),
        "pipe_pool_counts": {k: len(v) for k,v in pools.items()},
        "test_pipe_pool_disjoint_from_train": True,
    }
    design_path = out / "benchmark_design.json"
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")

    core = [protocol_copy, sensor_path, pools_path, manifest_path, corruption_path, design_path]
    hashes = {p.name: sha256(p) for p in core}

    freeze = {
        "status": "FROZEN_BENCHMARK",
        "artifact_sha256": hashes,
        "net3_sha256": EXPECTED_NET3,
        "wntr_version": EXPECTED_WNTR,
        "note": "Frozen before full scenario generation or model training."
    }
    freeze_path = out / "FROZEN_BENCHMARK_RECORD.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA CONTROLLED BENCHMARK FREEZE v0.3",
        "=" * 78,
        "STATUS: PASS / FROZEN",
        f"WNTR version: {EXPECTED_WNTR}",
        f"Net3 SHA-256: {EXPECTED_NET3}",
        "",
        f"Base contexts: {N_CONTEXTS}",
        f"Matched variants/context: 2 (CLEAN + LEAK)",
        f"Total hydraulic simulations: {N_CONTEXTS*2}",
        f"Splits: {SPLIT_COUNTS}",
        "",
        "SENSORS",
        f"  pressure: {len(pressure)}",
        f"  flow: {len(flow)} ({flow_method})",
        f"  tank level: {len(tanks)}",
        f"  pressure IDs: {', '.join(pressure)}",
        f"  flow IDs: {', '.join(flow)}",
        f"  tank IDs: {', '.join(tanks)}",
        "",
        "LEAK PIPE POOLS",
        f"  eligible pipes: {len(eligible_names)}",
    ]
    for split in ["train","validation","calibration","test"]:
        lines.append(f"  {split}: {len(pools[split])} pipes")
    lines += [
        "  train/test pipe overlap: 0",
        "",
        "DESIGN",
        "  matched demand context for CLEAN/LEAK pair: YES",
        "  test leak locations unseen in train: YES",
        "  global demand multiplier: U[0.85, 1.15]",
        "  per-junction multiplier: lognormal sigma=0.05, clip [0.85,1.15]",
        "  leak diameter fraction: {0.10,0.20,0.30} x pipe diameter",
        "  leak start hours: {4,6,8,10,12}",
        "  leak durations: {6,8,10,12} h, constrained to end <=24 h",
        "",
        "ARTIFACT HASHES"
    ]
    for k,v in hashes.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        "NEXT",
        "  Retain benchmark_summary.txt with the project record.",
        "  Do not generate the 480 simulations until this freeze is reviewed."
    ]
    summary_path = out / "benchmark_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nBENCHMARK FREEZE SUCCESS")
    print(f"Summary: {summary_path}")
    print(f"Freeze:  {freeze_path}")

if __name__ == "__main__":
    main()
