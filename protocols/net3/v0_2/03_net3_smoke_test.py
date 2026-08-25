
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import wntr
except ImportError:
    print("ERROR: required packages are missing.")
    print("Run: py -m pip install wntr pandas numpy")
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure(wn):
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 300
    wn.options.time.pattern_timestep = 300
    wn.options.hydraulic.demand_model = "PDD"
    # Explicit non-default pressures, recorded for smoke test only.
    # Full benchmark values will be frozen after this smoke test.
    wn.options.hydraulic.minimum_pressure = 0.0
    wn.options.hydraulic.required_pressure = 15.0
    return wn


def run_clean(inp):
    wn = wntr.network.WaterNetworkModel(str(inp))
    configure(wn)
    sim = wntr.sim.WNTRSimulator(wn)
    res = sim.run_sim()
    return wn, res


def run_leak(inp, pipe_name="123", area=0.0005):
    wn = wntr.network.WaterNetworkModel(str(inp))
    configure(wn)

    if pipe_name not in wn.pipe_name_list:
        # deterministic fallback: first ordinary pipe
        pipe_name = sorted(wn.pipe_name_list)[0]

    new_pipe = f"{pipe_name}_SMOKE_B"
    leak_node_name = f"{pipe_name}_SMOKE_NODE"
    wn = wntr.morph.split_pipe(
        wn,
        pipe_name,
        new_pipe,
        leak_node_name,
        split_at_point=0.5,
        return_copy=True,
    )
    leak_node = wn.get_node(leak_node_name)
    leak_node.add_leak(
        wn,
        area=area,
        discharge_coeff=0.75,
        start_time=8*3600,
        end_time=16*3600,
    )
    sim = wntr.sim.WNTRSimulator(wn)
    res = sim.run_sim()
    return wn, res, pipe_name, leak_node_name


def dataframe_ok(df):
    arr = df.to_numpy(dtype=float)
    return np.isfinite(arr).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net3", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp = Path(args.net3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Net3 file not found: {inp}")

    print("AQUA-VERA WNTR/NET3 SMOKE TEST")
    print("=" * 72)
    print(f"WNTR version: {getattr(wntr, '__version__', 'unknown')}")
    print(f"Net3 SHA-256: {sha256(inp)}")

    print("\n[1/2] Running clean 24h PDD simulation...")
    clean_wn, clean = run_clean(inp)

    print("[2/2] Running matched pipe-leak simulation...")
    leak_wn, leak, pipe_name, leak_node_name = run_leak(inp)

    cp = clean.node["pressure"]
    lp = leak.node["pressure"]

    # Compare only node names shared between clean and leak networks.
    shared_nodes = sorted(set(cp.columns).intersection(lp.columns))
    cp2 = cp[shared_nodes]
    lp2 = lp[shared_nodes]

    if not dataframe_ok(cp2):
        raise SystemExit("Clean pressure results contain non-finite values.")
    if not dataframe_ok(lp2):
        raise SystemExit("Leak pressure results contain non-finite values.")

    common_times = cp2.index.intersection(lp2.index)
    cp2 = cp2.loc[common_times]
    lp2 = lp2.loc[common_times]

    abs_diff = (lp2 - cp2).abs()
    max_pressure_diff = float(abs_diff.to_numpy().max())
    mean_pressure_diff = float(abs_diff.to_numpy().mean())

    if max_pressure_diff <= 1e-8:
        raise SystemExit(
            "Leak scenario produced no measurable pressure difference. "
            "Do not proceed."
        )

    # Leak-demand output, if provided by simulator.
    leak_demand_max = None
    if "leak_demand" in leak.node:
        ld = leak.node["leak_demand"]
        if leak_node_name in ld.columns:
            leak_demand_max = float(ld[leak_node_name].max())

    # Save a compact sample, not all simulation data.
    sample_times = [0, 8*3600, 12*3600, 16*3600, 24*3600]
    sample_times = [t for t in sample_times if t in common_times]
    pressure_sample = pd.DataFrame({
        "time_s": sample_times
    })
    # Deterministic first ten shared nodes
    for n in shared_nodes[:10]:
        pressure_sample[f"clean_{n}"] = [float(cp2.loc[t, n]) for t in sample_times]
        pressure_sample[f"leak_{n}"] = [float(lp2.loc[t, n]) for t in sample_times]
    pressure_sample.to_csv(out / "pressure_smoke_sample.csv", index=False)

    info = {
        "status": "PASS",
        "wntr_version": getattr(wntr, "__version__", "unknown"),
        "net3_sha256": sha256(inp),
        "simulation_hours": 24,
        "report_timestep_seconds": 300,
        "demand_model": "PDD",
        "clean_nodes": int(len(clean_wn.node_name_list)),
        "clean_links": int(len(clean_wn.link_name_list)),
        "leak_pipe": pipe_name,
        "leak_node": leak_node_name,
        "leak_area_m2": 0.0005,
        "leak_start_hour": 8,
        "leak_end_hour": 16,
        "shared_pressure_nodes": len(shared_nodes),
        "pressure_rows": len(common_times),
        "max_abs_pressure_difference": max_pressure_diff,
        "mean_abs_pressure_difference": mean_pressure_diff,
        "max_leak_demand": leak_demand_max,
        "note": "Smoke-test parameters are not yet the frozen full-benchmark parameters."
    }

    (out / "smoke_test.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA WNTR/NET3 SMOKE TEST",
        "=" * 72,
        "STATUS: PASS",
        f"WNTR version: {info['wntr_version']}",
        f"Net3 SHA-256: {info['net3_sha256']}",
        f"clean nodes / links: {info['clean_nodes']} / {info['clean_links']}",
        f"pressure rows: {info['pressure_rows']}",
        f"leak pipe: {info['leak_pipe']}",
        f"leak node: {info['leak_node']}",
        f"leak area m2: {info['leak_area_m2']}",
        f"leak active: hour {info['leak_start_hour']} -> {info['leak_end_hour']}",
        f"max abs pressure difference: {info['max_abs_pressure_difference']:.8g}",
        f"mean abs pressure difference: {info['mean_abs_pressure_difference']:.8g}",
        f"max leak demand: {info['max_leak_demand']}",
        "",
        "NOTE:",
        "  This validates the simulator path only.",
        "  Scenario counts, leak distributions, sensor selection and seeds are",
        "  NOT frozen by this smoke test.",
        "",
        "NEXT:",
        "  Retain smoke_test_summary.txt with the project record.",
    ]
    (out / "smoke_test_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nSMOKE TEST PASS")
    print(f"Summary: {out / 'smoke_test_summary.txt'}")


if __name__ == "__main__":
    main()
