
import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx

H_PROTOCOL = "058bc380aac09ca5b6874fbd5d58155dfbdebb4e499dd608209358d0ebeb525d"
H_GRAPH_INPUT = "89bd64f6758b46af9ba39ea575ab78ee0da042c94a6b706855c89e69141411ff"
H_MLP = "d3a11328c1884481d10177bf624b542e2be0882760e74bb137d6ca28ac385f08"
H_GCN = "da1b7d0365d2b2a97a7b9b1b9795767163d7a90daaa979e973aa67e907522358"
H_THRESHOLDS = "243ae4956479e7cb244b3a0f877b8826263ee7cd54cda517b20478e6dc6050b2"
H_CENTROIDS = "c017d1e933b4378b576d02f13a3a2f4c98c7275fb99ae28e895bdd9e92f6df46"
H_RECON = "49b2983f040c3e282cb6feddbb4f1b3c2ce326b29d9c4d0da11ae7d460cf2ce7"
H_NORM = "44cab3f6ef97752245bdce42d1cce70db929b6c1325b6e979650067da3f1ee0f"
H_L_TOWN = "a7551b86745f4cc3433c78e60023077fa1386947d4d35372ffd3a0405622b436"

MISSING = {"low": 3, "medium": 8, "high": 13}
TOPOLOGY = {"low": 1, "medium": 3, "high": 5}
FIXED_PRESSURE_COUNT = 8
SEVERITIES = ["low", "medium", "high"]


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
            f"{label} HASH MISMATCH\nexpected={expected}\ngot={got}"
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
            f"Expected one {filename} under {root}, found {len(matches)}"
        )
    return matches[0]


def digest(*parts):
    return hashlib.sha256(
        "|".join(str(x) for x in parts).encode("utf-8")
    ).hexdigest()


def uint63(*parts):
    return int(digest(*parts)[:16], 16) % (2**63 - 1)


def rank_ids(family, ids):
    return sorted(
        ids,
        key=lambda x: digest(H_PROTOCOL, family, x),
    )


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

    nodes = []
    for sec in ["[JUNCTIONS]", "[RESERVOIRS]", "[TANKS]"]:
        for row in sections.get(sec, []):
            p = row.split()
            if p:
                nodes.append(p[0])

    links = []
    for sec, typ in [
        ("[PIPES]", "pipe"),
        ("[PUMPS]", "pump"),
        ("[VALVES]", "valve"),
    ]:
        for row in sections.get(sec, []):
            p = row.split()
            if len(p) >= 3:
                links.append({
                    "link_id": p[0],
                    "u": p[1],
                    "v": p[2],
                    "type": typ,
                })
    return nodes, links


def select_topology(nodes, links):
    ordinary = [x for x in links if x["type"] == "pipe"]
    ranked = rank_ids(
        "TOPOLOGY_MISMATCH",
        [x["link_id"] for x in ordinary],
    )
    by_id = {x["link_id"]: x for x in links}
    active = list(links)
    chosen = []

    for lid in ranked:
        if len(chosen) >= 5:
            break

        r = by_id[lid]

        # Do not remove an edge when another parallel link remains between the
        # same endpoints; such a removal is graph-topologically invisible.
        if any(
            x["link_id"] != lid and {x["u"], x["v"]} == {r["u"], r["v"]}
            for x in active
        ):
            continue

        trial = [x for x in active if x["link_id"] != lid]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from((x["u"], x["v"]) for x in trial)

        if nx.is_connected(G):
            chosen.append(lid)
            active = trial

    if len(chosen) != 5:
        raise SystemExit(
            f"Could freeze only {len(chosen)} connectivity-preserving "
            "ordinary-pipe removals; expected 5."
        )
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-freeze", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tf = Path(args.train_freeze)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA BATTLEDIM FINAL-EVALUATION IMPLEMENTATION FREEZE v0.16c")
    print("=" * 78)
    print("PRE-2019. This run does not open/check/hash any 2019 file.")
    print()

    verify(tf / "FROZEN_2018_NORMALIZATION.json", H_NORM, "2018 normalization")
    verify(tf / "FROZEN_LTown_GRAPH_INPUT.json", H_GRAPH_INPUT, "L-Town graph input")
    verify(tf / "BattLeDIM_MLP_v0_16b.pt", H_MLP, "frozen MLP")
    verify(tf / "BattLeDIM_GCN_GRU_v0_16b.pt", H_GCN, "frozen GCN-GRU")
    verify(
        tf / "FROZEN_2018_RELIABILITY_THRESHOLDS.json",
        H_THRESHOLDS,
        "frozen reliability thresholds",
    )
    verify(
        tf / "FROZEN_2018_ATTRIBUTION_CENTROIDS.json",
        H_CENTROIDS,
        "frozen attribution centroids",
    )
    verify(
        tf / "FROZEN_2018_PRESSURE_RECONSTRUCTION.json",
        H_RECON,
        "frozen pressure reconstruction",
    )

    inp = find_unique(data, "L-TOWN.inp")
    verify(inp, H_L_TOWN, "L-TOWN.inp")

    graph_input = json.loads(
        (tf / "FROZEN_LTown_GRAPH_INPUT.json").read_text(encoding="utf-8")
    )
    pressure_ids = [
        row["sensor_id"]
        for row in graph_input["pressure_mapping"]
    ]
    if len(pressure_ids) != 33 or len(set(pressure_ids)) != 33:
        raise SystemExit("Expected exactly 33 unique pressure sensor IDs")

    sensor_selection = {}

    ranked_missing = rank_ids("MISSING_SENSOR", pressure_ids)
    sensor_selection["MISSING_SENSOR"] = {
        severity: ranked_missing[:count]
        for severity, count in MISSING.items()
    }

    for fam in ["SENSOR_NOISE", "SENSOR_BIAS", "STALE_SENSOR"]:
        ranked = rank_ids(fam, pressure_ids)
        sensor_selection[fam] = {
            severity: ranked[:FIXED_PRESSURE_COUNT]
            for severity in SEVERITIES
        }

    nodes, links = parse_inp(inp)
    topo_rank = select_topology(nodes, links)
    topology_selection = {
        severity: topo_rank[:count]
        for severity, count in TOPOLOGY.items()
    }

    noise_seeds = {
        severity: uint63(H_PROTOCOL, "SENSOR_NOISE", severity)
        for severity in SEVERITIES
    }
    bias_sign = {
        sid: (
            1 if int(digest(H_PROTOCOL, "SENSOR_BIAS", sid)[-1], 16) % 2 == 0
            else -1
        )
        for sid in sensor_selection["SENSOR_BIAS"]["low"]
    }

    implementation = {
        "status": "FROZEN_PRE_2019_FINAL_EVALUATION",
        "version": "0.16c",
        "protocol_sha256": H_PROTOCOL,
        "2019_checked": False,
        "2019_opened": False,
        "2019_hashed": False,
        "corruption_selection_rule": (
            "SHA-256 ranking using frozen protocol hash + family + item ID; "
            "selection is global, label-independent and window-independent."
        ),
        "pressure_sensor_selection": sensor_selection,
        "topology_edge_selection": topology_selection,
        "noise_rng_seed_by_severity": noise_seeds,
        "bias_sign_by_selected_sensor": bias_sign,
        "noise_application": (
            "Corrupt the 2019 5-minute time series once per severity before "
            "window extraction, so overlapping windows see the same corrupted "
            "measurement at the same timestamp."
        ),
        "bias_application": (
            "Apply one persistent signed offset per selected sensor to the "
            "entire 2019 time series."
        ),
        "stale_application": (
            "Shift the selected factual pressure stream by 6/12/24 5-minute "
            "steps; initial unavailable samples use the first factual value."
        ),
        "missing_application": (
            "Set selected pressure values to frozen 2018 mean; GCN observation "
            "mask is zero for those sensors."
        ),
        "topology_application": (
            "Remove the frozen connectivity-preserving ordinary pipes from the "
            "model-visible GCN graph only; SCADA remains factual; model-visible "
            "degree is recomputed."
        ),
        "attribution_subset_rule": "candidate 2019 window_index mod 10 == 0, then require eligibility",
        "controller_reporting": {
            "raw_always_act_error": (
                "error of raw degraded model prediction before any prescribed "
                "missing-sensor reconstruction"
            ),
            "recovery_prescribed_forced_error": (
                "error after prescribed reconstruction for MISSING_SENSOR; "
                "identical to raw prediction for all other families"
            ),
            "unsafe_proceed": (
                "final controller action is PROCEED and final prediction is incorrect"
            ),
            "proceed_rate": "fraction with final controller action PROCEED",
        },
        "inference_policy": (
            "descriptive external evaluation; no post-2019 confidence intervals "
            "or threshold optimization are introduced."
        ),
        "primary_replication_questions": [
            "entropy evidence-failure alignment",
            "attribution divergence versus entropy across conditions",
            "architecture-dependent missing-sensor reconstruction",
            "controller unsafe-proceed reduction versus raw always-act",
        ],
    }

    out_json = out / "FROZEN_FINAL_EVAL_IMPLEMENTATION_v0_16c.json"
    out_json.write_text(json.dumps(implementation, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA BATTLEDIM FINAL-EVALUATION IMPLEMENTATION FREEZE v0.16c",
        "=" * 78,
        "STATUS: PASS / FINAL 2019 EVALUATION IMPLEMENTATION FROZEN",
        "",
        "BOUNDARY",
        "  2019 checked/opened/hashed: NO",
        "  frozen models changed: NO",
        "  frozen reliability thresholds changed: NO",
        "  frozen reconstruction changed: NO",
        "",
        "DETERMINISTIC PRESSURE SENSOR SELECTION",
    ]
    for fam in ["MISSING_SENSOR", "SENSOR_NOISE", "SENSOR_BIAS", "STALE_SENSOR"]:
        for severity in SEVERITIES:
            lines.append(
                f"  {fam} {severity}: "
                + ",".join(sensor_selection[fam][severity])
            )

    lines += [
        "",
        "TOPOLOGY MISMATCH EDGE SELECTION",
    ]
    for severity in SEVERITIES:
        lines.append(
            f"  {severity}: " + ",".join(topology_selection[severity])
        )

    lines += [
        "",
        "NOISE RNG SEEDS",
    ]
    for severity in SEVERITIES:
        lines.append(f"  {severity}: {noise_seeds[severity]}")

    lines += [
        "",
        "REPORTING DEFINITIONS",
        "  raw always-act error: raw degraded prediction error before recovery",
        "  recovery-prescribed forced error: after missing reconstruction only",
        "  unsafe proceed: final PROCEED and final prediction incorrect",
        "  attribution subset: candidate window_index mod 10 == 0 AND eligible",
        "  inference: descriptive; no post-2019 CI/threshold tuning",
        "",
        f"FROZEN IMPLEMENTATION SHA-256: {sha256(out_json)}",
        "",
        "NEXT",
        "  Retain BattLeDIM_final_eval_freeze_summary_v0_16c.txt with the project record.",
        "  If PASS, 2019 may be opened exactly once for final external evaluation.",
    ]

    summary = out / "BattLeDIM_final_eval_freeze_summary_v0_16c.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
