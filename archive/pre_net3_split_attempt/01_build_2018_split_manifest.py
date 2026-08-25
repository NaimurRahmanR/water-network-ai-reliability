
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required.")
    print("Run: py -m pip install pandas numpy")
    sys.exit(1)

EXPECTED_2018 = {
    "2018_scada_model_ready.csv":
        "00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e",
    "2018_leakages_model_ready.csv":
        "743ea7b2845b3dced2ada4fba1a54194c54fc9ce18bf7180a1b3afa4b3eaf2f8",
}
EXPECTED_PROTOCOL_HASH = "5ea530772b1b4df01c3a437afc10929d570a1cf28fbffcdbc03db9bd0459bc41"

WINDOW_STEPS = 24          # 2 hours at 5-minute cadence
STRIDE_STEPS = 6           # one candidate every 30 minutes
SEED = 271828
EPS = 1e-12

TARGET_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "calibration": 0.15,
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path: Path, expected: str):
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH: {path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] hash {path.name}")


def load_protocol_record(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing frozen protocol record: {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "FROZEN":
        raise SystemExit("Protocol record is not marked FROZEN.")
    got_hash = rec.get("protocol_sha256")
    if got_hash != EXPECTED_PROTOCOL_HASH:
        raise SystemExit(
            "Protocol hash does not match the frozen v0.1 protocol.\n"
            f"expected={EXPECTED_PROTOCOL_HASH}\n"
            f"got={got_hash}"
        )
    print(f"[OK] frozen protocol hash {got_hash}")
    return rec


def contiguous_true_intervals(mask, timestamps):
    """Return [(start_idx, end_idx, start_time, end_time)] inclusive."""
    arr = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(arr)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(idx) - 1]
    out = []
    for a, b in zip(starts, ends):
        sidx, eidx = int(idx[a]), int(idx[b])
        out.append((sidx, eidx, timestamps.iloc[sidx], timestamps.iloc[eidx]))
    return out


def merge_intervals(intervals):
    """
    intervals: list of dicts with start_idx/end_idx/event_ids
    Merge overlapping/touching buffered intervals.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x["start_idx"], x["end_idx"]))
    merged = [dict(ordered[0])]
    merged[0]["event_ids"] = set(merged[0]["event_ids"])
    for x in ordered[1:]:
        cur = merged[-1]
        if x["start_idx"] <= cur["end_idx"] + 1:
            cur["end_idx"] = max(cur["end_idx"], x["end_idx"])
            cur["event_ids"].update(x["event_ids"])
        else:
            y = dict(x)
            y["event_ids"] = set(y["event_ids"])
            merged.append(y)
    return merged


def greedy_assign(groups, weight_key, seed):
    """
    Deterministic greedy group assignment by weight.
    Keeps groups intact. Largest groups assigned first to the split
    with the largest remaining target deficit.
    """
    rng = random.Random(seed)
    groups = [dict(g) for g in groups]
    rng.shuffle(groups)
    groups.sort(key=lambda g: g[weight_key], reverse=True)

    total = sum(g[weight_key] for g in groups)
    targets = {k: total * v for k, v in TARGET_FRACTIONS.items()}
    current = {k: 0 for k in TARGET_FRACTIONS}
    assignment = {}

    # Ensure each split receives one group when possible.
    splits = list(TARGET_FRACTIONS)
    if len(groups) >= len(splits):
        for split, g in zip(splits, groups[:len(splits)]):
            assignment[g["group_id"]] = split
            current[split] += g[weight_key]
        remaining = groups[len(splits):]
    else:
        remaining = groups

    for g in remaining:
        deficits = {
            s: targets[s] - current[s]
            for s in TARGET_FRACTIONS
        }
        split = max(deficits, key=lambda s: (deficits[s], TARGET_FRACTIONS[s]))
        assignment[g["group_id"]] = split
        current[split] += g[weight_key]

    return assignment, current, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ready", required=True)
    ap.add_argument("--protocol-record", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Path(args.model_ready)
    protocol_record = Path(args.protocol_record)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA 2018 SPLIT FREEZE")
    print("=" * 72)

    load_protocol_record(protocol_record)
    for name, expected in EXPECTED_2018.items():
        require_hash(model / name, expected)

    scada_path = model / "2018_scada_model_ready.csv"
    leak_path = model / "2018_leakages_model_ready.csv"

    print("[READ] 2018 SCADA")
    scada = pd.read_csv(scada_path, parse_dates=["Timestamp"])
    print("[READ] 2018 leakage labels")
    leaks = pd.read_csv(leak_path, parse_dates=["Timestamp"])

    if len(scada) != 105120 or len(leaks) != 105120:
        raise SystemExit(
            f"Expected 105,120 rows; got SCADA={len(scada)}, leaks={len(leaks)}"
        )
    if not scada["Timestamp"].equals(leaks["Timestamp"]):
        raise SystemExit("SCADA/leakage timestamps are not identical.")

    ts = scada["Timestamp"].reset_index(drop=True)
    leak_cols = [c for c in leaks.columns if c != "Timestamp"]
    if not leak_cols:
        raise SystemExit("No leakage columns found.")

    leak_values = leaks[leak_cols].astype(float)

    # Scientific sanity checks on label files.
    if not np.isfinite(leak_values.to_numpy()).all():
        raise SystemExit("Leakage labels contain non-finite values.")
    min_value = float(leak_values.min().min())
    if min_value < -EPS:
        raise SystemExit(
            f"Negative leakage rate found ({min_value}). "
            "Active-leak rule (>0) requires review."
        )

    # Active leak at a timestamp = any reported leakage rate > 0.
    active_matrix = leak_values.to_numpy() > EPS
    active_any = active_matrix.any(axis=1)
    active_count = active_any.sum()

    if active_count == 0:
        raise SystemExit("No positive leakage timestamps found in 2018.")

    # Build a catalog of contiguous positive intervals per leak column.
    raw_events = []
    event_catalog_rows = []
    event_num = 0
    for j, col in enumerate(leak_cols):
        mask = active_matrix[:, j]
        intervals = contiguous_true_intervals(mask, ts)
        for local_i, (sidx, eidx, st, et) in enumerate(intervals, start=1):
            event_num += 1
            eid = f"E{event_num:03d}"
            vals = leak_values.iloc[sidx:eidx+1, j]
            event_catalog_rows.append({
                "event_id": eid,
                "leak_column": col,
                "start_idx": sidx,
                "end_idx": eidx,
                "start_time": st,
                "end_time": et,
                "duration_steps": eidx - sidx + 1,
                "duration_hours": (eidx - sidx + 1) * 5.0 / 60.0,
                "max_rate": float(vals.max()),
                "mean_positive_rate": float(vals[vals > EPS].mean()),
            })

            # Guard interval: any 2-hour window whose input could share rows
            # with the event neighborhood stays in this event group.
            guard = WINDOW_STEPS - 1
            raw_events.append({
                "start_idx": max(0, sidx - guard),
                "end_idx": min(len(ts)-1, eidx + guard),
                "event_ids": {eid},
            })

    event_catalog = pd.DataFrame(event_catalog_rows)
    if len(event_catalog) < 3:
        raise SystemExit(
            f"Only {len(event_catalog)} contiguous leak event(s) found. "
            "Cannot make event-group train/validation/calibration split."
        )

    # Merge event guard intervals that overlap, so shared raw rows can never
    # land in different splits.
    components = merge_intervals(raw_events)

    # Candidate windows use final indices spaced every 30 minutes.
    candidate_final_idxs = list(range(WINDOW_STEPS - 1, len(ts), STRIDE_STEPS))

    rows = []
    event_groups = []
    event_component_for_idx = np.full(len(ts), -1, dtype=int)

    for ci, comp in enumerate(components):
        event_component_for_idx[comp["start_idx"]:comp["end_idx"]+1] = ci
        gid = f"EVENT_COMPONENT_{ci+1:03d}"
        comp["group_id"] = gid

    # Event-associated windows:
    # final timestamp inside buffered event component. The buffer makes the
    # full observation neighborhoods isolated from other splits.
    for ci, comp in enumerate(components):
        gid = comp["group_id"]
        idxs = [
            i for i in candidate_final_idxs
            if comp["start_idx"] <= i <= comp["end_idx"]
        ]
        if not idxs:
            continue
        positives = int(active_any[idxs].sum())
        event_groups.append({
            "group_id": gid,
            "group_type": "event_component",
            "candidate_windows": len(idxs),
            "positive_windows": positives,
            "event_ids": ",".join(sorted(comp["event_ids"])),
            "start_time": str(ts.iloc[comp["start_idx"]]),
            "end_time": str(ts.iloc[comp["end_idx"]]),
        })

    if len(event_groups) < 3:
        raise SystemExit(
            f"After guard-zone merging there are only {len(event_groups)} "
            "independent event components. Split requires review."
        )

    # Positive-bearing event groups get assigned first.
    positive_groups = [g for g in event_groups if g["positive_windows"] > 0]
    if len(positive_groups) < 3:
        raise SystemExit(
            f"Only {len(positive_groups)} independent positive event groups remain. "
            "Cannot guarantee positives in all development splits."
        )

    event_assignment, event_current, event_targets = greedy_assign(
        positive_groups, "positive_windows", SEED
    )

    # Event groups with no positive target windows (pure guard windows) inherit
    # assignment by nearest positive component if possible; otherwise train.
    component_order = {g["group_id"]: k for k, g in enumerate(event_groups)}
    for g in event_groups:
        if g["group_id"] in event_assignment:
            continue
        k = component_order[g["group_id"]]
        neighbors = []
        for other in positive_groups:
            ok = component_order[other["group_id"]]
            neighbors.append((abs(k-ok), other["group_id"]))
        event_assignment[g["group_id"]] = (
            event_assignment[min(neighbors)[1]] if neighbors else "train"
        )

    # Negative-only candidate windows:
    # Must have no active leak anywhere in the 2-hour input window,
    # must lie outside event guard components, and must not cross weekly block
    # boundaries. Group by Monday-based calendar week.
    negative_groups = {}
    for final_idx in candidate_final_idxs:
        start_idx = final_idx - WINDOW_STEPS + 1

        # skip anything covered by event guard
        if event_component_for_idx[final_idx] >= 0:
            continue

        # require the entire observation window to contain no active leakage
        if active_any[start_idx:final_idx+1].any():
            continue

        start_time = ts.iloc[start_idx]
        final_time = ts.iloc[final_idx]

        # ISO week grouping; require whole window inside same week.
        iso_s = start_time.isocalendar()
        iso_f = final_time.isocalendar()
        key_s = (int(iso_s.year), int(iso_s.week))
        key_f = (int(iso_f.year), int(iso_f.week))
        if key_s != key_f:
            continue

        gid = f"NEG_WEEK_{key_f[0]}_{key_f[1]:02d}"
        negative_groups.setdefault(gid, {
            "group_id": gid,
            "group_type": "negative_week",
            "candidate_windows": 0,
            "positive_windows": 0,
            "event_ids": "",
            "start_time": str(start_time),
            "end_time": str(final_time),
        })
        g = negative_groups[gid]
        g["candidate_windows"] += 1
        g["end_time"] = str(final_time)

    neg_list = list(negative_groups.values())
    if len(neg_list) < 3:
        raise SystemExit(
            f"Only {len(neg_list)} independent negative week groups found."
        )
    neg_assignment, neg_current, neg_targets = greedy_assign(
        neg_list, "candidate_windows", SEED + 1
    )

    # Materialize window manifest.
    event_lookup = {
        comp["group_id"]: comp for comp in components
    }

    # Map final indices to event group.
    final_to_event_gid = {}
    for comp in components:
        gid = comp["group_id"]
        for i in candidate_final_idxs:
            if comp["start_idx"] <= i <= comp["end_idx"]:
                final_to_event_gid[i] = gid

    # Map negative final index to week group if valid.
    final_to_neg_gid = {}
    for final_idx in candidate_final_idxs:
        if final_idx in final_to_event_gid:
            continue
        start_idx = final_idx - WINDOW_STEPS + 1
        if active_any[start_idx:final_idx+1].any():
            continue
        start_time = ts.iloc[start_idx]
        final_time = ts.iloc[final_idx]
        iso_s, iso_f = start_time.isocalendar(), final_time.isocalendar()
        key_s = (int(iso_s.year), int(iso_s.week))
        key_f = (int(iso_f.year), int(iso_f.week))
        if key_s != key_f:
            continue
        gid = f"NEG_WEEK_{key_f[0]}_{key_f[1]:02d}"
        if gid in neg_assignment:
            final_to_neg_gid[final_idx] = gid

    wid = 0
    for final_idx in candidate_final_idxs:
        gid = None
        gtype = None
        split = None

        if final_idx in final_to_event_gid:
            gid = final_to_event_gid[final_idx]
            gtype = "event_component"
            split = event_assignment[gid]
        elif final_idx in final_to_neg_gid:
            gid = final_to_neg_gid[final_idx]
            gtype = "negative_week"
            split = neg_assignment[gid]
        else:
            continue

        start_idx = final_idx - WINDOW_STEPS + 1
        wid += 1
        active_cols = [
            leak_cols[j]
            for j in np.flatnonzero(active_matrix[final_idx])
        ]

        rows.append({
            "window_id": f"W{wid:06d}",
            "split": split,
            "group_id": gid,
            "group_type": gtype,
            "start_idx": start_idx,
            "final_idx": final_idx,
            "start_time": ts.iloc[start_idx],
            "final_time": ts.iloc[final_idx],
            "target_active_leak": int(active_any[final_idx]),
            "active_leak_columns_at_target": ",".join(active_cols),
        })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise SystemExit("No windows generated.")

    # Hard leakage checks.
    # 1) group cannot appear in more than one split.
    cross = manifest.groupby("group_id")["split"].nunique()
    if (cross > 1).any():
        bad = cross[cross > 1].index.tolist()
        raise SystemExit(f"Group leakage across splits: {bad[:10]}")

    # 2) each split must contain positives and negatives.
    stats = {}
    for split in TARGET_FRACTIONS:
        sub = manifest[manifest["split"] == split]
        pos = int(sub["target_active_leak"].sum())
        neg = int((sub["target_active_leak"] == 0).sum())
        if pos == 0 or neg == 0:
            raise SystemExit(
                f"{split} lacks a class: positives={pos}, negatives={neg}"
            )
        stats[split] = {
            "windows": int(len(sub)),
            "positive": pos,
            "negative": neg,
            "positive_fraction": float(pos / len(sub)),
            "groups": int(sub["group_id"].nunique()),
        }

    # 3) no duplicate final timestamps across splits.
    dup = manifest["final_time"].duplicated().sum()
    if dup:
        raise SystemExit(f"Duplicate final timestamps in manifest: {dup}")

    # Save event catalog.
    event_catalog_out = out / "2018_event_catalog.csv"
    event_catalog.to_csv(event_catalog_out, index=False)

    group_catalog = pd.DataFrame(event_groups + neg_list)
    group_catalog["split"] = group_catalog["group_id"].map(
        {**event_assignment, **neg_assignment}
    )
    group_catalog_out = out / "2018_group_catalog.csv"
    group_catalog.to_csv(group_catalog_out, index=False)

    manifest_out = out / "2018_split_manifest.csv"
    manifest.to_csv(manifest_out, index=False, date_format="%Y-%m-%d %H:%M:%S")

    # Diagnostics about labels.
    label_diag = {
        "active_rule": "target_active_leak = 1 iff any 2018 leakage-rate column > 1e-12 at final timestamp",
        "leak_columns": len(leak_cols),
        "raw_contiguous_events": int(len(event_catalog)),
        "independent_event_components_after_2h_guard_merge": int(len(event_groups)),
        "positive_timestamps_5min": int(active_count),
        "positive_timestamp_fraction": float(active_count / len(leaks)),
        "minimum_leak_rate": float(leak_values.min().min()),
        "maximum_leak_rate": float(leak_values.max().max()),
        "window_steps": WINDOW_STEPS,
        "window_minutes": WINDOW_STEPS * 5,
        "stride_steps": STRIDE_STEPS,
        "stride_minutes": STRIDE_STEPS * 5,
    }

    metadata = {
        "status": "FROZEN_SPLIT",
        "seed": SEED,
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "source_hashes": EXPECTED_2018,
        "target_fractions": TARGET_FRACTIONS,
        "label_diagnostics": label_diag,
        "split_stats": stats,
        "files": {
            "manifest": manifest_out.name,
            "event_catalog": event_catalog_out.name,
            "group_catalog": group_catalog_out.name,
        },
        "scientific_rules": [
            "2018 only used to build train/validation/calibration split.",
            "2019 is not read by this script.",
            "2-hour windows; 30-minute stride.",
            "All windows around a contiguous leak event stay in one event component.",
            "Event components use a 2-hour guard zone to prevent overlapping raw input rows across splits.",
            "Pure-negative windows are grouped by ISO calendar week and cannot cross week boundaries.",
            "All assignment is deterministic with fixed seed 271828.",
        ],
    }

    metadata_out = out / "2018_split_metadata.json"
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Freeze hashes of all core artifacts.
    core = [
        manifest_out,
        event_catalog_out,
        group_catalog_out,
        metadata_out,
    ]
    artifact_hashes = {p.name: sha256(p) for p in core}
    freeze = {
        "status": "FROZEN_SPLIT",
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "seed": SEED,
        "artifact_sha256": artifact_hashes,
        "note": "2018 split frozen before model training. Script does not read 2019."
    }
    freeze_out = out / "FROZEN_2018_SPLIT_RECORD.json"
    freeze_out.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    # Human-readable summary.
    lines = []
    lines.append("AQUA-VERA 2018 SPLIT FREEZE")
    lines.append("=" * 78)
    lines.append("STATUS: PASS / FROZEN")
    lines.append(f"Protocol SHA-256: {EXPECTED_PROTOCOL_HASH}")
    lines.append(f"Seed: {SEED}")
    lines.append("")
    lines.append("LABEL / WINDOW DIAGNOSTICS")
    lines.append(f"  leakage columns: {label_diag['leak_columns']}")
    lines.append(f"  contiguous leak events: {label_diag['raw_contiguous_events']}")
    lines.append(
        "  independent event components after 2h guard merge: "
        f"{label_diag['independent_event_components_after_2h_guard_merge']}"
    )
    lines.append(
        f"  positive 5-min timestamps: {label_diag['positive_timestamps_5min']:,} "
        f"({label_diag['positive_timestamp_fraction']:.4%})"
    )
    lines.append(
        f"  leakage-rate min/max: {label_diag['minimum_leak_rate']:.8g} / "
        f"{label_diag['maximum_leak_rate']:.8g}"
    )
    lines.append(f"  window: {label_diag['window_minutes']} minutes")
    lines.append(f"  stride: {label_diag['stride_minutes']} minutes")
    lines.append("")
    lines.append("SPLITS")
    for split in ["train", "validation", "calibration"]:
        s = stats[split]
        lines.append(
            f"  {split}: windows={s['windows']:,}, positive={s['positive']:,}, "
            f"negative={s['negative']:,}, positive_fraction={s['positive_fraction']:.4%}, "
            f"groups={s['groups']}"
        )
    lines.append("")
    lines.append("LEAKAGE SAFETY CHECKS")
    lines.append("  group appears in only one split: PASS")
    lines.append("  each split has positive + negative examples: PASS")
    lines.append("  duplicate final timestamps across splits: 0")
    lines.append("  2019 read by script: NO")
    lines.append("")
    lines.append("ARTIFACT HASHES")
    for k, v in artifact_hashes.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("NEXT")
    lines.append("  Send split_summary.txt to the research record.")
    lines.append("  Do not train until the split summary has been reviewed.")

    summary_out = out / "split_summary.txt"
    summary_out.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("SPLIT FREEZE SUCCESS")
    print(f"Summary: {summary_out}")
    print(f"Freeze:  {freeze_out}")
    print("")
    print("Send split_summary.txt next. Do not train yet.")


if __name__ == "__main__":
    main()
