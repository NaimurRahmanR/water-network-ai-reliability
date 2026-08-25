
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

H18S = "00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e"
H18L = "743ea7b2845b3dced2ada4fba1a54194c54fc9ce18bf7180a1b3afa4b3eaf2f8"

WINDOW_STEPS = 24
STRIDE = 6
ONSET_HORIZON_STEPS = 24 * 60 // 5
REPAIR_GUARD_STEPS = 24 * 60 // 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_unique(root, name):
    root = Path(root)
    direct = root / name
    if direct.exists():
        return direct
    m = list(root.rglob(name))
    if len(m) != 1:
        raise SystemExit(f"Expected one {name} under {root}, found {len(m)}")
    return m[0]


def verify(path, expected):
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH {path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {path.name}")


def transitions(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return starts, ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ready", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mr = Path(args.model_ready)
    protocol = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scada_path = find_unique(mr, "2018_scada_model_ready.csv")
    leaks_path = find_unique(mr, "2018_leakages_model_ready.csv")

    verify(scada_path, H18S)
    verify(leaks_path, H18L)

    scada = pd.read_csv(scada_path)
    leaks = pd.read_csv(leaks_path)

    scada["Timestamp"] = pd.to_datetime(scada["Timestamp"])
    leaks["Timestamp"] = pd.to_datetime(leaks["Timestamp"])

    if len(scada) != 105120 or len(leaks) != 105120:
        raise SystemExit("2018 row count mismatch")
    if not np.array_equal(scada.Timestamp.to_numpy(), leaks.Timestamp.to_numpy()):
        raise SystemExit("2018 timestamps differ")

    leak_cols = [c for c in leaks.columns if c != "Timestamp"]
    values = leaks[leak_cols].to_numpy(dtype=float)
    active = values > 0.0

    event_rows = []
    onset_indices = []
    end_indices = []

    for j, col in enumerate(leak_cols):
        starts, ends = transitions(active[:, j])
        if len(starts) != 1 or len(ends) != 1:
            raise SystemExit(
                f"v0.15 expected one contiguous positive interval per 2018 "
                f"leak column based on v0.14b discovery; {col} has "
                f"starts={len(starts)} ends={len(ends)}"
            )
        s, e = int(starts[0]), int(ends[0])
        onset_indices.append(s)
        end_indices.append(e)
        event_rows.append({
            "leak_column": col,
            "onset_idx": s,
            "onset_timestamp": str(leaks.Timestamp.iloc[s]),
            "end_idx": e,
            "end_timestamp": str(leaks.Timestamp.iloc[e]),
            "duration_hours": float((e - s + 1) * 5 / 60),
        })

    event_df = pd.DataFrame(event_rows).sort_values("onset_idx").reset_index(drop=True)
    event_df["event_order"] = np.arange(1, len(event_df) + 1)
    event_path = out / "2018_frozen_onset_events.csv"
    event_df.to_csv(event_path, index=False)

    # Candidate windows and frozen target semantics.
    window_rows = []
    onset_indices = np.asarray(onset_indices, dtype=int)
    end_indices = np.asarray(end_indices, dtype=int)

    for final_idx in range(WINDOW_STEPS - 1, len(scada), STRIDE):
        start_idx = final_idx - WINDOW_STEPS + 1

        # Positive iff an onset occurred in the previous 24 hours including now.
        recent = (
            (onset_indices <= final_idx)
            & (onset_indices > final_idx - ONSET_HORIZON_STEPS)
        )
        label = int(recent.any())

        # Exclude 24 hours beginning at each end/repair transition.
        in_repair_guard = bool(
            (
                (end_indices <= final_idx)
                & (end_indices > final_idx - REPAIR_GUARD_STEPS)
            ).any()
        )

        eligible = int(not in_repair_guard)

        window_rows.append({
            "window_index": len(window_rows),
            "start_idx": start_idx,
            "final_idx": final_idx,
            "start_timestamp": str(scada.Timestamp.iloc[start_idx]),
            "final_timestamp": str(scada.Timestamp.iloc[final_idx]),
            "target_recent_leak_onset_24h": label,
            "repair_guard_excluded": int(in_repair_guard),
            "eligible_2018_development": eligible,
            "attribution_subset": int(len(window_rows) % 10 == 0),
        })

    windows = pd.DataFrame(window_rows)
    windows_path = out / "2018_frozen_window_manifest_v0_15.csv"
    windows.to_csv(windows_path, index=False)

    eligible = windows[windows.eligible_2018_development == 1]
    positives = int(eligible.target_recent_leak_onset_24h.sum())
    negatives = int(len(eligible) - positives)

    if positives < 200:
        raise SystemExit(
            f"Too few eligible positive windows for the frozen 24h onset task: "
            f"{positives}. Stop and review before training."
        )
    if negatives < positives:
        raise SystemExit(
            "Eligible 2018 negatives are fewer than positives; unexpected."
        )

    protocol_hash = sha256(protocol)

    record = {
        "status": "PASS_PROTOCOL_FROZEN_PRETRAINING",
        "version": "0.15",
        "protocol_sha256": protocol_hash,
        "2018_scada_sha256": H18S,
        "2018_leakages_sha256": H18L,
        "2019_opened": False,
        "2019_hashed": False,
        "models_trained": False,
        "target": "recent leak onset within previous 24 hours",
        "window_steps": WINDOW_STEPS,
        "stride_steps": STRIDE,
        "repair_guard_hours": 24,
        "events_2018": int(len(event_df)),
        "candidate_windows": int(len(windows)),
        "eligible_windows": int(len(eligible)),
        "eligible_positive_windows": positives,
        "eligible_negative_windows": negatives,
        "eligible_positive_fraction": float(positives / len(eligible)),
        "attribution_subset_windows_eligible": int(
            eligible.attribution_subset.sum()
        ),
        "predictive_threshold": 0.5,
        "cross_source_conflict_external": "omitted",
        "scientific_note": (
            "The external target is leak-onset detection because active-leak "
            "presence was 97.8% positive in the 2018 discovery audit."
        ),
    }

    record_path = out / "FROZEN_BATTLEDIM_PROTOCOL_RECORD_v0_15.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    protocol_copy = out / "BattLeDIM_external_protocol_v0_15.md"
    protocol_copy.write_bytes(protocol.read_bytes())

    hashes = {
        event_path.name: sha256(event_path),
        windows_path.name: sha256(windows_path),
        record_path.name: sha256(record_path),
        protocol_copy.name: sha256(protocol_copy),
    }

    lines = [
        "AQUA-VERA BATTLEDIM/L-TOWN EXTERNAL PROTOCOL FREEZE v0.15",
        "=" * 78,
        "STATUS: PASS / PROTOCOL FROZEN PRETRAINING",
        f"Protocol SHA-256: {protocol_hash}",
        "",
        "BOUNDARY",
        "  2018 parsed: YES",
        "  2019 opened/hashed: NO",
        "  models trained: NO",
        "  predictive threshold optimized: NO",
        "",
        "FROZEN EXTERNAL TASK",
        "  target: at least one leak onset in previous 24 hours",
        "  input window: 2 hours",
        "  stride: 30 minutes",
        "  repair/end guard: first 24 hours after end transition excluded",
        "  existing older active leaks without new onset: valid negatives",
        "",
        "2018 DEVELOPMENT ACCOUNTING",
        f"  leakage onset events: {len(event_df)}",
        f"  candidate windows: {len(windows):,}",
        f"  eligible windows: {len(eligible):,}",
        f"  positive windows: {positives:,}",
        f"  negative windows: {negatives:,}",
        f"  positive fraction: {positives/len(eligible):.4%}",
        f"  eligible attribution-subset windows: "
        f"{int(eligible.attribution_subset.sum()):,}",
        "",
        "EXTERNAL DEGRADATIONS",
        "  MISSING_SENSOR",
        "  SENSOR_NOISE",
        "  SENSOR_BIAS",
        "  STALE_SENSOR",
        "  TOPOLOGY_MISMATCH (GCN-GRU only)",
        "  CROSS_SOURCE_CONFLICT omitted: no matched counterfactual in BattLeDIM",
        "",
        "MODEL POLICY",
        "  transferred AQUA-VERA architectures/training recipes",
        "  MLP fixed 25 epochs",
        "  GCN-GRU fixed 28 epochs",
        "  predictive threshold 0.5",
        "  no BattLeDIM architecture or epoch tuning",
        "",
        "ARTIFACT HASHES",
    ]
    for n, h in hashes.items():
        lines.append(f"  {n}: {h}")

    lines += [
        "",
        "NEXT",
        "  Retain BattLeDIM_protocol_freeze_summary_v0_15.txt with the project record.",
        "  If PASS, train/freeze the two 2018 models and reliability references.",
        "  Do not inspect 2019.",
    ]

    summary = out / "BattLeDIM_protocol_freeze_summary_v0_15.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
