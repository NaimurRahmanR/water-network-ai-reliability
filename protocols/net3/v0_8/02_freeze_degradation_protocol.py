
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PREDICTIONS = "9e75780befaf5502bd38c3c02a1a1b62a3ec8c972eb8eab6d1be4b552fb040ce"
EXPECTED_CORRUPTION_CONFIG = "37f648b414d12c7a845ecea784376ece85d5a3c9417c4b585e5c96eb287c5fbd"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require(path, expected, label):
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"{label} HASH MISMATCH\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {label}: {got}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--frozen-benchmark", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    predictions = Path(args.predictions)
    frozen_benchmark = Path(args.frozen_benchmark)
    protocol = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    require(
        predictions,
        EXPECTED_PREDICTIONS,
        "clean held-out prediction file",
    )
    require(
        frozen_benchmark / "corruption_config.json",
        EXPECTED_CORRUPTION_CONFIG,
        "v0.3 corruption config",
    )

    protocol_hash = sha256(protocol)

    record = {
        "status": "DEGRADATION_PROTOCOL_FROZEN",
        "protocol_version": "0.8",
        "protocol_sha256": protocol_hash,
        "clean_heldout_prediction_sha256": EXPECTED_PREDICTIONS,
        "v0_3_corruption_config_sha256": EXPECTED_CORRUPTION_CONFIG,
        "root_corruption_seed": 8675309,
        "development_split_for_degradation_rules": "calibration only",
        "degraded_test_used_before_reliability_rule_freeze": False,
        "models_retrained_after_clean_test": False,
        "clean_model_thresholds_changed": False,
        "attribution_method": "Input x Gradient",
        "attribution_reference": "calibration predicted-class centroids",
        "attribution_divergence": "Jensen-Shannon divergence",
        "note": (
            "v0.8 resolves previously underspecified sensor-selection rules "
            "before any degradation examples are generated."
        ),
    }

    rec_path = out / "FROZEN_DEGRADATION_PROTOCOL_RECORD.json"
    rec_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    protocol_copy = out / "degradation_protocol_v0_8.md"
    protocol_copy.write_text(protocol.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "AQUA-VERA DEGRADATION & RELIABILITY PROTOCOL v0.8",
        "=" * 78,
        "STATUS: PASS / FROZEN",
        f"Protocol SHA-256: {protocol_hash}",
        f"Clean held-out prediction SHA-256: {EXPECTED_PREDICTIONS}",
        f"v0.3 corruption config SHA-256: {EXPECTED_CORRUPTION_CONFIG}",
        "",
        "BOUNDARY",
        "  clean models changed after held-out result: NO",
        "  clean thresholds changed: NO",
        "  degradation development split: CALIBRATION ONLY",
        "  degraded held-out test opened: NO",
        "",
        "CORRUPTIONS",
        "  MISSING_SENSOR: 2 / 6 / 10 pressure sensors",
        "  SENSOR_NOISE: 6 sensors; 0.25 / 0.50 / 1.00 train SD",
        "  SENSOR_BIAS: 6 sensors; 0.50 / 1.00 / 2.00 train SD",
        "  STALE_SENSOR: 6 sensors; 30 / 60 / 120 min",
        "  TOPOLOGY_MISMATCH: 1 / 3 / 5 removable edges; GCN only",
        "  CROSS_SOURCE_CONFLICT: 2 / 6 / 10 pressure sensors",
        "",
        "RELIABILITY SIGNALS",
        "  normalized binary predictive entropy",
        "  Input x Gradient attribution profile",
        "  predicted-class calibration centroid",
        "  Jensen-Shannon attribution-profile divergence",
        "",
        "NEXT",
        "  Retain degradation_protocol_summary.txt with the project record.",
        "  Next execution stage: calibration degradations only.",
    ]

    summary = out / "degradation_protocol_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))

if __name__ == "__main__":
    main()
