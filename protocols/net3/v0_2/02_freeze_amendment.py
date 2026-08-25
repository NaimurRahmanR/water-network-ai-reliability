
import argparse, hashlib, json
from pathlib import Path

V01 = "5ea530772b1b4df01c3a437afc10929d570a1cf28fbffcdbc03db9bd0459bc41"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v01-record", required=True)
    ap.add_argument("--amendment", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec = json.loads(Path(args.v01_record).read_text(encoding="utf-8"))
    if rec.get("protocol_sha256") != V01:
        raise SystemExit(
            "Original v0.1 protocol hash mismatch.\n"
            f"expected={V01}\ngot={rec.get('protocol_sha256')}"
        )

    amendment = Path(args.amendment)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ah = sha256(amendment)
    record = {
        "status": "AMENDED_BEFORE_TRAINING",
        "original_protocol_v0_1_sha256": V01,
        "amendment_v0_2_sha256": ah,
        "reason": "2018 BattLeDIM leak-event guard-zone graph collapsed to one independent component; strict event-group development split infeasible.",
        "training_started_before_amendment": False,
        "design_change": "Primary controlled benchmark moved to WNTR/Net3; BattLeDIM retained as external benchmark."
    }
    p = out / "FROZEN_AMENDMENT_v0_2_RECORD.json"
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"AMENDMENT FROZEN: {p}")
    print(f"Amendment SHA-256: {ah}")

if __name__ == "__main__":
    main()
