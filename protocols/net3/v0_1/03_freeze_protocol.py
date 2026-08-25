
import argparse, hashlib, json
from pathlib import Path

EXPECTED = {
    "2018_scada_model_ready.csv": "00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e",
    "2018_leakages_model_ready.csv": "743ea7b2845b3dced2ada4fba1a54194c54fc9ce18bf7180a1b3afa4b3eaf2f8",
    "2019_scada_model_ready.csv": "a021ade90c3cfa17362520f0fd7e9794a6c816e4787d7e49a3e52fdd6aa0d15d",
    "2019_leakages_model_ready.csv": "76fa8c50f19b6dffc2010b4413681179c43da5aa984374b9062897b6fe45c27f",
}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ready", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.model_ready)
    protocol = Path(args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verified = {}
    for name, expected in EXPECTED.items():
        p = d / name
        if not p.exists():
            raise SystemExit(f"Missing: {p}")
        got = sha256(p)
        if got != expected:
            raise SystemExit(f"HASH MISMATCH {name}\nexpected={expected}\ngot={got}")
        verified[name] = got
        print(f"[OK] {name}")

    protocol_hash = sha256(protocol)
    record = {
        "status": "FROZEN",
        "protocol_file": protocol.name,
        "protocol_sha256": protocol_hash,
        "verified_data_sha256": verified,
        "note": "This record freezes the protocol before model training and before 2019 evaluation."
    }
    target = out / "FROZEN_PROTOCOL_RECORD.json"
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nFROZEN: {target}")
    print(f"Protocol SHA-256: {protocol_hash}")

if __name__ == "__main__":
    main()
