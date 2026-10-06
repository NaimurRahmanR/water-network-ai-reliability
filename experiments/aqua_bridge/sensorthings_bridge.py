from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests


@dataclass
class FROSTClient:
    base_url: str
    timeout: float = 15.0

    def _post(self, entity_set: str, payload: dict) -> dict:
        r = requests.post(
            f"{self.base_url.rstrip('/')}/{entity_set}",
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        if r.content:
            return r.json()
        location = r.headers.get("Location") or r.headers.get("location")
        return {"location": location}

    def create_thing_bundle(self, name: str, observed_property: str, unit_symbol: str = "1") -> dict:
        thing = self._post("Things", {
            "name": name,
            "description": "AQUA-BRIDGE experimental WDS sensor",
            "properties": {"experiment": "AQUA-BRIDGE", "claimBoundary": "simulated Net3 sensor"},
            "Locations": [{
                "name": f"{name} location",
                "description": "Logical Net3 sensor location",
                "encodingType": "application/vnd.geo+json",
                "location": {"type": "Point", "coordinates": [0.0, 0.0]},
            }],
            "Datastreams": [{
                "name": f"{name} value",
                "description": "AQUA-BRIDGE replay datastream",
                "observationType": "http://www.opengis.net/def/observationType/OGC-OM/2.0/OM_Measurement",
                "unitOfMeasurement": {"name": observed_property, "symbol": unit_symbol, "definition": "urn:aqua-bridge:unit"},
                "ObservedProperty": {
                    "name": observed_property,
                    "description": f"Experimental {observed_property}",
                    "definition": f"urn:aqua-bridge:observed-property:{observed_property}",
                },
                "Sensor": {
                    "name": f"{name} simulated sensor",
                    "description": "WNTR/Net3 replay source",
                    "encodingType": "application/pdf",
                    "metadata": "urn:aqua-bridge:simulated-sensor",
                },
            }],
        })
        return thing

    def create_observation(self, datastream_id: str | int, value: float, phenomenon_time: str, properties: dict | None = None):
        return self._post("Observations", {
            "phenomenonTime": phenomenon_time,
            "result": float(value),
            "Datastream": {"@iot.id": datastream_id},
            "parameters": properties or {},
        })


def load_datastream_ids(base_url: str) -> Dict[str, int | str]:
    r = requests.get(f"{base_url.rstrip('/')}/Datastreams?$top=1000", timeout=15)
    r.raise_for_status()
    result = {}
    for item in r.json().get("value", []):
        result[item["name"]] = item["@iot.id"]
    return result


def replay_context(base_url: str, generated_dir: Path, context_id: str, variant: str, sleep_s: float = 0.0):
    layout = json.loads((generated_dir / "tensor_layout.json").read_text(encoding="utf-8"))
    names: List[str] = layout["sensor_feature_order"]
    key = "clean_sensor" if variant.upper() == "CLEAN" else "leak_sensor"
    path = generated_dir / "contexts" / f"{context_id}.npz"
    with np.load(path) as z:
        x = np.asarray(z[key], dtype=float)
        times = np.asarray(z["time_s"], dtype=int)

    ids = load_datastream_ids(base_url)
    client = FROSTClient(base_url)
    missing = []
    for name in names:
        ds_name = f"{name} value"
        if ds_name not in ids:
            missing.append(ds_name)
    if missing:
        raise RuntimeError(
            "SensorThings datastreams have not been bootstrapped. Missing examples: " + ", ".join(missing[:5])
        )

    for ti, sec in enumerate(times):
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(sec)))
        for fi, name in enumerate(names):
            client.create_observation(
                ids[f"{name} value"], x[ti, fi], timestamp,
                {"context_id": context_id, "variant": variant, "feature_index": fi},
            )
        if sleep_s > 0:
            time.sleep(sleep_s)


def bootstrap(base_url: str, generated_dir: Path):
    layout = json.loads((generated_dir / "tensor_layout.json").read_text(encoding="utf-8"))
    client = FROSTClient(base_url)
    for name in layout["sensor_feature_order"]:
        if name.startswith("pressure::"):
            op, unit = "pressure", "m"
        elif name.startswith("flow::"):
            op, unit = "flow", "m3/s"
        else:
            op, unit = "tank_level", "m"
        client.create_thing_bundle(name, op, unit)
        print(f"created SensorThings bundle for {name}")


def main():
    ap = argparse.ArgumentParser(description="Replay existing Net3 WDS observations through an OGC SensorThings/FROST endpoint")
    ap.add_argument("--base-url", default="http://localhost:8080/FROST-Server/v1.1")
    ap.add_argument("--generated-dir", type=Path, required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    rp = sub.add_parser("replay")
    rp.add_argument("--context-id", required=True)
    rp.add_argument("--variant", choices=["CLEAN", "LEAK"], default="CLEAN")
    rp.add_argument("--sleep-s", type=float, default=0.0)
    args = ap.parse_args()
    if args.command == "bootstrap":
        bootstrap(args.base_url, args.generated_dir)
    else:
        replay_context(args.base_url, args.generated_dir, args.context_id, args.variant, args.sleep_s)


if __name__ == "__main__":
    main()
