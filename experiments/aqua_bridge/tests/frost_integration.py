from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import requests

from sensorthings_bridge import FROSTClient, bootstrap, load_datastream_ids


BASE = "http://localhost:8080/FROST-Server/v1.1"


def wait_http(timeout_s: float = 90.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(BASE, timeout=3)
            if r.ok:
                return
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(2)
    raise RuntimeError(f"FROST did not become ready: {last}")


def main():
    wait_http()

    fixture = Path("_aqua_bridge/frost_fixture")
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "tensor_layout.json").write_text(json.dumps({
        "sensor_feature_order": ["pressure::P01"]
    }), encoding="utf-8")

    bootstrap(BASE, fixture)
    ids = load_datastream_ids(BASE)
    name = "pressure::P01 value"
    assert name in ids, ids
    ds_id = ids[name]

    received = []
    ready = threading.Event()
    got = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "is_failure", False):
            raise RuntimeError(f"MQTT connection failed: {reason_code}")
        client.subscribe(f"v1.1/Datastreams({ds_id})/Observations", qos=1)
        ready.set()

    def on_message(client, userdata, msg):
        received.append((msg.topic, msg.payload.decode("utf-8", errors="replace")))
        got.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("localhost", 1883, keepalive=30)
    client.loop_start()
    try:
        assert ready.wait(15), "MQTT subscription did not become ready"
        FROSTClient(BASE).create_observation(
            ds_id,
            42.125,
            "2026-10-06T12:00:00Z",
            {"edge_id": "edge_1", "integration_test": True},
        )
        assert got.wait(20), "No MQTT notification received for posted SensorThings Observation"

        r = requests.get(f"{BASE}/Datastreams({ds_id})/Observations?$top=10", timeout=10)
        r.raise_for_status()
        values = r.json().get("value", [])
        assert any(abs(float(o["result"]) - 42.125) < 1e-9 for o in values), values

        out = Path("outputs/aqua_bridge_frost")
        out.mkdir(parents=True, exist_ok=True)
        (out / "integration_result.json").write_text(json.dumps({
            "status": "PASS",
            "sensorthings_http": True,
            "mqtt_notification": True,
            "datastream_id": ds_id,
            "mqtt_topic": received[0][0] if received else None,
            "observation_result": 42.125,
            "claim_boundary": "Containerised FROST integration test; not physical edge hardware or utility deployment."
        }, indent=2), encoding="utf-8")
        print(json.dumps(json.loads((out / "integration_result.json").read_text()), indent=2))
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
