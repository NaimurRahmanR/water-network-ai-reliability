from __future__ import annotations

import argparse
import json
from pathlib import Path

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser(description="Minimal AQUA-BRIDGE edge MQTT consumer for FROST SensorThings observation updates")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="v1.1/Observations")
    ap.add_argument("--edge-id", default="edge_1")
    ap.add_argument("--out", type=Path, default=Path("outputs/aqua_bridge/mqtt_edge_1.jsonl"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"connected reason={reason_code}; subscribing to {args.topic}")
        client.subscribe(args.topic, qos=1)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = {"raw": msg.payload.decode("utf-8", errors="replace")}
        record = {"edge_id": args.edge_id, "topic": msg.topic, "payload": payload}
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
