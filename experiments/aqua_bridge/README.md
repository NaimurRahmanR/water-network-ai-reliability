# AQUA-BRIDGE

Reliability-aware edge-central water supervision with an experimental Incident Hub.

AQUA-BRIDGE is an application-grade extension of the frozen Net3 reliability work already in this repository. It does not modify the parent Net3 protocols, trained models or reported results.

## Research question

When WDS sensor evidence is missing, stale, biased, lost in communication or conflicts with another physical state, when is local edge reconstruction sufficient and when should a wider central view verify or escalate the case?

A secondary question is whether those failures and recoveries can be recorded as structured incidents and profiled into traceable, reusable experimental lessons.

## Implemented in v0.1

- deterministic three-edge partition of the existing pressure/flow/tank sensor feature space;
- edge-local Ridge reconstruction using only features assigned to that edge;
- central reconstruction using the wider model-visible sensor set;
- four architectures: central-only, edge-only, naive hybrid, reliability-aware hybrid;
- controlled missing sensor, stale sensor, persistent bias, packet loss and cross-source conflict interventions;
- frozen parent TRAIN / CALIBRATION / TEST context assignments from generation_index.csv;
- calibration-frozen residual and edge-central disagreement gates;
- PROCEED, reconstruction, verification and ESCALATE outcomes;
- SQLite Incident Hub plus CSV export;
- unsupervised incident profiling with silhouette-selected K-means;
- deterministic benchmark-scoped lessons with explicit claim boundaries;
- optional OGC SensorThings/FROST replay of existing Net3 sensor matrices;
- optional MQTT edge consumer.

See PROTOCOL_v0_1.md for the frozen TEST experiment definition and PROTOCOL_v0_2.md for the untouched-VALIDATION Incident-Hub confirmation. Executed results and claim boundaries are summarised in RESULTS.md.

## CPU smoke run

From the repository root:

    python experiments/aqua_bridge/run_aqua_bridge.py --smoke --out outputs/aqua_bridge_smoke

The smoke path validates code flow only. It is not a scientific Net3 result.

## Net3 run

AQUA-BRIDGE consumes the local artifacts produced by src/net3/v0_3a/01_generate_scenarios.py. The generated directory must contain generation_index.csv, tensor_layout.json and contexts/*.npz.

    python experiments/aqua_bridge/run_aqua_bridge.py --generated-dir PATH/TO/generated_metadata_v0_3a --out outputs/aqua_bridge_net3

The default run deterministically downsamples each clean/leak context for bounded CPU execution. The existing frozen split assignments are preserved.

Outputs include architecture_metrics.csv, incidents.csv, incident_hub.sqlite, frozen_reconstruction_thresholds.csv, clustered_incidents.csv, cluster_summary.csv, lessons.json and run_manifest.json.

## SensorThings / FROST

The compose file follows the current FROST all-in-one pattern with PostgreSQL/PostGIS and exposes HTTP 8080 and MQTT 1883.

    docker compose -f experiments/aqua_bridge/docker-compose.frost.yml up -d
    pip install -r experiments/aqua_bridge/requirements.txt

Create SensorThings entities from the existing sensor layout:

    python experiments/aqua_bridge/sensorthings_bridge.py --generated-dir PATH/TO/generated_metadata_v0_3a bootstrap

Replay one generated context:

    python experiments/aqua_bridge/sensorthings_bridge.py --generated-dir PATH/TO/generated_metadata_v0_3a replay --context-id C0001 --variant CLEAN

Subscribe an edge consumer to a FROST MQTT observation topic:

    python experiments/aqua_bridge/mqtt_edge_consumer.py --edge-id edge_1 --topic 'v1.1/Observations'

A Datastream-specific FROST MQTT topic can be supplied through --topic when required by the local query configuration.

## Executed evidence

The full Net3 v0.1 run, the containerised FROST SensorThings/MQTT integration, and the v0.2 untouched-VALIDATION confirmation have all completed successfully in GitHub Actions.

Compact executed evidence is retained under `evidence/`:

- `frost_integration_evidence.json`;
- `net3_v0_1_results.json`;
- `validation_v0_2_results.json`.

The v0.2 confirmation selected two descriptive profiles with silhouette 0.7011 and bootstrap ARI mean 0.9469. The reliability-aware hybrid reproduced the coverage-risk trade-off on VALIDATION: mean coverage 0.9608, mean escalation 0.0392, and zero observed benchmark-defined unsafe proceeds in that confirmation. This is not a zero-operational-risk claim.

See `RESULTS.md` for the exact interpretation.

## Claim boundaries

This is a research prototype, not a utility deployment. Logical/containerised edge nodes are not physical edge devices. Controlled sensor and communication failures are benchmark interventions. Incident-derived lessons are not operator-approved utility rules. SensorThings/MQTT code must not be described as executed deployment evidence until a runtime record has actually been produced.
