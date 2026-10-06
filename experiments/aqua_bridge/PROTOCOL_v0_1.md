# AQUA-BRIDGE Protocol v0.1

Status: frozen before inspection of any AQUA-BRIDGE Net3 held-out test result.

## Objective

AQUA-BRIDGE extends the existing frozen Net3 reliability benchmark with a bounded edge-versus-central reconstruction experiment and an experimental Incident Hub.

Primary question: when a WDS pressure observation is unreliable or unavailable, does a reconstruction based only on an edge-local sensor subset behave differently from one using the wider network observation set, and can disagreement between the two be used to decide when autonomous recovery should stop?

Secondary question: can the resulting failures, recoveries and escalations be stored as structured incidents and grouped into reproducible behavioural profiles without using the known corruption label as a clustering feature?

## Parent evidence and split discipline

The experiment consumes the existing v0.3a Net3 generated contexts. It must reuse generation_index.csv split labels.

- TRAIN contexts: fit Ridge reconstruction models.
- CALIBRATION contexts: freeze per-sensor residual thresholds and the edge-versus-central disagreement threshold.
- TEST contexts: final architecture comparison only.
- VALIDATION contexts: not used by AQUA-BRIDGE v0.1.

No threshold or model choice may be changed after inspecting AQUA-BRIDGE TEST outcomes without creating a new protocol version.

## Architectures

1. central_only: reconstruct using all non-target sensors when the target observation is missing or fails the central residual gate.
2. edge_only: reconstruct using only the target edge partition when the target observation is missing or fails the edge residual gate.
3. hybrid_naive: accept the edge reconstruction automatically when the edge gate fails.
4. hybrid_reliability_aware: allow bounded recovery only when edge and central reconstruction agree within a CALIBRATION-frozen threshold; otherwise ESCALATE.

## Edge partition

The existing model-visible sensor order is deterministically partitioned across three logical edge nodes. The partition is a controlled experimental abstraction, not a claim about a real utility topology or physical edge deployment.

## Interventions

- CLEAN: no intervention.
- MISSING_SENSOR: target pressure observation removed.
- STALE_SENSOR: target replaced by the previous retained observation; recorded delay is the raw Net3 report-index gap multiplied by the 300-second report interval.
- SENSOR_BIAS: target shifted by 0.5, 1.0 or 2.0 TRAIN standard deviations.
- PACKET_LOSS: targeted observation removed and explicitly recorded as a communication loss.
- CROSS_SOURCE_CONFLICT: target replaced by the matched opposite clean/leak physical variant at the same context and report index where available.

These are controlled benchmark interventions. They do not establish general real-world causal effects.

## Reconstruction models

Ridge regression with alpha=1.0 is fixed in v0.1.

For each pressure sensor:

- edge model features: all non-target model-visible features assigned to the same logical edge;
- central model features: all non-target model-visible sensor features.

Missing predictor values are replaced only with TRAIN means.

## Frozen gates

For each pressure sensor, CALIBRATION data freezes:

- edge absolute reconstruction residual q95;
- central absolute reconstruction residual q95;
- edge-versus-central reconstruction disagreement q95;
- q99 residuals used to define the v0.1 unsafe-recovery error boundary.

The unsafe-recovery boundary is two times the larger edge/central q99 residual, with a numerical floor of 1e-6.

## Primary outcomes

Reported for architecture x intervention x severity:

- autonomous coverage;
- escalation rate;
- unsafe-proceed rate over all evaluated cases;
- selective unsafe risk conditional on proceeding;
- mean absolute recovery error conditional on proceeding;
- p95 recovery error conditional on proceeding.

No architecture is required to win every metric. A coverage-risk trade-off is an acceptable result.

## Incident Hub

Every evaluated architecture decision is written to SQLite and CSV with context, variant, report index, intervention, sensor, logical edge, action, observed value, recovered value, clean reference, reconstruction residuals, edge-central disagreement, packet-loss marker, stale-delay field and unsafe-proceed marker.

## Incident profiling

Only hybrid_reliability_aware non-clean incidents are profiled. Known corruption labels are not clustering features.

Features include edge residual, central residual, edge-central disagreement, recovery error, packet-loss marker, stale delay, proceed status and unsafe-proceed status.

K-means k=2..5 is compared using silhouette score on a seed-fixed maximum sample of 5,000 incidents. The selected clustering is descriptive, not a claim of naturally true incident classes.

Lessons generated from cluster summaries are benchmark-scoped and must state that operator validation is required before operational use.

## SensorThings / MQTT extension

The repository includes an optional FROST SensorThings replay path and MQTT observation consumer. This infrastructure layer is evaluated separately from the offline scientific comparison. Successful code presence alone must not be described as an executed IoT deployment; execution evidence should be retained when the service is actually run.

## Claim boundaries

AQUA-BRIDGE v0.1 is not:

- a water-utility deployment;
- physical edge hardware;
- an operator-validated Incident Hub;
- a production distributed database architecture;
- a water-quality model;
- evidence of general causal effects outside the controlled benchmark.
