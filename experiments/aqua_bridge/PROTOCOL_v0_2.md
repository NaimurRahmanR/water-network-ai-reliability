# AQUA-BRIDGE Protocol v0.2 — Incident-Profile Confirmation

Status: frozen after v0.1 TEST architecture results were inspected, but before any v0.2 VALIDATION incident-profile result was generated or inspected.

## Why this amendment exists

AQUA-BRIDGE v0.1 successfully compared the four reconstruction/control architectures on the frozen Net3 TEST split. Its architecture results remain unchanged.

The v0.1 Incident-Hub clustering was descriptive, but its threshold-based natural-language lesson templates did not distinguish the observed profiles sufficiently. Because the v0.1 TEST outcomes have already been inspected, this amendment does not retune the architecture, reconstruction models, reliability gates, corruption definitions, or v0.1 TEST clustering.

Instead, v0.2 uses the previously unused parent VALIDATION contexts as an untouched confirmation set for the Incident-Hub profiling and lesson-description layer only.

## Data discipline

The parent v0.3a Net3 split assignments are reused exactly.

- TRAIN: fit the same Ridge reconstruction models as v0.1.
- CALIBRATION: freeze the same per-sensor residual and edge-central disagreement thresholds as v0.1.
- TEST: not used by v0.2; the already-inspected v0.1 TEST results remain historical evidence.
- VALIDATION: used once as the v0.2 confirmation set for incident profiling and descriptive lesson generation.

No v0.2 method choice may be changed after inspecting VALIDATION outputs without creating a new protocol version.

## Architectures and interventions

v0.2 reuses the v0.1 implementation without changing:

- central_only;
- edge_only;
- hybrid_naive;
- hybrid_reliability_aware;
- CLEAN;
- MISSING_SENSOR;
- STALE_SENSOR;
- SENSOR_BIAS;
- PACKET_LOSS;
- CROSS_SOURCE_CONFLICT.

The v0.2 confirmation analysis focuses on non-clean incidents produced by hybrid_reliability_aware.

## Incident profiling

The clustering feature set remains the v0.1 feature set and does not include the known intervention label:

- edge reconstruction residual;
- central reconstruction residual;
- edge-central disagreement;
- absolute recovery error;
- packet-loss marker;
- stale delay;
- proceeded marker;
- unsafe-proceed marker.

A deterministic seed-fixed maximum of 5,000 incidents is used.

K-means k=2..5 is compared by silhouette score. The selected partition is descriptive and is not treated as a naturally true taxonomy.

## Cluster stability

After selecting k, stability is measured with 25 seed-derived bootstrap resamples.

For each bootstrap replicate:

1. sample the standardized incident-profile rows with replacement;
2. fit K-means with the already selected k;
3. predict cluster labels for the full standardized confirmation sample;
4. compare the replicate labels with the base clustering using Adjusted Rand Index (ARI).

Report mean, median, 5th percentile, minimum and maximum ARI.

This stability analysis assesses partition sensitivity within this benchmark. It does not establish cross-utility cluster validity.

## Descriptive lesson generation

The v0.1 absolute thresholds for natural-language lesson text are not used.

Each confirmation cluster is instead described by:

- support size;
- dominant intervention for context only;
- dominant action;
- escalation rate;
- benchmark-defined unsafe-proceed rate;
- median edge-central disagreement;
- median recovery error;
- packet-loss rate;
- median stale delay;
- the three largest standardized mean differences from the overall confirmation sample.

A deterministic rule maps the relative profile to one benchmark-scoped lesson:

1. If the cluster has the highest edge-central disagreement and either escalation or unsafe-proceed is above the confirmation-sample average, recommend wider-network verification before autonomous reuse.
2. Else if the cluster has the highest packet-loss rate above the sample average, recommend treating communication loss as evidence-quality provenance and requesting fresh evidence before reuse.
3. Else if the cluster has the highest stale delay above the sample average, recommend fresh-measurement verification before reuse.
4. Else if the cluster has below-average disagreement, below-average recovery error and at-least-average proceed rate, describe bounded reconstruction as comparatively stable while retaining provenance and monitoring.
5. Otherwise, recommend retaining provenance and wider-network verification when the profile recurs.

These are descriptive benchmark lessons, not operator-approved operating rules.

## Confirmation outputs

v0.2 writes:

- validation architecture metrics;
- clustered validation incidents;
- cluster profiles;
- cluster stability metrics;
- descriptive lessons;
- a compact confirmation summary;
- a run manifest with claim boundaries.

## Claim boundaries

AQUA-BRIDGE v0.2 is not:

- a utility deployment;
- physical edge hardware;
- an operator-validated Incident Hub;
- proof of naturally true incident classes;
- a production distributed database architecture;
- a water-quality model;
- evidence of general causal effects outside the controlled benchmark.

The v0.2 lesson layer is explicitly post-v0.1 and confirmed on previously unused VALIDATION contexts.
