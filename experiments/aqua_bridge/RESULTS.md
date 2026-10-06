# AQUA-BRIDGE Executed Results

AQUA-BRIDGE extends the frozen Net3 water-network reliability benchmark with a controlled edge-versus-central reconstruction study, an executed SensorThings/MQTT integration, and an experimental Incident Hub.

## Scientific design

The parent TRAIN/CALIBRATION/TEST/VALIDATION context assignments are preserved.

- TRAIN fits Ridge reconstruction models.
- CALIBRATION freezes residual and edge-central disagreement gates.
- TEST is the frozen v0.1 architecture comparison.
- VALIDATION was unused by v0.1 and was opened only for the v0.2 Incident-Hub confirmation after PROTOCOL_v0_2.md was committed.

The four architectures are central-only, edge-only, naive hybrid, and reliability-aware hybrid. Controlled interventions cover missing sensors, stale sensors, persistent bias, packet loss, and cross-source conflict.

## v0.1 frozen TEST architecture result

The full 240-context Net3 benchmark was regenerated from the verified WNTR 1.5.0 Net3 input in GitHub Actions and the v0.1 experiment completed successfully.

| Architecture | Mean coverage | Mean benchmark-defined unsafe-proceed rate | Mean recovery MAE when proceeding |
| --- | ---: | ---: | ---: |
| central_only | 1.0000 | 0.001171 | 0.1102 |
| edge_only | 1.0000 | 0.002685 | 0.3422 |
| hybrid_naive | 1.0000 | 0.002685 | 0.3422 |
| hybrid_reliability_aware | **0.9669** | **0.000234** | 0.1479 |

The result is a coverage-risk trade-off, not a universal architecture win. The reliability-aware hybrid escalated a small subset of cases and had a lower benchmark-defined unsafe-recovery rate than the always-proceed alternatives.

Workflow evidence: https://github.com/NaimurRahmanR/water-network-ai-reliability/actions/runs/37484690354

## Executed SensorThings / MQTT integration

A containerised FROST SensorThings integration was executed in CI. The retained evidence records:

- SensorThings HTTP success;
- MQTT notification success;
- a created Datastream;
- delivery on a Datastream-specific observation topic.

This is executed containerised integration evidence, not physical edge hardware or a utility deployment.

Workflow evidence: https://github.com/NaimurRahmanR/water-network-ai-reliability/actions/runs/37485671791

## v0.2 untouched VALIDATION confirmation

The v0.2 protocol was committed after v0.1 TEST inspection but before any v0.2 VALIDATION result was generated. It left the reconstruction models, intervention definitions, architecture policies and gates unchanged.

The confirmation reproduced the qualitative architecture trade-off:

| Architecture | Mean coverage | Mean escalation | Mean benchmark-defined unsafe-proceed rate | Mean recovery MAE |
| --- | ---: | ---: | ---: | ---: |
| central_only | 1.0000 | 0.0000 | 0.000585 | **0.0833** |
| edge_only | 1.0000 | 0.0000 | 0.001295 | 0.3100 |
| hybrid_naive | 1.0000 | 0.0000 | 0.001295 | 0.3100 |
| hybrid_reliability_aware | **0.9608** | **0.0392** | **0.000000** | 0.1411 |

Zero here means zero benchmark-defined unsafe proceeds were observed in this VALIDATION confirmation. It is not a claim of zero operational risk.

## Incident profiles

The v0.2 confirmation selected two descriptive incident profiles without using the known intervention label as a clustering feature.

- silhouette score: **0.7011**;
- bootstrap ARI mean: **0.9469**;
- bootstrap ARI median: **0.9743**;
- bootstrap ARI 5th percentile: **0.8414**.

The larger profile (n=4,799 sampled incident records) had median edge-central disagreement 0.0742 and median recovery error 0.0314.

The smaller profile (n=201) had median edge-central disagreement 2.0001 and median recovery error 1.5683. Its strongest standardized differences were higher recovery error and higher edge/central reconstruction residuals. The benchmark-scoped lesson is therefore to retain provenance and use wider-network verification rather than unconditional reuse when that high-error profile recurs.

The other profile retained a separate communication-loss lesson because packet-loss records were relatively more prominent in that profile. These lessons are descriptive summaries of the benchmark and require operator validation before operational use.

Workflow evidence: https://github.com/NaimurRahmanR/water-network-ai-reliability/actions/runs/37487539408

## What the experiment now supports

AQUA-BRIDGE provides implemented and executed prototype evidence for:

- WNTR/EPANET-family hydraulic benchmark integration;
- pressure/flow/tank sensor streams;
- local versus central reconstruction;
- logical edge partitioning;
- central-only, edge-only and hybrid comparison;
- reliability-aware escalation;
- missing/stale/biased/conflicting evidence;
- packet-loss intervention;
- SensorThings/FROST integration;
- MQTT observation delivery;
- structured Incident Hub records;
- incident clustering and bootstrap stability analysis;
- benchmark-scoped traceable lessons.

## Claim boundaries

AQUA-BRIDGE does **not** establish:

- real water-utility deployment;
- physical edge-device deployment;
- operator-validated lessons;
- production-scale distributed database performance;
- water-quality modelling;
- cross-utility validity of the incident profiles;
- general causal effects outside the controlled benchmark.

Those remain legitimate future-work areas rather than hidden gaps.
