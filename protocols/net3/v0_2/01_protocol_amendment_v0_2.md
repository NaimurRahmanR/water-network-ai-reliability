# AQUA-VERA Protocol Amendment v0.2

## Status
AMENDMENT BEFORE MODEL TRAINING.

This document does **not** overwrite protocol v0.1. It records a pre-training
design correction discovered by the split-validation gate.

Original protocol v0.1 SHA-256:

`5ea530772b1b4df01c3a437afc10929d570a1cf28fbffcdbc03db9bd0459bc41`

## Trigger for amendment

The frozen v0.1 protocol assumed that 2018 BattLeDIM leak episodes could be
partitioned into independent train / validation / calibration event groups.

The deterministic split builder disproved that assumption:

> After two-hour guard-zone merging, only one independent event component
> remained.

No model was trained and 2019 was not used to make this decision.

## Scientific interpretation

BattLeDIM 2018 is a historical training/calibration year containing multiple
leak processes that can overlap or persist. Treating it as a set of independent
event episodes for a strict event-held-out three-way development split is not
supported by the observed label structure.

An arbitrary random-window split would create substantial temporal/event
dependence and is therefore rejected.

## Amended benchmark architecture

### A. Primary controlled benchmark — WNTR / EPANET Net3

Use EPA WNTR with Net3 to generate independent hydraulic scenarios.

Each physical scenario is generated independently and assigned to exactly one
split before model training:

- training
- validation
- calibration
- held-out controlled test

The final scenario count, split counts, sensor placement rule, leak-area
distribution, demand perturbation distribution, seeds and simulation duration
will be frozen **after a simulator smoke test and before full benchmark
generation**.

The controlled benchmark is the primary source for:

- graph-model training;
- independent scenario-held-out evaluation;
- matched clean/degraded evidence pairs;
- missing/noisy/biased/stale sensor experiments;
- topology mismatch;
- controlled evidence conflict;
- reconstruction;
- explanation instability;
- reliability-controller evaluation.

### B. BattLeDIM L-Town — external benchmark

BattLeDIM is retained as an external benchmark:

- 2018: development / adaptation only;
- 2019: final external evaluation only.

BattLeDIM 2019 may not be used for architecture selection, hyperparameter
selection, corruption severity tuning, explanation threshold tuning or
controller threshold tuning.

The BattLeDIM external analysis will be clearly separated from the controlled
Net3 benchmark.

## Primary task

Binary leak-presence decision over a fixed observation window remains the
primary task.

Leak localisation may be reported as a secondary task if implemented without
changing the primary analysis after test inspection.

## Controlled evidence degradations

The v0.1 families remain:

1. MISSING_SENSOR
2. SENSOR_NOISE
3. SENSOR_BIAS
4. STALE_SENSOR
5. TOPOLOGY_MISMATCH
6. CROSS_SOURCE_CONFLICT

These are applied to the model-visible evidence for an already generated
physical scenario. They do not alter the hidden physical ground truth.

## Reliability signals and controller

The v0.1 reliability signals and controller actions remain unchanged in
principle. Exact implementations and thresholds must be frozen using only
training/validation/calibration scenarios from the controlled benchmark.

## Claim boundary

Primary claims will concern a controlled simulation benchmark using WNTR/Net3.

BattLeDIM results will be described as external benchmark transfer/validation,
not live utility deployment.

## Amendment integrity

This correction occurred:
- before model training;
- after a failed pre-training split-validity check;
- without inspecting BattLeDIM 2019 performance for model selection.

The failed v0.1 split attempt should be retained in the repository as an audit
artifact rather than deleted.
