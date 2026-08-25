# AQUA-VERA Controlled Benchmark Freeze v0.3

## Status
To be frozen after the successful WNTR/Net3 smoke test and before full benchmark generation.

This document extends the v0.2 amendment. It does not erase v0.1 or v0.2.

## Validated simulator path
- WNTR version: 1.5.0
- Network: EPANET Example Network 3 (Net3)
- Net3 SHA-256:
  `ea3e825c4fef0b5cba47fb06301bc85253f18b6364dc96c44d9fb492c40faa52`
- Simulator: WNTRSimulator
- Hydraulic model: pressure-dependent demand (PDD)
- Required pressure: 15 m
- Minimum pressure: 0 m
- Hydraulic/report timestep: 300 seconds
- Scenario duration: 24 hours

The smoke test demonstrated finite hydraulic outputs and a measurable pressure response to a timed pipe leak.

## Independent base contexts
Total base contexts: **240**

Each base context has two matched physical variants:
1. CLEAN
2. LEAK

Total hydraulic simulations: **480**

The CLEAN and LEAK variants share the same demand perturbation and initial network context. The only intended physical difference is the leak intervention.

## Frozen split counts
Base contexts:
- train: 150
- validation: 30
- calibration: 30
- held-out test: 30

Both matched variants of a base context remain in the same split.

## Leak-location separation
Eligible leak locations:
- ordinary Net3 pipes only;
- diameter <= 0.9144 m (36 inches), following the eligibility cutoff used in the official WNTR stochastic leak example.

Eligible pipes are deterministically shuffled and partitioned into disjoint pipe pools:
- train pipe pool: 60%
- validation pipe pool: 15%
- calibration pipe pool: 10%
- test pipe pool: remaining 15%

A scenario may use only a pipe from its split's pipe pool.

Therefore, held-out test leak locations are unseen during training.

## Demand-context variation
For each base context:
- global demand multiplier sampled uniformly from [0.85, 1.15];
- per-junction demand multipliers generated from a fixed scenario-specific seed;
- per-junction multiplicative noise sampled lognormally with sigma=0.05 and clipped to [0.85, 1.15];
- demand perturbations are fixed for the complete 24-hour scenario.

No held-out test context is used for development decisions.

## Leak intervention
One leak per LEAK variant.

For the paired leak pipe:
- leak point: midpoint of the pipe;
- discharge coefficient: 0.75;
- leak diameter fraction sampled from {0.10, 0.20, 0.30} × pipe diameter;
- leak area = pi * (leak_diameter / 2)^2;
- start hour sampled from {4, 6, 8, 10, 12};
- duration sampled from {6, 8, 10, 12} hours;
- combinations are constrained so end hour <= 24.

The discrete severity levels are benchmark design choices, not claims about real-world leak-frequency distributions.

## Observation schedule
- 5-minute hydraulic/report timestep;
- 24-hour scenario;
- 289 reported timestamps including t=0 and t=24h.

Primary model windows:
- 24 observations = 2 hours;
- final-timestamp target;
- stride for model examples will be frozen in the dataset-construction stage, before model training.

Primary target:
`active_leak = 1` iff the simulated leak is active at the final timestamp of the observation window.

## Sensor layout
Sensor placement is derived from Net3 topology only, before simulation labels exist.

### Pressure sensors
24 junction pressure sensors selected using deterministic farthest-point graph sampling:
1. first node = highest-degree junction;
2. each next node maximizes shortest-path distance to the current selected set;
3. lexical node ID breaks ties.

### Tank-level sensors
All Net3 tanks are observed.

### Flow sensors
6 ordinary pipes selected deterministically by edge betweenness centrality where available; deterministic diameter/name fallback is used if needed.

The final exact sensor IDs are written to `sensor_layout.json` and hashed before benchmark generation.

Model-visible dynamic evidence:
- pressure at selected junction sensors;
- tank level/pressure at all tanks;
- flow at selected pipe sensors;
- explicit observation masks.

Latent full-network simulation outputs may be retained for analysis/verifier development but are not automatically model-visible.

## Controlled evidence degradation
Applied to model-visible held-out evidence after clean physical simulation.

### MISSING_SENSOR
Fraction of pressure sensors masked:
- low: 0.10
- medium: 0.25
- high: 0.40

### SENSOR_NOISE
Independent Gaussian noise using per-sensor training standard deviation:
- low: 0.25 SD
- medium: 0.50 SD
- high: 1.00 SD

### SENSOR_BIAS
Persistent offset using per-sensor training standard deviation:
- low: 0.50 SD
- medium: 1.00 SD
- high: 2.00 SD

### STALE_SENSOR
Selected sensors replay older readings:
- low: 30 minutes
- medium: 60 minutes
- high: 120 minutes

### TOPOLOGY_MISMATCH
Model-visible graph removes non-bridge ordinary pipes while physical simulation remains unchanged:
- low: 1 edge
- medium: 3 edges
- high: 5 edges

Removal must preserve graph connectivity.

### CROSS_SOURCE_CONFLICT
For matched CLEAN/LEAK pairs, selected pressure evidence may be substituted from the paired counterfactual variant while flow/tank evidence remains from the factual variant:
- low: 10% of pressure sensors
- medium: 25%
- high: 40%

The paired counterfactual is used only to generate controlled conflicting evidence. It is never exposed to the model/controller as hidden clean truth.

## Reliability evaluation
Primary reliability comparisons remain:
- always proceed;
- uncertainty-only;
- evidence-integrity-only;
- hybrid controller.

Primary reliability signals:
- predictive uncertainty;
- explanation instability;
- reconstruction residual;
- independent topology/hydraulic consistency where implementable without oracle information.

All thresholds are selected on calibration contexts only.

## Reconstruction
Missing/anomalous model-visible sensor evidence will be reconstructed using:
1. simple temporal/local baseline;
2. graph-aware reconstruction.

Outcomes:
- sensor reconstruction error;
- downstream leak-decision recovery.

## Held-out integrity
The held-out controlled test is not used for:
- architecture selection;
- feature selection;
- early stopping;
- threshold selection;
- severity tuning;
- explainer selection;
- reconstruction-method selection;
- controller calibration.

## External benchmark
BattLeDIM/L-Town remains separate:
- 2018: external development/adaptation;
- 2019: final external evaluation.

Results from Net3 and L-Town must not be pooled into a single accuracy denominator.

## Claim boundary
This is a controlled simulation benchmark plus external benchmark validation.
It is not live utility deployment or field validation.
