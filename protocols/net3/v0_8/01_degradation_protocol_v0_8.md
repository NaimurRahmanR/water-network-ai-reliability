# AQUA-VERA Degradation & Reliability Protocol v0.8

## Status
Freeze before generating any controlled degradation examples.

## Existing clean held-out result
The clean held-out comparison is final and will not be used to redesign either
model. The non-graph MLP outperformed the frozen GCN-GRU on the clean held-out
set. Both models remain frozen.

The reliability study will evaluate BOTH frozen models:
- MLP = non-graph comparator
- GCN-GRU = topology-aware system under study

No attempt will be made to retrain the GCN-GRU because it lost the clean
comparison.

## Development/test boundary

Controlled degradations are developed in two stages:

1. CALIBRATION degradation stage
   - generate frozen corruptions on calibration contexts;
   - fit/reference reliability signals;
   - select reconstruction/controller thresholds;
   - freeze all reliability rules.

2. HELD-OUT degradation stage
   - generate the same frozen corruption families on held-out test contexts;
   - evaluate once;
   - no post-test tuning.

Clean held-out results are already known, but degraded held-out outcomes are
not used for any design choice.

## Deterministic corruption assignment

Root corruption seed:
`8675309`

Pressure sensor selection is deterministic per:
- physical context_id
- corruption family

Selection is independent of:
- CLEAN/LEAK physical variant;
- model prediction;
- ground-truth leak label;
- corruption outcome.

The same sensor subset is therefore used for the paired CLEAN/LEAK variants
of a physical context.

Sensor ranking rule:
SHA-256 of:
`root_seed | context_id | corruption_family | sensor_id`

Lowest lexical digest ranks first.

There are 24 frozen pressure sensors.

## Controlled evidence degradations

### 1. MISSING_SENSOR

Nominal frozen severities from v0.3:
- low: 10%
- medium: 25%
- high: 40%

Realized pressure-sensor counts:
- low: 2 / 24 = 8.33%
- medium: 6 / 24 = 25.00%
- high: 10 / 24 = 41.67%

For selected sensors:
- numerical value is replaced by its frozen training mean;
- MLP receives the mean-imputed value;
- GCN-GRU receives the mean-imputed value AND pressure observation mask = 0.

No hidden clean value is exposed to either model/controller.

### 2. SENSOR_NOISE

Exactly 6 / 24 pressure sensors are corrupted at every severity.

Independent Gaussian noise is added at each 5-minute timestamp using the
frozen training standard deviation for that sensor:

- low: sigma = 0.25 training SD
- medium: sigma = 0.50 training SD
- high: sigma = 1.00 training SD

A deterministic context/family/severity random seed is used.

The same standardized noise realization is used for matched CLEAN/LEAK
physical variants of a context.

### 3. SENSOR_BIAS

Exactly 6 / 24 pressure sensors are corrupted at every severity.

A persistent offset is applied for the complete scenario:

- low: 0.50 training SD
- medium: 1.00 training SD
- high: 2.00 training SD

Offset sign (+/-) is deterministic per context and sensor and is the same for
the matched physical variants.

### 4. STALE_SENSOR

Exactly 6 / 24 pressure sensors are stale at every severity.

Lag:
- low: 30 min = 6 samples
- medium: 60 min = 12 samples
- high: 120 min = 24 samples

At time t the stale stream reports the factual stream from max(0, t-lag).
Only past values from the same physical variant are used.

### 5. TOPOLOGY_MISMATCH

Physical hydraulic simulation and numeric sensor evidence remain unchanged.

The GCN-GRU model-visible graph removes:
- low: 1 ordinary pipe edge
- medium: 3 ordinary pipe edges
- high: 5 ordinary pipe edges

Only removals that preserve graph connectivity are allowed.
The deterministic edge order is derived from:
`root_seed | context_id | TOPOLOGY_MISMATCH | edge_id`

The MLP has no topology input. TOPOLOGY_MISMATCH is therefore marked
NOT-APPLICABLE for the MLP rather than counted as an MLP robustness success.

### 6. CROSS_SOURCE_CONFLICT

Nominal severities:
- low: 10%
- medium: 25%
- high: 40%

Realized pressure-sensor counts:
- low: 2 / 24
- medium: 6 / 24
- high: 10 / 24

For selected pressure sensors only:
- factual CLEAN physical variant receives pressure evidence from its matched
  LEAK counterfactual;
- factual LEAK physical variant receives pressure evidence from its matched
  CLEAN counterfactual.

Flow and tank evidence remain factual.

The paired counterfactual is used ONLY to manufacture controlled conflicting
evidence. It is never available as a runtime reliability signal or recovery
oracle.

## Prediction reliability signal

Binary predictive entropy:

H(p) = -p log p - (1-p) log(1-p)

Normalized by log(2), so the score is in [0,1].

This uses only the frozen model probability.

## Attribution-profile signal

Method:
Input × Gradient with respect to the model's leak logit.

Reason:
- compatible with both frozen differentiable PyTorch models;
- one forward/backward attribution pass;
- feasible for calibration and degraded evaluation at scale;
- no extra explainer model is fitted.

### Comparable 33-sensor attribution space

Attribution is computed with respect to the same normalized raw input tensor:

`[24 time steps, 33 frozen sensor features]`

For the GCN-GRU, a differentiable wrapper maps this raw sensor tensor to the
frozen 97-node graph representation before forwarding through the frozen
GCN-GRU.

No ground-truth label is used for attribution.

### Attribution vector

For each window:
1. compute Input × Gradient for the leak logit;
2. take absolute value;
3. sum over the 24 time steps;
4. obtain a 33-dimensional nonnegative sensor-attribution vector;
5. add epsilon = 1e-12;
6. normalize to sum to one.

### Calibration reference

For each frozen model, form TWO reference centroids from UNCORRUPTED
calibration windows:
- windows the model predicts negative using its already frozen threshold;
- windows the model predicts positive using its already frozen threshold.

For a runtime window:
- use the model's CURRENT predicted class;
- compare its normalized attribution profile to the corresponding calibration
  predicted-class centroid.

Ground truth is not used to select the reference centroid.

### Attribution-profile divergence

Use Jensen-Shannon divergence between:
- current normalized 33-sensor attribution profile;
- frozen calibration predicted-class centroid.

Natural logarithms are used.

This is called:
`attribution_profile_divergence`

It is a specific operationalization of explanation instability, not a claim
that it captures all forms of explanation instability.

## Attribution evaluation subset

Prediction, confidence, corruption, reconstruction and controller metrics are
computed on all eligible windows.

To keep full-gradient attribution tractable and pre-specified, attribution
analysis uses a deterministic temporal subset:

Candidate final-index positions:
`[0,5,10,15,20,25,30,35,40,44]`

within the frozen list of 45 candidate windows per physical variant.

This yields:
- 10 attribution windows / physical variant / context;
- 20 attribution windows per physical context;
- all calibration contexts for reference/detector development;
- all held-out contexts for final attribution analysis.

No attribution window is selected based on labels, predictions, or failures.

## Evidence-failure labels

For reliability-signal evaluation only:
- uncorrupted evidence: evidence_failure = 0
- any generated degradation: evidence_failure = 1

The corruption family/severity is NEVER supplied to the predictive model or
runtime reliability signal.

## Next stage after this freeze

Generate CALIBRATION degradations only.

From calibration data only:
- build attribution centroids;
- evaluate predictive entropy and attribution-profile divergence;
- build simple and graph-aware reconstruction candidates;
- select/freeze reliability-controller thresholds and recovery rules.

Only after those are frozen may degraded TEST evaluation begin.

## Claim boundary

AQUA-VERA remains a controlled WNTR/Net3 simulation benchmark with a separate
external BattLeDIM analysis planned later.

No live-utility validation claim is permitted.
