# AQUA-VERA BattLeDIM/L-Town External Validation Protocol v0.15

## Status
Freeze before BattLeDIM model training and before inspecting 2019 leakage outcomes.

## Motivation from 2018-only discovery

The diagnostic label `any leakage column > 0 at timestamp` is unsuitable:
2018 is positive for approximately 97.8% of 5-minute rows and the 14
per-column active intervals merge into one global active component.

Therefore the external task is NOT active-leak presence.

## Primary external task

**Recent leak-onset detection**

At a candidate window final timestamp `t`, the binary target is:

`1` iff at least one 2018 leak column transitions from non-positive to positive
during `(t - 24 hours, t]`.

Otherwise the target is `0`, except for windows excluded by the repair guard
defined below.

This is a retrospective event-onset task using the fully released BattLeDIM
ground-truth leakage traces. It is not a claim that the full traces were
available to competition participants in 2018/2019.

### Why 24 hours

This external study is about evidence-reliability replication, not reproducing
the BattLeDIM competition score. A 24-hour recent-onset horizon provides enough
positive training windows from 14 historical 2018 leakage events while retaining
a temporally meaningful event-detection task.

Do NOT call the target "early warning" or "instant detection."

## Repair/end guard

For each leakage column, a positive-to-non-positive transition is treated as a
repair/end transition.

Candidate windows with final timestamps in:

`[repair_time, repair_time + 24 hours)`

are excluded from predictive-model training/evaluation so that recovery from a
repair is not mislabeled as an ordinary stable negative period.

Pre-onset windows remain valid negatives. Existing older active leaks are also
valid negatives if no NEW onset occurred within the previous 24 hours.

That is intentional: the task asks whether a new leak event has recently begun,
not whether the system contains any leak whatsoever.

## Windowing

- input length: 24 samples = 2 hours
- source resolution: 5 minutes
- stride: 6 samples = 30 minutes
- final-timestamp recent-onset label
- 2018 only for development/training
- 2019 only for final external evaluation

## Development-year usage

2018 is the complete development year.

No held-out 2018 test claim will be made.

The final external claim is entirely based on sealed 2019.

## Normalization

Feature mean and standard deviation:
- computed from all unique 2018 SCADA rows only;
- no 2019 values;
- no leakage labels used in normalization.

## Predictive models

The external validation deliberately transfers the already-fixed AQUA-VERA
training recipe instead of tuning architectures on BattLeDIM.

### Non-graph MLP

Input:
- 24 x 37 normalized SCADA values = 888 features.

Architecture:
- 888 -> 256 -> 64 -> 1
- ReLU
- dropout 0.20

Training:
- weighted binary cross entropy
- positive weight computed from eligible 2018 windows
- AdamW
- learning rate 0.001
- weight decay 0.0001
- batch size 128
- exactly 25 epochs
- seed 314159

The 25-epoch duration is transferred from the previously frozen Net3 MLP
training result. It is not selected using BattLeDIM outcomes.

### Sparse-observation L-Town GCN-GRU

Graph:
- L-TOWN.inp
- 785 nodes
- 909 edges
- 33 pressure sensors mapped to nodes
- 3 observed-flow sensors mapped to links
- 1 tank-level stream mapped to its tank node

Node features retain the AQUA-VERA graph representation:
1. normalized observed pressure
2. pressure observation mask
3. normalized tank level
4. tank observation mask
5. signed normalized observed-flow aggregate
6. observed-flow incidence count
7. normalized graph degree
8. junction indicator
9. tank indicator
10. reservoir indicator

Architecture:
- GCN 10 -> 32 -> 32
- mean + max graph pooling
- GRU hidden size 64
- head 64 -> 32 -> 1
- dropout 0.20

Training:
- same weighted BCE as MLP
- AdamW learning rate 0.0008
- weight decay 0.0001
- batch size 128
- exactly 28 epochs
- seed 2718281

The 28-epoch duration is transferred from the previously frozen Net3
GCN-GRU result and is not tuned on BattLeDIM.

## Predictive decision threshold

Primary predictive threshold for both models:
`0.5`

No BattLeDIM label-based threshold optimization is permitted.

AUPRC/AUROC remain threshold-free descriptive metrics.

## External controlled evidence degradations

The final 2019 evaluation will use five real-data-compatible degradation
families.

### 1. MISSING_SENSOR
Pressure sensors selected deterministically by physical window-independent
context key.

Counts among 33 pressure sensors:
- low: 3
- medium: 8
- high: 13

Values are mean-imputed using 2018 means.
For GCN-GRU their pressure observation mask becomes `0`.

### 2. SENSOR_NOISE
8 / 33 pressure sensors at each severity.

Independent Gaussian noise:
- low: 0.25 x 2018 training SD
- medium: 0.50 x SD
- high: 1.00 x SD

### 3. SENSOR_BIAS
8 / 33 pressure sensors at each severity.

Persistent offset:
- low: 0.50 x 2018 SD
- medium: 1.00 x SD
- high: 2.00 x SD

Sign is deterministic per selected sensor.

### 4. STALE_SENSOR
8 / 33 pressure sensors.

Lag:
- low: 30 min
- medium: 60 min
- high: 120 min

Only factual past values are used.

### 5. TOPOLOGY_MISMATCH
GCN-GRU only.

Remove:
- low: 1 ordinary pipe edge
- medium: 3
- high: 5

Only connectivity-preserving removals are allowed.
Physical SCADA evidence remains unchanged.
Topology-derived degree is recomputed from the mismatched model-visible graph.

## Omitted degradation

`CROSS_SOURCE_CONFLICT` is NOT included in BattLeDIM external validation.

Reason:
the observational dataset does not provide the matched factual/counterfactual
physical pair required by AQUA-VERA's controlled conflict construction.

We will not invent a weaker replacement and call it equivalent.

## Reliability signals

### Predictive entropy

Normalized binary entropy:
`H(p) / log(2)`.

### Attribution-profile divergence

Same frozen method as core AQUA-VERA:
- Input x Gradient with respect to leak-onset logit;
- absolute attribution;
- sum across 24 time steps;
- normalize across 37 SCADA streams;
- predicted-class clean-2018 reference centroid;
- Jensen-Shannon divergence.

## Attribution subset

For computational tractability, use candidate windows with:

`window_index mod 10 == 0`

This deterministic 10% temporal subset is used for:
- 2018 attribution reference distributions;
- final 2019 attribution analysis;
- dual-signal controller evaluation.

It is label-independent.

## Reliability thresholds

After the final 2018 models are trained:

For each model separately:
- entropy threshold = 95th percentile of CLEAN 2018 attribution-subset entropy;
- attribution threshold = 95th percentile of CLEAN 2018 attribution divergence.

No degraded 2018 labels are used to choose reliability thresholds.

## Missing-sensor reconstruction

Fit 33 independent Ridge regressions using unique normalized 2018 SCADA rows.

For each pressure sensor:
- target = that pressure sensor;
- predictors = the other 36 model-visible streams;
- alpha = 1.0.

For multiple missing sensors:
- mean-impute all missing pressure inputs first;
- reconstruct all selected pressures simultaneously from that same vector;
- one pass only.

For GCN-GRU:
- reconstructed pressure mask remains `0`.

## Controller

Same action semantics as core AQUA-VERA:

- entropy low + attribution low -> PROCEED
- entropy high only -> VERIFY
- attribution high only -> REQUEST_SENSOR
- both high -> ESCALATE
- explicit missing sensor -> RECONSTRUCT, then re-evaluate

`VERIFY`, `REQUEST_SENSOR`, and `ESCALATE` are safe deferrals in this benchmark;
they do not fabricate a corrected decision.

Unsafe proceed:
`final action == PROCEED AND final prediction incorrect`.

## External evaluation boundary

2019 may be opened only after:
1. v0.15 protocol freeze passes;
2. final 2018 MLP and GCN-GRU are trained;
3. normalization is frozen;
4. attribution centroids are frozen;
5. reliability thresholds are frozen;
6. reconstruction models are frozen;
7. all code/hash checks pass.

Once 2019 outcomes are opened:
- no retraining;
- no threshold changes;
- no corruption changes;
- no controller changes.

## Primary replication questions

R1. Does predictive entropy remain poorly aligned with controlled evidence
degradation on BattLeDIM/L-Town?

R2. Does attribution-profile divergence outperform entropy across a majority
of real-data-compatible degradation conditions?

R3. Is missing-sensor reconstruction architecture-dependent rather than
uniformly beneficial?

R4. Does the frozen dual-signal controller reduce unsafe proceeding relative
to raw always-act prediction under degraded evidence?

These are replication questions, not guaranteed hypotheses.

## Claim boundary

This is:
- retrospective external validation on a real-network-derived benchmark;
- not live utility deployment;
- not BattLeDIM leaderboard reproduction;
- not SOTA leak detection;
- not early-warning validation.
