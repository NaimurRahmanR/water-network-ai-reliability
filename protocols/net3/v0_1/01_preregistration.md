# AQUA-VERA Experimental Protocol Freeze v0.1

## Status
This protocol is to be frozen **before any model training or inspection of 2019 performance**.

## Research objective
Evaluate whether graph-based water-distribution AI remains reliable when operational evidence is degraded, and whether reliability signals and recovery/control mechanisms can reduce unsafe proceeding.

## Data
Primary benchmark: BattLeDIM L-Town.

Model-ready source files and expected SHA-256 hashes:

- 2018_scada_model_ready.csv  
  `00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e`
- 2018_leakages_model_ready.csv  
  `743ea7b2845b3dced2ada4fba1a54194c54fc9ce18bf7180a1b3afa4b3eaf2f8`
- 2019_scada_model_ready.csv  
  `a021ade90c3cfa17362520f0fd7e9794a6c816e4787d7e49a3e52fdd6aa0d15d`
- 2019_leakages_model_ready.csv  
  `76fa8c50f19b6dffc2010b4413681179c43da5aa984374b9062897b6fe45c27f`

L-Town topology is taken from the previously checksum-verified `L-TOWN.inp`.

## Data separation
- **2018**: development year only.
- **2019**: final untouched held-out evaluation year.
- 2019 may not be used for:
  - feature selection,
  - architecture selection,
  - early stopping,
  - threshold selection,
  - corruption severity tuning,
  - controller tuning,
  - explanation threshold tuning,
  - reconstruction method selection.

2019 labels may be opened only for final scoring after all development choices are frozen.

## Development splitting rule within 2018
We will construct windows from 2018 only.

Primary temporal input window:
- 24 observations = 2 hours at 5-minute sampling.

Primary target:
- **binary active-leak detection at the final timestamp of the window**.

Secondary analyses:
- sensor-level evidence-failure detection;
- decision recovery after reconstruction;
- explanation instability;
- leak localisation only if a scientifically defensible localisation target can be derived without excluding unseen 2019 leaks.

To reduce event leakage:
- windows belonging to the same contiguous leak episode must remain in one split;
- event groups are assigned with a fixed seed;
- negative windows are assigned in blocked chronological groups, not individually shuffled.

Development split target:
- train: approximately 70% of 2018 groups;
- validation: approximately 15%;
- calibration: approximately 15%.

The exact counts are generated deterministically from the grouping rule and recorded before training.

## Inputs
Dynamic SCADA features:
- 33 pressure sensors;
- 3 flow sensors;
- 1 tank-level sensor.

Graph:
- L-Town EPANET topology.
- Static node/edge attributes may be derived from the `.inp` file.
- Static features must not include hidden leak labels.

## Baselines
At minimum:

1. **Non-graph temporal baseline**
   - MLP/temporal model using the same observed SCADA evidence.

2. **Graph model**
   - graph-based model using the L-Town topology and identical observation windows.

No architecture may be selected using 2019 performance.

## Controlled evidence degradation
Applied to clean held-out inputs without changing the physical leak ground truth.

Frozen degradation families:

1. `MISSING_SENSOR`
   - mask a fixed fraction of observed pressure sensors.

2. `SENSOR_NOISE`
   - additive noise scaled using **2018 training statistics only**.

3. `SENSOR_BIAS`
   - persistent offset scaled using **2018 training statistics only**.

4. `STALE_SENSOR`
   - replace selected current values with earlier values from the same observed stream.

5. `TOPOLOGY_MISMATCH`
   - perturb model-visible topology only; physical labels and SCADA stream remain unchanged.
   - perturbations must preserve a valid graph and be generated with a fixed seed.

6. `CROSS_SOURCE_CONFLICT`
   - introduce a controlled inconsistency between selected sensor evidence and another model-visible evidence source without changing hidden physical ground truth.

Severity levels must be fixed using 2018 development/calibration data before 2019 scoring.

## Reliability signals
Primary candidate signals:

- predictive probability / entropy;
- ensemble disagreement (if ensemble is used);
- explanation instability relative to a calibration reference;
- reconstruction residual / sensor-integrity score;
- independent topology/hydraulic-consistency signal where implementable without oracle information.

Any signal threshold must be set using 2018 calibration only.

## Reconstruction experiment
For missing/anomalous sensors:

- reconstruct selected sensor evidence using only model-visible information;
- evaluate both:
  1. sensor reconstruction error;
  2. downstream leak-decision recovery.

Primary comparison:
- no reconstruction;
- simple local/temporal reconstruction baseline;
- graph-aware reconstruction.

## Sensor-prioritisation / information-value experiment
When multiple sensors are unreliable or unavailable:

Compare:
- random sensor request;
- largest reconstruction residual;
- model attribution;
- combined reliability score.

Primary outcome:
- downstream decision recovery after **one** sensor request.

This is described as **VoI-inspired sensor prioritisation** unless a formal Value-of-Information objective is explicitly implemented.

## Reliability controller
Actions:

- `PROCEED`
- `RECONSTRUCT`
- `VERIFY`
- `REQUEST_SENSOR`
- `ESCALATE`

Controllers:

1. always proceed;
2. uncertainty-only;
3. evidence-integrity-only;
4. hybrid reliability controller.

Controller policy and thresholds must be frozen on 2018 calibration before final 2019 evaluation.

## Primary metrics
### Prediction
- balanced accuracy;
- macro F1;
- AUROC;
- AUPRC.

### Calibration
- Brier score;
- negative log-likelihood;
- ECE (with binning rule fixed before final test).

### Evidence-failure detection
- AUROC;
- AUPRC.

### Reconstruction
- MAE/RMSE on reconstructed sensors;
- downstream decision recovery rate.

### Controller
- accuracy;
- unsafe-proceed rate;
- escalation/request rate;
- extra evidence/reconstruction cost.

## Primary safety-style definition
`unsafe_proceed = 1` when:
- the controller selects `PROCEED`, and
- the underlying leak decision is incorrect.

This definition is frozen before 2019 evaluation.

## Hypotheses
H1. Controlled evidence degradation reduces leak-decision reliability relative to matched clean inputs.

H2. Predictive confidence/entropy alone will not perfectly identify evidence failures.

H3. Explanation instability contains useful evidence-failure information beyond raw predictive confidence.

H4. Graph-aware reconstruction improves downstream decision reliability under missing/anomalous sensor evidence relative to no reconstruction.

H5. Evidence-aware sensor prioritisation improves one-query decision recovery over random sensor selection.

H6. A hybrid controller reduces unsafe proceeding relative to always-proceed and uncertainty-only control.

All hypotheses may fail; negative results will be reported.

## Leakage / oracle prohibitions
The model/controller must never receive:
- 2019 labels during development;
- corruption condition identity as an input feature;
- hidden clean values before an explicit reconstruction/request operation;
- true leak status or true leak location as a reliability signal;
- future timestamps beyond the observation window.

## Reproducibility
Record:
- Python version;
- package versions;
- fixed seeds;
- exact data hashes;
- exact split manifest;
- exact corruption manifest;
- model configs;
- raw predictions;
- calibration thresholds;
- controller traces;
- evaluation outputs.

## Claim boundary
This is a controlled benchmark study on public/simulated or benchmark WDS data. It must not be described as live utility deployment or field validation.
