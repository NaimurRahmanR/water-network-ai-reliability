# Method

## Controlled Net3 benchmark

The controlled benchmark uses WNTR/EPANET Net3. I generated 240 physical base contexts and simulated matched clean/leak variants, producing 480 simulations.

The frozen split contains:

| Split | Physical contexts |
| --- | ---: |
| Train | 150 |
| Validation | 30 |
| Calibration | 30 |
| Test | 30 |

Matched clean/leak variants stay in the same split. Test leak pipes do not appear in training.

Each model receives a 2-hour window sampled every 5 minutes, with a 30-minute stride.

### Models

**MLP**

- flattened 24-step sensor window;
- two hidden layers;
- weighted binary cross-entropy;
- decision threshold selected on calibration and frozen before test.

**GCN-GRU**

- 97-node Net3 graph;
- sensor-derived node features;
- two graph-convolution layers;
- graph pooling;
- GRU temporal aggregation;
- frozen calibration threshold.

The graph model was not retuned after the clean held-out comparison.

## Controlled evidence degradation

The final held-out study applies frozen interventions to the evidence layer:

- missing pressure sensors;
- Gaussian sensor noise;
- persistent sensor bias;
- stale sensor readings;
- graph-topology mismatch;
- matched cross-source conflict.

The physical test context and target remain fixed while the model-visible evidence changes.

## Reliability signals

### Predictive entropy

Binary predictive entropy is normalized by `log(2)`.

### Attribution-profile divergence

For each selected window:

1. compute Input×Gradient with respect to the leak logit;
2. take absolute attribution;
3. sum over time to obtain a sensor-level profile;
4. normalize the profile;
5. compare it with the clean calibration centroid for the model's current predicted class using Jensen-Shannon divergence.

## Missing-sensor reconstruction

One Ridge regression is trained for each pressure sensor using training-only data. When multiple sensors are missing, the missing values are first set to the training mean and all selected sensors are reconstructed simultaneously in one pass.

For the GCN-GRU, reconstructed pressure values do not change the observation mask from missing to observed.

## Reliability controller

The final action is based on the frozen entropy and attribution-divergence thresholds.

| Signal state | Action |
| --- | --- |
| low entropy, low attribution divergence | `PROCEED` |
| high entropy only | `VERIFY` |
| high attribution divergence only | `REQUEST_SENSOR` |
| both high | `ESCALATE` |
| explicit missing pressure sensor | `RECONSTRUCT`, then re-evaluate |

`VERIFY`, `REQUEST_SENSOR`, and `ESCALATE` are deferral actions in this experiment. Only `RECONSTRUCT` performs an implemented evidence transformation.

## External BattLeDIM experiment

The original idea of classifying whether any leak is active was rejected before model training because it was positive for almost all 2018 time points.

The external target was therefore fixed prospectively as:

> a positive window if at least one leak onset occurred within the previous 24 hours.

The input window remains 2 hours with a 30-minute stride. The first 24 hours after a leak end are excluded.

The MLP and GCN-GRU training recipes, threshold `0.5`, attribution method, reliability thresholds, missing-sensor reconstruction, degradation implementation, RNG seeds, and controller reporting definitions were frozen before 2019 was opened.
