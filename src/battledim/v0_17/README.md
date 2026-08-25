# AQUA-VERA BattLeDIM/L-Town Final External 2019 Evaluation v0.17

This is the **final external evaluation** of the separate BattLeDIM extension.
It does not retrain or tune anything.

Before reading 2019, the evaluator verifies the frozen hashes for:

- 2018 normalization;
- L-Town graph input;
- v0.16b MLP;
- v0.16b GCN-GRU;
- 2018 reliability thresholds;
- 2018 attribution centroids;
- 2018 Ridge pressure reconstruction;
- v0.16c final-evaluation implementation freeze;
- `L-TOWN.inp`.

It then verifies and opens the already-reserved 2019 model-ready files by their
pre-existing SHA-256 hashes.

## Frozen external target

Same semantics as v0.15:

- input: 2-hour SCADA window;
- stride: 30 minutes;
- positive iff a new leak onset occurred in the previous 24 hours;
- windows in the first 24 hours after an observed leak end are excluded.

A year-boundary detail is fixed in this evaluator **before execution**:

- a leak already active at the first 2019 sample is treated as pre-existing,
  not as an artificial onset at midnight;
- a leak still active at the last 2019 sample is not assigned an unobserved end
  beyond the data.

This follows the frozen target wording of an observed non-positive -> positive
transition.

## Frozen degradation evaluation

- MISSING_SENSOR: MLP + GCN-GRU
- SENSOR_NOISE: MLP + GCN-GRU
- SENSOR_BIAS: MLP + GCN-GRU
- STALE_SENSOR: MLP + GCN-GRU
- TOPOLOGY_MISMATCH: GCN-GRU only

Exact sensors, topology edges, noise seeds and bias signs come only from v0.16c.

## Reliability evaluation

- normalized predictive entropy;
- Input x Gradient sensor-profile divergence to frozen predicted-class 2018
  centroids;
- frozen 95th-percentile 2018 reliability thresholds;
- frozen 2018 Ridge pressure reconstruction;
- frozen proceed / verify / request-sensor / escalate controller.

Controller reporting separately preserves:

1. raw always-act error before recovery;
2. recovery-prescribed forced error;
3. unsafe proceed after the controller.

## Checkpoint/resume behavior

This run is intentionally computationally heavy because the frozen GCN-GRU has
785 graph nodes and attribution requires gradients.

The evaluator writes deterministic per-condition checkpoints and prints
within-condition progress. If the process is interrupted, rerunning the same BAT
reuses only checkpoints created by the **same evaluator source hash and same
v0.16c freeze hash**. An attempt ledger records every started/completed run.

Checkpointing is operational resilience only. It does not permit any model,
threshold, corruption, target or controller change.

## Run

`run_battledim_final_external_2019_v0_17_windows.bat`

## Send back

`C:\Users\<you>\Downloads\AQUA_VERA_BattLeDIM_external_2019_v0_17\BattLeDIM_final_external_2019_summary_v0_17.txt`

After this run, do not tune from 2019. The next stage is a Reproducibility external
validation audit and final research interpretation.
