# AQUA-VERA Reliability Controller Freeze v0.11

This is the final pre-degraded-test methodological gate.

## Thresholds

No threshold is optimized against degraded labels.

For each frozen model:

- predictive-entropy threshold =
  95th percentile of CLEAN calibration entropy;
- attribution-profile-divergence threshold =
  95th percentile of CLEAN calibration Jensen-Shannon divergence.

The dual-signal controller therefore has a pre-specified clean false-alarm
budget rather than a threshold selected to maximize degraded-condition scores.

## Actions

- entropy low + attribution divergence low -> PROCEED
- entropy high only -> VERIFY
- attribution divergence high only -> REQUEST_SENSOR
- both high -> ESCALATE
- explicit missing-sensor mask -> RECONSTRUCT, then apply the same frozen
  dual-signal decision rule

`VERIFY`, `REQUEST_SENSOR`, and `ESCALATE` are safe deferral actions in this
controlled benchmark. They do not invent a recovered prediction.

## Missing-sensor reconstruction

24 independent pressure reconstruction models are fitted using TRAIN only.

For each pressure sensor:
- target = that normalized pressure stream;
- predictors = the other 32 model-visible normalized streams;
- model = Ridge(alpha=1.0).

Training rows:
150 train contexts x 2 physical variants x 289 timestamps = 86,700 unique rows.

For multiple missing pressure sensors:
- mean-impute them first (= normalized zero);
- reconstruct all selected targets simultaneously from that same pre-recovery
  vector;
- do one pass only.

For GCN-GRU, a reconstructed sensor remains `observation_mask=0`. The model is
never told that an imputed value was directly observed.

## Calibration evaluation

The script evaluates missing-sensor reconstruction and the full dual-signal
controller on the already-frozen attribution subset.

This evaluation is descriptive and remains pre-test.

## Run

Double-click:

`run_controller_freeze_windows.bat`

## Send back

`C:\Users\<you>\Downloads\AQUA_VERA_controller_frozen_v0_11\controller_freeze_summary.txt`

If this passes, the reliability policy is frozen and degraded held-out test
evaluation can begin.
