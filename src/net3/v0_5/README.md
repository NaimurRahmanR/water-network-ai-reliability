# AQUA-VERA Baseline MLP v0.5

This is the first actual model-training stage.

## Why the held-out test remains sealed

The graph model has not yet been trained/frozen. If we evaluated this baseline
on held-out test now, then used that information while designing the graph
model, the final model comparison would no longer be clean.

Therefore this script has a hard guard:
- loads TRAIN features
- loads VALIDATION features
- loads CALIBRATION features
- refuses to load TEST features

## Fixed model

Input:
- 24 time steps
- 33 model-visible sensors
- flattened dimension = 792

Architecture:
- 792 -> 256 -> 64 -> 1
- ReLU
- dropout 0.20

Training:
- weighted BCE from train-only class ratio
- AdamW
- lr = 1e-3
- weight decay = 1e-4
- batch size = 256
- seed = 314159
- max 50 epochs
- early stopping patience 7
- selection metric = validation AUPRC

Calibration:
- threshold selected on CALIBRATION only
- objective = maximum balanced accuracy
- deterministic tie-break

## Output

`C:\Users\<you>\Downloads\AQUA_VERA_baseline_mlp_v0_5\`

Expected:
- baseline_mlp_v0_5.pt
- baseline_config.json
- pretest_metrics.json
- training_history.csv
- validation_predictions.csv
- calibration_predictions.csv
- FROZEN_BASELINE_RECORD.json
- baseline_summary.txt

Retain `baseline_summary.txt` with the project record.

Next: graph model using train/validation/calibration only.
The held-out test remains sealed until both model families are frozen.
