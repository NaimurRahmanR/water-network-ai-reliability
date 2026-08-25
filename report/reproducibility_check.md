# Reproducibility and consistency check

After the controlled Net3 experiment was complete, I recomputed the reported results from the frozen output files and checked the split, corruption, attribution, reconstruction, and controller accounting.

This check did not rerun or retune the experiment.

## Artifact integrity

- 59 collected verification-package files matched their recorded SHA-256 values.
- All 13 final v0.12 scientific artifacts matched their recorded hashes.
- The final evaluator records the exact frozen v0.9b, v0.10b, and v0.11 source hashes before generating degraded-test outputs.

## Split isolation

| Split | Contexts | Windows | Positive windows |
| --- | ---: | ---: | ---: |
| Train | 150 | 13,500 | 2,636 |
| Validation | 30 | 2,700 | 580 |
| Calibration | 30 | 2,700 | 572 |
| Test | 30 | 2,700 | 520 |

Checks:

- zero physical-context overlap across splits;
- matched clean/leak variants remain in the same split;
- 18 test leak pipes have zero overlap with the 68 training leak pipes;
- model thresholds were selected using calibration rather than test data.

The first failed clean-test evaluator had already loaded test contexts before failing, but it produced no predictions or metrics. No model or threshold was changed before the corrected evaluation.

## Clean held-out metrics

The raw prediction file independently reproduces:

| Metric | MLP | GCN-GRU |
| --- | ---: | ---: |
| Accuracy | 0.937037 | 0.900741 |
| Balanced accuracy | 0.897309 | 0.830169 |
| F1 | 0.835907 | 0.735178 |
| AUPRC | 0.903734 | 0.826005 |
| AUROC | 0.944324 | 0.898841 |
| Brier | 0.057147 | 0.078914 |

The 2,000-resample context-cluster bootstrap also reproduces the stored comparison.

## Final degradation accounting

- 94,500 prediction rows;
- 35 model-visible conditions;
- 2,700 windows and 30 physical contexts per condition;
- no duplicated `(model, family, severity, window)` records;
- 21,000 attribution rows;
- 600 attribution windows per condition.

The realized corruption counts match the frozen definitions.

## Reliability signals

Recomputed entropy evidence-failure AUROC:

- min 0.160137
- median 0.494997
- max 0.577355

Recomputed attribution-divergence AUROC:

- min 0.479386
- median 0.548489
- max 0.978678
- attribution > entropy in 27/33 conditions

The Jensen-Shannon divergence values recompute from the saved profiles and frozen calibration centroids.

## Missing-sensor reconstruction

All stored pre/post accuracy values reproduce exactly. The MLP improves at all three severities; the GCN-GRU becomes progressively worse.

## Controller accounting

The controller actions reproduce from the frozen thresholds with zero action mismatches.

The reporting keeps three quantities separate:

1. raw degraded prediction error before recovery;
2. forced prediction error after the prescribed missing-sensor reconstruction;
3. unsafe proceed after the reliability controller.

On degraded attribution-subset windows:

| Model | Raw error | Post-reconstruction forced error | Unsafe proceed |
| --- | ---: | ---: | ---: |
| MLP | 0.247111 | 0.218000 | 0.170111 |
| GCN-GRU | 0.201574 | 0.225833 | 0.079722 |

This separation is important because reconstruction helps the MLP and harms the GCN-GRU.

## Numerical precision

Discrete metrics reproduce exactly to floating-point precision. A few ranking metrics differ from the stored tables only at very small levels after CSV serialization:

- maximum condition-metric difference: 3.51e-5;
- maximum entropy-detection difference: 1.31e-6;
- maximum attribution-detection difference: 2.78e-6.

These differences do not change any reported conclusion.

## Result

The controlled Net3 result is internally consistent with the frozen split, raw predictions, degradation manifest, attribution files, reconstruction outputs, controller outcomes, and stored thresholds.

The machine-readable recomputation is preserved in `outputs/verification/recomputed_net3_verification.json`.
