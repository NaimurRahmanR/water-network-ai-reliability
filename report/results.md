# Results

## Clean Net3 held-out test

| Metric | MLP | GCN-GRU |
| --- | ---: | ---: |
| Accuracy | 0.937037 | 0.900741 |
| Balanced accuracy | 0.897309 | 0.830169 |
| F1 | 0.835907 | 0.735178 |
| AUPRC | 0.903734 | 0.826005 |
| AUROC | 0.944324 | 0.898841 |
| Brier | 0.057147 | 0.078914 |

A paired 2,000-resample bootstrap clustered by the 30 physical contexts gave the following MLP-minus-GCN differences:

- balanced accuracy: +0.067117, 95% CI [0.007222, 0.125348];
- F1: +0.101334, [0.017702, 0.187264];
- AUPRC: +0.078651, [-0.000540, 0.153912];
- AUROC: +0.045932, [-0.002292, 0.095922];
- Brier advantage: +0.021783, [0.002730, 0.039595].

The clean MLP advantage is therefore supported for balanced accuracy, F1, and Brier under this bootstrap. AUPRC and AUROC point estimates favour the MLP, but their intervals cross zero.

## Evidence-failure detection

Held-out predictive-entropy AUROC across the applicable degradation conditions:

- minimum: 0.160137
- median: 0.494997
- maximum: 0.577355

Attribution-profile divergence on the identical frozen subset:

- minimum: 0.479386
- median: 0.548489
- maximum: 0.978678
- attribution AUROC > entropy AUROC in 27/33 conditions

The result supports a narrow conclusion: attribution-profile divergence carried more evidence-failure information than predictive entropy in most tested conditions, but the signal was highly condition dependent.

## Missing-sensor reconstruction

| Model | Missingness | Before | After | Accuracy change |
| --- | --- | ---: | ---: | ---: |
| MLP | Low | 0.707037 | 0.928519 | +0.221481 |
| MLP | Medium | 0.561481 | 0.691481 | +0.130000 |
| MLP | High | 0.454074 | 0.567037 | +0.112963 |
| GCN-GRU | Low | 0.884815 | 0.865556 | -0.019259 |
| GCN-GRU | Medium | 0.801852 | 0.669630 | -0.132222 |
| GCN-GRU | High | 0.710741 | 0.488519 | -0.222222 |

The same recovery method behaved differently across architectures.

## Reliability controller

On the frozen attribution subset:

| Model | State | Proceed rate | Unsafe proceed |
| --- | --- | ---: | ---: |
| MLP | Clean | 0.860000 | 0.026667 |
| MLP | Degraded | 0.824889 | 0.170111 |
| GCN-GRU | Clean | 0.903333 | 0.058333 |
| GCN-GRU | Degraded | 0.775093 | 0.079722 |

For degraded windows, the raw prediction error before recovery was 0.247111 for the MLP and 0.201574 for the GCN-GRU. After the prescribed missing-sensor reconstruction, the forced-action error was 0.218000 and 0.225833 respectively. This distinction matters because reconstruction helped the MLP and harmed the GCN-GRU.

## BattLeDIM 2019 external evaluation

2019 contains 17,181 eligible windows, including 887 positive recent-onset windows.

| Model | Balanced accuracy | F1 | AUPRC | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.483765 | 0.084387 | 0.053443 | 0.498085 | 0.249789 |
| GCN-GRU | 0.472635 | 0.087472 | 0.053366 | 0.483409 | 0.307372 |

Both predictors failed to transfer usefully to 2019.

The external reliability-signal ordering was still similar in part:

- entropy evidence-failure AUROC median: 0.498144;
- attribution-divergence AUROC median: 0.532679;
- attribution > entropy in 23/27 conditions.

These results are descriptive. They do not make the external system useful because the underlying clean predictor itself is near chance.

## Post-hoc transfer diagnostic

The read-only 2018→2019 diagnostic found:

- PSI >= 0.25 in 32/37 features;
- median PSI 0.749274;
- maximum PSI 0.960571;
- median KS 0.355289;
- maximum KS 0.402416.

The diagnostic characterizes the transfer failure. It is not used to retune the reported external evaluation.
