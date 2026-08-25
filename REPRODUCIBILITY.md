# Reproducibility

The experiment was developed as a sequence of frozen stages so that later held-out results could not be used to tune earlier decisions.

## Controlled Net3 study

Run in order:

1. `protocols/net3/v0_1/` — initial protocol freeze.
2. `protocols/net3/v0_2/` — pre-training amendment moving the controlled benchmark to Net3 after the original BattLeDIM event split proved unsuitable.
3. `protocols/net3/v0_3/` — Net3 benchmark freeze.
4. `src/net3/v0_3a/` — generate the frozen Net3 simulations.
5. `src/net3/v0_4/` — freeze windows and training-only normalization.
6. `src/net3/v0_5/` — train the MLP.
7. `src/net3/v0_6/` — train the GCN-GRU.
8. `src/net3/v0_7b/` and `v0_7c/` — clean held-out evaluation and context-cluster bootstrap.
9. `protocols/net3/v0_8/` — freeze degradation definitions.
10. `src/net3/v0_9b/` — calibration degradations.
11. `src/net3/v0_10b/` — Input×Gradient attribution references.
12. `src/net3/v0_11/` — Ridge reconstruction and reliability-controller freeze.
13. `src/net3/v0_12/` — final degraded held-out evaluation.
14. `report/reproducibility_check.md` — independent post-experiment recomputation.

Final held-out accounting:

- 30 physical test contexts;
- 2,700 prediction windows per condition;
- 600 attribution/controller windows per condition;
- 94,500 final prediction rows;
- 21,000 attribution rows.

## BattLeDIM/L-Town external study

This is a separate extension.

1. `src/battledim/v0_14b/` — 2018-only event-structure discovery.
2. `protocols/battledim/v0_15/` — freeze the external recent-onset task.
3. `src/battledim/v0_16b/` — train/freeze the 2018 MLP, GCN-GRU, attribution references, thresholds, and pressure reconstructors.
4. `src/battledim/v0_16c/` — freeze exact corruption selections, topology removals, RNG seeds, and reporting definitions.
5. `src/battledim/v0_17/` — open 2019 once for the final external evaluation.
6. `src/battledim/v0_18/` — post-hoc read-only transfer diagnostic.

The v0.18 diagnostic does not retrain either model and does not alter the reported external result.

## Environment

Exact versions recorded in the Net3 model freeze:

- Python 3.12.2
- Windows 11
- NumPy 2.5.0
- pandas 3.0.3
- PyTorch 2.11.0+cpu
- WNTR 1.5.0

Later stages also use scikit-learn and NetworkX. Their exact historical versions were not preserved in the available freeze records, so I do not invent version pins for them.

## Repository integrity

```bash
python tools/verify_repository.py
```

This checks the repository manifest and several frozen scientific artifacts against their recorded SHA-256 values.

## External raw-output mirror

The exact v0.16b and v0.17 local output directories were not available in the workspace used to assemble this repository snapshot. Their source code, final summaries, and recorded artifact hashes are preserved.

If the original Windows output directories are still available, `tools/collect_external_artifacts_windows.bat` packages them read-only for a later Git commit. It does not rerun the experiment.


## Repository-release source cleanup

Some experiment-stage scripts originally printed workflow-only messages about where
generated summaries should be reviewed or retained. Those messages were not part of
model training, inference, data generation, corruption definitions, metric computation,
threshold selection, or controller logic.

For the public repository I removed or reworded those workflow-only messages. Historical
source-file SHA values recorded during the experiment therefore refer to the original
frozen experiment-stage files, not to these cleaned public copies.

Frozen numerical artifacts, model weights, prediction files, corruption manifests,
attribution profiles, reconstruction outputs, and controller outputs were not regenerated
for this repository cleanup.
