# Experiment history

The repository keeps the scientifically relevant development history because several failed stages changed the design.

## Initial BattLeDIM split attempt

The first plan was to use BattLeDIM 2018 directly for the primary benchmark. Under the proposed guarded event split, all active leakage periods merged into one component. I did not force a split that would violate the intended separation.

## Net3 amendment

Before model training, the controlled benchmark moved to WNTR/Net3. BattLeDIM was retained for a later external evaluation.

## Clean-test evaluator failure

The first Net3 clean-test evaluator loaded test contexts but failed before producing predictions or metrics because of a graph checkpoint compatibility problem. No test performance was observed and no model or threshold was changed before the corrected evaluation.

The public claim is therefore **no test performance was observed before the successful frozen evaluation**, not that the test files were literally never accessed.

## Calibration implementation fixes

Two later failures were implementation issues rather than scientific redesign:

- a Windows line-ending hash mismatch stopped one degradation run before outputs were generated;
- the first attribution audit helper used the wrong lookup scope and stopped before the attribution freeze.

Both were corrected without changing the frozen scientific method.

## BattLeDIM external task

A 2018-only discovery stage showed that `any active leak` was positive for almost the entire year. Before training the external models, the task was changed and frozen as recent leak-onset detection.

## External training implementation

The first L-Town GCN implementation was operationally too slow and gave little progress feedback. The replacement used mathematically equivalent sparse aggregation, fixed progress reporting, and epoch-boundary resume checkpoints. The model architecture, optimizer, seeds, batch size, and epoch count were unchanged.

## 2019 result

The 2019 external test was not rescued after failure. The subsequent transfer analysis is read-only and is kept separate from the frozen external result.
