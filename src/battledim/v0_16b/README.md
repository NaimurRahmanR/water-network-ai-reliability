# AQUA-VERA BattLeDIM Train + Reliability Freeze v0.16b

This is an **implementation-only performance patch** to v0.16.

The scientific protocol is unchanged:
- 2018 only;
- same frozen v0.15 target/window manifest;
- same MLP and GCN-GRU architectures;
- same seeds;
- same weighted BCE;
- same optimizer hyperparameters;
- same effective batch size 128;
- same 25 MLP epochs / 28 GCN-GRU epochs;
- same predictive threshold 0.5;
- same attribution, reliability-threshold, and reconstruction rules;
- 2019 remains untouched.

## Why v0.16 looked frozen

After MLP epoch 25, v0.16 entered the much heavier 785-node GCN-GRU stage.
It printed nothing until an entire GCN epoch completed.

## What v0.16b changes

1. Replaces gather + `index_add_` GCN aggregation with
   `torch.sparse.mm` using the exact same frozen symmetric normalized adjacency.
2. Prints progress inside every GCN epoch.
3. Uses unbuffered Python console output.
4. Writes an epoch-boundary GCN resume checkpoint. If interrupted, rerun the
   same BAT and it resumes after the last completed GCN epoch. The checkpoint
   is deleted after the final GCN model is successfully frozen.

No model/data/scientific rule is changed.

## Run

Stop the old v0.16 run with Ctrl+C, extract this package, then run:

`run_battledim_train_freeze_v0_16b_windows.bat`

Final output:

`C:\Users\<you>\Downloads\AQUA_VERA_BattLeDIM_train_frozen_v0_16b\BattLeDIM_train_freeze_summary_v0_16b.txt`
