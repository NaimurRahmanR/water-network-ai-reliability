# AQUA-VERA 2018 Split Freeze v0.1

Run this **after** the protocol freeze and **before any model training**.

## What it does

- Verifies the frozen protocol hash:
  `5ea530772b1b4df01c3a437afc10929d570a1cf28fbffcdbc03db9bd0459bc41`
- Verifies the exact 2018 model-ready data hashes.
- Reads **2018 only**. It never opens the 2019 files.
- Defines the primary target at each final timestamp:
  active leak iff any leakage-rate column is > 1e-12.
- Detects contiguous leak episodes independently by leakage column.
- Adds a two-hour guard around episodes.
- Merges overlapping guarded episodes so shared raw input rows cannot be split.
- Uses two-hour observation windows.
- Uses a 30-minute candidate stride to reduce extreme overlap/redundancy.
- Groups pure-negative windows by ISO calendar week.
- Assigns whole groups deterministically to train/validation/calibration.
- Fails if it cannot preserve positive and negative examples in all three splits.
- Produces immutable manifests and hashes them.

## Run

Double-click:

`run_2018_split_freeze_windows.bat`

## Output

`C:\Users\<you>\Downloads\AQUA_VERA_2018_split_frozen\`

Expected files:

- `2018_event_catalog.csv`
- `2018_group_catalog.csv`
- `2018_split_manifest.csv`
- `2018_split_metadata.json`
- `FROZEN_2018_SPLIT_RECORD.json`
- `split_summary.txt`

## Gate

Send `split_summary.txt` back to the research record before model training.

If the script fails because too few independent event components remain after
guard merging, that is a scientific constraint in the data and should be
reviewed rather than bypassed.
