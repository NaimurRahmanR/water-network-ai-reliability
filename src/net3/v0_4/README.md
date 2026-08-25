# AQUA-VERA Window + Normalization Freeze v0.4

This is the final data-design gate before model training.

## Frozen window design

- 2-hour observation windows
- 24 samples per window
- 30-minute stride
- 45 windows per physical variant
- CLEAN and LEAK variants retained
- target = active leak at the final timestamp
- matched pair ID = same context + same final timestamp

The context-level train/validation/calibration/test split from benchmark v0.3
is inherited unchanged.

## Frozen normalization

Normalization is computed from:
- TRAIN contexts only
- both CLEAN and LEAK physical variants
- unique 5-minute rows, not overlapping windows

Validation, calibration and test values do not influence mean or scale.

## Why windows are not copied

The script writes an immutable `window_manifest.csv` pointing to slices inside
the already checksum-verified context NPZ files. This avoids duplicating tens
of thousands of overlapping window tensors while keeping every sample exactly
reconstructible.

## Run

Double-click:

`run_window_freeze_windows.bat`

## Output

`C:\Users\<you>\Downloads\AQUA_VERA_windows_frozen_v0_4\`

Expected:
- window_manifest.csv
- normalization_train_only.json
- window_config.json
- FROZEN_WINDOW_RECORD.json
- window_summary.txt

Retain `window_summary.txt` with the project record.

If this passes review, model training can begin.
