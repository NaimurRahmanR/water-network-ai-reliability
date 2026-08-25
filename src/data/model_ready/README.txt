AQUA-VERA MODEL-READY PACK

Purpose
-------
The structural audit passed, but the original BattLeDIM sensor values are not
being parsed as ordinary decimal-point numbers by pandas. This preprocessor:

1. Reads the original semicolon-delimited files as strings.
2. Tests decimal-point and European decimal-comma parsing for every numeric column.
3. FAILS rather than silently coercing malformed values.
4. Validates exact 5-minute timestamps and 105,120 rows per year.
5. Merges pressure, flow and tank level by timestamp.
6. Creates model-ready comma-separated, decimal-point files.
7. Preserves 2018 and 2019 as separate years.
8. Creates SHA-256 hashes and a manifest.

It DOES NOT modify the original BattLeDIM files.

Run on Windows
--------------
Double-click:
    run_model_ready_windows.bat

Output
------
C:\Users\<you>\Downloads\AQUA_VERA_model_ready\

Expected:
- 2018_scada_model_ready.csv
- 2018_leakages_model_ready.csv
- 2019_scada_model_ready.csv
- 2019_leakages_model_ready.csv
- model_ready_manifest.json
- model_ready_summary.txt

Scientific gate
---------------
2019 is reserved for final held-out evaluation.
Do not use 2019 for threshold choice, early stopping, feature selection,
hyperparameter tuning, or reliability-controller calibration.

Retain model_ready_summary.txt back with the project record. before training.
