# AQUA-VERA Fast Held-Out Bootstrap Finalizer v0.7c

Use this if v0.7b reaches:

`[BOOTSTRAP] paired 30-context cluster bootstrap`

and appears stuck.

The slowdown is caused by the original bootstrap repeatedly filtering and
concatenating pandas DataFrames.

v0.7c does NOT rerun either model and does NOT regenerate predictions.

It reads:

`C:\Users\<you>\Downloads\AQUA_VERA_heldout_test_v0_7b\heldout_test_predictions.csv`

That file was saved before the slow bootstrap started.

It then performs the same 2,000-resample paired bootstrap clustered by the 30
held-out contexts using precomputed NumPy row indices.

Run:

`run_fast_bootstrap_finalize_windows.bat`

Send back:

`C:\Users\<you>\Downloads\AQUA_VERA_heldout_test_v0_7c\heldout_test_summary_v0_7c.txt`
