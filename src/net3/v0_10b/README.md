# AQUA-VERA Calibration Attribution v0.10b

## Why v0.10 failed

The scientific attribution method was not the problem.

For GCN-GRU degraded attribution, the implementation intentionally processes
one physical context at a time because missing-sensor masks and topology can
be context-specific.

Each context contributes 20 attribution windows:
- 10 CLEAN-variant windows
- 10 LEAK-variant windows

However, v0.10's audit helper created a lookup for all 600 frozen attribution
windows in the condition and then incorrectly required:

`len(full_condition_lookup) == len(current_context_batch)`

That compared `600` against `20` and stopped at the first GCN
`MISSING_SENSOR/low` context.

The error occurred before the per-window probability reproduction audit.

## v0.10b correction

The helper now:

1. takes the exact window IDs in the current attribution batch;
2. verifies every one exists in the frozen v0.9b condition predictions;
3. rejects duplicate IDs;
4. checks regenerated probability against the matching frozen probability for
   each window using the unchanged `2e-6` tolerance.

No scientific setting changes:
- same frozen v0.8 protocol;
- same v0.9b degraded predictions;
- same frozen MLP and GCN-GRU;
- same Input×Gradient method;
- same 10-window deterministic subset;
- same predicted-class centroids;
- same Jensen-Shannon divergence;
- same thresholds;
- degraded held-out test remains sealed.

v0.10 did not reach the output-freeze stage, so v0.10b produces the first
complete frozen attribution artifact set.

## Run

Double-click:

`run_calibration_attribution_v0_10b_windows.bat`

## Send back

`C:\Users\<you>\Downloads\AQUA_VERA_calibration_attribution_v0_10b\calibration_attribution_summary.txt`
