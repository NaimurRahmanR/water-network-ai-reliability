# AQUA-VERA Final Degraded Held-Out Test v0.12

This is the first/final frozen degraded held-out evaluation.

Before generating any held-out corruption, the script:

1. verifies the exact source hashes of the already executed v0.9b corruption, v0.10b attribution, and v0.11 controller scripts;
2. verifies all frozen data/model/controller artifacts;
3. regenerates every calibration corruption selection with the frozen v0.9b code and requires it to match the frozen calibration corruption manifest.

Only then does it generate held-out degradations.

Reporting boundary:
- prediction, entropy, robustness, and missing-sensor predictive recovery: all 2,700 windows/condition;
- Input×Gradient attribution and the dual-signal controller: the pre-specified 600-window attribution subset/condition.

No new subset, model, threshold, corruption severity, reconstruction model, centroid, or action rule is selected here.

Run `run_final_degraded_test_windows.bat`.

Send back:
`C:\Users\<you>\Downloads\AQUA_VERA_final_degraded_test_v0_12\final_degraded_test_summary.txt`
