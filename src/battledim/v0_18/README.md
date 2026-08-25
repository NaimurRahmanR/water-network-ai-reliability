# AQUA-VERA External Transfer Failure Diagnostic v0.18

Read-only post-hoc analysis of the completed BattLeDIM 2019 external test.

It does NOT retrain models, change thresholds, regenerate degradations, or
replace the external result.

It measures:
- 2018 -> 2019 feature drift (PSI, KS, standardized mean shift);
- clean score-distribution shift;
- calibration degradation;
- monthly 2019 performance;
- event-window detection behavior.

Run:
`run_external_failure_diagnostic_v0_18_windows.bat`

Send:
`C:\Users\<you>\Downloads\AQUA_VERA_external_failure_diagnostic_v0_18\transfer_failure_diagnostic_summary_v0_18.txt`
