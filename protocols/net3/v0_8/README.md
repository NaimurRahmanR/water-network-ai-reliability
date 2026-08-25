# AQUA-VERA Degradation Protocol Freeze v0.8

This is a protocol-only gate.

It does not generate corrupted examples and does not run either model.

It freezes:
- exact deterministic sensor-selection rules;
- corruption counts/magnitudes;
- topology mismatch semantics;
- counterfactual-conflict semantics;
- predictive entropy definition;
- Input x Gradient attribution-profile definition;
- calibration-only attribution reference;
- Jensen-Shannon attribution-profile divergence;
- deterministic attribution evaluation subset.

It also verifies that the clean held-out prediction file is still the exact
byte-for-byte v0.7b output.

Run:

`run_degradation_protocol_freeze_windows.bat`

Send back:

`C:\Users\<you>\Downloads\AQUA_VERA_degradation_protocol_frozen_v0_8\degradation_protocol_summary.txt`
