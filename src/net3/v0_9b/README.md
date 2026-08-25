# AQUA-VERA Calibration Degradations v0.9b

## Why v0.9 failed

The v0.8 freeze recorded the SHA-256 of the original packaged protocol file:

`0203ffcb0a456cbdb09c1186a9d669f0266cc85b200e13dbfcd53a62dd746b2c`

The freeze script then copied the protocol using Python `Path.write_text()` on
Windows. The text was unchanged, but Windows newline translation rewrote LF
line endings as CRLF. The resulting copied file therefore has a different
byte-level SHA-256:

`9c378c9396c6432ff555b6274606099ff262bc18fb6ac2e206631ca4b9266815`

v0.9 incorrectly expected the copied CRLF file itself to retain the original
LF file's hash.

## v0.9b correction

v0.9b verifies two separate facts:

1. `FROZEN_DEGRADATION_PROTOCOL_RECORD.json` must still declare the original
   source protocol hash `0203...`.
2. The Windows-copied `degradation_protocol_v0_8.md` must have the known CRLF
   byte hash `9c378...`.

No protocol content, corruption rule, seed, model, threshold, or split boundary
is changed.

The failed v0.9 run stopped before generating calibration degradation outputs.
Degraded held-out test remains sealed.

## Run

Double-click:

`run_calibration_degradations_v0_9b_windows.bat`

## Send back

`C:\Users\<you>\Downloads\AQUA_VERA_calibration_degradations_v0_9b\calibration_degradation_summary.txt`
