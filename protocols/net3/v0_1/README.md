# AQUA-VERA Protocol Freeze Pack v0.1

This is the next step after the model-ready preprocessing PASS.

## Run
Extract the ZIP and double-click:

`run_protocol_freeze_windows.bat`

It will:
1. verify all four model-ready CSV hashes;
2. hash the preregistration;
3. write a `FROZEN_PROTOCOL_RECORD.json`;
4. copy the frozen protocol/config into:

`C:\Users\<you>\Downloads\AQUA_VERA_protocol_frozen`

## Important
Do not train any model before this freeze succeeds.

After this, the next script will build the deterministic 2018 train/validation/calibration event-group split. 2019 remains untouched.
