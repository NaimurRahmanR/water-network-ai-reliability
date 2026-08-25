# AQUA-VERA BattLeDIM Final-Evaluation Implementation Freeze v0.16c

This is the last integrity gate before opening 2019.

The v0.15 scientific protocol already froze the external task, models,
degradation families, reliability signals and controller. v0.16b froze the
trained 2018 models and reliability references.

v0.16c freezes the remaining implementation-level choices that must not be
chosen after seeing 2019:

- exact deterministic pressure-sensor selections;
- exact topology-mismatch edge selections;
- Gaussian-noise RNG seeds;
- persistent sensor-bias signs;
- time-series-level corruption semantics for overlapping windows;
- attribution-subset accounting;
- raw always-act vs recovery-prescribed vs unsafe-proceed reporting definitions;
- descriptive-only external inference policy.

It verifies the frozen v0.16b artifact hashes and L-Town model, and it does not
check/open/hash any 2019 file.

Run:

`run_battledim_final_eval_freeze_v0_16c_windows.bat`

Send:

`C:\Users\<you>\Downloads\AQUA_VERA_BattLeDIM_final_eval_frozen_v0_16c\BattLeDIM_final_eval_freeze_summary_v0_16c.txt`
