# AQUA-VERA Controlled Benchmark Freeze v0.3

Run this after the successful Net3 smoke test.

It does NOT run the full benchmark. It freezes the design first.

## Frozen design

- 240 independent base hydraulic contexts.
- Matched CLEAN + LEAK variant for every context.
- 480 total hydraulic simulations.
- 150 / 30 / 30 / 30 base-context split:
  train / validation / calibration / held-out test.
- Held-out leak pipe pool is disjoint from training leak pipe pool.
- 24 pressure sensors selected from topology only.
- 6 flow sensors selected from topology only.
- all Net3 tank levels observed.
- deterministic seeds and scenario manifest.
- exact corruption severities frozen.

## Run

Double-click:

`run_benchmark_freeze_windows.bat`

## Output

`C:\Users\<you>\Downloads\AQUA_VERA_benchmark_frozen\`

Expected files:
- benchmark_protocol_v0_3.md
- sensor_layout.json
- leak_pipe_pools.json
- scenario_manifest.csv
- corruption_config.json
- benchmark_design.json
- FROZEN_BENCHMARK_RECORD.json
- benchmark_summary.txt

Retain `benchmark_summary.txt` with the project record before generating the 480 hydraulic simulations.
