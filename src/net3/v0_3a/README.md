# AQUA-VERA Scenario Generator v0.3a

This is the first full data-generation step after the benchmark v0.3 freeze.

## Scientific constraints

The script verifies the exact frozen artifact hashes before running.

It does not:
- redesign the split;
- choose new sensors;
- choose new leak pipes;
- inspect BattLeDIM 2019;
- train any model;
- tune any threshold.

## Important implementation lock

Hydraulic and report timesteps are 300 seconds.

The original Net3 demand-pattern timestep from `Net3.inp` is preserved.
The earlier smoke test temporarily set pattern timing differently, but the
smoke test was explicitly not the full benchmark freeze.

The generator records the native pattern timestep in every generation record.

## Outputs

Each of the 240 base contexts produces one compressed `.npz` containing both
matched variants:

- CLEAN model-visible sensors
- LEAK model-visible sensors
- CLEAN full original-node pressure
- LEAK full original-node pressure
- CLEAN full original-link flow
- LEAK full original-link flow
- CLEAN active-leak label
- LEAK active-leak label
- time in seconds

This corresponds to 480 hydraulic simulations.

The original network node/link order and sensor feature order are recorded in:
- `tensor_layout.json`
- `graph_nodes.csv`
- `graph_links.csv`

## Resume behavior

Generation is resume-safe.

If interrupted, rerun the batch file. A context is skipped only when its stored
NPZ SHA-256 matches its context metadata.

## Run

Double-click:

`run_generate_windows.bat`

## Output

`C:\Users\<you>\Downloads\AQUA_VERA_generated_v0_3a\`

When complete, send:
`generation_summary.txt`

Do not train yet. The next step creates immutable 2-hour model windows and
training-only normalization statistics.
