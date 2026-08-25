# AQUA-VERA Amendment v0.2 + Net3 Smoke Test

## Why this exists

The v0.1 BattLeDIM 2018 split gate failed because all buffered leak episodes
collapsed into one independent component. We do not bypass that scientific
constraint with a random split.

Instead:

- v0.1 remains preserved.
- v0.2 records the design correction before training.
- WNTR/Net3 becomes the primary controlled scenario benchmark.
- BattLeDIM remains the external benchmark.

## Run

Extract and double-click:

`run_amendment_and_smoke_windows.bat`

The script will:

1. verify the v0.1 frozen protocol hash;
2. freeze the v0.2 amendment;
3. install WNTR if needed;
4. load your existing Net3.inp;
5. run one clean 24-hour PDD simulation;
6. split an ordinary pipe and add a timed leak;
7. run a matched leak simulation;
8. verify finite hydraulic outputs and a measurable pressure change.

## Output

Amendment:
`C:\Users\<you>\Downloads\AQUA_VERA_protocol_amended_v0_2\`

Smoke test:
`C:\Users\<you>\Downloads\AQUA_VERA_net3_smoke\smoke_test_summary.txt`

Retain `smoke_test_summary.txt` with the project record.

The smoke parameters are NOT the final benchmark parameters. Exact scenario
counts, distributions, sensor placement and seeds will be frozen only after
the smoke test passes.
