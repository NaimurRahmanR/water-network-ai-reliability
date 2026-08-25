# AQUA-VERA Graph GCN-GRU v0.6

This trains the frozen topology-aware comparison model.

## Test integrity

The script has a hard guard against loading the held-out TEST contexts.

It uses:
- TRAIN for parameter learning
- VALIDATION for early stopping
- CALIBRATION for the frozen decision threshold
- TEST: not loaded

Only after this graph model and the non-graph MLP are both frozen should the
controlled held-out test be opened.

## Same dynamic evidence as the MLP

The graph model uses exactly the same 33 model-visible sensor streams:
- 24 pressure
- 6 flow
- 3 tank level

It adds only fixed Net3 topology/static node information.

## Node representation

97 Net3 nodes.

Dynamic channels:
1. normalized pressure value
2. pressure mask
3. normalized tank-level value
4. tank mask
5. signed normalized observed flow aggregate
6. observed-flow incidence count

Static channels:
7. normalized graph degree
8. junction indicator
9. tank indicator
10. reservoir indicator

Observed flow direction convention:
- negative at the original link start node
- positive at the original link end node

## Architecture

At every one of the 24 time steps:
- 2-layer normalized GCN: 10 -> 32 -> 32
- global mean + max graph pooling -> 64

Across time:
- GRU hidden size 64
- binary head 64 -> 32 -> 1

Training:
- weighted BCE
- AdamW
- lr 8e-4
- weight decay 1e-4
- batch size 128
- dropout 0.20
- seed 2718281
- max 40 epochs
- patience 6
- early stopping metric: validation AUPRC

Calibration:
- threshold chosen on calibration only
- objective: balanced accuracy

## Run

Double-click:
`run_graph_gcn_gru_windows.bat`

## Output

`C:\Users\<you>\Downloads\AQUA_VERA_graph_gcn_gru_v0_6\`

Send:
`graph_summary.txt`

If it passes review, both model families are frozen and we can open the
controlled held-out test once.
