# BattLeDIM/L-Town External Validation Protocol Freeze v0.15

The 2018 discovery audit showed that `any active leak` is not a useful target:
approximately 97.8% of 2018 rows were positive and the 14 leakage intervals
merged into one global active component.

v0.15 therefore prospectively freezes the external task as:

**recent leak-onset detection: did at least one new leak begin during the
previous 24 hours?**

This is done before model training and before opening 2019 outcomes.

The run:
- verifies only 2018 input hashes;
- derives the 14 2018 onset/end transitions;
- freezes the 2-hour / 30-minute-stride window target;
- excludes the first 24 hours after repair/end transitions;
- freezes the transferred MLP and GCN-GRU training recipes;
- freezes five real-data-compatible degradation families;
- freezes the reliability-signal/controller definitions;
- explicitly omits CROSS_SOURCE_CONFLICT because BattLeDIM has no matched
  counterfactual physical pair.

It does **not** train models or open/hash 2019.

Run:

`run_battledim_protocol_freeze_v0_15_windows.bat`

Send:

`C:\Users\<you>\Downloads\AQUA_VERA_BattLeDIM_protocol_frozen_v0_15\BattLeDIM_protocol_freeze_summary_v0_15.txt`
