AQUA-VERA DATA AUDIT v3

This replaces v1 and v2.

v1 issue:
- BattLeDIM CSVs are semicolon-delimited, but v1 assumed commas.

v2 issue:
- v2 selected pandas' Python engine while also passing low_memory=False.
- pandas does not support low_memory with the Python engine, so CSV parsing
  stopped before reading the data.

v3 fix:
- Detects semicolon delimiter.
- Tries pandas C engine with strict parsing.
- Falls back to Python engine without unsupported options.
- Checks timestamps, sampling regularity, missing values, numeric parsing,
  sensor columns, graph mappings and leak-link mappings.

RUN:
    run_audit_windows_v3.bat

OUTPUT:
    C:\Users\<you>\Downloads\AQUA_VERA_audit_v3\audit_summary_v3.txt

Retain audit_summary_v3.txt back with the project record.
