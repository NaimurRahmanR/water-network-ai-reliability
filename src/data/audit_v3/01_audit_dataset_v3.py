
import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.")
    print("Run: py -m pip install pandas pyyaml")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is not installed.")
    print("Run: py -m pip install pandas pyyaml")
    sys.exit(1)


def detect_delimiter(path: Path):
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:65536]
    first = sample.splitlines()[0] if sample else ""
    if ";" in first:
        return ";"
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except Exception:
        return ","


def read_csv_strict(path: Path):
    sep = detect_delimiter(path)

    # First try the fast C engine. This supports low_memory=False.
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            encoding="utf-8-sig",
            engine="c",
            on_bad_lines="error",
            low_memory=False,
        )
        engine = "c"
    except Exception as first_error:
        # Fallback to Python engine WITHOUT low_memory (unsupported there).
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                encoding="utf-8-sig",
                engine="python",
                on_bad_lines="error",
            )
            engine = "python"
        except Exception:
            raise first_error

    return df, sep, engine


def parse_timestamp(series):
    # Try standard parsing first, then day-first fallback.
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().mean() < 0.95:
        s2 = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if s2.notna().mean() > s.notna().mean():
            s = s2
    return s


def read_csv_info(path: Path):
    info = {"file": path.name, "size_bytes": path.stat().st_size}
    df, sep, engine = read_csv_strict(path)
    info["delimiter"] = sep
    info["engine"] = engine
    info["rows"] = int(len(df))
    info["columns"] = [str(c) for c in df.columns]
    info["n_columns"] = int(len(df.columns))
    info["missing_total"] = int(df.isna().sum().sum())
    info["missing_by_column_top20"] = {
        str(k): int(v)
        for k, v in df.isna().sum().sort_values(ascending=False).head(20).items()
    }

    # Timestamp detection
    ts_candidates = [
        c for c in df.columns
        if any(x in str(c).strip().lower() for x in ["timestamp", "datetime", "date", "time"])
    ]
    if not ts_candidates and len(df.columns):
        ts_candidates = [df.columns[0]]

    parsed = None
    parsed_col = None
    for c in ts_candidates:
        s = parse_timestamp(df[c])
        if s.notna().mean() > 0.95:
            parsed = s
            parsed_col = c
            break

    if parsed is not None:
        valid = parsed.dropna().sort_values()
        info["timestamp_column"] = str(parsed_col)
        info["timestamp_parse_rate"] = float(parsed.notna().mean())
        info["timestamp_min"] = str(valid.iloc[0]) if len(valid) else None
        info["timestamp_max"] = str(valid.iloc[-1]) if len(valid) else None
        info["duplicate_timestamps"] = int(valid.duplicated().sum())
        diffs = valid.diff().dropna()
        if len(diffs):
            modes = diffs.mode()
            common = modes.iloc[0] if len(modes) else None
            info["most_common_interval"] = str(common) if common is not None else None
            info["nonstandard_intervals"] = int((diffs != common).sum()) if common is not None else None
    else:
        info["timestamp_column"] = None

    non_ts = [c for c in df.columns if c != parsed_col]
    info["data_columns"] = [str(c) for c in non_ts]
    info["n_data_columns"] = len(non_ts)

    numeric_rates = {}
    for c in non_ts[:100]:
        vals = pd.to_numeric(df[c], errors="coerce")
        numeric_rates[str(c)] = float(vals.notna().mean())
    info["numeric_parse_rate_first100"] = numeric_rates

    return info


def parse_inp(path: Path):
    sections = {}
    current = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line.upper()
                sections.setdefault(current, [])
                continue
            if current:
                content = line.split(";", 1)[0].strip()
                if content:
                    sections[current].append(content)

    junctions = sections.get("[JUNCTIONS]", [])
    reservoirs = sections.get("[RESERVOIRS]", [])
    tanks = sections.get("[TANKS]", [])
    pipes = sections.get("[PIPES]", [])
    pumps = sections.get("[PUMPS]", [])
    valves = sections.get("[VALVES]", [])

    node_ids = []
    for bucket in [junctions, reservoirs, tanks]:
        for line in bucket:
            toks = line.split()
            if toks:
                node_ids.append(toks[0])

    edge_ids = []
    for bucket in [pipes, pumps, valves]:
        for line in bucket:
            toks = line.split()
            if len(toks) >= 3:
                edge_ids.append(toks[0])

    return {
        "file": path.name,
        "junctions": len(junctions),
        "reservoirs": len(reservoirs),
        "tanks": len(tanks),
        "nodes_total": len(node_ids),
        "pipes": len(pipes),
        "pumps": len(pumps),
        "valves": len(valves),
        "edges_total": len(edge_ids),
        "all_node_ids": node_ids,
        "all_edge_ids": edge_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    bdir = data / "BattLeDIM_LTown"
    ndir = data / "WNTR_Net3"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not bdir.exists():
        raise SystemExit(f"Missing folder: {bdir}")

    results = {
        "dataset_root": str(data.resolve()),
        "csvs": {},
        "yaml": {},
        "networks": {},
        "cross_checks": {},
    }

    for p in sorted(bdir.glob("*.csv")):
        print(f"[AUDIT] {p.name}")
        try:
            results["csvs"][p.name] = read_csv_info(p)
        except Exception as e:
            results["csvs"][p.name] = {"file": p.name, "error": repr(e)}

    yml = bdir / "dataset_configuration.yaml"
    if yml.exists():
        with yml.open("r", encoding="utf-8", errors="ignore") as f:
            results["yaml"] = yaml.safe_load(f)

    ltown = bdir / "L-TOWN.inp"
    if ltown.exists():
        results["networks"]["L-TOWN.inp"] = parse_inp(ltown)

    net3 = ndir / "Net3.inp"
    if net3.exists():
        results["networks"]["Net3.inp"] = parse_inp(net3)

    # Cross-check graph IDs against configuration.
    linfo = results["networks"].get("L-TOWN.inp", {})
    node_ids = set(linfo.get("all_node_ids", []))
    edge_ids = set(linfo.get("all_edge_ids", []))
    cfg = results.get("yaml") or {}

    if isinstance(cfg, dict):
        for key in ["pressure_sensors", "level_sensors", "amrs"]:
            vals = cfg.get(key) or []
            if isinstance(vals, list):
                results["cross_checks"][f"{key}_not_in_graph"] = [
                    str(x) for x in vals if str(x) not in node_ids
                ]

        vals = cfg.get("flow_sensors") or []
        if isinstance(vals, list):
            results["cross_checks"]["flow_sensors_not_in_graph"] = [
                str(x) for x in vals if str(x) not in edge_ids
            ]

        leakages = cfg.get("leakages") or []
        leak_links = []
        if isinstance(leakages, list):
            for item in leakages:
                if isinstance(item, str):
                    leak_links.append(item.split(",")[0].strip())
                elif isinstance(item, (list, tuple)) and item:
                    leak_links.append(str(item[0]))
                elif isinstance(item, dict):
                    for k in ["linkID", "link_id", "link"]:
                        if k in item:
                            leak_links.append(str(item[k]))
                            break
        if leak_links:
            results["cross_checks"]["configured_leak_links_not_in_graph"] = [
                x for x in leak_links if x not in edge_ids
            ]

        # SCADA columns vs configured sensors
        for year in ["2018", "2019"]:
            for kind, cfg_key in [
                ("SCADA_Pressures", "pressure_sensors"),
                ("SCADA_Flows", "flow_sensors"),
                ("SCADA_Levels", "level_sensors"),
            ]:
                fname = f"{year}_{kind}.csv"
                info = results["csvs"].get(fname, {})
                cols = set(info.get("data_columns", []))
                expected = set(str(x) for x in (cfg.get(cfg_key) or []))
                if cols and expected:
                    results["cross_checks"][f"{fname}_configured_missing_from_csv"] = sorted(expected - cols)
                    results["cross_checks"][f"{fname}_unexpected_csv_columns"] = sorted(cols - expected)

    safe = json.loads(json.dumps(results))
    for n in safe["networks"].values():
        n.pop("all_node_ids", None)
        n.pop("all_edge_ids", None)

    (out / "audit.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")

    lines = []
    lines.append("AQUA-VERA DATA AUDIT v3")
    lines.append("=" * 78)
    lines.append(f"Dataset root: {data.resolve()}")
    lines.append("")

    any_error = False
    for name, info in safe["csvs"].items():
        lines.append(name)
        if "error" in info:
            any_error = True
            lines.append(f"  ERROR: {info['error']}")
            lines.append("")
            continue
        lines.append(f"  delimiter: {repr(info.get('delimiter'))}")
        lines.append(f"  parser engine: {info.get('engine')}")
        lines.append(f"  rows: {info.get('rows'):,}")
        lines.append(f"  columns: {info.get('n_columns')}")
        lines.append(f"  data columns: {info.get('n_data_columns')}")
        lines.append(f"  missing cells: {info.get('missing_total'):,}")
        lines.append(f"  timestamp column: {info.get('timestamp_column')}")
        if info.get("timestamp_column"):
            lines.append(f"  range: {info.get('timestamp_min')} -> {info.get('timestamp_max')}")
            lines.append(f"  common interval: {info.get('most_common_interval')}")
            lines.append(f"  nonstandard intervals: {info.get('nonstandard_intervals')}")
            lines.append(f"  duplicate timestamps: {info.get('duplicate_timestamps')}")
        lines.append(f"  first columns: {', '.join(info.get('columns', [])[:12])}")
        rates = info.get("numeric_parse_rate_first100", {})
        bad_numeric = {k: round(v, 6) for k, v in rates.items() if v < 0.99}
        lines.append(f"  numeric parse issues: {bad_numeric if bad_numeric else 'NONE'}")
        lines.append("")

    for name, info in safe["networks"].items():
        lines.append(name)
        lines.append(f"  nodes: {info.get('nodes_total')}")
        lines.append(f"  edges: {info.get('edges_total')}")
        lines.append(f"  junctions/reservoirs/tanks: {info.get('junctions')}/{info.get('reservoirs')}/{info.get('tanks')}")
        lines.append(f"  pipes/pumps/valves: {info.get('pipes')}/{info.get('pumps')}/{info.get('valves')}")
        lines.append("")

    lines.append("CROSS-CHECKS")
    cross_bad = False
    for k, v in safe["cross_checks"].items():
        if v:
            cross_bad = True
        lines.append(f"  {k}: {v if v else 'OK'}")

    lines.append("")
    if any_error:
        lines.append("OVERALL: CSV PARSE ERROR — DO NOT PROCEED")
    elif cross_bad:
        lines.append("OVERALL: PARSED, BUT CROSS-CHECK MISMATCHES NEED REVIEW")
    else:
        lines.append("OVERALL: BASIC DATA AUDIT PASSED")

    (out / "audit_summary_v3.txt").write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("AUDIT COMPLETE")
    print(f"Summary: {out / 'audit_summary_v3.txt'}")
    print(f"JSON:    {out / 'audit.json'}")
    print("")
    print("Send me audit_summary_v3.txt next.")


if __name__ == "__main__":
    main()
