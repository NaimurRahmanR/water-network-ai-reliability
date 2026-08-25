
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.")
    print("Run: py -m pip install pandas")
    sys.exit(1)


SCADA_FILES = {
    "2018": [
        ("2018_SCADA_Pressures.csv", "pressure"),
        ("2018_SCADA_Flows.csv", "flow"),
        ("2018_SCADA_Levels.csv", "level"),
    ],
    "2019": [
        ("2019_SCADA_Pressures.csv", "pressure"),
        ("2019_SCADA_Flows.csv", "flow"),
        ("2019_SCADA_Levels.csv", "level"),
    ],
}
LEAK_FILES = {
    "2018": "2018_Leakages.csv",
    "2019": "2019_Leakages.csv",
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_numeric_frame(path: Path):
    # Read original as strings so we can diagnose formatting explicitly.
    raw = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        engine="c",
    )

    if "Timestamp" not in raw.columns:
        raise ValueError(f"{path.name}: Timestamp column not found")

    ts = pd.to_datetime(raw["Timestamp"], errors="coerce")
    if ts.isna().any():
        bad = raw.loc[ts.isna(), "Timestamp"].head(5).tolist()
        raise ValueError(f"{path.name}: unparseable timestamps, examples={bad}")

    out = pd.DataFrame({"Timestamp": ts})

    diagnostics = {}
    for c in raw.columns:
        if c == "Timestamp":
            continue

        s = raw[c].astype(str).str.strip()

        # Try dot-decimal as-is.
        dot = pd.to_numeric(s, errors="coerce")
        dot_rate = float(dot.notna().mean())

        # Try European decimal comma -> decimal point.
        comma_norm = s.str.replace(",", ".", regex=False)
        comma = pd.to_numeric(comma_norm, errors="coerce")
        comma_rate = float(comma.notna().mean())

        if comma_rate > dot_rate:
            chosen = comma
            mode = "decimal_comma"
            rate = comma_rate
        else:
            chosen = dot
            mode = "decimal_point"
            rate = dot_rate

        # Strict scientific gate: do not silently coerce bad tokens.
        if rate < 0.99999:
            bad_mask = chosen.isna()
            examples = s[bad_mask].drop_duplicates().head(10).tolist()
            diagnostics[c] = {
                "mode": mode,
                "parse_rate": rate,
                "bad_examples": examples,
            }
            raise ValueError(
                f"{path.name}:{c}: numeric parse rate {rate:.6f}; "
                f"bad examples={examples}"
            )

        out[c] = chosen.astype("float64")
        diagnostics[c] = {"mode": mode, "parse_rate": rate}

    if out["Timestamp"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate timestamps")

    diffs = out["Timestamp"].sort_values().diff().dropna()
    if len(diffs):
        modal = diffs.mode().iloc[0]
        if modal != pd.Timedelta(minutes=5):
            raise ValueError(f"{path.name}: expected 5-minute interval, got {modal}")
        nonstandard = int((diffs != modal).sum())
        if nonstandard:
            raise ValueError(f"{path.name}: {nonstandard} nonstandard intervals")

    return out, diagnostics


def prefix_features(df, prefix):
    rename = {c: f"{prefix}__{c}" for c in df.columns if c != "Timestamp"}
    return df.rename(columns=rename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    source = data / "BattLeDIM_LTown"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise SystemExit(f"Missing source folder: {source}")

    manifest = {
        "source": str(source.resolve()),
        "years": {},
        "notes": [
            "Original BattLeDIM files are preserved unchanged.",
            "Model-ready outputs use comma-separated CSV and decimal points.",
            "Timestamp remains explicit and is not converted to a numeric feature here.",
            "2019 is not to be used for threshold selection or hyperparameter tuning.",
        ],
    }

    for year in ["2018", "2019"]:
        print(f"\n=== {year} ===")
        merged = None
        year_diag = {}

        for fname, prefix in SCADA_FILES[year]:
            path = source / fname
            print(f"[READ] {fname}")
            df, diag = parse_numeric_frame(path)
            year_diag[fname] = diag
            df = prefix_features(df, prefix)

            if merged is None:
                merged = df
            else:
                before = len(merged)
                merged = merged.merge(df, on="Timestamp", how="inner", validate="one_to_one")
                if len(merged) != before:
                    raise ValueError(
                        f"{year}: timestamp mismatch while merging {fname}: "
                        f"{before} -> {len(merged)} rows"
                    )

        leak_path = source / LEAK_FILES[year]
        print(f"[READ] {leak_path.name}")
        leaks, leak_diag = parse_numeric_frame(leak_path)
        leaks = prefix_features(leaks, "leak")
        year_diag[leak_path.name] = leak_diag

        if len(merged) != 105120:
            raise ValueError(f"{year}: expected 105120 SCADA rows, got {len(merged)}")
        if len(leaks) != 105120:
            raise ValueError(f"{year}: expected 105120 leakage rows, got {len(leaks)}")

        if not merged["Timestamp"].equals(leaks["Timestamp"]):
            raise ValueError(f"{year}: SCADA and leakage timestamps are not identical")

        scada_out = outdir / f"{year}_scada_model_ready.csv"
        leaks_out = outdir / f"{year}_leakages_model_ready.csv"

        merged.to_csv(scada_out, index=False, date_format="%Y-%m-%d %H:%M:%S")
        leaks.to_csv(leaks_out, index=False, date_format="%Y-%m-%d %H:%M:%S")

        # Snapshot statistics only — no modelling decisions.
        feature_cols = [c for c in merged.columns if c != "Timestamp"]
        leak_cols = [c for c in leaks.columns if c != "Timestamp"]

        manifest["years"][year] = {
            "rows": len(merged),
            "scada_feature_count": len(feature_cols),
            "pressure_features": len([c for c in feature_cols if c.startswith("pressure__")]),
            "flow_features": len([c for c in feature_cols if c.startswith("flow__")]),
            "level_features": len([c for c in feature_cols if c.startswith("level__")]),
            "leak_columns": len(leak_cols),
            "timestamp_min": str(merged["Timestamp"].min()),
            "timestamp_max": str(merged["Timestamp"].max()),
            "scada_file": scada_out.name,
            "scada_sha256": sha256(scada_out),
            "leak_file": leaks_out.name,
            "leak_sha256": sha256(leaks_out),
            "parse_diagnostics": year_diag,
        }

        print(f"[DONE] {scada_out}")
        print(f"[DONE] {leaks_out}")

    manifest_path = outdir / "model_ready_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = []
    lines.append("AQUA-VERA MODEL-READY PREPROCESSING")
    lines.append("=" * 78)
    lines.append("STATUS: PASS")
    lines.append("")
    for year in ["2018", "2019"]:
        y = manifest["years"][year]
        lines.append(f"{year}")
        lines.append(f"  rows: {y['rows']:,}")
        lines.append(f"  SCADA features: {y['scada_feature_count']} "
                     f"(pressure={y['pressure_features']}, flow={y['flow_features']}, level={y['level_features']})")
        lines.append(f"  leak columns: {y['leak_columns']}")
        lines.append(f"  range: {y['timestamp_min']} -> {y['timestamp_max']}")
        lines.append(f"  SCADA SHA-256: {y['scada_sha256']}")
        lines.append(f"  leak SHA-256: {y['leak_sha256']}")
        lines.append("")

    lines.append("IMPORTANT")
    lines.append("  Original downloaded files were not modified.")
    lines.append("  2019 remains reserved for final held-out evaluation.")
    lines.append("  Do not tune thresholds/hyperparameters on 2019.")
    lines.append("")
    lines.append("NEXT: retain this summary with the project record. before training.")

    summary = outdir / "model_ready_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print("\nPREPROCESSING PASS")
    print(f"Summary:  {summary}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
