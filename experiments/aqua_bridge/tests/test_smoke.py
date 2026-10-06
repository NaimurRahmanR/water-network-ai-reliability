from pathlib import Path
import json
import sqlite3
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from run_aqua_bridge import _make_smoke, run


def test_smoke(tmp_path: Path):
    layout, train, calib, test = _make_smoke(1234)
    metrics = run(layout, train, calib, test.iloc[:16].copy(), tmp_path, 1234)
    assert set(metrics["architecture"]) == {
        "central_only", "edge_only", "hybrid_naive", "hybrid_reliability_aware"
    }
    assert (tmp_path / "incident_hub.sqlite").exists()
    assert (tmp_path / "architecture_metrics.csv").exists()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert "Experimental benchmark only; not utility deployment." in manifest["claim_boundary"]
    with sqlite3.connect(tmp_path / "incident_hub.sqlite") as conn:
        n = conn.execute("select count(*) from incidents").fetchone()[0]
    assert n > 0
