from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_lessons_v0_2 import _build_profiles_and_lessons


def test_profiles_produce_distinct_lessons_and_stability():
    rng = np.random.default_rng(7)
    rows = []

    for cluster in (0, 1):
        for i in range(40):
            high = cluster == 1
            disagreement = rng.normal(1.2 if high else 0.08, 0.02)
            edge_resid = rng.normal(1.0 if high else 0.05, 0.01)
            central_resid = rng.normal(0.45 if high else 0.04, 0.01)
            abs_error = rng.normal(0.8 if high else 0.06, 0.01)
            action = "ESCALATE" if high and i < 20 else "PROCEED"
            rows.append(
                {
                    "cluster": cluster,
                    "condition": "CROSS_SOURCE_CONFLICT" if high else "MISSING_SENSOR",
                    "action": action,
                    "proceeded": 0 if action == "ESCALATE" else 1,
                    "unsafe_proceed": 1 if high and i < 8 else 0,
                    "edge_residual": max(edge_resid, 0),
                    "central_residual": max(central_resid, 0),
                    "edge_central_disagreement": max(disagreement, 0),
                    "abs_error": max(abs_error, 0),
                    "packet_lost": 0,
                    "delay_ms": 0,
                }
            )

    df = pd.DataFrame(rows)
    profiles, lessons, stability = _build_profiles_and_lessons(df, seed=271828, bootstrap_reps=5)

    assert len(profiles) == 2
    assert len(lessons) == 2
    assert len({item["lesson"] for item in lessons}) == 2
    assert any(item["lesson_category"] == "WIDER_NETWORK_VERIFICATION" for item in lessons)
    assert 0.0 <= stability["adjusted_rand_index_min"] <= 1.0
    assert 0.0 <= stability["base_silhouette"] <= 1.0
