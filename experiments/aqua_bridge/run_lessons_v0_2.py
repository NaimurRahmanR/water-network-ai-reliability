from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from run_aqua_bridge import (
    _context_paths,
    _load_layout,
    _rows_from_contexts,
    run as run_v0_1_core,
)

PROFILE_FEATURES = (
    "edge_residual",
    "central_residual",
    "disagreement",
    "abs_error",
    "packet_lost",
    "delay_seconds",
    "proceeded",
    "unsafe",
)


def _split_paths(generated_dir: Path, paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    index = pd.read_csv(generated_dir / "generation_index.csv", usecols=["context_id", "split"])
    split_map = dict(zip(index.context_id.astype(str), index.split.astype(str).str.lower()))
    buckets = {"train": [], "calibration": [], "validation": []}
    for path in paths:
        split = split_map.get(path.stem)
        if split in buckets:
            buckets[split].append(path)
    if any(not buckets[key] for key in buckets):
        counts = {key: len(value) for key, value in buckets.items()}
        raise ValueError(f"Could not recover untouched v0.2 split set: {counts}")
    return buckets["train"], buckets["calibration"], buckets["validation"]


def _safe_fill(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    med = values.median()
    if not np.isfinite(med):
        med = 0.0
    return values.fillna(float(med)).astype(float)


def _profile_matrix(work: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_residual": _safe_fill(work.edge_residual),
            "central_residual": _safe_fill(work.central_residual),
            "disagreement": _safe_fill(work.edge_central_disagreement),
            "abs_error": _safe_fill(work.abs_error),
            "packet_lost": _safe_fill(work.packet_lost),
            "delay_seconds": _safe_fill(work.delay_ms) / 1000.0,
            "proceeded": _safe_fill(work.proceeded),
            "unsafe": _safe_fill(work.unsafe_proceed),
        },
        index=work.index,
    )


def _standardize(numeric: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = numeric.mean()
    scale = numeric.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return (numeric - mean) / scale, mean, scale


def _largest_effects(
    numeric: pd.DataFrame,
    cluster_mask: pd.Series,
    overall_mean: pd.Series,
    overall_scale: pd.Series,
    topn: int = 3,
) -> list[dict]:
    cluster_mean = numeric.loc[cluster_mask].mean()
    effects = ((cluster_mean - overall_mean) / overall_scale).replace([np.inf, -np.inf], 0).fillna(0)
    ordered = effects.reindex(effects.abs().sort_values(ascending=False).index)
    out = []
    for feature, value in ordered.iloc[:topn].items():
        out.append(
            {
                "feature": str(feature),
                "direction": "higher" if value >= 0 else "lower",
                "standardized_difference": float(value),
            }
        )
    return out


def _human_feature(name: str) -> str:
    return {
        "edge_residual": "edge reconstruction residual",
        "central_residual": "central reconstruction residual",
        "disagreement": "edge-central disagreement",
        "abs_error": "recovery error",
        "packet_lost": "communication loss",
        "delay_seconds": "stale-data delay",
        "proceeded": "autonomous proceed rate",
        "unsafe": "benchmark-defined unsafe proceed",
    }.get(name, name.replace("_", " "))


def _lesson_for_cluster(profile: dict, all_profiles: list[dict], effects: list[dict]) -> tuple[str, str]:
    avg_escalation = float(np.mean([p["escalation_rate"] for p in all_profiles]))
    avg_unsafe = float(np.mean([p["unsafe_rate"] for p in all_profiles]))
    avg_disagreement = float(np.mean([p["median_edge_central_disagreement"] for p in all_profiles]))
    avg_error = float(np.mean([p["median_abs_error"] for p in all_profiles]))
    avg_packet = float(np.mean([p["packet_loss_rate"] for p in all_profiles]))
    avg_delay = float(np.mean([p["median_stale_delay_seconds"] for p in all_profiles]))
    avg_proceed = float(np.mean([p["proceed_rate"] for p in all_profiles]))

    max_disagreement = max(p["median_edge_central_disagreement"] for p in all_profiles)
    max_packet = max(p["packet_loss_rate"] for p in all_profiles)
    max_delay = max(p["median_stale_delay_seconds"] for p in all_profiles)

    if (
        profile["median_edge_central_disagreement"] >= max_disagreement
        and profile["median_edge_central_disagreement"] > avg_disagreement
        and (profile["escalation_rate"] > avg_escalation or profile["unsafe_rate"] > avg_unsafe)
    ):
        category = "WIDER_NETWORK_VERIFICATION"
        core = (
            "This profile combines relatively high edge-central disagreement with elevated "
            "escalation or benchmark-defined unsafe recovery. Prefer wider-network verification "
            "before autonomous reuse of a local reconstruction."
        )
    elif profile["packet_loss_rate"] >= max_packet and profile["packet_loss_rate"] > avg_packet:
        category = "FRESH_EVIDENCE_AFTER_COMMUNICATION_LOSS"
        core = (
            "Communication loss is comparatively prominent in this profile. Preserve that loss "
            "as evidence-quality provenance and request fresh evidence before reusing a recovered value."
        )
    elif profile["median_stale_delay_seconds"] >= max_delay and profile["median_stale_delay_seconds"] > avg_delay:
        category = "FRESH_MEASUREMENT_VERIFICATION"
        core = (
            "Stale evidence is comparatively prominent in this profile. Verify a fresh measurement "
            "before reusing the reconstructed state."
        )
    elif (
        profile["median_edge_central_disagreement"] <= avg_disagreement
        and profile["median_abs_error"] <= avg_error
        and profile["proceed_rate"] >= avg_proceed
    ):
        category = "BOUNDED_RECONSTRUCTION_WITH_PROVENANCE"
        core = (
            "This profile shows comparatively lower disagreement and recovery error. Bounded "
            "reconstruction appears more stable here, but provenance should remain attached and "
            "the reliability gates should remain active."
        )
    else:
        category = "PROVENANCE_AND_VERIFICATION"
        core = (
            "This profile does not support unconditional reuse. Retain evidence provenance and "
            "use wider-network verification when the same profile recurs."
        )

    if effects:
        descriptors = ", ".join(
            f"{_human_feature(item['feature'])} {item['direction']}"
            for item in effects[:2]
        )
        lesson = f"{core} The strongest relative profile signals were {descriptors}."
    else:
        lesson = core
    return category, lesson


def _build_profiles_and_lessons(work: pd.DataFrame, seed: int, bootstrap_reps: int = 25):
    if "cluster" not in work:
        raise ValueError("clustered incident table must contain cluster labels")
    numeric = _profile_matrix(work)
    z, overall_mean, overall_scale = _standardize(numeric)
    labels = work.cluster.astype(int).to_numpy()
    unique = sorted(np.unique(labels).tolist())
    if len(unique) < 2:
        raise ValueError("Need at least two incident profiles for confirmation")

    base_silhouette = float(silhouette_score(z, labels))
    raw_profiles = []
    effects_by_cluster: dict[int, list[dict]] = {}

    for cluster in unique:
        mask = work.cluster.astype(int) == cluster
        d = work.loc[mask]
        effects = _largest_effects(numeric, mask, overall_mean, overall_scale)
        effects_by_cluster[int(cluster)] = effects
        raw_profiles.append(
            {
                "cluster": int(cluster),
                "n": int(len(d)),
                "dominant_condition": str(d.condition.value_counts().idxmax()),
                "dominant_action": str(d.action.value_counts().idxmax()),
                "proceed_rate": float(pd.to_numeric(d.proceeded).mean()),
                "escalation_rate": float((d.action == "ESCALATE").mean()),
                "unsafe_rate": float(pd.to_numeric(d.unsafe_proceed).mean()),
                "median_edge_central_disagreement": float(pd.to_numeric(d.edge_central_disagreement).median()),
                "median_abs_error": float(pd.to_numeric(d.abs_error, errors="coerce").median()),
                "packet_loss_rate": float(pd.to_numeric(d.packet_lost).mean()),
                "median_stale_delay_seconds": float(pd.to_numeric(d.delay_ms).median() / 1000.0),
            }
        )

    lessons = []
    profile_rows = []
    for profile in raw_profiles:
        cluster = profile["cluster"]
        category, lesson = _lesson_for_cluster(profile, raw_profiles, effects_by_cluster[cluster])
        row = dict(profile)
        row["lesson_category"] = category
        row["top_distinguishing_features"] = json.dumps(effects_by_cluster[cluster], sort_keys=True)
        profile_rows.append(row)
        lessons.append(
            {
                "cluster": cluster,
                "support_n": profile["n"],
                "dominant_condition": profile["dominant_condition"],
                "dominant_action": profile["dominant_action"],
                "lesson_category": category,
                "lesson": lesson,
                "distinguishing_features": effects_by_cluster[cluster],
                "scope_note": (
                    "Confirmed only on the AQUA-BRIDGE Net3 VALIDATION benchmark; "
                    "not an operator-approved utility operating rule."
                ),
            }
        )

    rng = np.random.default_rng(seed + 2002)
    ari_values = []
    x = z.to_numpy(dtype=float)
    k = len(unique)
    for rep in range(bootstrap_reps):
        sample_idx = rng.integers(0, len(x), size=len(x))
        km = KMeans(n_clusters=k, random_state=seed + 3000 + rep, n_init=20)
        km.fit(x[sample_idx])
        predicted = km.predict(x)
        ari_values.append(float(adjusted_rand_score(labels, predicted)))

    stability = {
        "bootstrap_reps": int(bootstrap_reps),
        "adjusted_rand_index_mean": float(np.mean(ari_values)),
        "adjusted_rand_index_median": float(np.median(ari_values)),
        "adjusted_rand_index_p05": float(np.quantile(ari_values, 0.05)),
        "adjusted_rand_index_min": float(np.min(ari_values)),
        "adjusted_rand_index_max": float(np.max(ari_values)),
        "base_silhouette": base_silhouette,
    }
    return pd.DataFrame(profile_rows), lessons, stability


def main():
    ap = argparse.ArgumentParser(description="AQUA-BRIDGE v0.2 untouched-validation incident-profile confirmation")
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/aqua_bridge_v0_2_validation"))
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--max-rows-per-variant", type=int, default=12)
    args = ap.parse_args()

    layout = _load_layout(args.generated_dir)
    paths = _context_paths(args.generated_dir)
    train_paths, calibration_paths, validation_paths = _split_paths(args.generated_dir, paths)

    train = _rows_from_contexts(train_paths, args.max_rows_per_variant)
    calibration = _rows_from_contexts(calibration_paths, args.max_rows_per_variant)
    validation = _rows_from_contexts(validation_paths, args.max_rows_per_variant)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = run_v0_1_core(layout, train, calibration, validation, args.out, args.seed)
    metrics.to_csv(args.out / "validation_architecture_metrics.csv", index=False)

    clustered_path = args.out / "clustered_incidents.csv"
    if not clustered_path.exists():
        raise RuntimeError("v0.2 confirmation did not produce clustered incidents")
    clustered = pd.read_csv(clustered_path)

    profiles, lessons, stability = _build_profiles_and_lessons(clustered, args.seed)
    profiles.to_csv(args.out / "cluster_profiles_v0_2.csv", index=False)
    (args.out / "lessons_v0_2.json").write_text(
        json.dumps(
            {
                "confirmation_split": "VALIDATION",
                "lesson_method": "relative-profile descriptive rules frozen in PROTOCOL_v0_2.md",
                "lessons": lessons,
                "claim_boundary": (
                    "Benchmark-scoped descriptive lessons only; operator validation is required "
                    "before any operational use."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out / "cluster_stability_v0_2.json").write_text(
        json.dumps(stability, indent=2),
        encoding="utf-8",
    )

    means = metrics.groupby("architecture", sort=True).agg(
        mean_coverage=("coverage", "mean"),
        mean_escalation_rate=("escalation_rate", "mean"),
        mean_unsafe_proceed_rate_all=("unsafe_proceed_rate_all", "mean"),
        mean_mae_when_proceeding=("mae_when_proceeding", "mean"),
    )
    summary = {
        "protocol": "AQUA-BRIDGE v0.2",
        "confirmation_split": "VALIDATION",
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "validation_rows": int(len(validation)),
        "profile_count": int(len(profiles)),
        "unique_lesson_categories": sorted({item["lesson_category"] for item in lessons}),
        "cluster_stability": stability,
        "architecture_means": {
            str(idx): {key: float(value) for key, value in row.items()}
            for idx, row in means.to_dict(orient="index").items()
        },
        "claim_boundaries": [
            "v0.1 TEST architecture results were not retuned.",
            "v0.2 uses previously unused VALIDATION contexts for incident-profile confirmation.",
            "Logical edge nodes are not physical edge hardware.",
            "Lessons are benchmark-scoped and not operator-approved utility rules.",
            "No general causal claim is made outside controlled benchmark interventions.",
        ],
    }
    (args.out / "confirmation_summary_v0_2.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if len(lessons) > 1 and len({item["lesson"] for item in lessons}) == 1:
        raise RuntimeError("All v0.2 profiles collapsed to identical lesson text")

    print("AQUA-BRIDGE v0.2 confirmation completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
