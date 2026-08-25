
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERROR: numpy and pandas are required.")
    print("Run: py -m pip install numpy pandas")
    sys.exit(1)

EXPECTED_CORE = {
    "generation_index.csv": "bc33af176da705ed87850e6032dc313761e2c839faf04aa03c59bd12deb5848a",
    "tensor_layout.json": "33220279d7e1ff6c92caacd54785040764168c576d5eb218401515be206a9261",
    "graph_nodes.csv": "dfb90c5fa1b7c80935edcca5294ec0b87b22cb3788732f4432d58135c8e6f295",
    "graph_links.csv": "d92ab147d4678c8aff999a86f761f15deeb843a738b3021acc3062b824860c1e",
}

WINDOW_STEPS = 24
STRIDE_STEPS = 6
EXPECTED_TIMESTEPS = 289
EXPECTED_FEATURES = 33
EXPECTED_CONTEXTS = 240
EXPECTED_SPLITS = {"train":150, "validation":30, "calibration":30, "test":30}
EPS_STD = 1e-8


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, expected: str):
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH: {path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {path.name}")


def variant_arrays(npz, variant):
    if variant == "CLEAN":
        sensor = npz["clean_sensor"]
        label = npz["clean_active_leak"]
    elif variant == "LEAK":
        sensor = npz["leak_sensor"]
        label = npz["leak_active_leak"]
    else:
        raise ValueError(variant)
    return sensor, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gen = Path(args.generated)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("AQUA-VERA WINDOW + NORMALIZATION FREEZE v0.4")
    print("=" * 78)

    for name, expected in EXPECTED_CORE.items():
        verify_file(gen / name, expected)

    index = pd.read_csv(gen / "generation_index.csv")
    if len(index) != EXPECTED_CONTEXTS:
        raise SystemExit(f"Expected {EXPECTED_CONTEXTS} contexts, got {len(index)}")

    got_splits = index["split"].value_counts().to_dict()
    if got_splits != EXPECTED_SPLITS:
        raise SystemExit(f"Split count mismatch: {got_splits}")

    layout = json.loads((gen / "tensor_layout.json").read_text(encoding="utf-8"))
    feature_order = layout["sensor_feature_order"]
    if len(feature_order) != EXPECTED_FEATURES:
        raise SystemExit(
            f"Expected {EXPECTED_FEATURES} model-visible features, got {len(feature_order)}"
        )

    contexts_dir = gen / "contexts"
    if not contexts_dir.exists():
        raise SystemExit(f"Missing contexts folder: {contexts_dir}")

    # Verify all context NPZ files against the hashes frozen in generation_index.csv.
    print("[VERIFY] context NPZ hashes...")
    for _, row in index.sort_values("context_id").iterrows():
        cid = str(row["context_id"])
        p = contexts_dir / f"{cid}.npz"
        if not p.exists():
            raise SystemExit(f"Missing context NPZ: {p}")
        got = sha256(p)
        expected = str(row["npz_sha256"])
        if got != expected:
            raise SystemExit(
                f"Context hash mismatch {cid}\nexpected={expected}\ngot={got}"
            )

    # Frozen candidate final indices.
    # 24 samples = 2h, stride 6 = 30m.
    final_indices = list(range(WINDOW_STEPS - 1, EXPECTED_TIMESTEPS, STRIDE_STEPS))
    if len(final_indices) != 45:
        raise SystemExit(f"Expected 45 windows/variant, got {len(final_indices)}")

    # Freeze normalization from TRAIN contexts only, using unique 5-minute rows
    # from both CLEAN and LEAK physical variants (no overlap duplication from windows).
    sum_x = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    sum_x2 = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    n_rows = 0

    train_index = index[index["split"] == "train"].sort_values("context_id")
    if len(train_index) != 150:
        raise SystemExit(f"Expected 150 train contexts, got {len(train_index)}")

    print("[STATS] training-only normalization...")
    for _, row in train_index.iterrows():
        cid = str(row["context_id"])
        with np.load(contexts_dir / f"{cid}.npz", allow_pickle=False) as z:
            for variant in ["CLEAN", "LEAK"]:
                x, _ = variant_arrays(z, variant)
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (EXPECTED_TIMESTEPS, EXPECTED_FEATURES):
                    raise SystemExit(f"{cid}/{variant}: unexpected sensor shape {x.shape}")
                if not np.isfinite(x).all():
                    raise SystemExit(f"{cid}/{variant}: non-finite sensor values")
                sum_x += x.sum(axis=0)
                sum_x2 += np.square(x).sum(axis=0)
                n_rows += x.shape[0]

    mean = sum_x / n_rows
    var = np.maximum(sum_x2 / n_rows - np.square(mean), 0.0)
    std = np.sqrt(var)
    constant_mask = std < EPS_STD
    scale = std.copy()
    scale[constant_mask] = 1.0

    norm = {
        "status": "FROZEN",
        "source_split": "train only",
        "source_contexts": 150,
        "source_physical_variants_per_context": 2,
        "source_unique_5min_rows": int(n_rows),
        "feature_count": EXPECTED_FEATURES,
        "feature_order": feature_order,
        "mean": [float(x) for x in mean],
        "population_std": [float(x) for x in std],
        "scale_used": [float(x) for x in scale],
        "constant_or_near_constant_features": [
            feature_order[i] for i, flag in enumerate(constant_mask) if flag
        ],
        "std_definition": "population standard deviation (ddof=0)",
        "constant_feature_rule": "if std < 1e-8, use scale=1.0 and record feature",
        "no_validation_calibration_test_data_used": True,
    }
    norm_path = out / "normalization_train_only.json"
    norm_path.write_text(json.dumps(norm, indent=2), encoding="utf-8")

    # Build immutable window manifest. No window tensors are duplicated; each row
    # references a frozen context NPZ + slice indices.
    rows = []
    wid = 0
    per_split_variant = {}
    positives = {}

    for _, row in index.sort_values(["split", "context_id"]).iterrows():
        cid = str(row["context_id"])
        split = str(row["split"])

        with np.load(contexts_dir / f"{cid}.npz", allow_pickle=False) as z:
            time_s = np.asarray(z["time_s"], dtype=np.int64)
            if time_s.shape != (EXPECTED_TIMESTEPS,):
                raise SystemExit(f"{cid}: unexpected time shape {time_s.shape}")
            expected_time = np.arange(0, 24*3600 + 1, 300, dtype=np.int64)
            if not np.array_equal(time_s, expected_time):
                raise SystemExit(f"{cid}: timestamp grid mismatch")

            for variant in ["CLEAN", "LEAK"]:
                x, labels = variant_arrays(z, variant)
                x = np.asarray(x)
                labels = np.asarray(labels, dtype=np.uint8)

                if x.shape != (EXPECTED_TIMESTEPS, EXPECTED_FEATURES):
                    raise SystemExit(f"{cid}/{variant}: sensor shape {x.shape}")
                if labels.shape != (EXPECTED_TIMESTEPS,):
                    raise SystemExit(f"{cid}/{variant}: label shape {labels.shape}")

                key = f"{split}:{variant}"
                per_split_variant[key] = per_split_variant.get(key, 0) + 45

                for final_idx in final_indices:
                    start_idx = final_idx - WINDOW_STEPS + 1
                    label = int(labels[final_idx])
                    if label not in (0, 1):
                        raise SystemExit(f"{cid}/{variant}: invalid label {label}")
                    wid += 1
                    pair_id = f"{cid}_T{final_idx:03d}"
                    rows.append({
                        "window_id": f"W{wid:06d}",
                        "pair_id": pair_id,
                        "context_id": cid,
                        "variant": variant,
                        "split": split,
                        "start_idx": start_idx,
                        "final_idx": final_idx,
                        "start_time_s": int(time_s[start_idx]),
                        "final_time_s": int(time_s[final_idx]),
                        "target_active_leak": label,
                        "npz_file": f"contexts/{cid}.npz",
                    })
                    positives[split] = positives.get(split, 0) + label

    manifest = pd.DataFrame(rows)
    expected_windows = EXPECTED_CONTEXTS * 2 * 45
    if len(manifest) != expected_windows:
        raise SystemExit(
            f"Expected {expected_windows} windows, got {len(manifest)}"
        )

    # Hard split integrity checks.
    if manifest["window_id"].duplicated().any():
        raise SystemExit("Duplicate window_id")
    group_leak = manifest.groupby("context_id")["split"].nunique()
    if (group_leak != 1).any():
        raise SystemExit("Context appears in multiple splits")
    if manifest.groupby(["context_id","final_idx"])["variant"].nunique().min() != 2:
        raise SystemExit("Matched CLEAN/LEAK pair missing for one or more timestamps")

    expected_split_windows = {
        "train": 150 * 2 * 45,
        "validation": 30 * 2 * 45,
        "calibration": 30 * 2 * 45,
        "test": 30 * 2 * 45,
    }
    got_split_windows = manifest["split"].value_counts().to_dict()
    if got_split_windows != expected_split_windows:
        raise SystemExit(f"Window split counts mismatch: {got_split_windows}")

    # Clean windows must always be negative.
    clean_pos = int(
        manifest.loc[manifest["variant"] == "CLEAN", "target_active_leak"].sum()
    )
    if clean_pos != 0:
        raise SystemExit(f"CLEAN windows unexpectedly contain {clean_pos} positives")

    # Every split must have positive and negative labels.
    split_stats = {}
    for split in ["train","validation","calibration","test"]:
        s = manifest[manifest["split"] == split]
        pos = int(s["target_active_leak"].sum())
        neg = int(len(s) - pos)
        if pos == 0 or neg == 0:
            raise SystemExit(f"{split} missing class: pos={pos}, neg={neg}")
        split_stats[split] = {
            "windows": int(len(s)),
            "positive": pos,
            "negative": neg,
            "positive_fraction": float(pos / len(s)),
            "contexts": int(s["context_id"].nunique()),
        }

    manifest_path = out / "window_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    config = {
        "status": "FROZEN",
        "window_steps": WINDOW_STEPS,
        "window_minutes": WINDOW_STEPS * 5,
        "stride_steps": STRIDE_STEPS,
        "stride_minutes": STRIDE_STEPS * 5,
        "windows_per_variant": len(final_indices),
        "final_indices": final_indices,
        "target": "active_leak at final timestamp",
        "variant_policy": "both CLEAN and LEAK physical variants retained",
        "pairing": "same context_id and final_idx define a matched CLEAN/LEAK pair",
        "window_storage": "manifest references frozen context NPZ slices; windows are not duplicated",
        "normalization": "training contexts only, unique raw 5-minute rows, both physical variants",
        "split_integrity": "context-level split inherited from frozen v0.3 scenario manifest",
        "heldout_test_used_for_normalization": False,
    }
    config_path = out / "window_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    core = [manifest_path, norm_path, config_path]
    hashes = {p.name: sha256(p) for p in core}

    freeze = {
        "status": "FROZEN_PRETRAINING_DATA",
        "generator_core_hashes_verified": EXPECTED_CORE,
        "artifact_sha256": hashes,
        "window_count": int(len(manifest)),
        "split_stats": split_stats,
        "normalization_source": "train only",
        "note": "Frozen before any model training."
    }
    freeze_path = out / "FROZEN_WINDOW_RECORD.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    lines = [
        "AQUA-VERA WINDOW + NORMALIZATION FREEZE v0.4",
        "=" * 78,
        "STATUS: PASS / FROZEN",
        "",
        "WINDOW DESIGN",
        f"  observation window: {WINDOW_STEPS} steps = {WINDOW_STEPS*5} minutes",
        f"  stride: {STRIDE_STEPS} steps = {STRIDE_STEPS*5} minutes",
        f"  windows per physical variant: {len(final_indices)}",
        f"  total windows: {len(manifest):,}",
        f"  model-visible features: {EXPECTED_FEATURES}",
        "",
        "NORMALIZATION",
        "  source: TRAIN contexts only",
        "  physical variants used: CLEAN + LEAK",
        f"  unique 5-min rows used: {n_rows:,}",
        f"  near-constant features: {len(norm['constant_or_near_constant_features'])}",
        "  validation/calibration/test used: NO",
        "",
        "SPLIT / CLASS COUNTS",
    ]
    for split in ["train","validation","calibration","test"]:
        s = split_stats[split]
        lines.append(
            f"  {split}: windows={s['windows']:,}, "
            f"positive={s['positive']:,}, negative={s['negative']:,}, "
            f"positive_fraction={s['positive_fraction']:.4%}, "
            f"contexts={s['contexts']}"
        )
    lines += [
        "",
        "INTEGRITY",
        "  context appears in only one split: PASS",
        "  matched CLEAN/LEAK pair at each candidate timestamp: PASS",
        "  CLEAN positive windows: 0",
        "",
        "ARTIFACT HASHES",
    ]
    for k,v in hashes.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        "NEXT",
        "  Retain window_summary.txt with the project record.",
        "  If reviewed cleanly, model training may begin."
    ]
    summary_path = out / "window_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nWINDOW FREEZE SUCCESS")
    print(f"Summary: {summary_path}")
    print(f"Freeze:  {freeze_path}")


if __name__ == "__main__":
    main()
