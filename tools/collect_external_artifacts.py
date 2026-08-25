"""Read-only collector for exact frozen BattLeDIM output directories.

This does NOT rerun the experiment. It packages existing local output files so
they can be mirrored into the Git repository.
"""
from pathlib import Path
import hashlib, json, shutil, zipfile

HOME = Path.home()
DL = HOME / "Downloads"

SOURCES = {
    "v0_16b": DL / "AQUA_VERA_BattLeDIM_train_frozen_v0_16b",
    "v0_17": DL / "AQUA_VERA_BattLeDIM_external_2019_v0_17",
    "v0_18": DL / "AQUA_VERA_external_failure_diagnostic_v0_18",
}

OUT = DL / "AQUA_VERA_external_exact_artifacts_for_github.zip"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

missing = [str(p) for p in SOURCES.values() if not p.exists()]
if missing:
    raise SystemExit("Missing frozen output directory/directories:\n" + "\n".join(missing))

manifest = []
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for version, root in SOURCES.items():
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            arc = Path(version) / p.relative_to(root)
            z.write(p, arc)
            manifest.append({
                "version": version,
                "path": str(arc).replace("\\", "/"),
                "sha256": sha256(p),
                "size_bytes": p.stat().st_size,
            })
    z.writestr(
        "COLLECTION_MANIFEST.json",
        json.dumps({"read_only": True, "files": manifest}, indent=2),
    )

print(f"Created: {OUT}")
print(f"SHA-256: {sha256(OUT)}")
print(f"Files: {len(manifest)}")
