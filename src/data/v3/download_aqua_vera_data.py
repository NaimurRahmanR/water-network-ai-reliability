#!/usr/bin/env python3
"""Reliable downloader for AQUA-VERA public source datasets.

Downloads AQUA-VERA public source datasets with resumable transfers.

Version 3 distinguishes three modes clearly:
- default/core: research-ready BattLeDIM CSV core + L-Town + EPA Net3
- --full-battledim: EVERY file in the official BattLeDIM Zenodo record (~550.5 MB)
- --everything: full BattLeDIM + EPA Net3 + full LeakDB.zip (~13.9 GB)

Transfers resume from .part files, validate reported byte counts, and verify
MD5 checksums before promoting files to final names.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BATTLEDIM_BASE = "https://zenodo.org/records/4017659/files"
LEAKDB_URL = "https://zenodo.org/records/13985057/files/LeakDB.zip?download=1"
LEAKDB_MD5 = "d607929ef7caea432e663aee8071f2d7"
NET3_URL = "https://raw.githubusercontent.com/USEPA/WNTR/main/examples/networks/Net3.inp"

# Checksums from the official BattLeDIM Zenodo record (DOI 10.5281/zenodo.4017659).
BATTLEDIM = {
    "2018_Fixed_Leakages_Report.txt": "50b1f8b7ba0db5a19cedbb060c63d912",
    "2018_Leakages.csv": "c2c5fab90420da44f02e29775050abfe",
    "2018_SCADA.xlsx": "20c1224db8e379e31eba301d852e5cf7",
    "2018_SCADA_Demands.csv": "8803afcb1da2428de169237be3d911fb",
    "2018_SCADA_Flows.csv": "d0602c06946b46287e956f007e4264ee",
    "2018_SCADA_Levels.csv": "92a512d6749ddf66fd6d99f2bdde6cc4",
    "2018_SCADA_Pressures.csv": "d389d8541350c19ff0bfc6b80f246d35",
    "2019_Leakages.csv": "e1f0a43683813a90a9fec8562dde599b",
    "2019_SCADA.xlsx": "80242d5f59d39d0ef2306e2351c28367",
    "2019_SCADA_Demands.csv": "b3e111de397a1b06d33b8a9f13bc997d",
    "2019_SCADA_Flows.csv": "28fc99fdcbf80fcd26079e7fe602d6dc",
    "2019_SCADA_Levels.csv": "e5a8050bd38729e4b7648b9d52abd5b1",
    "2019_SCADA_Pressures.csv": "5ea1e46d3f2f0a89a3f98d6fd39a851d",
    "dataset_configuration.yaml": "48486401f5b4d0447023f5ce5d242c52",
    "L-TOWN.inp": "bfbb16b8b463e0576200ba6ad360f68e",
    "L-TOWN_Real.inp": "40fb38de25ed692a5ddd30e0f5fae332",
    "README.txt": "7ad5cdc2eebfe78197a3412e80d07ddc",
}

STARTER = [
    "2018_Leakages.csv",
    "2018_SCADA_Flows.csv",
    "2018_SCADA_Levels.csv",
    "2018_SCADA_Pressures.csv",
    "dataset_configuration.yaml",
    "L-TOWN.inp",
    "README.txt",
]

CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.I)


def md5sum(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float | int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{n} B"


def parse_total_size(status: int, headers, requested_start: int) -> int | None:
    """Return full-object size when it can be established from HTTP headers."""
    cr = headers.get("Content-Range")
    if cr:
        m = CONTENT_RANGE_RE.fullmatch(cr.strip())
        if m:
            actual_start = int(m.group(1))
            if actual_start != requested_start:
                raise RuntimeError(
                    f"server resumed at byte {actual_start}, expected {requested_start}"
                )
            if m.group(3) != "*":
                return int(m.group(3))
    if status == 200:
        cl = headers.get("Content-Length")
        if cl and cl.isdigit():
            return int(cl)
    return None


def download(
    url: str,
    dest: Path,
    expected_md5: str | None = None,
    retries: int = 12,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected_md5:
        got = md5sum(dest)
        if got.lower() == expected_md5.lower():
            print(f"[OK] {dest.name} already exists and checksum matches")
            return
        print(f"[WARN] {dest.name} exists but checksum is wrong; moving it back to .part")
        part = dest.with_suffix(dest.suffix + ".part")
        if part.exists():
            part.unlink()
        dest.replace(part)
    elif dest.exists() and dest.stat().st_size > 0 and not expected_md5:
        print(f"[SKIP] {dest.name} already exists (no checksum configured)")
        return

    part = dest.with_suffix(dest.suffix + ".part")
    headers_base = {
        "User-Agent": "AQUA-VERA dataset downloader/2.0",
        # Avoid transparent content coding so Range offsets refer to stored bytes.
        "Accept-Encoding": "identity",
    }

    last_err: Exception | None = None
    checksum_resets = 0

    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        headers = dict(headers_base)
        if start:
            headers["Range"] = f"bytes={start}-"
            print(f"[RESUME] {dest.name} from {human(start)}")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                status = getattr(r, "status", r.getcode())

                # If a server ignores Range, restart cleanly instead of appending a full file.
                if start and status != 206:
                    print(f"[INFO] {dest.name}: server ignored Range; restarting cleanly")
                    part.unlink(missing_ok=True)
                    raise RuntimeError("Range request was not honoured; clean restart scheduled")

                total = parse_total_size(status, r.headers, start)
                mode = "ab" if start else "wb"
                t0 = time.time()
                with part.open(mode) as f:
                    while True:
                        block = r.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        current = f.tell()
                        if total:
                            pct = min(100.0, 100.0 * current / total)
                            print(
                                f"\r  {dest.name}: {pct:6.2f}% "
                                f"({human(current)}/{human(total)})",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r  {dest.name}: {human(current)}",
                                end="",
                                flush=True,
                            )
                elapsed = max(time.time() - t0, 0.001)
                current = part.stat().st_size
                print(f"  [{human(max(0, current - start) / elapsed)}/s]")

            current = part.stat().st_size
            if total is not None and current < total:
                raise RuntimeError(
                    f"incomplete response: have {current} bytes of {total}; will resume"
                )
            if total is not None and current > total:
                part.unlink(missing_ok=True)
                raise RuntimeError(
                    f"received too many bytes ({current} > {total}); restarting cleanly"
                )

            if expected_md5:
                got = md5sum(part)
                if got.lower() != expected_md5.lower():
                    # At the reported full size, a checksum mismatch is not a truncation.
                    # Reset once to protect against a bad/cached response, then fail if repeated.
                    checksum_resets += 1
                    msg = (
                        f"MD5 mismatch for complete {dest.name}: "
                        f"expected {expected_md5}, got {got}"
                    )
                    if checksum_resets <= 2:
                        print(f"[WARN] {msg}; restarting from byte 0")
                        part.unlink(missing_ok=True)
                        raise RuntimeError(msg)
                    raise RuntimeError(msg)

            part.replace(dest)
            print(f"[DONE] {dest}")
            return

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            ConnectionResetError,
            http.client.IncompleteRead,
            OSError,
            RuntimeError,
        ) as e:
            last_err = e
            kept = part.stat().st_size if part.exists() else 0
            print(
                f"\n[RETRY {attempt}/{retries}] {dest.name}: {e}"
                + (f" | kept {human(kept)} for resume" if kept else "")
            )
            if attempt < retries:
                time.sleep(min(2 ** min(attempt, 4), 15))

    raise RuntimeError(
        f"Failed to download {dest.name} after {retries} attempts: {last_err}\n"
        f"Run the downloader again; any .part file will be resumed."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Download AQUA-VERA source datasets")
    ap.add_argument("--output", default="AQUA_VERA_data", help="Destination directory")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--starter", action="store_true", help="Download the smaller BattLeDIM starter subset")
    g.add_argument("--full-battledim", action="store_true", help="Download ALL 17 files in the official BattLeDIM record (~550.5 MB)")
    g.add_argument("--everything", action="store_true", help="Download full BattLeDIM plus full LeakDB.zip (~13.9 GB)")
    ap.add_argument("--leakdb-full", action="store_true", help="Also download LeakDB.zip (~13.9 GB) with the selected BattLeDIM mode")
    ap.add_argument("--retries", type=int, default=12, help="Maximum attempts per file (default: 12)")
    args = ap.parse_args()

    root = Path(args.output).expanduser().resolve()
    bd = root / "BattLeDIM_LTown"
    wn = root / "WNTR_Net3"
    lk = root / "LeakDB"
    root.mkdir(parents=True, exist_ok=True)

    # Default remains the practical CSV/core set, while --full-battledim is now
    # literally the complete Zenodo BattLeDIM record.
    core_names = [
        "2018_Leakages.csv", "2018_SCADA_Demands.csv", "2018_SCADA_Flows.csv",
        "2018_SCADA_Levels.csv", "2018_SCADA_Pressures.csv",
        "2019_Leakages.csv", "2019_SCADA_Demands.csv", "2019_SCADA_Flows.csv",
        "2019_SCADA_Levels.csv", "2019_SCADA_Pressures.csv",
        "dataset_configuration.yaml", "L-TOWN.inp", "README.txt",
    ]
    if args.starter:
        names = STARTER
    elif args.full_battledim or args.everything:
        names = list(BATTLEDIM.keys())
    else:
        names = core_names

    print("AQUA-VERA resumable downloader v3")
    print(f"Destination: {root}")
    print(f"BattLeDIM files selected: {len(names)}")
    print("Existing checksum-valid files will be skipped.\n")

    for name in names:
        url = f"{BATTLEDIM_BASE}/{name}?download=1"
        download(url, bd / name, BATTLEDIM[name], retries=args.retries)

    print("\nDownloading EPA WNTR Net3 network...")
    download(NET3_URL, wn / "Net3.inp", retries=args.retries)

    if args.leakdb_full or args.everything:
        print("\nWARNING: LeakDB.zip is approximately 13.9 GB.")
        download(LEAKDB_URL, lk / "LeakDB.zip", expected_md5=LEAKDB_MD5, retries=args.retries)
    else:
        lk.mkdir(parents=True, exist_ok=True)
        (lk / "FULL_DATA_NOT_DOWNLOADED.txt").write_text(
            "LeakDB full archive is ~13.9 GB and is intentionally opt-in.\n"
            "Run: python download_aqua_vera_data.py --output AQUA_VERA_data --leakdb-full\n"
            "Official record: https://zenodo.org/records/13985057\n",
            encoding="utf-8",
        )

    print("\nSUCCESS: selected downloads completed.")
    print("BattLeDIM files were checksum-verified before being accepted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled. Re-run later; any .part file will resume.")
        raise SystemExit(130)
