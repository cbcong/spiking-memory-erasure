"""Download the pre-announced SHD files with strict size/checksum limits.

No framework installs or implicit home-directory caches. The raw HDF5 files
and original archives stay in a single explicit project data directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile


FILES = (
    {"name": "shd_train.h5.zip", "archive_bytes": 130863613, "raw_bytes": 268826085,
     "md5": "f3252aeb598ac776c1b526422d90eecb", "crc32": "94bf93b0"},
    {"name": "shd_test.h5.zip", "archive_bytes": 38141465, "raw_bytes": 78719235,
     "md5": "1503a5064faa34311c398fb0a1ed0a6f", "crc32": "81e01e06"},
)
BASE = "https://zenkelab.org/datasets/shd/"


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2**20):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def run(destination: Path):
    started = time.perf_counter()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    if free < 2 * 2**30:
        raise OSError("Require 2 GiB free before starting; forecast itself is under 0.8 GB")
    entries = []
    transferred = 0
    for item in FILES:
        archive = destination / item["name"]
        url = BASE + item["name"]
        if not archive.exists():
            part = archive.with_suffix(archive.suffix + ".part")
            if part.exists():
                raise FileExistsError(f"Partial file retained for inspection: {part}")
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            count, next_report = 0, 16 * 2**20
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200 or int(response.headers["Content-Length"]) != item["archive_bytes"]:
                    raise ValueError("Remote archive changed; re-check and announce the disk forecast")
                with part.open("xb") as stream:
                    while chunk := response.read(2**20):
                        count += len(chunk)
                        if count > item["archive_bytes"]:
                            raise ValueError("Remote transfer exceeds announced size")
                        stream.write(chunk)
                        if count >= next_report:
                            print(f'{item["name"]}: {count / 1e6:.1f} MB', flush=True)
                            next_report += 16 * 2**20
            if count != item["archive_bytes"] or digests(part)["md5"] != item["md5"]:
                raise ValueError(f"Archive length/checksum mismatch: {part}")
            part.rename(archive)
            transferred += count
        archive_hashes = digests(archive)
        if archive.stat().st_size != item["archive_bytes"] or archive_hashes["md5"] != item["md5"]:
            raise ValueError(f"Existing archive mismatch; preserved without overwriting: {archive}")
        raw = destination / item["name"][:-4]
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if len(members) != 1 or members[0].filename != raw.name:
                raise ValueError("Unexpected ZIP members; extraction refused")
            info = members[0]
            if info.file_size != item["raw_bytes"] or f"{info.CRC:08x}" != item["crc32"]:
                raise ValueError("Uncompressed size or CRC differs from inspected metadata")
            if not raw.exists():
                partial_raw = raw.with_suffix(raw.suffix + ".part")
                count = 0
                with z.open(info) as stream, partial_raw.open("xb") as out:
                    while chunk := stream.read(2**20):
                        count += len(chunk)
                        if count > item["raw_bytes"]:
                            raise ValueError("Extracted size exceeds announced maximum")
                        out.write(chunk)
                if count != item["raw_bytes"]:
                    raise ValueError("Incomplete HDF5 extraction")
                partial_raw.rename(raw)
        if raw.stat().st_size != item["raw_bytes"]:
            raise ValueError("Existing HDF5 has wrong size; retained for inspection")
        entries.append({**item, "url": url, "archive_sha256": archive_hashes["sha256"],
                        "raw_name": raw.name, "raw_sha256": digests(raw)["sha256"]})
        print(f"{raw.name}: verified and ready", flush=True)
    manifest = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/",
        "checksum_source": BASE + "md5sums.txt",
        "license": "CC BY 4.0",
        "citation": "Cramer et al. (2022), doi:10.1109/TNNLS.2020.3044364",
        "preannounced_storage_forecast_bytes": 800_000_000,
        "free_bytes_before": free, "download_bytes_this_run": transferred,
        "stored_archive_plus_raw_bytes": sum(i["archive_bytes"] + i["raw_bytes"] for i in FILES),
        "wall_seconds": time.perf_counter() - started, "files": entries,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "directory": str(destination),
                      "download_bytes": transferred, "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/shd/raw"))
    args = parser.parse_args()
    run(args.output)
