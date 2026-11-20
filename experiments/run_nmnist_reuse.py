"""Frozen real-event-to-memory transfer pilot on the N-MNIST official test set."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import requests

from experiments.run_locked_erasure_scale import plans_at
from integer_snn.network import NetworkSpec
from integer_snn.topology_certificate import MODEL
from tasks.delayed_query import (IdealBitRegister, erase_and_recover, fresh_network,
                                 prepare_query, query_after_delay, write_pattern)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/nmnist_reuse_20261119.json"
DEFAULT_ARCHIVE = ROOT / "data/nmnist/Test.zip"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_archive(path, config):
    if path.stat().st_size != config["size_bytes"] or digest(path) != config["sha256"]:
        raise ValueError("Test.zip does not match the publisher SHA256 and length")


def fetch(path, config):
    if path.exists():
        check_archive(path, config)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".zip.part")
    offset = temp.stat().st_size if temp.exists() else 0
    length = config["size_bytes"]
    with requests.Session() as session, temp.open("ab") as out:
        while offset < length:
            last = min(offset + 4 * 1024**2, length) - 1
            with session.get(config["url"], params={"offset": offset},
                             headers={"Range": f"bytes={offset}-{last}"},
                             timeout=(12, 60), stream=True) as response:
                response.raise_for_status()
                if response.headers.get("Content-Range") != f"bytes {offset}-{last}/{length}":
                    raise ValueError("publisher did not serve the requested byte range")
                block = response.content
                if len(block) != last - offset + 1:
                    raise ValueError("incomplete byte range; retry to resume")
                out.write(block)
                offset += len(block)
    check_archive(temp, config)
    os.replace(temp, path)


def descriptor(raw):
    events = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 5)
    events = events[events[:, 1] != 240]
    if not len(events) or events[:, :2].max() >= 34:
        raise ValueError("invalid 34 x 34 event coordinates")
    bands = 4 * events[:, 1].astype(int) // 34
    right = np.bincount(bands[events[:, 0] >= 17], minlength=4)
    left = np.bincount(bands[events[:, 0] < 17], minlength=4)
    return tuple(map(int, right > left)), len(events)


def select_files(archive):
    groups = [[] for _ in range(10)]
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix.lower() == ".bin" and path.stem.isdigit() and path.parent.name.isdigit():
            groups[int(path.parent.name)].append(name)
    if sum(map(len, groups)) != 10000 or min(map(len, groups)) < 32:
        raise ValueError("unexpected archive file tree")
    key = lambda name: hashlib.sha256(("NN-NMNIST-20261119-v1|" + name).encode()).digest()
    return [sorted(group, key=key)[:32] for group in groups]


def read(spec, states, delay, reference):
    ready = prepare_query(spec, states, delay, reference=reference)
    return query_after_delay(ready, 0, reference=reference)


def run(path, output):
    config = json.loads(DESIGN.read_bytes())
    check_archive(path, config)
    spec = NetworkSpec(MODEL, tuple(tuple(int((j+1) % 4 == i) for j in range(4))
                                    for i in range(4)))
    plans = plans_at(spec, config["deadline"])
    locked = plans["locked"]
    arms = {"locked": locked, "row_tight": plans["row_tight"],
            "no_suppression_equal_wait": (0, locked[1], locked[2])}
    episodes = []
    with ZipFile(path) as archive:
        names = select_files(archive)
        for digit in range(10):
            for j in range(16):
                old_path = names[digit][j]
                new_path = names[(digit+1) % 10][16+j]
                old, old_events = descriptor(archive.read(old_path))
                new, new_events = descriptor(archive.read(new_path))
                reference = j == 0
                start = write_pattern(spec, fresh_network(spec), old, reference=reference)
                old_read = read(spec, start, config["delay_steps"], reference)
                outcomes = {}
                for arm, plan in arms.items():
                    erased = erase_and_recover(spec, old_read.next_states, *plan, reference=reference)
                    rewritten = write_pattern(spec, erased.states, new, reference=reference)
                    new_read = read(spec, rewritten, config["delay_steps"], reference)
                    outcomes[arm] = dict(old_read=old_read.all_predictions, zero=erased.zero,
                                         new_read=new_read.all_predictions)
                reg = IdealBitRegister((0,)*4).write(old)
                cleared = reg.clear()
                outcomes["ideal_register"] = dict(old_read=reg.bits, zero=not any(cleared.bits),
                                                   new_read=cleared.write(new).bits)
                episodes.append(dict(old_path=old_path, new_path=new_path,
                                     old_events=old_events, new_events=new_events,
                                     old=old, new=new, outcomes=outcomes))
    used = [r[role] for r in episodes for role in ("old_path","new_path")]
    assert len(set(used)) == 320
    metrics = {}
    for arm in config["arms"]:
        counts = Counter()
        for row in episodes:
            outcome = row["outcomes"][arm]
            counts["episodes"] += 1
            counts["old_correct"] += outcome["old_read"] == row["old"]
            counts["zero"] += outcome["zero"]
            counts["new_correct"] += outcome["new_read"] == row["new"]
            counts["bit_errors"] += sum(a != b for a,b in zip(row["new"],outcome["new_read"]))
            changed = row["new"] != row["old"]
            counts["different"] += changed
            counts["old_intrusions"] += changed and outcome["new_read"] == row["old"]
        metrics[arm] = dict(counts)
    result = dict(archive_sha256=config["sha256"], design_sha256=digest(DESIGN),
                  script_sha256=digest(Path(__file__)), plans=arms, metrics=metrics,
                  distinct_codes=len({r[key] for r in episodes for key in ("old","new")}),
                  total_events=sum(r["old_events"]+r["new_events"] for r in episodes),
                  fraction_crosschecked_episodes=10, rows=episodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x",encoding="utf8") as out:
        json.dump(result,out,indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--fetch", action="store_true", help="download the published 169674850-byte test zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.archive, json.loads(DESIGN.read_bytes()))
    run(args.archive, args.output)
