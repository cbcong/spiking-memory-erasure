"""Bounded shared front-end features; raw files are never modified.

This is an offline utterance encoder input, not an end-to-end streaming SNN.
The future memory-query task will use the same fitted encoder for all memories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from experiments.run_pilot import peak_memory_bytes


def extract(path, channel_groups=70, time_bins=20, seconds=1.5):
    rows, hashes = [], []
    stats = {"samples": 0, "events": 0, "events_at_or_after_horizon": 0,
             "empty_samples": 0, "maximum_event_time_seconds": 0.0}
    with h5py.File(path, "r") as f:
        labels = np.asarray(f["labels"][:], dtype=np.int64)
        speakers = np.asarray(f["extra/speaker"][:], dtype=np.int64)
        if len(labels) != len(speakers) or not np.isin(labels, np.arange(20)).all():
            raise ValueError("unexpected label/speaker format")
        for i in range(len(labels)):
            times = np.asarray(f["spikes/times"][i], dtype=np.float64)
            units = np.asarray(f["spikes/units"][i], dtype=np.int64)
            if (len(times) != len(units) or not np.isfinite(times).all()
                    or (times < 0).any() or (units < 0).any() or (units >= 700).any()):
                raise ValueError(f"Invalid events at {path.name}:{i}")
            stats["events"] += len(times)
            stats["empty_samples"] += int(len(times) == 0)
            if len(times):
                stats["maximum_event_time_seconds"] = max(stats["maximum_event_time_seconds"], float(times.max()))
            stats["events_at_or_after_horizon"] += int(np.count_nonzero(times >= seconds))
            # Late events are explicitly accumulated in the last bin, not dropped.
            time_index = np.minimum((times * time_bins / seconds).astype(np.int64), time_bins - 1)
            channel_index = units * channel_groups // 700
            count = np.bincount(time_index * channel_groups + channel_index,
                                minlength=channel_groups * time_bins).astype(np.float32)
            rows.append(np.log1p(count))
            sha = hashlib.sha256(times.tobytes() + units.tobytes()).hexdigest()
            hashes.append(sha)
        stats["samples"] = len(labels)
    return np.stack(rows), labels, speakers, hashes, stats


def run(raw: Path, output: Path):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("choose a fresh features directory")
    output.mkdir(parents=True)
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    splits, summaries, sample_hashes = {}, {}, {}
    for split in ("train", "test"):
        filename = f"shd_{split}.h5"
        expected = next(f for f in manifest["files"] if f["raw_name"] == filename)
        sha = hashlib.sha256()
        with (raw / filename).open("rb") as stream:
            while chunk := stream.read(2**20):
                sha.update(chunk)
        if sha.hexdigest() != expected["raw_sha256"]:
            raise ValueError("raw data hash changed")
        x, y, speaker, hashes, stats = extract(raw / filename)
        splits[split] = (x, y, speaker)
        sample_hashes[split] = hashes
        summaries[split] = stats
        print(f"{split}: {len(y)} samples, {stats['events']} events", flush=True)
    train_x, train_y, train_speaker = splits["train"]
    test_x, test_y, test_speaker = splits["test"]
    validation = np.isin(train_speaker, [0, 1])
    training = ~validation
    if set(train_speaker[training]) & set(train_speaker[validation]):
        raise AssertionError("speaker leakage in development split")
    train_hashes = {sample_hashes["train"][i] for i in np.flatnonzero(training)}
    valid_hashes = {sample_hashes["train"][i] for i in np.flatnonzero(validation)}
    test_hashes = set(sample_hashes["test"])
    overlap = {"train_validation": len(train_hashes & valid_hashes),
               "development_official_test": len(set(sample_hashes["train"]) & test_hashes)}
    if any(overlap.values()):
        raise ValueError(f"exact event record leakage detected: {overlap}")
    # Normalization is fitted only on the chosen eight training speakers.
    mean = train_x[training].mean(axis=0)
    scale = np.maximum(train_x[training].std(axis=0), np.float32(0.1))
    normalized_train = np.clip((train_x - mean) / scale, -5, 5).astype(np.float32)
    normalized_test = np.clip((test_x - mean) / scale, -5, 5).astype(np.float32)
    np.savez_compressed(output / "features.npz", train_x=normalized_train[training],
                        train_y=train_y[training], train_speaker=train_speaker[training],
                        valid_x=normalized_train[validation], valid_y=train_y[validation],
                        valid_speaker=train_speaker[validation], test_x=normalized_test,
                        test_y=test_y, test_speaker=test_speaker, mean=mean, scale=scale,
                        training_original_indices=np.flatnonzero(training),
                        validation_original_indices=np.flatnonzero(validation))
    summary = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "raw_manifest": str((raw / "manifest.json").resolve()),
        "configuration": {"channel_groups": 70, "time_bins": 20, "horizon_seconds": 1.5,
                          "count_transform": "log1p", "validation_speakers": [0, 1],
                          "normalization": "train-speaker-only mean/std; std floor 0.1; clip [-5,5]"},
        "sample_counts": {"train": int(training.sum()), "validation": int(validation.sum()),
                          "official_test": len(test_y)},
        "training_speakers": sorted(map(int, set(train_speaker[training]))),
        "validation_speakers": [0, 1],
        "official_test_unseen_in_original_train_speakers": [4, 5],
        "exact_record_overlap": overlap, "source_summaries": summaries,
        "features_sha256": hashlib.sha256((output / "features.npz").read_bytes()).hexdigest(),
        "scope": ["offline event histogram front end, not end-to-end SNN",
                  "test labels are not used for fitting, model selection or normalization",
                  "future query identities are absent from encoder features",
                  "official test has both seen and unseen speakers; report subsets separately",
                  "the task built later is a derived memory benchmark, not the original SHD leaderboard"],
        "wall_seconds": time.perf_counter() - started,
        "peak_process_working_set_bytes": peak_memory_bytes(),
        "files_bytes_before_summary": sum(p.stat().st_size for p in output.iterdir()),
        "numpy_version": np.__version__, "h5py_version": h5py.__version__,
    }
    (output / "sample_event_hashes.json").write_text(json.dumps(sample_hashes), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/shd/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/shd/features_v1"))
    args = parser.parse_args()
    run(args.raw, args.output)
