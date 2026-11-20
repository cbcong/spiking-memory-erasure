"""Train five small shared front ends on CPU with a fixed speaker split."""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from encoders.linear import fit, evaluate
from experiments.run_pilot import peak_memory_bytes


def run(features: Path, config_path: Path, output: Path):
    started, cpu_started = time.perf_counter(), time.process_time()
    if output.exists():
        raise FileExistsError("choose a new run directory")
    output.mkdir(parents=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (output / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if config["seeds"] != [0, 1, 2, 3, 4]:
        raise ValueError("this preregistered pilot uses exactly five seeds")
    f = np.load(features / "features.npz")
    x, y, vx, vy = [f[k] for k in ("train_x", "train_y", "valid_x", "valid_y")]
    fitted = []
    kwargs = {k: config[k] for k in ("max_epochs", "batch_size", "learning_rate", "l2", "patience")}
    for seed in config["seeds"]:
        fit_started = time.perf_counter()
        w, b, record = fit(x, y, vx, vy, seed, **kwargs)
        training, _ = evaluate(x, y, w, b)
        validation, _ = evaluate(vx, vy, w, b)
        record.update(seed=seed, training=training, validation=validation,
                      wall_seconds=time.perf_counter() - fit_started)
        fitted.append((seed, w, b, record))
        np.savez_compressed(output / f"encoder_seed{seed}.npz", weight=w, bias=b)
        (output / f"training_seed{seed}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"seed": seed, "train": training["accuracy"], "validation": validation["accuracy"],
                          "selected_epoch": record["selected_epoch"], "seconds": record["wall_seconds"]}), flush=True)
    # Only now is the official test array used by evaluation.
    tx, ty, speaker = [f[k] for k in ("test_x", "test_y", "test_speaker")]
    results = []
    for seed, w, b, record in fitted:
        test, prediction = evaluate(tx, ty, w, b)
        unseen = np.isin(speaker, [4, 5])
        subgroup = {name: {"n": int(mask.sum()), "accuracy": float(np.mean(prediction[mask] == ty[mask]))}
                    for name, mask in (("unseen_speakers_4_5", unseen), ("other_speakers", ~unseen))}
        np.save(output / f"test_predictions_seed{seed}.npy", prediction.astype(np.uint8))
        results.append({"seed": seed, "training": record["training"], "validation": record["validation"],
                        "test": test, "subgroups": subgroup, "selected_epoch": record["selected_epoch"]})
    accuracies = np.array([r["test"]["accuracy"] for r in results])
    sd = float(accuracies.std(ddof=1))
    source_files = [Path("encoders/linear.py"), Path(__file__), config_path,
                    Path("experiments/build_shd_features.py"), Path("tests/test_linear_encoder.py")]
    summary = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "features_summary": json.loads((features / "summary.json").read_text(encoding="utf-8")),
        "seeds": results, "test_accuracy_mean": float(accuracies.mean()), "test_accuracy_seed_sd": sd,
        "test_accuracy_seed_mean_t95_interval": [float(accuracies.mean() - 2.776445105 * sd / np.sqrt(5)),
                                                float(accuracies.mean() + 2.776445105 * sd / np.sqrt(5))],
        "interval_scope": "5 optimizer/initialization seeds on the same test set; not dataset sampling uncertainty",
        "encoder_parameter_count": int(fitted[0][1].size + fitted[0][2].size),
        "encoder_parameter_bytes_float32": int(fitted[0][1].nbytes + fitted[0][2].nbytes),
        "scope": ["shared offline float32 linear encoder; integer memory is evaluated separately",
                  "not original SHD benchmark SOTA; validation speakers are held out of official train",
                  "no query or delayed-task answer enters encoder features",
                  "same preprocessing and seed-specific encoder to be used for all memory baselines",
                  "do not select the best seed or tune using these test results"],
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "peak_process_working_set_bytes": peak_memory_bytes(), "gpu_used": False, "download_bytes": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "test_accuracy_mean", "test_accuracy_seed_sd",
                                             "wall_seconds", "cpu_seconds", "peak_process_working_set_bytes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/shd/features_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/shd_encoder_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.config, args.output)
