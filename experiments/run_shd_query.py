"""Paired real-source memory/reuse study, with a shared frozen front end."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np

from experiments.run_pilot import peak_memory_bytes
from integer_snn.model import Params
from integer_snn.network import NetworkSpec
from tasks.delayed_query import fresh_network, write_pattern, prepare_query, erase_and_recover
from tasks.shd_query import encode_classes, query_class_after_delay


def run(features, encoders, config_path, output):
    started, cpu_started = time.perf_counter(), time.process_time()
    if output.exists():
        raise FileExistsError("select a new output directory")
    output.mkdir(parents=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["slots"], config["bits_per_class"], config["module_count"]) != (4, 5, 20):
        raise ValueError("this task implementation requires exactly four 5-bit classes and 20 modules")
    if config["encoder_seeds"] != [0, 1, 2, 3, 4] or config["erase_amplitudes"] != [3, 2]:
        raise ValueError("this paired pilot fixes all five encoder seeds and the E3/E2 policies")
    for field in ("erase_duration", "recovery_ticks", "episodes_per_condition"):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if not config["delays"] or any(type(d) is not int or d < 0 for d in config["delays"]):
        raise ValueError("delays must be a nonempty list of nonnegative integers")
    (output / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    labels = np.load(features / "features.npz")["test_y"]
    m = config["module_count"]
    spec = NetworkSpec(Params(5, 4, 3, 2, 3, 2, 6, 4, 14),
                       tuple(tuple(int(j == (i - 1) % m) for j in range(m)) for i in range(m)))
    rows, episodes, audit, sampled_ids = [], [], {}, set()
    for seed in config["encoder_seeds"]:
        prediction = np.load(encoders / f"test_predictions_seed{seed}.npy")
        if len(prediction) != len(labels):
            raise ValueError("encoder and data predictions have different lengths")
        for delay in config["delays"]:
            # Paired across delays: identical utterance indices and addresses.
            # Addresses are still generated only after the physical holds below.
            pattern_rng = random.Random(config["task_seed_base"] + seed)
            query_rng = random.Random(config["query_seed_base"] + seed)
            for episode in range(config["episodes_per_condition"]):
                ids = pattern_rng.sample(range(len(labels)), 8)
                old_ids, new_ids = ids[:4], ids[4:]
                sampled_ids.update(ids)
                old_classes = tuple(int(prediction[i]) for i in old_ids)
                new_classes = tuple(int(prediction[i]) for i in new_ids)
                # One shared physical old-memory trajectory for the paired policies.
                reference = episode == 0 and delay == config["delays"][0]
                states = write_pattern(spec, fresh_network(spec), encode_classes(old_classes),
                                       reference=reference, audit=audit)
                ready = prepare_query(spec, states, delay, start_tick=8, reference=reference, audit=audit)
                old_slot = query_rng.randrange(4)  # not drawn before prepare_query completes
                old_read = query_class_after_delay(ready, old_slot, reference=reference, audit=audit)
                prepared, erased = {}, {}
                for e in config["erase_amplitudes"]:
                    erased[e] = erase_and_recover(spec, old_read.next_states, amplitude=e,
                                                  duration=config["erase_duration"],
                                                  recovery=config["recovery_ticks"],
                                                  reference=reference, audit=audit)
                    rewritten = write_pattern(spec, erased[e].states, encode_classes(new_classes),
                                              reference=reference, audit=audit)
                    prepared[e] = prepare_query(spec, rewritten, delay,
                                                 start_tick=(old_read.end_tick + config["erase_duration"]
                                                             + config["recovery_ticks"] + 8),
                                                 reference=reference, audit=audit)
                # Same new address for both policies, drawn after both holds complete.
                new_slot = query_rng.randrange(4)
                episode_key = f"s{seed}_d{delay}_e{episode}"
                episodes.append({"id": episode_key, "old_test_indices": old_ids,
                                 "new_test_indices": new_ids, "old_query_slot": old_slot,
                                 "new_query_slot": new_slot, "old_encoder_classes": old_classes,
                                 "new_encoder_classes": new_classes})
                for e in config["erase_amplitudes"]:
                    read = query_class_after_delay(prepared[e], new_slot, reference=reference, audit=audit)
                    old_enc, new_enc = old_classes[old_slot], new_classes[new_slot]
                    different = old_classes[new_slot] != new_enc
                    rows.append({
                        "episode": episode_key, "seed": seed, "delay": delay, "method": f"integer_E{e}",
                        "old_query_memory_correct": old_read.prediction == old_enc,
                        "old_query_label_correct": old_read.prediction == int(labels[old_ids[old_slot]]),
                        "erase_zero": erased[e].zero,
                        "rewrite_memory_correct": read.prediction == new_enc,
                        "rewrite_label_correct": read.prediction == int(labels[new_ids[new_slot]]),
                        "shared_encoder_rewrite_label_correct": new_enc == int(labels[new_ids[new_slot]]),
                        "rewrite_invalid": read.prediction is None,
                        "rewrite_phase_valid": read.phase_valid,
                        "intrusion_eligible": different,
                        "old_content_intrusion": different and read.prediction == old_classes[new_slot],
                        "new_prediction": read.prediction, "target_encoded_class": new_enc,
                        "target_true_class": int(labels[new_ids[new_slot]])})
                # Functional reference. This direct copy/clear has different physical permissions.
                rows.append({
                    "episode": episode_key, "seed": seed, "delay": delay, "method": "ideal_register",
                    "old_query_memory_correct": True,
                    "old_query_label_correct": old_classes[old_slot] == int(labels[old_ids[old_slot]]),
                    "erase_zero": True, "rewrite_memory_correct": True,
                    "rewrite_label_correct": new_classes[new_slot] == int(labels[new_ids[new_slot]]),
                    "shared_encoder_rewrite_label_correct": new_classes[new_slot] == int(labels[new_ids[new_slot]]),
                    "rewrite_invalid": False,
                    "rewrite_phase_valid": None,
                    "intrusion_eligible": old_classes[new_slot] != new_classes[new_slot],
                    "old_content_intrusion": False,
                    "new_prediction": new_classes[new_slot], "target_encoded_class": new_classes[new_slot],
                    "target_true_class": int(labels[new_ids[new_slot]])})
            print(f"seed={seed} delay={delay} completed", flush=True)
    metrics = ("old_query_memory_correct", "old_query_label_correct", "erase_zero",
               "rewrite_memory_correct", "rewrite_label_correct", "shared_encoder_rewrite_label_correct",
               "rewrite_invalid")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["delay"], row["seed"])].append(row)
    grouped = []
    for (method, delay, seed), values in sorted(groups.items()):
        eligible = sum(r["intrusion_eligible"] for r in values)
        grouped.append({"method": method, "delay": delay, "seed": seed, "n": len(values),
                        **{key: sum(r[key] for r in values) / len(values) for key in metrics},
                        "intrusion_eligible_n": eligible,
                        "old_content_intrusion_rate": (sum(r["old_content_intrusion"] for r in values) / eligible
                                                      if eligible else None)})
    paired_summary = []
    for method in ("integer_E3", "integer_E2", "ideal_register"):
        selected = [r for r in rows if r["method"] == method]
        eligible = sum(r["intrusion_eligible"] for r in selected)
        per_seed = [statistics.mean(r["rewrite_label_correct"] for r in selected if r["seed"] == seed)
                    for seed in config["encoder_seeds"]]
        mean, sd = statistics.mean(per_seed), statistics.stdev(per_seed)
        paired_summary.append({"method": method, "n": len(selected),
                               **{key: statistics.mean(r[key] for r in selected) for key in metrics},
                               "rewrite_phase_valid_rate": (statistics.mean(r["rewrite_phase_valid"] for r in selected)
                                                            if method != "ideal_register" else None),
                               "intrusion_eligible_n": eligible,
                               "old_content_intrusion_rate": (sum(r["old_content_intrusion"] for r in selected) / eligible
                                                             if eligible else None),
                               "rewrite_accuracy_seed_sd": sd,
                               "rewrite_accuracy_seed_t95": [mean - 2.776445105 * sd / (5 ** .5),
                                                            mean + 2.776445105 * sd / (5 ** .5)]})
    with (output / "episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "episode_manifest.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    source_files = [Path(__file__), Path("tasks/shd_query.py"), Path("tasks/delayed_query.py"),
                    Path("integer_snn/network.py"), Path("integer_snn/model.py"), config_path]
    summary = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "source_encoder_summary": str((encoders / "summary.json").resolve()),
        "paired_episodes": len(episodes), "strategy_rows": len(rows),
        "unique_official_test_samples_used": len(sampled_ids), "metrics": paired_summary,
        "seed_delay_metrics": grouped, "audit": audit,
        "logical_storage": {
            "memory_content_bits": 20, "snn_dynamic_bits": spec.persisted_state_bits,
            "five_pairs_of_two_bit_spike_counters_bits": 20,
            "query_slot_bits": 2, "query_output_bits": 5, "window_counter_bits": 2,
            "schedule_counter_bits_for_max_delay": max(config["delays"] + [14, 8]).bit_length(),
            "phase_bits": 3, "external_encoded_input_bits": 20,
            "shared_encoder_float32_parameter_bytes": 112080,
            "shared_feature_normalizer_float32_bytes": 11200,
            "shared_front_end_feature_vector_float32_bytes_during_encoding": 5600,
            "weights_topology_and_parameter_storage": "not included in dynamic total; count in hardware implementation",
            "ideal_register_content_bits": 20,
            "scope": "logical dynamic state plus explicit accessories; no RAM/energy superiority claim"},
        "limitations": [
            "derived four-utterance class-selection task, not the original SHD benchmark",
            "same frozen float32 encoder estimates shared by all memories; no end-to-end SNN training",
            "query is future slot selection among encoded classes, not arbitrary semantic content",
            "test utterances can recur across episodes; episode count is not independent source sample count",
            "identical utterance/address episodes reused across delays; paired delay cells are not independent samples",
            "five seeds vary encoder and episode/query generation; interval is seed variation only",
            "E2 is a paired erase-budget counterfactual, not a strong alternative architecture",
            "v1 used overly strict content readout; v2 corrected it but used unpaired delays; v3 pairs delays with new fixed task seeds after audit",
            "ideal register uses direct write/clear and is a functional reference only",
            "no end-to-end resource or energy advantage established",
        ],
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "peak_process_working_set_bytes": peak_memory_bytes(), "gpu_used": False, "download_bytes": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "wall_seconds": summary["wall_seconds"],
                      "metrics": paired_summary}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/shd/features_v1"))
    parser.add_argument("--encoders", type=Path, default=Path("results/shd_encoder_20261112_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/shd_query_v3.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.encoders, args.config, args.output)
