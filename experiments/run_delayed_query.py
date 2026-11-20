"""Paired delayed unknown-address queries and real erasure/reuse, CPU only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from integer_snn.model import Params
from integer_snn.network import NetworkSpec
from tasks.delayed_query import (fresh_network, write_pattern, prepare_query, query_after_delay,
                                 erase_and_recover, IdealBitRegister, logical_storage_ledger)
from experiments.run_pilot import peak_memory_bytes


ROOT = Path(__file__).resolve().parents[1]


def make_spec(modules, topology):
    weights = [[0] * modules for _ in range(modules)]
    if topology == "ring":
        for target in range(modules):
            weights[target][(target - 1) % modules] = 1
    elif topology != "uncoupled":
        raise ValueError("topology must be ring or uncoupled")
    return NetworkSpec(Params(5, 4, 3, 2, 3, 2, 6, 4, 14), tuple(tuple(row) for row in weights))


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _scores(rows):
    n = len(rows)
    changed = sum(row["queried_bit_changed"] for row in rows)
    return {
        "episodes": n,
        "initial_single_query_accuracy": sum(r["old_query_correct"] for r in rows) / n,
        "initial_full_pattern_accuracy": sum(r["old_pattern_correct"] for r in rows) / n,
        "erase_all_zero_rate": sum(r["erase_all_zero"] for r in rows) / n,
        "rewrite_single_query_accuracy": sum(r["new_query_correct"] for r in rows) / n,
        "rewrite_full_pattern_accuracy": sum(r["new_pattern_correct"] for r in rows) / n,
        "rewrite_query_error_count": sum(not r["new_query_correct"] for r in rows),
        "invalid_rewrite_queries": sum(r["new_prediction"] is None for r in rows),
        "noncanonical_rewrite_spike_windows": sum(r["new_phase_valid"] is False for r in rows),
        "changed_query_episodes": changed,
        "old_content_intrusions": sum(r["old_content_intrusion"] for r in rows),
        "old_content_intrusion_rate_given_changed_query": (
            sum(r["old_content_intrusion"] for r in rows) / changed if changed else None),
    }


def run(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh result directory")
    if not 1 <= args.episodes <= 16:
        raise ValueError("episodes must be in 1..16")
    output.mkdir(parents=True)
    config = {
        "evaluation_seeds": [0, 1, 2, 3, 4], "delays": [0, 1, 16, 64, 256],
        "modules": [2, 4, 8, 16], "topologies": ["uncoupled", "ring"],
        "episodes_per_condition": args.episodes,
        "episode_count_choice": "eight predefined episodes by default to keep all conditions under the 120-second CPU run budget; no outcome-based stopping",
        "write_amplitude": 14, "write_ticks": 2, "warmup_ticks": 6,
        "read_window_ticks": 2, "erase_duration": 4, "erase_recovery": 10,
        "primary_readout": "unique winner of two spike counts; ties invalid; exact (1,0)/(0,1) phase validity reported separately",
        "policies": ["snn_e3", "snn_e2_counterfactual", "ideal_bit_register"],
        "pattern_rng": "10000000 + seed*10000 + modules*10 + topology_index",
        "query_rng": "20000000 + seed*10000 + modules*10 + topology_index",
        "rng_pairing": "reinitialize each condition; identical pattern and query streams across delay and policy comparisons",
        "query_generation": "q1 drawn only after initial prepare_query returns; q2 drawn only after BOTH physical rewrite prepare_query calls return",
        "reference_spot_checks": "every update in seed0 episode0 for M2 and M4, both topologies and all delays",
        "training_performed": False, "evaluation_seeds_are_non_training": True,
        "hard_timeout_seconds": args.hard_timeout_seconds,
    }
    # Frozen before any episode is run; resource choices cannot follow outcomes.
    _json(output / "config.json", config)
    rows, failures, ledgers = [], [], []
    audit = {"network_steps": 0, "reference_steps": 0}
    condition_count = 0
    for seed, modules, topology, delay in itertools.product(
            config["evaluation_seeds"], config["modules"], config["topologies"], config["delays"]):
        if time.perf_counter() - started > args.hard_timeout_seconds:
            raise TimeoutError("Run budget reached between task conditions")
        spec = make_spec(modules, topology)
        topology_index = config["topologies"].index(topology)
        offset = seed * 10000 + modules * 10 + topology_index
        pattern_rng = random.Random(10000000 + offset)
        query_rng = random.Random(20000000 + offset)
        if seed == 0:
            ledgers.append({"modules": modules, "topology": topology, "delay": delay,
                            "weights": [list(row) for row in spec.weights],
                            "ledger": logical_storage_ledger(spec, delay)})
        for episode in range(args.episodes):
            old_pattern = tuple(pattern_rng.randrange(2) for _ in range(modules))
            new_pattern = tuple(pattern_rng.randrange(2) for _ in range(modules))
            reference = seed == 0 and episode == 0 and modules <= 4
            # Only the initial power-on uses fresh_network; never used for reuse.
            states = write_pattern(spec, fresh_network(spec), old_pattern,
                                   reference=reference, audit=audit)
            old_ready = prepare_query(spec, states, delay, start_tick=8,
                                      reference=reference, audit=audit)
            q1 = query_rng.randrange(modules)
            old_read = query_after_delay(old_ready, q1, reference=reference, audit=audit)
            prepared = {}
            for policy, amplitude in (("snn_e3", 3), ("snn_e2_counterfactual", 2)):
                erased = erase_and_recover(spec, old_read.next_states, amplitude,
                                           reference=reference, audit=audit)
                rewritten = write_pattern(spec, erased.states, new_pattern,
                                           reference=reference, audit=audit)
                ready = prepare_query(spec, rewritten, delay, start_tick=old_read.end_tick + 14 + 8,
                                      reference=reference, audit=audit)
                prepared[policy] = (erased, ready)
            q2 = query_rng.randrange(modules)
            base = {"seed": seed, "modules": modules, "topology": topology, "delay": delay,
                    "episode": episode, "old_pattern": "".join(map(str, old_pattern)),
                    "new_pattern": "".join(map(str, new_pattern)), "old_query": q1, "new_query": q2,
                    "old_query_ready_tick": old_ready.ready_tick,
                    "new_query_ready_tick": prepared["snn_e3"][1].ready_tick,
                    "queried_bit_changed": old_pattern[q2] != new_pattern[q2]}
            for policy in ("snn_e3", "snn_e2_counterfactual"):
                erased, ready = prepared[policy]
                read = query_after_delay(ready, q2, reference=reference, audit=audit)
                row = {**base, "policy": policy,
                       "old_prediction": old_read.prediction,
                       "new_prediction": read.prediction,
                       "old_query_correct": old_read.prediction == old_pattern[q1],
                       "old_pattern_correct": old_read.all_predictions == old_pattern,
                       "old_phase_valid": old_read.phase_valid,
                       "erase_all_zero": erased.zero,
                       "new_query_correct": read.prediction == new_pattern[q2],
                       "new_pattern_correct": read.all_predictions == new_pattern,
                       "new_phase_valid": read.phase_valid,
                       "old_content_intrusion": base["queried_bit_changed"] and read.prediction == old_pattern[q2],
                       "selected_counts_0": read.selected_counts[0], "selected_counts_1": read.selected_counts[1]}
                rows.append(row)
                if (not row["erase_all_zero"] or not row["new_pattern_correct"]) and len(failures) < 16:
                    failures.append({"metrics": row, "erased_states": [asdict(s) for s in erased.states],
                                     "rewrite_query_states": [asdict(s) for s in ready.states],
                                     "two_tick_all_counts": read.all_counts})
            # Different transition permissions are explicit, not presented as an
            # input-driven neural model or a fair energy-comparison participant.
            register = IdealBitRegister((0,) * modules).write(old_pattern)
            old_bit = register.read(q1)
            old_full = register.bits
            register = register.clear()
            erased_zero = all(bit == 0 for bit in register.bits)
            register = register.write(new_pattern)
            new_bit = register.read(q2)
            rows.append({**base, "policy": "ideal_bit_register", "old_prediction": old_bit,
                         "new_prediction": new_bit, "old_query_correct": old_bit == old_pattern[q1],
                         "old_pattern_correct": old_full == old_pattern, "erase_all_zero": erased_zero,
                         "old_phase_valid": None,
                         "new_query_correct": new_bit == new_pattern[q2],
                         "new_pattern_correct": register.bits == new_pattern,
                         "new_phase_valid": None,
                         "old_content_intrusion": base["queried_bit_changed"] and new_bit == old_pattern[q2],
                         "selected_counts_0": None, "selected_counts_1": None})
        condition_count += 1
        if condition_count % 40 == 0:
            print(f"Completed {condition_count}/200 task conditions, elapsed {time.perf_counter()-started:.2f}s", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["modules"], row["topology"], row["delay"], row["policy"])].append(row)
    grouped = [{"seed": key[0], "modules": key[1], "topology": key[2], "delay": key[3],
                "policy": key[4], **_scores(value)} for key, value in sorted(groups.items())]
    policies = {policy: _scores([row for row in rows if row["policy"] == policy]) for policy in config["policies"]}
    summary = {
        "status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _hash(config), "paired_episode_count": len(rows) // 3,
        "condition_count": condition_count, "rows": len(rows), "policy_metrics": policies,
        "scientific_rows_sha256": _hash(rows), "audit": audit,
        "training_performed": False, "seed_selection": "five prespecified non-training evaluation seeds; no learned parameters",
        "query_timing_checked_by_api": True,
        "scope": [
            "synthetic unknown-address-after-delay storage and reuse task, not SHD/SSC or trained classification",
            "q1 generated after old delay and q2 after both paired rewrite delays; labels only drive write input and scoring",
            "both SNN policies use identical patterns, query addresses and normal write pulses",
            "the physical pre-erase write/hold/read trajectory is shared since policies are identical up to the erase intervention",
            "E2 residual states are actually rewritten, never replaced by zero",
            "topology/coupling remains enabled through write, hold, query, erase, recovery and rewrite",
            "external noise is zero; this is not a whole-pipeline noisy-writing result",
            "whole-pattern readout is additional offline instrumentation, separate from one addressed readout",
            "digital register has direct assignment and clear privileges; no fair RAM/energy superiority claim",
        ],
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "peak_process_working_set_bytes": peak_memory_bytes(), "download_bytes": 0, "gpu_used": False,
    }
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _json(output / "condition_metrics.json", grouped)
    _json(output / "failure_examples.json", failures)
    _json(output / "storage_ledgers.json", ledgers)
    _json(output / "summary.json", summary)
    paths = [ROOT / "tasks/__init__.py", ROOT / "tasks/delayed_query.py", ROOT / "experiments/run_delayed_query.py",
             ROOT / "tests/test_delayed_query.py", ROOT / "integer_snn/network.py", ROOT / "integer_snn/model.py"]
    _json(output / "source_hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size > 5_000_000:
        raise AssertionError("Output exceeds five MB")
    if time.perf_counter() - started > args.hard_timeout_seconds:
        raise TimeoutError("Run exceeded declared 120-second budget")
    print(json.dumps({"output": str(output), "output_bytes": size, **summary}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--hard-timeout-seconds", type=float, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
