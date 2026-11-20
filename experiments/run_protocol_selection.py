"""Frozen 213-case, three-arm fixed-protocol selection with Fraction audits.

CPU only; standard library only; no downloads.  The output directory must not
exist.  A fixed first-32-case development round precedes the complete run and
is not included in the scientific aggregates.  Every accepted (case,E,D) is
audited on the original two-symbol domain with independent Fraction dynamics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

from integer_snn.fixed_protocol import (METHODS, enumerate_domain, reconstruct_case,
                                        select_protocol, state_key)
from integer_snn.model import zero_state
from integer_snn.reference import reference_step


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "amplitudes": [0, 64], "durations": [1, 32], "recovery_limit": 32,
    "candidate_order": "lexicographic increasing E, then D",
    "e0_rule": "exclude by independently reconstructed nonzero periodic retention for every arm",
    "budgets_ms": [20, 100], "repetitions": 5, "sample_target": 128,
    "method_order_seed": 2026111301, "sample_seed_base": 2026111302,
    "max_boxes": 128, "development_indices": list(range(32)),
    "development_repetitions": 1, "case_count": 213,
    "formal_run_limit_seconds": 600, "output_limit_bytes": 10000000,
    "thread_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    "scope": "retrospective comparison on 213 previously explored sparse-period configurations",
    "sampling": "all phase corners plus phase-balanced random without replacement to 128, or full domain",
    "budget_semantics": "complete preprocessing and complete candidate calls charged in wall time; reject over-budget results",
    "deploy_schedule": "all accepted protocols use actual pulse D followed by 32 zero-input recovery ticks",
    "method_optimizations": "per-run pulse prefix, transition and/or recovery caches; no cross-run cache",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "mean": statistics.mean(values), "p95": values[int((len(values) - 1) * .95)],
            "max": values[-1], "sum": sum(values)}


def audit_protocol(params, states, amplitude, duration, recovery_limit=32):
    """Independent Fraction images retain multiplicities of original starts.

    The images merge equal states only after adding their original-state
    counts.  Thus failure counts and recovery underestimation are counts of
    original starts, not merely counts of distinct release states.
    """
    started, cpu_started = time.perf_counter(), time.process_time()
    transition_cache, transitions = {}, 0
    image = {s: (1, s) for s in states}

    def advance(image, erase):
        nonlocal transitions
        out = {}
        for state, (count, original) in image.items():
            key = (state, erase)
            if key not in transition_cache:
                transition_cache[key] = reference_step(params, state, erase=erase)
                transitions += 1
            nxt = transition_cache[key]
            previous, representative = out.get(nxt, (0, original))
            out[nxt] = (previous + count, representative)
        return out

    for _ in range(duration):
        image = advance(image, amplitude)
    release_states = len(image)
    total = len(states)
    nonzero_counts = []
    max_actual_recovery = None
    for recovery in range(recovery_limit + 1):
        nonzero = total - image.get(zero_state(), (0, None))[0]
        nonzero_counts.append(nonzero)
        if nonzero == 0 and max_actual_recovery is None:
            max_actual_recovery = recovery
        if recovery < recovery_limit:
            image = advance(image, 0)
    failure_examples, failure_statuses = [], Counter()
    for state, (count, original) in sorted(image.items(), key=lambda item: state_key(item[0])):
        if state == zero_state():
            continue
        current, seen, path = state, {}, []
        status, period = "unresolved_after_256_additional_ticks", None
        for tick in range(257):
            if current == zero_state():
                status = "late_zero_after_recovery_limit"
                break
            if current in seen:
                status, period = "nonzero_cycle", tick - seen[current]
                break
            seen[current] = tick
            path.append(current)
            current = next(iter(advance({current: (1, original)}, 0)))
        failure_statuses[status] += count
        if len(failure_examples) < 3:
            failure_examples.append({"initial_state": asdict(original), "at_recovery_limit": asdict(state),
                                     "represented_initial_states": count, "status": status, "period": period})
    return {"passed": nonzero_counts[-1] == 0, "original_initial_states": total,
            "release_image_states": release_states, "max_actual_recovery": max_actual_recovery,
            "actual_total": duration + max_actual_recovery if max_actual_recovery is not None else None,
            "deploy_total": duration + recovery_limit, "failed_original_states": nonzero_counts[-1],
            "nonzero_original_states_by_recovery_tick": nonzero_counts,
            "failure_status_counts": dict(failure_statuses), "failure_examples": failure_examples,
            "fraction_transition_evaluations": transitions,
            "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
            "independent_fraction": True}


def selection_round(cases, repeats, progress, deadline=None):
    rng = random.Random(CONFIG["method_order_seed"])
    rows = []
    for case_index, case in enumerate(cases):
        for budget_ms in CONFIG["budgets_ms"]:
            for repetition in range(repeats):
                order = list(METHODS)
                rng.shuffle(order)
                for order_index, method in enumerate(order):
                    if deadline is not None and time.perf_counter() > deadline:
                        raise TimeoutError("formal run deadline exceeded between atomic calls")
                    sample_seed = CONFIG["sample_seed_base"] + 1009 * case_index + repetition
                    result = select_protocol(case, method, budget_ms / 1000, sample_seed,
                                             max_amplitude=CONFIG["amplitudes"][1],
                                             max_duration=CONFIG["durations"][1],
                                             recovery_limit=CONFIG["recovery_limit"],
                                             max_boxes=CONFIG["max_boxes"], sample_target=CONFIG["sample_target"])
                    protocol = result.pop("protocol")
                    events = result.pop("native_event_counts")
                    last = result.pop("last_candidate")
                    row = {"case_index": case_index, "budget_ms": budget_ms,
                           "repetition": repetition, "random_order": order_index, **result}
                    row.update(protocol or {"amplitude": None, "duration": None, "reported_recovery": None,
                                            "deploy_total": None, "recovery_kind": None})
                    row.update({"last_amplitude": last["amplitude"] if last else None,
                                "last_duration": last["duration"] if last else None,
                                "last_candidate_accepted": last["candidate_accepted"] if last else None})
                    row.update({f"event_{key}": value for key, value in events.items()})
                    rows.append(row)
        if (case_index + 1) % 8 == 0 or case_index + 1 == len(cases):
            progress(case_index + 1, len(cases), rows)
    return rows


def aggregates(rows):
    output = []
    for budget_ms in CONFIG["budgets_ms"]:
        for method in METHODS:
            group = [r for r in rows if r["budget_ms"] == budget_ms and r["method"] == method]
            accepted = [r for r in group if r["status"] == "accepted"]
            valid = [r for r in accepted if r["audit_passed"]]
            false = [r for r in accepted if not r["audit_passed"]]
            per_repeat = []
            for repetition in range(CONFIG["repetitions"]):
                selected = [r for r in group if r["repetition"] == repetition]
                per_repeat.append({"repetition": repetition, "cases": len(selected),
                                   "accepted": sum(r["status"] == "accepted" for r in selected),
                                   "valid_found": sum(r.get("audit_passed") is True for r in selected),
                                   "false_acceptances": sum(r.get("audit_passed") is False for r in selected),
                                   "valid_found_rate": sum(r.get("audit_passed") is True for r in selected) / len(selected)})
            output.append({"budget_ms": budget_ms, "method": method, "selection_runs": len(group),
                           "accepted_runs": len(accepted), "valid_found_runs": len(valid),
                           "valid_found_rate": len(valid) / len(group),
                           "false_acceptance_runs": len(false),
                           "false_acceptance_rate_among_accepted": len(false) / len(accepted) if accepted else None,
                           "missed_original_states_across_accepted_runs": sum(r["failed_original_states"] for r in accepted),
                           "recovery_underestimation_runs": sum(r["recovery_underestimated"] for r in accepted),
                           "underestimated_original_states_across_runs": sum(r["nonzero_at_reported_recovery"] for r in accepted),
                           "empirical_full_domain_runs": sum(r.get("full_domain") is True for r in group) if method == "empirical" else None,
                           "wall_seconds": stats([r["wall_seconds"] for r in group]),
                           "cpu_seconds": stats([r["cpu_seconds"] for r in group]),
                           "preprocessing_seconds": stats([r["preprocessing_seconds"] for r in group]),
                           "overrun_seconds": stats([r["overrun_seconds"] for r in group]),
                           "accepted_amplitude": stats([r["amplitude"] for r in valid]),
                           "accepted_duration": stats([r["duration"] for r in valid]),
                           "actual_total": stats([r["actual_total"] for r in valid]),
                           "deploy_total": stats([r["deploy_total"] for r in valid]),
                           "per_repeat": per_repeat})
    return output


def run(output):
    entire_run_start = time.perf_counter()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are immutable")
    source = ROOT / "results/erasure_study_20261112_v3/case_details.json"
    source_bytes = source.read_bytes()
    cases = json.loads(source_bytes)
    if len(cases) != CONFIG["case_count"]:
        raise AssertionError("Frozen source must contain all 213 cases")
    output.mkdir(parents=True)
    for key, value in CONFIG["thread_environment"].items():
        os.environ[key] = value
    source_files = [ROOT / p for p in (
        "integer_snn/fixed_protocol.py", "tests/test_fixed_protocol.py", "experiments/run_protocol_selection.py",
        "integer_snn/model.py", "integer_snn/reference.py", "integer_snn/phase_erasure.py",
        "integer_snn/certificates.py", "integer_snn/budgeted_erasure.py")]
    frozen = {"configuration": CONFIG, "configuration_sha256": digest(CONFIG),
              "frozen_before_development_utc": datetime.now(timezone.utc).isoformat(),
              "source_case_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    write_json(output / "frozen_config.json", frozen)
    write_json(output / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
               "processor": platform.processor(), "thread_environment": CONFIG["thread_environment"],
               "wall_clock": vars(time.get_clock_info("perf_counter")), "gpu_used": False, "download_bytes": 0})
    # Fixed warm-up uses the first case; it is excluded and disclosed.
    for method in METHODS:
        select_protocol(cases[0], method, .020, CONFIG["sample_seed_base"])
    dev_start = time.perf_counter()
    def dev_progress(count, total, rows):
        print(f"development {count}/{total}: {time.perf_counter()-dev_start:.2f}s; accepted {sum(r['status']=='accepted' for r in rows)}", flush=True)
    development = selection_round(cases[:32], 1, dev_progress)
    dev_seconds = time.perf_counter() - dev_start
    projected = dev_seconds * (213 / 32) * 5
    write_csv(output / "development_32.csv", development)
    write_json(output / "development_summary.json", {"case_indices": list(range(32)), "repeats": 1,
               "wall_seconds": dev_seconds, "linear_selection_projection_seconds": projected,
               "not_in_formal_aggregates": True, "algorithm_changed_after_development": False})
    if projected > 560:
        raise RuntimeError("Fixed development round projects too little audit headroom; no formal run started")
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = entire_run_start + CONFIG["formal_run_limit_seconds"]
    def progress(count, total, rows):
        print(f"formal selection {count}/{total}: {time.perf_counter()-started:.2f}s; accepted {sum(r['status']=='accepted' for r in rows)}", flush=True)
    rows = selection_round(cases, CONFIG["repetitions"], progress, deadline)
    selection_wall = time.perf_counter() - started
    write_csv(output / "selection_pre_audit.csv", rows)
    accepted_keys = sorted({(r["case_index"], r["amplitude"], r["duration"]) for r in rows if r["status"] == "accepted"})
    audits, lookup, manifests = [], {}, []
    audit_start = time.perf_counter()
    for case_index, case in enumerate(cases):
        params, groups, boxes = reconstruct_case(case)
        original = enumerate_domain(groups[0] + groups[1])
        canonical = enumerate_domain(boxes)
        if len(original) != case["initial_states"]:
            raise AssertionError("Original source state count changed")
        manifests.append({"case_index": case_index, "params": case["params"],
                          "original_states": len(original), "canonical_states": len(canonical),
                          "original_boxes_sha256": digest(groups), "canonical_boxes_sha256": digest(boxes)})
        for _, amplitude, duration in (key for key in accepted_keys if key[0] == case_index):
            if time.perf_counter() > deadline:
                raise TimeoutError("Formal deadline exceeded before complete audit")
            audited = {"case_index": case_index, "amplitude": amplitude, "duration": duration,
                       **audit_protocol(params, original, amplitude, duration)}
            lookup[(case_index, amplitude, duration)] = len(audits)
            audits.append(audited)
        if (case_index + 1) % 16 == 0 or case_index + 1 == len(cases):
            print(f"Fraction audit {case_index+1}/{len(cases)}: {len(audits)} protocols, formal elapsed {time.perf_counter()-started:.2f}s", flush=True)
    for row in rows:
        if row["status"] != "accepted":
            row.update(audit_index=None, audit_passed=None, failed_original_states=None,
                       actual_total=None, recovery_underestimated=None, nonzero_at_reported_recovery=None)
            continue
        index = lookup[(row["case_index"], row["amplitude"], row["duration"])]
        audit = audits[index]
        nonzero = audit["nonzero_original_states_by_recovery_tick"][row["reported_recovery"]]
        row.update(audit_index=index, audit_passed=audit["passed"],
                   failed_original_states=audit["failed_original_states"], actual_total=audit["actual_total"],
                   recovery_underestimated=nonzero > 0, nonzero_at_reported_recovery=nonzero)
    write_csv(output / "selection_results.csv", rows)
    write_json(output / "fraction_audits.json", audits)
    write_json(output / "case_manifest.json", manifests)
    pairs = []
    indexed = {(r["case_index"], r["budget_ms"], r["repetition"], r["method"]): r for r in rows}
    for budget_ms in CONFIG["budgets_ms"]:
        for other in ("empirical", "exact"):
            counts = Counter()
            for case_index in range(len(cases)):
                for repetition in range(CONFIG["repetitions"]):
                    certified = indexed[(case_index, budget_ms, repetition, "certified")]
                    comparator = indexed[(case_index, budget_ms, repetition, other)]
                    a, b = certified.get("audit_passed") is True, comparator.get("audit_passed") is True
                    counts["both_valid" if a and b else "certified_only" if a else "comparator_only" if b else "neither"] += 1
                    if a and b:
                        ea, eb = (certified["amplitude"], certified["duration"]), (comparator["amplitude"], comparator["duration"])
                        counts["same_E_D" if ea == eb else "certified_lexicographic_lower" if ea < eb else "comparator_lexicographic_lower"] += 1
                        counts["certified_faster" if certified["wall_seconds"] < comparator["wall_seconds"] else "comparator_faster"] += 1
            pairs.append({"budget_ms": budget_ms, "comparator": other, **counts})
    elapsed = time.perf_counter() - started
    summary = {"status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "configuration_sha256": digest(CONFIG), "scope": CONFIG["scope"],
               "formal_wall_seconds": elapsed, "formal_cpu_seconds": time.process_time() - cpu_started,
               "entire_run_wall_seconds": time.perf_counter() - entire_run_start,
               "selection_wall_seconds": selection_wall, "audit_wall_seconds": time.perf_counter() - audit_start,
               "development_wall_seconds": dev_seconds, "selection_runs": len(rows),
               "source_case_count": len(cases), "original_states_across_cases": sum(m["original_states"] for m in manifests),
               "distinct_protocols_audited": len(audits), "all_accepted_audited": True,
               "distinct_false_accepted_protocols": sum(not a["passed"] for a in audits),
               "fraction_transition_evaluations": sum(a["fraction_transition_evaluations"] for a in audits),
               "method_aggregates": aggregates(rows), "paired_comparisons": pairs,
               "measurement_limits": [
                   "wall accounting after complete calls; rejected calls can overshoot budget; no hard runtime guarantee",
                   "source JSON loading and one disclosed warm-up are common setup; per-run reconstruction and preprocessing are charged",
                   "5 real repeats use fixed random method order; timing decisions may differ across repeats",
                   "interval event counts and concrete event counts are not equivalent compute units",
                   "empirical observed maximum recovery is not a bound unless the whole finite domain was sampled",
                   "deployable common recovery is fixed at 32, so deploy completion is D+32 for all methods",
                   "different methods can accept different cases; use matched paired comparisons",
                   "213 source configurations were previously explored; this is not a new confirmatory sample",
                   "all normal inputs, noise and cross-module coupling are zero during pulse and recovery"],
               "gpu_used": False, "download_bytes": 0}
    write_json(output / "summary.json", summary)
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if time.perf_counter() > deadline or size > CONFIG["output_limit_bytes"]:
        raise AssertionError(f"Run resource bound exceeded: {elapsed}s, {size} bytes")
    print(json.dumps({"status": "completed", "output": str(output), "formal_wall_seconds": elapsed,
                      "output_bytes": size, "distinct_protocols_audited": len(audits),
                      "aggregates": [{k: a[k] for k in ("budget_ms", "method", "valid_found_runs", "selection_runs",
                                      "false_acceptance_runs", "recovery_underestimation_runs")} for a in summary["method_aggregates"]]}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/protocol_selection_20261113_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
