"""Frozen topology-amplitude validation on full-legal-state network samples.

Both row baselines have the same initial state and continuously enabled coupling.
Random samples validate implementation, not the universal mathematical theorem.
No dataset downloads; no GPU; source hashes saved before any simulation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from integer_snn.model import State, zero_state
from integer_snn.network import NetworkSpec, network_step, network_reference_step
from integer_snn.topology_certificate import MODEL, certify_topology_erasure, certify_row_sum_erasure
from experiments.run_pilot import peak_memory_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "seed": 2026111307, "simulated_sizes": [1, 2, 4, 8, 16],
    "certificate_only_sizes": [32, 64], "random_graph_sizes": [1, 2, 4, 8],
    "random_graphs_per_size": 8, "random_edge_probabilities": [0.15, 0.4],
    "samples_per_graph": 24, "structured_boundary_samples": 8,
    "negative_pulse_durations": [1, 4, 16, 64], "negative_release_horizon": 96,
    "positive_methods": ["topology_minimum", "row_lower_amplitude", "row_shorter_pulse"],
    "scope": "fixed MODEL; all-legal-state boundary and uniform scalar samples; no joint exhaustive-state claim",
    "graph_selection": "frozen structured families plus all random graphs including unsupported SCCs",
    "elapsed_limit_seconds": 240, "result_limit_bytes": 15000000,
    "runtime_interpretation": "descriptive CPU run, not isolated performance benchmark",
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def matrices(n):
    def zero():
        return [[0] * n for _ in range(n)]
    yield "empty", zero()
    for r in (1, 2, 4, 16):
        w = zero()
        for t in range(1, n):
            w[t][t-1] = r
        yield "chain_r" + str(r), w
        w = zero()
        for t in range(n):
            w[t][(t-1) % n] = r
        yield "ring_uniform_r" + str(r), w
        if n > 1:
            w[0][n-1] = 1
            yield "ring_weak1_r" + str(r), w
    for r in (1, 4, 16):
        w = zero()
        for t in range(n):
            for s in range(t):
                w[t][s] = r
        yield "dense_dag_r" + str(r), w
    # Multiple incoming edges between SCCs; each internal component is one ring.
    for r in (1, 4, 16):
        w = zero()
        groups = [list(range(i, min(i + 2, n))) for i in range(0, n, 2)]
        for gi, group in enumerate(groups):
            for pos, t in enumerate(group):
                w[t][group[(pos-1) % len(group)]] = 1 + (pos % 2)
                for prev in groups[:gi]:
                    for s in prev:
                        w[t][s] = r
        yield "cycle_scc_cascade_external" + str(r), w

def cases():
    out, seen = [], {}
    def add(label, matrix, simulate, selection):
        w = tuple(tuple(row) for row in matrix)
        key = digest(w)
        if key in seen:
            out[seen[key]]["aliases"].append(label)
            return
        seen[key] = len(out)
        out.append({"id": len(out), "label": label, "aliases": [], "weights": w,
                    "simulate": simulate, "selection": selection})
    for n in CONFIG["simulated_sizes"] + CONFIG["certificate_only_sizes"]:
        for name, matrix in matrices(n):
            add(str(n) + "_" + name, matrix, n in CONFIG["simulated_sizes"], "structured")
    rng = random.Random(CONFIG["seed"])
    for n in CONFIG["random_graph_sizes"]:
        for k in range(CONFIG["random_graphs_per_size"]):
            p = CONFIG["random_edge_probabilities"][k % 2]
            w = [[rng.randint(1, 4) if rng.random() < p else 0 for _ in range(n)] for _ in range(n)]
            add(f"{n}_random{k}_p{p}", w, True, "random")
    add("multiinput_periodic_counterexample", ((0, 0, 1), (0, 0, 1), (2, 2, 0)), True, "known_negative")
    return out

def initials(n, seed):
    rng = random.Random(seed)
    starts = []
    for sample in range(CONFIG["samples_per_graph"]):
        modules = []
        for m in range(n):
            cells = []
            for cell in (0, 1):
                if sample < 8:
                    spike = ((sample & 3) >> cell) & 1
                    if sample >= 4:
                        spike ^= (m & 1)
                    voltage = 0 if spike else (13 if sample % 2 == 0 else -15)
                    current = 15 if sample < 6 else (15 if (m+cell) % 2 else 0)
                else:
                    index = rng.randrange(480)
                    spike = int(index >= 464)
                    voltage = 0 if spike else index // 16 - 15
                    current = index % 16
                cells.append((voltage, current, spike))
            modules.append(State(tuple(x[0] for x in cells), tuple(x[1] for x in cells),
                                 tuple(x[2] for x in cells)))
        starts.append(tuple(modules))
    return starts

def audit_positive(spec, cert, starts, deadline):
    updates = fraction = 0
    latest_spike = latest_zero = 0
    first_zeros = []
    zero = (zero_state(),) * spec.module_count
    for sample, initial in enumerate(starts):
        state, first_zero = initial, None
        for tick in range(1, cert["total_steps"] + 1):
            erase = cert["amplitude"] if tick <= cert["duration"] else 0
            nxt = network_step(spec, state, erase=erase)
            if sample < 2 or spec.module_count <= 2:
                if nxt != network_reference_step(spec, state, erase=erase):
                    raise AssertionError(("Fraction mismatch", sample, tick))
                fraction += 1
            state = nxt
            updates += 1
            if any(any(s.spike) for s in state):
                latest_spike = max(latest_spike, tick)
                if tick > cert["last_spike_upper"]:
                    raise AssertionError(("spike exceeds theorem", sample, tick, cert))
            if tick >= cert["duration"] and state == zero and first_zero is None:
                first_zero = tick
        if state != zero:
            raise AssertionError(("false positive", sample, cert, state))
        first_zeros.append(first_zero)
        latest_zero = max(latest_zero, first_zero)
        if time.perf_counter() > deadline:
            raise TimeoutError("topology experiment exceeded fixed elapsed limit")
    return {"passed": True, "initial_states": len(starts), "network_updates": updates,
            "fraction_network_updates": fraction, "latest_observed_spike": latest_spike,
            "latest_observed_zero_after_release": latest_zero,
            "zero_times_after_release": first_zeros}

def audit_negative(spec, amplitude, deadline):
    initial = (zero_state(),) * spec.module_count
    fraction = 0
    for _ in range(5):
        initial = network_reference_step(spec, initial, inputs=((14, 0),) * spec.module_count)
        fraction += 1
    if not all(s.current[0] == 15 and s.spike[0] == 1 and s.current[1] == s.spike[1] == 0 for s in initial):
        raise AssertionError("positive write did not reach saturated active cells")
    rows = []
    for duration in CONFIG["negative_pulse_durations"]:
        state = initial
        for _ in range(duration):
            state = network_reference_step(spec, state, erase=amplitude)
            fraction += 1
        seen, period, zero_hit = {}, None, False
        for tick in range(CONFIG["negative_release_horizon"] + 1):
            if state == (zero_state(),) * spec.module_count:
                zero_hit = True
                break
            if state in seen:
                period = tick - seen[state]
                break
            seen[state] = tick
            if tick < CONFIG["negative_release_horizon"]:
                state = network_reference_step(spec, state)
                fraction += 1
        if zero_hit:
            raise AssertionError(("necessary amplitude disproved", amplitude, duration))
        rows.append({"duration": duration, "zero_hit": zero_hit,
                     "nonzero_release_cycle_period": period,
                     "claim_scope": "finite sample; all-duration impossibility comes from theorem"})
        if time.perf_counter() > deadline:
            raise TimeoutError("negative audit exceeded elapsed limit")
    return {"passed": True, "amplitude": amplitude, "positive_write": [14, 5],
            "protocols": rows, "fraction_network_updates": fraction}

def counterexample_audit():
    spec = NetworkSpec(MODEL, ((0, 0, 1), (0, 0, 1), (2, 2, 0)))
    state = (zero_state(),) * 3
    write_trace = []
    for tick in range(1, 10):
        inputs = ((14 if tick in (1, 2) else 0, 0),
                  (14 if tick in (2, 3) else 0, 0),
                  (14 if tick <= 5 else 0, 0))
        state = network_reference_step(spec, state, inputs=inputs)
        write_trace.append([asdict(s) for s in state])
    written = state
    a = State((0, -14), (9, 0), (1, 0))
    b = State((10, -15), (12, 0), (0, 0))
    c = State((0, -15), (15, 0), (1, 0))
    for _ in range(5):
        state = network_reference_step(spec, state, erase=3)
    if state not in ((a, b, c), (b, a, c)):
        raise AssertionError(("reachable periodic example mismatch", state))
    if network_reference_step(spec, network_reference_step(spec, state, erase=3), erase=3) != state:
        raise AssertionError("two-cycle failed")
    rows = []
    for duration in (0, 1, 2, 3, 4, 8, 16, 32):
        state = written
        for _ in range(duration):
            state = network_reference_step(spec, state, erase=3)
        seen = {}
        for tick in range(129):
            if state == (zero_state(),) * 3:
                raise AssertionError("known nonerasable example erased")
            if state in seen:
                rows.append({"duration": duration, "release_cycle_period": tick - seen[state]})
                break
            seen[state] = tick
            state = network_reference_step(spec, state)
        else:
            raise AssertionError("release example not periodic by fixed limit")
    return {"weights": spec.weights, "checker_status": certify_topology_erasure(spec, 3)["status"],
            "write_trace": write_trace, "pulse_cycle": [[asdict(s) for s in v] for v in ((a, b, c), (b, a, c))],
            "release_checks": rows, "passed": True,
            "interpretation": "empty threshold-2 static core and weak edges on every cycle do not suffice with multiinput SCCs"}

def run(output):
    started, cpu_started = time.perf_counter(), time.process_time()
    deadline = started + CONFIG["elapsed_limit_seconds"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = ["integer_snn/model.py", "integer_snn/reference.py", "integer_snn/network.py",
                    "integer_snn/topology_certificate.py", "experiments/run_topology_certificate.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    graph_cases = cases()
    save(output / "frozen_config.json", {"config": CONFIG, "params": asdict(MODEL),
         "created_utc": datetime.now(timezone.utc).isoformat(), "sources_sha256": hashes,
         "graph_count": len(graph_cases), "graph_selection_sha256": digest(graph_cases)})
    rows = []
    for case in graph_cases:
        spec = NetworkSpec(MODEL, case["weights"])
        topology = certify_topology_erasure(spec)
        row = {**case, "modules": spec.module_count, "row_sum": spec.max_row_sum,
               "topology_status": topology["status"], "topology_certificate": topology, "protocols": []}
        certs = [("row_lower_amplitude", certify_row_sum_erasure(spec)),
                 ("row_shorter_pulse", certify_row_sum_erasure(spec, shorter_pulse=True))]
        if topology["certified"]:
            certs.insert(0, ("topology_minimum", topology))
        starts = initials(spec.module_count, CONFIG["seed"] + case["id"]) if case["simulate"] else None
        for name, cert in certs:
            protocol = {"method": name, "amplitude": cert["amplitude"], "duration": cert["duration"],
                        "recovery": cert["recovery"], "total_steps": cert["total_steps"],
                        "amplitude_times_pulse_duration": cert["amplitude"] * cert["duration"],
                        "product_interpretation": "control magnitude-time proxy, not measured energy"}
            if starts is not None:
                protocol["audit"] = audit_positive(spec, cert, starts, deadline)
            row["protocols"].append(protocol)
        if starts is not None and topology["certified"]:
            row["below_threshold_audit"] = audit_negative(spec, topology["minimum_amplitude"] - 1, deadline)
        rows.append(row)
        if (case["id"] + 1) % 16 == 0:
            print(json.dumps({"graphs_completed": case["id"] + 1, "elapsed": time.perf_counter() - started}), flush=True)
    witness = counterexample_audit()
    positive = [p for r in rows for p in r["protocols"] if "audit" in p]
    negative = [r["below_threshold_audit"] for r in rows if "below_threshold_audit" in r]
    supported = [r for r in rows if r["topology_status"] == "certified"]
    tradeoffs = Counter()
    for r in supported:
        top, low, short = r["protocols"]
        tradeoffs["lower_amplitude_than_strong_row"] += top["amplitude"] < low["amplitude"]
        tradeoffs["equal_amplitude_to_strong_row"] += top["amplitude"] == low["amplitude"]
        tradeoffs["longer_constructive_total_than_strong_row"] += top["total_steps"] > low["total_steps"]
        tradeoffs["lower_magnitude_time_product_than_strong_row"] += top["amplitude_times_pulse_duration"] < low["amplitude_times_pulse_duration"]
    scientific = {"config": CONFIG, "rows": rows, "multiinput_counterexample": witness}
    summary = {
        "passed": True, "total_graphs": len(rows), "supported_graphs": len(supported),
        "unknown_graphs": len(rows) - len(supported),
        "simulated_graphs": sum(r["simulate"] for r in rows),
        "certificate_only_graphs": sum(not r["simulate"] for r in rows),
        "positive_protocols": len(positive), "positive_network_trajectories": sum(p["audit"]["initial_states"] for p in positive),
        "positive_network_updates": sum(p["audit"]["network_updates"] for p in positive),
        "positive_fraction_network_updates": sum(p["audit"]["fraction_network_updates"] for p in positive),
        "negative_graphs": len(negative), "negative_protocols": sum(len(r["protocols"]) for r in negative),
        "negative_fraction_network_updates": sum(r["fraction_network_updates"] for r in negative),
        "negative_cycles_found": sum(p["nonzero_release_cycle_period"] is not None for r in negative for p in r["protocols"]),
        "tradeoffs_all_supported_including_certificate_only": dict(tradeoffs),
        "scientific_sha256": digest(scientific), "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started, "peak_memory_bytes": peak_memory_bytes(),
        "python": platform.python_version(), "sources_sha256": hashes,
        "scope": CONFIG["scope"], "failures": 0,
        "limitations": ["fixed model only", "samples do not prove universal claims",
                       "duration is a conservative bound, not minimum time",
                       "amplitude-times-duration is not hardware energy",
                       "topology unknown is not infeasible"],
    }
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise AssertionError("frozen sources changed during run")
    save(output / "case_results.json", rows)
    save(output / "multiinput_counterexample.json", witness)
    save(output / "summary.json", summary)
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size > CONFIG["result_limit_bytes"]:
        raise AssertionError("result output exceeds fixed byte limit")
    save(output / "validation.json", {"source_hashes_match": True, "passed": True, "bytes_before_validation": size,
         "scientific_sha256": summary["scientific_sha256"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results/topology_certificate_20261113_v1"))
    args = parser.parse_args()
    run(args.output)

