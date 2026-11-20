"""Reproducible, bounded, standard-library-only integer SNN pilot.

Run from the project root: py -3.12 -m experiments.run_pilot
All reported certificates are conditional on the implemented integer semantics.
No gradient training, GPU work, or general-network T3 claim is made here.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from integer_snn.model import Params, State, iter_legal_states, step, zero_state
from integer_snn.reference import reference_step
from integer_snn.certificates import certify_retention, erasure_certificate
from integer_snn.reachability import reachable_closure, verify_coupled_product

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def peak_memory_bytes():
    if sys.platform != "win32":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("peak", "working", "quota_peak_paged", "quota_paged",
             "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if get_info(get_process(), ctypes.byref(counters), counters.cb):
        return int(counters.peak)
    return None


def environment():
    gpu = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            gpu = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"python": sys.version, "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(), "detected_gpu": gpu,
            "gpu_used": False, "external_dependencies": [],
            "worker_processes": 1, "pid": os.getpid()}


class Budget:
    def __init__(self, seconds):
        self.start = time.perf_counter()
        self.seconds = seconds

    def check(self):
        if time.perf_counter() - self.start > self.seconds:
            raise TimeoutError(f"Pilot exceeded its {self.seconds}s CPU-run wall-time budget.")


def find_cycle(params, initial, max_steps=192):
    seen, path = {}, []
    state = initial
    for _ in range(max_steps):
        if state in seen:
            start = seen[state]
            return path[:start], path[start:]
        seen[state] = len(path)
        path.append(state)
        state = step(params, state)
    return path, []


def write_symbol(params, symbol, duration, amplitude=None):
    state = zero_state()
    inputs = [0, 0]
    inputs[symbol] = params.threshold if amplitude is None else amplitude
    for _ in range(duration):
        state = step(params, state, inputs=tuple(inputs))
    return state


def cycle_label(cycle):
    active = [any(state.spike[i] for state in cycle) for i in range(2)]
    return ("both" if all(active) else "symbol_0" if active[0]
            else "symbol_1" if active[1] else "silent")


def one_spike_period(cycle, active=0):
    if not cycle or cycle_label(cycle) != f"symbol_{active}":
        return None
    indices = [i for i, state in enumerate(cycle) if state.spike[active]]
    if not indices:
        return None
    gaps = [(indices[(k + 1) % len(indices)] - index) % len(cycle) or len(cycle)
            for k, index in enumerate(indices)]
    return gaps[0] if len(set(gaps)) == 1 else None


def reference_audit(budget):
    p = Params(bits_v=3, bits_i=2, alpha_num=3, alpha_shift=2,
               beta_num=3, beta_shift=2, weight=2, inhibition=2, threshold=3)
    controls = [((0, 0), 0, (0, 0)), ((3, 0), 0, (0, 0)),
                ((0, 3), 0, (0, 0)), ((0, 0), 6, (0, 0)),
                ((0, 0), 0, (-1, 1)), ((-5, 5), 2, (1, -1))]
    comparisons, state_count = 0, 0
    for state in iter_legal_states(p):
        state_count += 1
        for inputs, erase, noise in controls:
            actual = step(p, state, inputs=inputs, erase=erase, noise=noise)
            expected = reference_step(p, state, inputs=inputs, erase=erase, noise=noise)
            if actual != expected:
                raise AssertionError(("integer_reference_mismatch", state, actual, expected))
            comparisons += 1
        if state_count % 256 == 0:
            budget.check()
    return {"params": asdict(p), "legal_states": state_count,
            "control_cases": len(controls), "comparisons": comparisons, "mismatches": 0,
            "scope": "all legal states of this small configuration, six declared controls"}


def full_state_graph(budget):
    p = Params(bits_v=3, bits_i=3, alpha_num=3, alpha_shift=2,
               beta_num=3, beta_shift=2, weight=2, inhibition=2, threshold=3)
    states = list(iter_legal_states(p))
    transitions = {state: step(p, state) for state in states}
    assigned, attractors = {}, []
    for source in states:
        if source in assigned:
            continue
        local, path, node = {}, [], source
        while node not in assigned and node not in local:
            local[node] = len(path)
            path.append(node)
            node = transitions[node]
        if node in assigned:
            attractor = assigned[node]
        else:
            cycle = path[local[node]:]
            attractor = len(attractors)
            attractors.append({"period": len(cycle), "label": cycle_label(cycle),
                               "cycle": [asdict(s) for s in cycle]})
        for state in path:
            assigned[state] = attractor
        budget.check()
    counts = Counter(assigned.values())
    for index, attractor in enumerate(attractors):
        attractor["basin_states"] = counts[index]
    return {"params": asdict(p), "logical_state_bits": p.state_bits,
            "legal_states": len(states), "transitions_checked": len(transitions),
            "attractor_count": len(attractors), "attractors": attractors,
            "scope": "all legal post-update encodings; not all states are reachable by the write protocol"}


def parameter_sweep(config, budget):
    rows = []
    combinations = itertools.product(
        config["sweep_bits"], config["leaks"], config["current_leaks"],
        config["thresholds"], config["weights"], config["inhibitions"])
    for bits, alpha, beta, threshold, weight, inhibition in combinations:
        if threshold > 2 ** (bits - 1) - 1:
            continue
        p = Params(bits_v=bits, bits_i=bits, alpha_num=alpha[0], alpha_shift=alpha[1],
                   beta_num=beta[0], beta_shift=beta[1], weight=weight,
                   inhibition=inhibition, threshold=threshold)
        row = {**asdict(p), "write_steps": 0, "cycle_period": 0,
               "spike_interval": 0, "label": "not_found", "cert_delta0": False,
               "cert_delta1": False, "joint_within_budget": False}
        for duration in config["write_durations"]:
            initial = write_symbol(p, 0, duration)
            transient, cycle = find_cycle(p, initial, config["cycle_search_steps"])
            period = one_spike_period(cycle)
            row.update(write_steps=duration, cycle_period=len(cycle),
                       spike_interval=period or 0, label=cycle_label(cycle) if cycle else "unknown")
            if period is None or period < 2:
                continue
            currents = [s.current[0] for s in cycle if s.spike[0]]
            interval = (min(currents), max(currents))
            certs = {delta: certify_retention(p, period, interval, noise_bound=delta)
                     for delta in config["noise_bounds"]}
            row["cert_delta0"] = certs[0]["certified"]
            row["cert_delta1"] = certs[1]["certified"]
            row.update(current_lo=interval[0], current_hi=interval[1],
                       transient_steps=len(transient))
            if certs[0]["certified"]:
                erase = erasure_certificate(p, certs[0]["imax"])
                row.update(erase_amplitude=erase["amplitude"], erase_duration=erase["duration"],
                           erase_recovery=erase["recovery"])
                row["joint_within_budget"] = (
                    erase["amplitude"] <= config["erase_amplitude_budget"]
                    and erase["duration"] <= config["erase_duration_budget"]
                    and erase["total_steps"] <= config["erase_total_budget"])
            break
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"Sweep: {len(rows)} configurations", flush=True)
            budget.check()
    summary = {"configurations": len(rows),
               "sparse_periodic": sum(row["spike_interval"] >= 2 for row in rows),
               "certified_delta0": sum(row["cert_delta0"] for row in rows),
               "certified_delta1": sum(row["cert_delta1"] for row in rows),
               "joint_within_budget": sum(row["joint_within_budget"] for row in rows),
               "joint_budget_noise_bound": 0,
               "classification_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
               "scope": "finite exploratory grid; first sparse write protocol per configuration; no population inference"}
    return rows, summary


def phase_box_states(cert):
    """Enumerate the T1 phase-box overapproximation, preserving certified spikes."""
    states = set()
    for phase in cert["phases"]:
        values = [range(pair[0], pair[1] + 1)
                  for pair in phase["voltage_post"] + phase["current"]]
        for v0, v1, i0, i1 in itertools.product(*values):
            states.add(State(v=(v0, v1), current=(i0, i1), spike=tuple(phase["expected_spikes"])))
    return states


def check_erasure_states(params, states, erase, budget):
    failed, max_first_zero, checks = 0, 0, 0
    for initial in sorted(states, key=lambda s: (s.v, s.current, s.spike)):
        state, first_zero = initial, None
        for t in range(1, erase["total_steps"] + 1):
            state = step(params, state, erase=erase["amplitude"] if t <= erase["duration"] else 0)
            if t >= erase["duration"] and state == zero_state() and first_zero is None:
                first_zero = t
        failed += state != zero_state()
        max_first_zero = max(max_first_zero, first_zero or erase["total_steps"] + 1)
        checks += 1
        if checks % 256 == 0:
            budget.check()
    if failed:
        raise AssertionError(("erasure_certificate_counterexample", failed))
    return {"initial_states": checks, "failures": failed,
            "latest_zero_step_from_pulse_start": max_first_zero,
            "latest_recovery_steps": max_first_zero - erase["duration"],
            "scope": "all supplied initial states; caller specifies the set; zero external noise during erase/recovery"}


def robust_phase_audit(params, cert, budget):
    """All local states x all integer disturbances; checks inclusion in next phase box."""
    phases = cert["phases"]
    delta, checks = cert["noise_bound"], 0
    violations = []
    for index, phase in enumerate(phases):
        # phases include times 0..p; skip last, whose return is enclosed by time 0.
        if index + 1 >= len(phases):
            break
        target = phases[index + 1]
        values = [range(pair[0], pair[1] + 1)
                  for pair in phase["voltage_post"] + phase["current"]]
        for v0, v1, i0, i1 in itertools.product(*values):
            state = State(v=(v0, v1), current=(i0, i1), spike=tuple(phase["expected_spikes"]))
            for noise in itertools.product(range(-delta, delta + 1), repeat=2):
                actual = reference_step(params, state, noise=noise)
                inside = all(lo <= x <= hi for x, (lo, hi) in
                             zip(actual.v + actual.current,
                                 target["voltage_post"] + target["current"]))
                inside = inside and actual.spike == tuple(target["expected_spikes"])
                checks += 1
                if not inside:
                    violations.append({"state": asdict(state), "noise": noise,
                                       "actual": asdict(actual), "target": target})
            if checks % 512 == 0:
                budget.check()
    if violations:
        raise AssertionError(("robust_phase_audit", violations[0]))
    return {"one_step_disturbance_checks": checks, "violations": 0,
            "scope": "all integer states in each declared box, every bounded disturbance pair"}


def trace_row(stage, t, state, inputs=(0, 0), erase=0):
    return {"stage": stage, "t": t, "v0": state.v[0], "v1": state.v[1],
            "i0": state.current[0], "i1": state.current[1],
            "s0": state.spike[0], "s1": state.spike[1],
            "u0": inputs[0], "u1": inputs[1], "erase": erase}


def witness_experiment(config, budget):
    p = Params(**config["preferred_witness"])
    # An invariant current interval is broader than one exact orbit.
    cert = certify_retention(p, 2, (7, 9), noise_bound=1)
    if not cert["certified"]:
        raise AssertionError(("preferred_witness_not_certified", cert))
    opposite = certify_retention(p, 2, (7, 9), noise_bound=1, active=1)
    if not opposite["certified"]:
        raise AssertionError("Symmetric memory symbol must also be certified.")
    erase = erasure_certificate(p, cert["imax"])
    states = phase_box_states(cert) | phase_box_states(opposite)
    audit = robust_phase_audit(p, cert, budget)
    erase_audit = check_erasure_states(p, states, erase, budget)
    write_checks = []
    for symbol in range(2):
        initial = write_symbol(p, symbol, 2)
        transient, cycle = find_cycle(p, initial)
        relevant = cert if symbol == 0 else opposite
        section_hits = [s for s in transient + cycle
                        if s.spike[symbol] and 7 <= s.current[symbol] <= 9
                        and s.current[1-symbol] == 0 and s.spike[1-symbol] == 0
                        and -p.qv <= s.v[1-symbol] <= p.threshold-1]
        if not section_hits or cycle_label(cycle) != f"symbol_{symbol}":
            raise AssertionError("External write did not reach the certified memory section.")
        write_checks.append({"symbol": symbol, "write_steps": 2, "write_amplitude": p.threshold,
                             "transient_steps": len(transient), "cycle_period": len(cycle),
                             "cycle": [asdict(s) for s in cycle],
                             "section_reached": asdict(section_hits[0])})
    # Trace a concrete write/hold/erase/rewrite sequence; no internal state reset.
    state, trace, t = zero_state(), [], 0
    schedule = [("write_0", 2, (p.threshold, 0), 0),
                ("hold_0", 20, (0, 0), 0),
                ("erase", erase["duration"], (0, 0), erase["amplitude"]),
                ("recover", erase["recovery"], (0, 0), 0),
                ("write_1", 2, (0, p.threshold), 0),
                ("hold_1", 20, (0, 0), 0)]
    trace.append(trace_row("initial", t, state))
    for stage, duration, inputs, amplitude in schedule:
        for _ in range(duration):
            t += 1
            state = step(p, state, inputs=inputs, erase=amplitude)
            trace.append(trace_row(stage, t, state, inputs, amplitude))
        if stage == "recover" and state != zero_state():
            raise AssertionError("Cycle reuse must erase to zero before writing the second symbol.")
    return {"params": asdict(p), "retention": cert, "opposite_retention": opposite,
            "erasure": erase, "write_reachability": write_checks,
            "robust_audit": audit, "erasure_audit": erase_audit,
            "state_bits": p.state_bits}, trace


def network_step(params, states, coupling, inputs=None, erase=0):
    """Two modules: matching branches receive c * other-module OLD spike."""
    inputs = inputs or [(0, 0), (0, 0)]
    return tuple(step(params, state,
                      inputs=tuple(inputs[a][i] + coupling * states[1-a].spike[i]
                                   for i in range(2)), erase=erase)
                 for a, state in enumerate(states))


def network_query(params, states, coupling, window=2):
    counts = [[0, 0], [0, 0]]
    for _ in range(window):
        states = network_step(params, states, coupling)
        for a in range(2):
            for i in range(2):
                counts[a][i] += states[a].spike[i]
    answers = tuple(0 if c[0] > c[1] else 1 if c[1] > c[0] else None for c in counts)
    return states, answers


def network_experiment(config, witness, budget):
    p = Params(**witness["params"])
    rows = []
    for coupling in (0, 1, 2, 4):
        # Bound +c holds per cell regardless of other module phase.
        erase = erasure_certificate(p, witness["retention"]["imax"],
                                    positive_input_bound=coupling)
        for symbols in itertools.product(range(2), repeat=2):
            for delay in config["hold_delays"]:
                states = (zero_state(), zero_state())
                inputs = [tuple(p.threshold if i == symbol else 0 for i in range(2))
                          for symbol in symbols]
                for _ in range(2):
                    states = network_step(p, states, coupling, inputs)
                # Fixed warm-up, known in advance, does not inspect the target to stop.
                for _ in range(8):
                    states = network_step(p, states, coupling)
                entry_checks = []
                for a, symbol in enumerate(symbols):
                    cert = witness["retention"] if symbol == 0 else witness["opposite_retention"]
                    entry_checks.append(any(
                        states[a].spike == tuple(phase["expected_spikes"])
                        and all(lo <= x <= hi for x, (lo, hi) in
                                zip(states[a].v + states[a].current,
                                    phase["voltage_post"] + phase["current"]))
                        for phase in cert["phases"]))
                if coupling <= 1 and not all(entry_checks):
                    raise AssertionError("Write/warm-up did not enter the composed retention contract.")
                for _ in range(delay):
                    states = network_step(p, states, coupling)
                states, answers = network_query(p, states, coupling)
                current_bound_valid = all(max(s.current) <= witness["retention"]["imax"] for s in states)
                for t in range(erase["total_steps"]):
                    states = network_step(p, states, coupling,
                                          erase=erase["amplitude"] if t < erase["duration"] else 0)
                erased = all(s == zero_state() for s in states)
                next_symbols = tuple(1-symbol for symbol in symbols)
                inputs = [tuple(p.threshold if i == symbol else 0 for i in range(2))
                          for symbol in next_symbols]
                for _ in range(2):
                    states = network_step(p, states, coupling, inputs)
                for _ in range(8 + delay):
                    states = network_step(p, states, coupling)
                _, next_answers = network_query(p, states, coupling)
                rows.append({"coupling": coupling, "symbols": "".join(map(str, symbols)),
                             "delay": delay, "retained": answers == symbols,
                             "erased": erased, "rewritten": next_answers == next_symbols,
                             "all_joint": answers == symbols and erased and next_answers == next_symbols,
                             "entered_retention_contract": all(entry_checks),
                             "erase_start_current_bound_valid": current_bound_valid,
                             "local_robust_bound_covers_coupling": coupling <= 1,
                             "external_noise_during_network_trial": 0,
                             "erase_amplitude": erase["amplitude"],
                             "erase_duration": erase["duration"]})
                budget.check()
    summary = []
    for coupling in (0, 1, 2, 4):
        subset = [r for r in rows if r["coupling"] == coupling]
        summary.append({"coupling": coupling, "trials": len(subset),
                        "retention_successes": sum(r["retained"] for r in subset),
                        "erasure_successes": sum(r["erased"] for r in subset),
                        "rewrite_successes": sum(r["rewritten"] for r in subset),
                        "joint_successes": sum(r["all_joint"] for r in subset),
                        "current_bound_violations": sum(not r["erase_start_current_bound_valid"] for r in subset),
                        "scope": "uniform bounded-input composition covers c<=1 only; c>1 is empirical stress"})
    if any(not r["all_joint"] for r in rows if r["coupling"] <= 1):
        raise AssertionError("Certified-range network trial failed; inspect write and phase assumptions.")
    return {"per_coupling": summary,
            "scope": "two explicit modules, all four contents, declared delays, 2-tick delayed symbol readout; not a full future-query benchmark or general T3"}, rows


def erasure_grid(witness, budget):
    p = Params(**witness["params"])
    cycles = [State(**{k: tuple(v) for k, v in item.items()})
              for check in witness["write_reachability"] for item in check["cycle"]]
    rows = []
    for amplitude in range(0, witness["erasure"]["amplitude"] + 1):
        for duration in (1, 2, 4, 8, 16):
            successes = 0
            for initial in cycles:
                state = initial
                for _ in range(duration):
                    state = step(p, state, erase=amplitude)
                for _ in range(32):
                    state = step(p, state)
                successes += state == zero_state()
            rows.append({"amplitude": amplitude, "duration": duration,
                         "recovery_observed": 32, "initial_cycle_states": len(cycles),
                         "exact_zero_successes": successes,
                         "all_cycle_phases_erased": successes == len(cycles)})
        budget.check()
    # Failure to certify or finite-time failure does not prove global impossibility.
    best = [r for r in rows if r["all_cycle_phases_erased"]]
    summary = {"tested_protocols": len(rows), "all_phase_success_protocols": len(best),
               "smallest_successful_tested_amplitude": min(r["amplitude"] for r in best) if best else None,
               "zero_amplitude_success_protocols": sum(r["all_cycle_phases_erased"] for r in rows if r["amplitude"] == 0),
               "scope": "exact nominal cycle phases, fixed tested durations and 32 recovery ticks; not all robust states"}
    return summary, rows


def exact_reachability_experiment(witness, budget):
    """Finite computational witnesses; keep their scope separate from T1/T2."""
    p = Params(**witness["params"])
    states_by_symbol, closure_rows, checked_edges = {}, [], 0
    for symbol in (0, 1):
        state = write_symbol(p, symbol, 2)
        for _ in range(6):
            state = step(p, state)
        result = reachable_closure(p, [state], noise_bound=1, max_states=4096)
        if not result["closed"]:
            raise AssertionError("Exact local reachable closure was not completed.")
        states = result["states"]
        for source in states:
            for noise in itertools.product((-1, 0, 1), repeat=2):
                actual = step(p, source, noise=noise)
                independent = reference_step(p, source, noise=noise)
                if actual != independent or actual not in states:
                    raise AssertionError("Independent exact-closure check failed.")
                checked_edges += 1
            if source.current[1-symbol] != 0 or source.spike[1-symbol] != 0:
                raise AssertionError("Inactive symbol became active.")
            if source.spike[symbol] == 1:
                valid = source.current[symbol] == 9 and source.v[symbol] == 0
            else:
                valid = source.current[symbol] == 12 and 11 <= source.v[symbol] <= 13
            if not valid:
                raise AssertionError("Expected two-phase event pattern was not invariant.")
        states_by_symbol[symbol] = states
        closure_rows.append({"symbol": symbol, "states": len(states),
                             "transition_checks": result["transition_checks"],
                             "closed": True,
                             "initial_state_after_write_and_warmup": asdict(state)})
        budget.check()
    union = states_by_symbol[0] | states_by_symbol[1]
    tighter = {"amplitude": 3, "duration": 10, "recovery": 16, "total_steps": 26}
    # Empirical finite-state exact check, not the generic T2 sufficient bound.
    tighter_check = check_erasure_states(p, union, tighter, budget)
    # Compare on exactly the SAME 526 phase-box states as the conservative T2 audit.
    broad_states = phase_box_states(witness["retention"]) | phase_box_states(witness["opposite_retention"])
    broad_tighter_check = check_erasure_states(p, broad_states, tighter, budget)
    failed_state = sorted(states_by_symbol[0], key=lambda s: (s.v, s.current, s.spike))[0]
    after = failed_state
    for _ in range(32):
        after = step(p, after, erase=2)
    transient, remaining_cycle = find_cycle(p, after)
    if not remaining_cycle or cycle_label(remaining_cycle) != "symbol_0":
        raise AssertionError("The declared low-amplitude erase counterexample did not persist.")
    product = verify_coupled_product(p, states_by_symbol, coupling=1,
                                     erase_amplitude=5, erase_duration=16,
                                     recovery_steps=16)
    if product["closure_failures"] or product["erase_failures"]:
        raise AssertionError("Exact coupled product verification failed.")
    short_write = write_symbol(p, 0, 1)
    _, short_cycle = find_cycle(p, short_write)
    return {"local_closures": closure_rows,
            "independent_reference_edge_checks": checked_edges,
            "nominal_write_steps": 2, "nominal_warmup_steps": 6,
            "noise_during_write_warmup": 0, "noise_during_hold": 1,
            "tighter_erasure_protocol": tighter, "tighter_erasure_check": tighter_check,
            "same_phase_boxes_tighter_erasure_check": broad_tighter_check,
            "failed_erasure": {"amplitude": 2, "duration": 32, "initial": asdict(failed_state),
                               "cycle_after_release": [asdict(s) for s in remaining_cycle],
                               "transient_after_release": len(transient)},
            "one_tick_write_final_cycle": [asdict(s) for s in short_cycle],
            "coupled_product": product,
            "scope": "finite exact reachable sets and their fixed coupled product; not a general tight-erasure theorem"}


def run(config, output, budget):
    scientific = {}
    for name, fn in (("reference_audit", lambda: reference_audit(budget)),
                     ("full_state_graph", lambda: full_state_graph(budget))):
        print(f"Running {name}", flush=True)
        scientific[name] = fn()
    print("Running bounded parameter sweep", flush=True)
    rows, scientific["parameter_sweep"] = parameter_sweep(config, budget)
    save_csv(output / "parameter_sweep.csv", rows)
    scientific["parameter_sweep_rows_sha256"] = canonical_hash(rows)
    print("Checking a sparse-period robust witness", flush=True)
    witness, trace = witness_experiment(config, budget)
    scientific["witness"] = witness
    save_csv(output / "witness_trace.csv", trace)
    print("Testing nonzero two-module coupling and erase/reuse", flush=True)
    scientific["network"], rows = network_experiment(config, witness, budget)
    save_csv(output / "network_trials.csv", rows)
    scientific["network_rows_sha256"] = canonical_hash(rows)
    scientific["erasure_grid"], rows = erasure_grid(witness, budget)
    save_csv(output / "erasure_grid.csv", rows)
    scientific["erasure_grid_rows_sha256"] = canonical_hash(rows)
    print("Checking exact noisy reachable sets and the coupled Cartesian product", flush=True)
    scientific["exact_reachability"] = exact_reachability_experiment(witness, budget)
    return scientific


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot")
    parser.add_argument("--compare-to", type=Path, help="Previous summary.json for exact deterministic comparison")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Seed recorded for future randomized extensions; this pilot is exhaustive/deterministic.
    random.seed(config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Use a fresh --output directory to preserve prior experiment evidence.")
    save_json(args.output / "config.json", config)
    env = environment()
    save_json(args.output / "environment.json", env)
    budget, cpu_start = Budget(config["hard_timeout_seconds"]), time.process_time()
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
               for directory in ("integer_snn", "experiments")
               for p in sorted((ROOT / directory).rglob("*.py"))}
    save_json(args.output / "source_hashes.json", sources)
    try:
        science = run(config, args.output, budget)
    except Exception as error:
        save_json(args.output / "failure.json",
                  {"type": type(error).__name__, "message": str(error),
                   "wall_seconds": time.perf_counter() - budget.start})
        raise
    result = {"schema_version": 1, "status": "completed",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "command": subprocess.list2cmdline([sys.executable, "-m", "experiments.run_pilot", *sys.argv[1:]]),
              "wall_seconds": time.perf_counter() - budget.start,
              "cpu_seconds": time.process_time() - cpu_start,
              "peak_process_working_set_bytes": peak_memory_bytes(),
              "gpu_used": False, "config_sha256": canonical_hash(config),
              "scientific_sha256": canonical_hash(science),
              "scientific_results": science}
    save_json(args.output / "summary.json", result)
    if args.compare_to:
        previous = json.loads(args.compare_to.read_text(encoding="utf-8"))
        comparison = {"baseline": str(args.compare_to.resolve()),
                      "same_config": previous["config_sha256"] == result["config_sha256"],
                      "same_scientific_results": previous["scientific_sha256"] == result["scientific_sha256"],
                      "excluded_nondeterministic_fields": ["timestamp", "timing", "memory", "pid"],
                      "current_scientific_sha256": result["scientific_sha256"]}
        save_json(args.output / "reproducibility.json", comparison)
        if not comparison["same_config"] or not comparison["same_scientific_results"]:
            raise AssertionError("Deterministic reproducibility check failed.")
    print(json.dumps({"status": "completed", "wall_seconds": result["wall_seconds"],
                      "peak_mb": (result["peak_process_working_set_bytes"] or 0) / 1024**2,
                      "sweep": science["parameter_sweep"], "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
