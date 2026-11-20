"""Finite CPU validation of a row-sum-controlled network erasure theorem.

Random network states are explicitly sampling evidence, not state enumeration.
Separate local input-branch and two-module product checks are exhaustive over
their declared finite domains.  No downloads, packages, training, or GPU use.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from integer_snn.model import Params, State, zero_state
from integer_snn.network import (NetworkSpec, network_step, network_reference_step,
                                 current_spike_contract, transition_contract)
from integer_snn.reachability import reachable_closure
from experiments.run_pilot import write_symbol, peak_memory_bytes
from integer_snn.model import step


ROOT = Path(__file__).resolve().parents[1]


def witness_params(inhibition=4):
    return Params(5, 4, 3, 2, 3, 2, 6, inhibition, 14)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_branch_audit(row_sum, inhibition):
    """Exact scalar abstraction, independently evaluated with Fraction.

    Every legal local (v,I,s) with s=1 => I<=9 and s=0 => I<=12 is included.
    Coupling c is ANY integer in [0,R] with c[t]+c[t-1]<=R.  Other-cell
    inhibition h can independently be either 0 or 1 on each step, a conservative
    enlargement.  The initial coupling has no fictitious prehistory restriction.
    """
    amplitude = max(3, row_sum + 2)
    image = {(v, current, 0, -1) for v in range(-15, 14) for current in range(13)}
    image |= {(0, current, 1, -1) for current in range(10)}
    initial_count = len(image)
    history, evaluations = [], 0
    no_consecutive = True
    for tick in range(1, 5):
        next_image = set()
        for voltage, current, spike, previous_c in image:
            for coupling in range(row_sum + 1):
                if previous_c >= 0 and coupling + previous_c > row_sum:
                    continue
                for inhibitory_spike in (0, 1):
                    new_current = min(15, int(Fraction(3 * current, 4)) + 6 * spike)
                    potential = max(-15, min(15, int(Fraction(3 * voltage, 4)) + new_current
                                                - inhibition * inhibitory_spike + coupling - amplitude))
                    fired = int(potential >= 14)
                    no_consecutive &= not (spike and fired)
                    next_image.add((0 if fired else potential, new_current, fired, coupling))
                    evaluations += 1
        image = next_image
        history.append({"tick": tick, "abstract_states": len(image),
                        "max_current": max(x[1] for x in image),
                        "max_voltage": max(x[0] for x in image), "min_voltage": min(x[0] for x in image),
                        "any_spike": any(x[2] for x in image)})
    if any(s or current > 6 or v > 13 for v, current, s, c in image) or not no_consecutive:
        raise AssertionError(("Local prefix candidate failed", row_sum, inhibition, history))
    # Global silence at t4 removes all coupling and inhibition inputs together.
    image = {(v, current, s) for v, current, s, c in image}
    for tick in range(5, 15):
        next_image = set()
        for voltage, current, spike in image:
            new_current = int(Fraction(3 * current, 4)) + 6 * spike
            potential = max(-15, min(15, int(Fraction(3 * voltage, 4)) + new_current))
            fired = int(potential >= 14)
            next_image.add((0 if fired else potential, new_current, fired))
            evaluations += 1
        image = next_image
        if any(x[2] for x in image):
            raise AssertionError("Candidate release rebounds in conservative scalar image")
        history.append({"tick": tick, "abstract_states": len(image),
                        "max_current": max(x[1] for x in image),
                        "max_voltage": max(x[0] for x in image), "min_voltage": min(x[0] for x in image),
                        "any_spike": False})
    if image != {(0, 0, 0)}:
        raise AssertionError("Scalar abstraction has not reached zero by t14")
    return {"row_sum_bound": row_sum, "inhibition": inhibition, "amplitude": amplitude,
            "initial_scalar_states": initial_count, "transition_evaluations": evaluations,
            "no_consecutive_spikes": no_consecutive, "zero_at_14": True, "history": history,
            "scope": "exhaustive scalar abstraction with weighted two-tick input contract; not arbitrary input sequences"}


def topologies(modules, row_sum, seed):
    """Distinct matrices: concentrated ring, distributed rows and seeded random.

    M=1 uses an explicit autapse when R>0.  Larger networks exclude self-loops
    here, although the general kernel and input-contract proof permit them.
    """
    rng = random.Random(seed)
    results, seen = [], set()
    for label in ("ring", "distributed", "random"):
        rows = []
        for target in range(modules):
            row = [0] * modules
            sources = [(target + k) % modules for k in range(1, modules)] or [target]
            for token in range(row_sum):
                source = sources[0] if label == "ring" else sources[token % len(sources)] if label == "distributed" else rng.choice(sources)
                row[source] += 1
            rows.append(tuple(row))
        matrix = tuple(rows)
        if matrix not in seen:
            results.append((label, matrix))
            seen.add(matrix)
    return results


def random_initial(spec, rng, sample):
    values = []
    for module in range(spec.module_count):
        voltages, currents, spikes = [], [], []
        for cell in (0, 1):
            if sample < 8:
                spike = (sample >> cell) & 1
                if sample >= 4:
                    spike ^= module % 2
                current = 9 if spike else 12
                voltage = 0 if spike else (-15 if sample == 7 else 13)
            else:
                spike = rng.randrange(2)
                current = rng.randrange(10 if spike else 13)
                voltage = 0 if spike else rng.randint(-15, 13)
            voltages.append(voltage)
            currents.append(current)
            spikes.append(spike)
        values.append(State(tuple(voltages), tuple(currents), tuple(spikes)))
    return tuple(values)


def sampled_network_case(spec, samples, seed, reference_samples=16):
    rng = random.Random(seed)
    amplitude = max(3, spec.max_row_sum + 2)
    latest_zero, checks, reference_checks, contract_checks = 0, 0, 0, 0
    initial_digest = hashlib.sha256()
    histories = [{"tick": tick, "max_current": 0, "max_voltage": -15,
                  "min_voltage": 15, "max_spikes": 0} for tick in range(15)]
    for sample in range(samples):
        states = random_initial(spec, rng, sample)
        initial_digest.update(repr(states).encode())
        if not current_spike_contract(spec, states):
            raise AssertionError("Generated state is outside declared initial theorem domain")
        # A zero visited while the pulse is still applied need not be absorbing.
        # Completion is recorded only at/after release, and final zero is checked.
        first_zero = None
        initial = states
        for tick in range(15):
            row = histories[tick]
            row["max_current"] = max(row["max_current"], *(x for state in states for x in state.current))
            row["max_voltage"] = max(row["max_voltage"], *(x for state in states for x in state.v))
            row["min_voltage"] = min(row["min_voltage"], *(x for state in states for x in state.v))
            row["max_spikes"] = max(row["max_spikes"], sum(sum(s.spike) for s in states))
            if tick == 4 and any(any(s.spike) or max(s.current) > 6 or max(s.v) > 13 for s in states):
                raise AssertionError(("Candidate D4 prefix failed", asdict(spec), initial, states))
            if tick == 9 and any(any(s.current) or max(map(abs, s.v)) > 6 for s in states):
                raise AssertionError(("Candidate t9 release bound failed", asdict(spec), initial, states))
            if tick == 14:
                break
            erase = amplitude if tick < 4 else 0
            successor = network_step(spec, states, erase=erase)
            checks += 1
            if spec.module_count <= 4 and sample < reference_samples:
                if successor != network_reference_step(spec, states, erase=erase):
                    raise AssertionError("Independent Fraction network reference disagrees")
                reference_checks += 1
            contract = transition_contract(spec, states, successor)
            if not contract["no_consecutive_spikes"] or not contract["weighted_window_bound"]:
                raise AssertionError(("Observed window contract violated", asdict(spec), initial, states, successor, contract))
            if not current_spike_contract(spec, successor):
                raise AssertionError("Current/spike domain is not invariant")
            contract_checks += 1
            states = successor
            if tick + 1 >= 4 and first_zero is None and all(s == zero_state() for s in states):
                first_zero = tick + 1
        if first_zero is None or any(s != zero_state() for s in states):
            raise AssertionError(("Candidate network theorem counterexample", asdict(spec), initial, states))
        latest_zero = max(latest_zero, first_zero)
    return {"params": asdict(spec.params), "weights": [list(row) for row in spec.weights],
            "modules": spec.module_count, "row_sums": list(spec.row_sums), "R": spec.max_row_sum,
            "amplitude": amplitude, "duration": 4, "recovery_bound": 10,
            "samples": samples, "seed": seed, "initial_states_sha256": initial_digest.hexdigest(),
            "latest_sampled_zero_tick": latest_zero, "network_transition_evaluations": checks,
            "independent_reference_network_steps": reference_checks, "observed_contract_checks": contract_checks,
            "history_bounds": histories, "all_passed": True,
            "scope": "fixed adversarial corners plus seeded random legal network states; not exhaustive"}


def exact_two_module_product():
    params = witness_params()
    locals_by_symbol = {}
    for symbol in (0, 1):
        state = write_symbol(params, symbol, 2)
        for _ in range(6):
            state = step(params, state)
        closure = reachable_closure(params, [state], noise_bound=1, max_states=4096)
        if not closure["closed"] or len(closure["states"]) != 32:
            raise AssertionError("Expected 32-state local noisy closure did not reconstruct")
        locals_by_symbol[symbol] = closure["states"]
    local_union = locals_by_symbol[0] | locals_by_symbol[1]
    image = set(itertools.product(local_union, repeat=2))
    initial_count = len(image)
    spec = NetworkSpec(params, ((0, 1), (1, 0)))
    first_zero, checks, history = None, 0, []
    for tick in range(1, 15):
        next_image = set()
        for previous in image:
            successor = network_step(spec, previous, erase=3 if tick <= 4 else 0)
            if successor != network_reference_step(spec, previous, erase=3 if tick <= 4 else 0):
                raise AssertionError("Product independent reference mismatch")
            contract = transition_contract(spec, previous, successor)
            if not contract["weighted_window_bound"] or not contract["no_consecutive_spikes"]:
                raise AssertionError("Product window contract failed")
            next_image.add(successor)
            checks += 1
        image = next_image
        history.append({"tick": tick, "image_states": len(image)})
        if image == {(zero_state(), zero_state())} and first_zero is None:
            first_zero = tick
    if image != {(zero_state(), zero_state())}:
        raise AssertionError("4096-pair product failed E3 D4 recovery10")
    return {"params": asdict(params), "weights": [[0, 1], [1, 0]],
            "local_closure_sizes": {str(k): len(v) for k, v in locals_by_symbol.items()},
            "initial_product_states": initial_count, "amplitude": 3, "duration": 4,
            "recovery_bound": 10, "latest_zero_tick": first_zero,
            "network_transition_evaluations": checks, "all_transitions_fraction_checked": True,
            "all_passed": True, "history": history,
            "scope": "exact four symbol-pair product of the two 32-state local closures; no Cartesian enumeration for larger M"}


def weight_two_counterexample():
    spec = NetworkSpec(witness_params(), ((0, 2), (2, 0)))
    states = (State((0, -5), (9, 0), (1, 0)),) * 2
    initial = states
    trace = [{"tick": 0, "states": [asdict(s) for s in states]}]
    for tick in range(1, 5):
        next_states = network_step(spec, states, erase=3)
        if next_states != network_reference_step(spec, states, erase=3):
            raise AssertionError("Counterexample reference disagreement")
        states = next_states
        trace.append({"tick": tick, "states": [asdict(s) for s in states]})
    release = states
    seen, path = {}, []
    for tick in range(128):
        if states in seen:
            cycle = path[seen[states]:]
            break
        if all(s == zero_state() for s in states):
            raise AssertionError("Requested W2/E3 counterexample unexpectedly erased")
        seen[states] = len(path)
        path.append(states)
        successor = network_step(spec, states)
        if successor != network_reference_step(spec, states):
            raise AssertionError("Counterexample recovery reference disagreement")
        states = successor
    else:
        raise AssertionError("Counterexample nonzero cycle not resolved within 128 ticks")
    return {"params": asdict(spec.params), "weights": [list(row) for row in spec.weights],
            "amplitude": 3, "duration": 4, "initial": [asdict(s) for s in initial],
            "pulse_trace": trace, "release_state": [asdict(s) for s in release],
            "release_transient_length": seen[states], "nonzero_cycle_period": len(cycle),
            "cycle": [[asdict(s) for s in network] for network in cycle],
            "all_transitions_fraction_checked": True,
            "scope": "E3 D4 failure at R2; does not contradict E=max(3,R+2)=4 guarantee"}


def run(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    if not 1 <= args.samples <= 4096:
        raise ValueError("samples must be in 1..4096")
    output.mkdir(parents=True)

    def deadline():
        if time.perf_counter() - started > args.hard_timeout_seconds:
            raise TimeoutError("Run deadline exceeded between network cases")

    local = [local_branch_audit(R, g) for R in range(5) for g in (0, 4, 16)]
    _json(output / "local_branch_audits.json", local)
    rows = []
    for modules, R, inhibition in itertools.product((1, 2, 4, 8, 16), range(5), (0, 4, 16)):
        for topology, matrix in topologies(modules, R, args.seed + modules * 100 + R):
            deadline()
            spec = NetworkSpec(witness_params(inhibition), matrix)
            case = sampled_network_case(spec, args.samples, args.seed + len(rows) * 1009)
            case["topology"] = topology
            case["case_index"] = len(rows)
            rows.append(case)
        if len(rows) and len(rows) % 25 == 0:
            print(f"Completed {len(rows)} network cases, elapsed {time.perf_counter() - started:.2f}s", flush=True)
    deadline()
    product_check = exact_two_module_product()
    counterexample = weight_two_counterexample()
    _json(output / "network_cases.json", rows)
    _json(output / "exact_product.json", product_check)
    _json(output / "counterexample_w2_e3.json", counterexample)
    scientific = {
        "network_cases": len(rows), "modules_tested": [1, 2, 4, 8, 16],
        "row_sum_bounds_tested": list(range(5)), "inhibitions_tested": [0, 4, 16],
        "initial_network_samples": sum(r["samples"] for r in rows),
        "network_transition_evaluations": sum(r["network_transition_evaluations"] for r in rows),
        "independent_sample_reference_steps": sum(r["independent_reference_network_steps"] for r in rows),
        "observed_contract_checks": sum(r["observed_contract_checks"] for r in rows),
        "latest_sampled_zero_tick": max(r["latest_sampled_zero_tick"] for r in rows),
        "local_branch_configurations": len(local),
        "local_branch_transition_evaluations": sum(r["transition_evaluations"] for r in local),
        "exact_product_initial_pairs": product_check["initial_product_states"],
        "exact_product_zero_tick": product_check["latest_zero_tick"],
        "exact_product_transition_evaluations": product_check["network_transition_evaluations"],
        "candidate_protocol": "E=max(3,R+2), D=4, recovery<=10, total<=14",
        "false_positive_cases": 0, "nonzero_cycle_counterexample_period": counterexample["nonzero_cycle_period"],
        "network_cases_sha256": _hash(rows), "local_branch_sha256": _hash(local),
        "exact_product_sha256": _hash(product_check), "counterexample_sha256": _hash(counterexample),
    }
    summary = {"status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "seed": args.seed, "samples_per_case": args.samples, "scientific_results": scientific,
               "scope": [
                   "all modules share bv5 bi4 alpha=beta3/4 w6 theta14; g is common and nonnegative",
                   "initial states are legal and satisfy spike1=>current<=9; spike0=>current<=12 per cell",
                   "same-index nonnegative integer W and simultaneous old spikes; explicit W[target][source]",
                   "common erase only, no external input/noise/cross-module signal beyond the stored W",
                   "network sampling is not an exhaustive proof; theorem supplied separately",
                   "local scalar abstraction is exhaustive only for the listed R,g and input-window contract",
                   "larger network Cartesian state spaces are never enumerated",
               ],
               "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
               "peak_process_working_set_bytes": peak_memory_bytes(), "gpu_used": False, "download_bytes": 0}
    _json(output / "summary.json", summary)
    paths = [ROOT / "integer_snn/network.py", ROOT / "experiments/run_coupled_erasure.py",
             ROOT / "tests/test_network.py", ROOT / "integer_snn/model.py", ROOT / "integer_snn/reference.py"]
    _json(output / "source_hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size > 10_000_000:
        raise AssertionError("Output exceeded 10 MB limit")
    deadline()
    print(json.dumps({"output": str(output), "output_bytes": size,
                      "wall_seconds": summary["wall_seconds"], **scientific}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--hard-timeout-seconds", type=float, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
