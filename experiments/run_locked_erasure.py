"""Compare topology-aware uniform erasure to exact free-duration optima."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.run_core_endpoint_witnesses import endpoint_certificate
from experiments.run_external_sat_erasure import query
from integer_snn.core_erasure import core_erasure_bounds
from integer_snn.deadline_erasure import deadline_peeling
from integer_snn.network import NetworkSpec
from integer_snn.release_tail import release_tail
from integer_snn.topology_certificate import MODEL, certify_row_sum_erasure


ROOT = Path(__file__).resolve().parents[1]
PANELS = ("free_duration_sat_20261119_v1", "free_duration_disjoint_20261119_v1")


def compare_protocols(spec, deadline):
    """Each source supplies an independently established permanent-pressure last spike."""
    if type(deadline) is not int or deadline < 0:
        raise ValueError("deadline must be a nonnegative integer")
    if spec.params != MODEL:
        raise ValueError("requires the fixed erasure model")
    bounds = []
    for short in (False, True):
        row = certify_row_sum_erasure(spec, shorter_pulse=short)
        bounds.append(("row", row["amplitude"], row["last_spike_upper"]))
    core = core_erasure_bounds(spec)
    # m synchronous removals have last-spike bound 4 + 5*(m-1).
    bounds.append(("endpoint", core["k"] + 3, 5 * len(core["layers"]) - 1))
    weak = endpoint_certificate(spec, core["lower"])
    if weak["certified"]:
        bounds.append(("endpoint", core["lower"], weak["last_spike_upper"]))

    # A tighter release tail can make a formerly late peeling plan admissible.
    # Archived peeling plans use D=L+4 and H=10. A plan whose new total is
    # at most T has L<=T, hence its old total is at most T+14.
    frontier = deadline_peeling(spec, range(deadline, deadline + 15))
    for plan in set(frontier.values()) - {None}:
        amplitude, pressure, recovery = plan
        if recovery != 10 or pressure < 4:
            raise AssertionError("peeling certificate contract changed")
        bounds.append(("peeling", amplitude, pressure - 4))

    groups = {"row_legacy": [], "row_tight": [], "endpoints_legacy": [],
              "peeling_legacy": [], "locked": []}
    for source, amplitude, last in set(bounds):
        a, h = release_tail(amplitude)
        old = (amplitude, last + 4, 10)
        new = (amplitude, last + a, h)
        if sum(old[1:]) <= deadline:
            groups["peeling_legacy"].append(old)
            if source != "peeling":
                groups["endpoints_legacy"].append(old)
            if source == "row":
                groups["row_legacy"].append(old)
        for plan in (old, new):
            if sum(plan[1:]) <= deadline:
                groups["locked"].append(plan)
                if source == "row":
                    groups["row_tight"].append(plan)
    # After minimizing peak amplitude, prefer the shorter total, then pulse.
    def rank(plan):
        e, d, h = plan
        return e, d + h, d
    return {name: min(plans, key=rank, default=None) for name, plans in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    if out.exists():
        parser.error("output already exists")
    records = []
    hashes = {}
    for panel in PANELS:
        source = ROOT / "results" / panel / "report.json"
        contents = source.read_bytes()
        hashes[panel] = hashlib.sha256(contents).hexdigest()
        for case in json.loads(contents)["cases"]:
            graph = tuple(map(tuple, case["weights"]))
            spec = NetworkSpec(MODEL, graph)
            deadline = case["horizon"]
            oracle = case["result"]
            if not oracle["minimum_proved"]:
                raise ValueError("an exact comparator is required")
            optimum = oracle["first_feasible"]["e"]
            plans = compare_protocols(spec, deadline)
            selected = plans["locked"]
            if selected is None or selected[0] < optimum:
                raise AssertionError((graph, deadline, selected, optimum))
            # Check the actual pulse and the claimed recovery, not just its peak.
            verdict = query(spec, *selected, 3_000_000)["status"]
            if verdict != "UNSAT":
                raise AssertionError((graph, deadline, selected, verdict))
            records.append(dict(panel=panel, weights=graph, deadline=deadline, optimum=optimum,
                                plans=plans, selected_protocol_check=verdict))
    sources = ("experiments/run_locked_erasure.py", "integer_snn/release_tail.py",
               "integer_snn/deadline_erasure.py", "integer_snn/core_erasure.py",
               "integer_snn/topology_certificate.py", "experiments/run_core_endpoint_witnesses.py",
               "experiments/run_external_sat_erasure.py")
    summary = {}
    for key in records[0]["plans"]:
        gaps = [r["plans"][key][0] - r["optimum"] for r in records if r["plans"][key] is not None]
        summary[key] = dict(solved=len(gaps), exact=gaps.count(0), gap_sum=sum(gaps),
                            max_gap=max(gaps, default=None))
    result = dict(inputs_sha256=hashes, source_sha256={
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in sources
    }, summary=summary, cases=records)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
