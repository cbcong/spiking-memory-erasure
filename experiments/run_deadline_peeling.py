"""Replay the frozen graph panel and compare paired protocols at common deadlines."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.run_core_scale_comparison import ROOT, weights
from experiments.run_extended_size_certificates import ExtendedSpec
from experiments.run_topology_certificate import initials
from integer_snn.deadline_erasure import deadline_peeling
from integer_snn.model import zero_state
from integer_snn.network import network_reference_step
from integer_snn.topology_certificate import MODEL


BUDGETS = (20, 40, 60, 100)


def run(panel):
    data = json.loads(panel.read_bytes())
    rows = []
    trajectories = steps = 0
    for record in data["rows"]:
        n, seed, family = record["n"], record["seed"], record["family"]
        matrix = weights(n, family, seed)
        if hashlib.sha256(json.dumps(matrix).encode()).hexdigest() != record["matrix_sha256"]:
            raise AssertionError("input matrix changed")
        spec = ExtendedSpec(MODEL, matrix)
        frontier = deadline_peeling(spec, BUDGETS)
        plans = {}
        for t in BUDGETS:
            old = min(tuple(p) for p in (record["base"], record["proposed"]) if sum(p[1:]) <= t)
            selected = min(old, frontier[t])
            plans[t] = dict(old=old, new=selected)
        if record["trial"] == 0:
            # Verify the new 40-step protocol on the two starts used in the frozen audit.
            e, d, h = plans[40]["new"]
            starts = initials(n, seed)
            for i in (3, 8):
                state = starts[i]
                for tick in range(d+h):
                    state = network_reference_step(spec, state, erase=e if tick < d else 0)
                if state != (zero_state(),)*n:
                    raise AssertionError((n, family, seed, i))
                trajectories += 1
                steps += d+h
        rows.append(dict(n=n, family=family, seed=seed, hash=record["matrix_sha256"],
                         baseline=record["base"], lower=record["lower"], plans=plans))

    def summarize(subset):
        base = sum(r["baseline"][0] for r in subset)
        out = {}
        for t in BUDGETS:
            old = sum(r["plans"][t]["old"][0] for r in subset)
            new = sum(r["plans"][t]["new"][0] for r in subset)
            out[t] = dict(n=len(subset), baseline_sum=base, old_sum=old,
                          new_sum=new, reduction_vs_baseline=1-new/base,
                          old_reduction_vs_baseline=1-old/base,
                          improved_vs_old=sum(r["plans"][t]["new"][0] < r["plans"][t]["old"][0]
                                              for r in subset),
                          equal_proved_lower=sum(r["plans"][t]["new"][0] == r["lower"] for r in subset))
        return out

    return dict(input=str(panel.relative_to(ROOT)), panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
                code_sha256={file: hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in
                             ["integer_snn/deadline_erasure.py", "experiments/run_deadline_peeling.py"]},
                original_protocols="min(row, previously selected endpoint plan) among those meeting deadline",
                new_protocols="original protocols plus bounded-round peeling, same D+H deadline",
                fraction_trajectories=trajectories, fraction_steps=steps, summary=summarize(rows),
                by_family={f:summarize([r for r in rows if r["family"] == f]) for f in data["families"]},
                by_size={n:summarize([r for r in rows if r["n"] == n]) for n in data["sizes"]}, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.panel.resolve())
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["summary"]))
