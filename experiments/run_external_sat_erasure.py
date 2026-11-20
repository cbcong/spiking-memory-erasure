"""Bounded full-state erasure by CNF model checking with PicoSAT (pycosat)."""
import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pycosat

from experiments.run_core_scale_comparison import examine
from integer_snn.model import State, zero_state
from integer_snn.network import NetworkSpec, network_reference_step
from integer_snn.topology_certificate import MODEL


class CNF:
    def __init__(self):
        self.last = 1
        self.clauses = [[1]]

    def var(self):
        self.last += 1
        return self.last

    def imply_equal(self, condition, left, right):
        prefix = [-literal for literal in condition]
        self.clauses.append(prefix + [-left, right])
        self.clauses.append(prefix + [left, -right])


class Order:
    """Integer represented by the monotone predicates value >= k."""

    def __init__(self, cnf, lo, hi):
        self.lo, self.hi = lo, hi
        self.bits = [cnf.var() for _ in range(hi - lo)]
        cnf.clauses.extend([a, -b] for a, b in zip(self.bits, self.bits[1:]))

    def ge(self, threshold):
        if threshold <= self.lo:
            return 1
        if threshold > self.hi:
            return -1
        return self.bits[threshold - self.lo - 1]

    def decode(self, positive):
        return self.lo + sum(bit in positive for bit in self.bits)


@dataclass
class Cell:
    v: Order
    i: Order
    s: int


def state(cnf, n):
    cells = []
    for _ in range(2 * n):
        cell = Cell(Order(cnf, -15, 13), Order(cnf, 0, 15), cnf.var())
        cnf.clauses.extend([[-cell.s, cell.v.ge(0)], [-cell.s, -cell.v.ge(1)]])
        cells.append(cell)
    return cells


def transition(cnf, previous, current, weights, erase):
    for ix, (old, new) in enumerate(zip(previous, current)):
        for fired in (0, 1):
            condition = [old.s if fired else -old.s]
            for k in range(1, 16):
                # floor(3*i/4) + 6*s >= k iff i >= ceil(4*(k-6*s)/3).
                minimum = -(-(4 * (k - 6 * fired)) // 3)
                cnf.imply_equal(condition, new.i.ge(k), old.i.ge(minimum))
        temporary = Order(cnf, -11, 24)
        for v in range(-15, 14):
            condition = [old.v.ge(v), -old.v.ge(v + 1)]
            decay = (1 if v >= 0 else -1) * (3 * abs(v) // 4)
            for k in range(-10, 25):
                cnf.imply_equal(condition, temporary.ge(k), new.i.ge(k - decay))
        controls = [(previous[ix ^ 1].s, -4)]
        controls += [(previous[2 * j + ix % 2].s, w)
                     for j, w in enumerate(weights[ix // 2]) if w]
        for assignment in itertools.product((0, 1), repeat=len(controls)):
            condition = [bit if val else -bit for (bit, _), val in zip(controls, assignment)]
            shift = sum(w * val for (_, w), val in zip(controls, assignment)) - erase
            cnf.imply_equal(condition, new.s, temporary.ge(14 - shift))
            for k in range(-14, 14):
                cnf.imply_equal(condition + [-new.s], new.v.ge(k), temporary.ge(k - shift))


def decode_initial(cells, solver_model):
    positive = {lit for lit in solver_model if lit > 0}
    return tuple(State(tuple(x.v.decode(positive) for x in cells[i:i + 2]),
                       tuple(x.i.decode(positive) for x in cells[i:i + 2]),
                       tuple(int(x.s in positive) for x in cells[i:i + 2]))
                 for i in range(0, len(cells), 2))


def query(spec, amplitude, press, release, limit):
    start = time.perf_counter()
    cnf = CNF()
    cells = initial = state(cnf, spec.module_count)
    for t in range(press + release):
        after = state(cnf, spec.module_count)
        transition(cnf, cells, after, spec.weights, amplitude if t < press else 0)
        cells = after
    cnf.clauses.append([lit for x in cells for lit in (-x.v.ge(0), x.v.ge(1), x.i.ge(1), x.s)])
    encoded = time.perf_counter()
    answer = pycosat.solve(cnf.clauses, prop_limit=limit)
    finished = time.perf_counter()
    result = dict(amplitude=amplitude, status=answer if isinstance(answer, str) else "SAT",
                  variables=cnf.last, clauses=len(cnf.clauses), encoding_s=encoded-start,
                  solving_s=finished-encoded)
    if result["status"] == "SAT":
        initial_state = trace = decode_initial(initial, answer)
        for t in range(press + release):
            trace = network_reference_step(spec, trace, erase=amplitude if t < press else 0)
        if trace == (zero_state(),) * spec.module_count:
            raise AssertionError("counterexample failed independent replay")
        result.update(initial=[asdict(x) for x in initial_state], final=[asdict(x) for x in trace])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prop-limit", type=int, default=3000000)
    parser.add_argument("--suite", choices=["pilot", "pair-grid"], default="pilot")
    args = parser.parse_args()
    cases = [("isolated", ((0,),), 8, 10), ("self_loop_weight_1", ((1,),), 10, 10),
             ("directed_pair_weight_3", ((0, 0), (3, 0)), 13, 10)]
    if args.suite == "pair-grid":
        cases = []
        for a, b in itertools.product(range(4), repeat=2):
            matrix = ((0, a), (b, 0))
            _, d, h = examine(NetworkSpec(MODEL, matrix))["proposed"]
            cases.append((f"pair_{a}_{b}", matrix, d, h))
    rows = []
    for name, matrix, d, h in cases:
        spec = NetworkSpec(MODEL, matrix)
        row = dict(name=name, weights=matrix, press=d, recovery=h, queries=[])
        for e in range(7):
            result = query(spec, e, d, h, args.prop_limit)
            row["queries"].append(result)
            print(name, e, result["status"], flush=True)
            if result["status"] == "UNSAT":
                break
        status = [r["status"] for r in row["queries"]]
        row["solver_minimum_at_fixed_timing"] = len(status) - 1 if status[-1] == "UNSAT" and all(
            s == "SAT" for s in status[:-1]) else None
        row["project_protocols"] = examine(spec)
        rows.append(row)
    report = dict(engine="PicoSAT via pycosat", pycosat_version=pycosat.__version__, suite=args.suite,
                  prop_limit=args.prop_limit, search_amplitudes="0 through first UNSAT or 6",
                  scope="all legal states; fixed press and recovery per case; no inputs or noise",
                  status_meaning="SAT is replayed failure, UNSAT is guaranteed success, UNKNOWN is unresolved",
                  cases=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
