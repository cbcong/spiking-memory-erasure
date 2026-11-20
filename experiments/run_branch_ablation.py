"""Single-process same-label, same-wall-budget ablation of interval merging."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

from integer_snn import fixed_protocol as fixed
from integer_snn.phase_erasure import _merge
from experiments.run_protocol_selection import audit_protocol, stats, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results/erasure_study_20261112_v3/case_details.json"
REGIMES = ("split", "hull")
BUDGETS = (20, 100)
REPEATS = 5
ORDER_SEED = 2026111905


@contextmanager
def with_policy(regime):
    """Bind the merge for initial, pulse and recovery boxes in an isolated process."""
    if regime not in REGIMES:
        raise ValueError("unknown branch policy")

    def merge_policy(boxes, default, counts):
        if default != "split":
            raise AssertionError("upstream checker changed policy semantics")
        return _merge(boxes, regime, counts)

    with patch.object(fixed, "_merge", new=merge_policy):
        yield


def select(case, regime, budget):
    with with_policy(regime):
        result = fixed.select_protocol(case, "certified", budget / 1000, sample_seed=0)
    protocol = result.pop("protocol")
    events = result.pop("native_event_counts")
    last = result.pop("last_candidate")
    return dict(**result, **(protocol or {}), **events, last=last)


def compare(rows, audits):
    by_arm = []
    comparison = []
    for budget in BUDGETS:
        subsets = {}
        for arm in REGIMES:
            selected = [r for r in rows if r["regime"] == arm and r["budget_ms"] == budget]
            subsets[arm] = {(r["case"], r["repeat"]):r for r in selected}
            valid = [r for r in selected if r["valid"]]
            by_arm.append(dict(budget_ms=budget, arm=arm, calls=len(selected), valid=len(valid),
                               valid_by_repeat=[sum(r["repeat"] == j for r in valid) for j in range(REPEATS)],
                               wall_ms=stats([1000*r["wall_seconds"] for r in selected]),
                               max_overrun_ms=1000*max(r["overrun_seconds"] for r in selected)))
        counts = Counter()
        for key, s in subsets["split"].items():
            h = subsets["hull"][key]
            if s["valid"] and h["valid"]:
                counts["both"] += 1
                a = (s["amplitude"], s["duration"])
                b = (h["amplitude"], h["duration"])
                counts["equal_ED" if a == b else "split_lower_ED" if a < b else "hull_lower_ED"] += 1
            else:
                counts["split_only" if s["valid"] else "hull_only" if h["valid"] else "neither"] += 1
        comparison.append(dict(budget_ms=budget, counts=dict(counts)))
    return dict(arms=by_arm, pairs=comparison, unique_audits=len(audits),
                failed_audits=sum(not a["passed"] for a in audits),
                underbound_rows=sum(r.get("nonzero_at_reported_H", 0)>0 for r in rows),
                unique_case_protocol_initial_state_pairs=sum(a["original_initial_states"] for a in audits),
                independent_fraction_updates=sum(a["fraction_transition_evaluations"] for a in audits))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(CASES.read_bytes())
    if len(cases) != 213:
        raise ValueError("case cohort changed")
    source_files = [Path(__file__).resolve(), ROOT/"tests/test_branch_ablation.py", CASES]
    source_files += [ROOT/"integer_snn"/name for name in
                     ("fixed_protocol.py", "phase_erasure.py", "budgeted_erasure.py", "model.py", "certificates.py", "reference.py")]
    source_files += [ROOT/"experiments/run_protocol_selection.py"]
    write_json(out/"design.json", dict(before_utc=datetime.now(timezone.utc).isoformat(),
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
               sizes=dict(cases=len(cases), repeats=REPEATS), budgets_ms=BUDGETS, arm_order_seed=ORDER_SEED,
               controls=dict(E=[0,64], D=[1,32], Hmax=32, box_cap=128),
               timing="sequential Python, case/budget/repeat blocks, randomized arm order, one warmup per arm; shared full-call wall clock; independent audit separate",
               domain="original two-symbol retention boxes canonicalized by swap for both arms; tests count unique (case,protocol) once",
               python=sys.version, platform=platform.platform()))
    rng = random.Random(ORDER_SEED)
    for arm in REGIMES:
        select(cases[0], arm, 20)
    started = time.perf_counter()
    rows = []
    for i, case in enumerate(cases):
        for budget in BUDGETS:
            for rep in range(REPEATS):
                order = list(REGIMES)
                rng.shuffle(order)
                for index, arm in enumerate(order):
                    result = select(case, arm, budget)
                    rows.append(dict(case=i, repeat=rep, order=index, budget_ms=budget, regime=arm, **result))
        if (i+1) % 40 == 0:
            print(f"timed {i+1}/{len(cases)}", flush=True)
    timed_seconds = time.perf_counter() - started
    audits, lookup = [], {}
    for i, case in enumerate(cases):
        p, groups, _ = fixed.reconstruct_case(case)
        original = fixed.enumerate_domain(groups[0]+groups[1])
        if len(original) != case["initial_states"]:
            raise AssertionError("wrong original domain")
        for r in (r for r in rows if r["case"] == i):
            r["valid"] = False
            if r["status"] != "accepted":
                continue
            if r["wall_seconds"] > r["budget_seconds"]:
                raise AssertionError("over-budget acceptance")
            key = (i, r["amplitude"], r["duration"])
            if key not in lookup:
                lookup[key] = len(audits)
                audits.append(dict(case=i, E=key[1], D=key[2], **audit_protocol(p, original, key[1], key[2])))
            r["audit_index"] = lookup[key]
            audit = audits[lookup[key]]
            r["nonzero_at_reported_H"] = audit["nonzero_original_states_by_recovery_tick"][r["reported_recovery"]]
            r["valid"] = audit["passed"] and not r["nonzero_at_reported_H"]
    write_csv(out/"selection.csv", rows)
    write_json(out/"fraction_audits.json", audits)
    summary = dict(**compare(rows, audits), timed_seconds=timed_seconds,
                   total_seconds=time.perf_counter()-started)
    write_json(out/"summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
