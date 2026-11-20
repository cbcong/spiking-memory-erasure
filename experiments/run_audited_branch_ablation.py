"""Measure first certified selection and complete, fresh Fraction audit in one deadline."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from experiments.run_branch_ablation import BUDGETS, CASES, REGIMES, REPEATS, ROOT, select
from experiments.run_protocol_selection import audit_protocol, write_csv, write_json
from integer_snn.fixed_protocol import enumerate_domain, reconstruct_case


def audited_call(case, arm, milliseconds):
    start = time.perf_counter()
    selection = select(case, arm, milliseconds)
    selected_at = time.perf_counter()
    fields = dict(status="no_selection", E=selection.get("amplitude"), D=selection.get("duration"),
                  H=selection.get("reported_recovery"), original_states=0, fraction_updates=0)
    if selection["status"] == "accepted":
        params, groups, _ = reconstruct_case(case)
        domain = enumerate_domain(groups[0]+groups[1])
        audit = audit_protocol(params, domain, fields["E"], fields["D"], fields["H"])
        valid = audit["passed"] and len(domain) == case["initial_states"]
        fields.update(status="verified" if valid else "incorrect", original_states=len(domain),
                      fraction_updates=audit["fraction_transition_evaluations"])
    end = time.perf_counter()
    total_ms = (end-start)*1000
    return dict(**fields, accepted=fields["status"] == "verified" and total_ms <= milliseconds,
                total_ms=total_ms, select_ms=(selected_at-start)*1000, audit_ms=(end-selected_at)*1000)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(CASES.read_bytes())
    if len(cases) != 213:
        raise ValueError("case count changed")
    files = [CASES, Path(__file__).resolve(), ROOT/"experiments/run_branch_ablation.py",
             ROOT/"experiments/run_protocol_selection.py"]
    files += [ROOT/"integer_snn"/name for name in ("fixed_protocol.py", "certificates.py", "model.py",
              "phase_erasure.py","reference.py","budgeted_erasure.py")]
    seed = 2026111906
    write_json(out/"design.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
               python=platform.python_version(), order_seed=seed, repetitions=REPEATS, budgets_ms=BUDGETS,
               contract="first certificate then fresh independent full original-domain Fraction audit through the claimed H, stop at the first certificate, atomic completion required by common wall deadline, no reoptimization and no cross-call cache",
               file_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    rng = random.Random(seed)
    for arm in REGIMES:
        audited_call(cases[0], arm, 20)
    started = time.perf_counter()
    rows = []
    for i, case in enumerate(cases):
        for budget in BUDGETS:
            for repeat in range(REPEATS):
                order = list(REGIMES)
                rng.shuffle(order)
                for j, arm in enumerate(order):
                    rows.append(dict(case=i,budget=budget,repeat=repeat,order=j,arm=arm,
                                     **audited_call(case, arm, budget)))
        if (i+1)%40==0:
            print(f"completed {i+1}/{len(cases)}", flush=True)
    write_csv(out/"calls.csv",rows)
    arms = []
    pairs = []
    for budget in BUDGETS:
        indices = {}
        for arm in REGIMES:
            subset = [r for r in rows if (r["arm"],r["budget"]) == (arm,budget)]
            indices[arm] = {(r["case"], r["repeat"]):r for r in subset}
            arms.append(dict(budget=budget,arm=arm,total=len(subset),accepted=sum(r["accepted"] for r in subset),
                             incorrect=sum(r["status"]=="incorrect" for r in subset),
                             selected=sum(r["status"]!="no_selection" for r in subset),
                             repeats=[sum(r["accepted"] and r["repeat"]==i for r in subset) for i in range(REPEATS)]))
        counts = Counter()
        for key, a in indices["split"].items():
            b = indices["hull"][key]
            counts["both" if a["accepted"] and b["accepted"] else "split_only" if a["accepted"] else
                   "hull_only" if b["accepted"] else "neither"] += 1
        pairs.append(dict(budget=budget,counts=dict(counts)))
    summary = dict(arms=arms,pairs=pairs,elapsed_seconds=time.perf_counter()-started)
    write_json(out/"summary.json",summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
