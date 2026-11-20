"""Paired comparison of the locked erasure method on a new frozen graph panel."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, quantiles

from experiments.run_core_scale_comparison import FAMILIES, ROOT, weights
from experiments.run_deadline_peeling import BUDGETS
from experiments.run_extended_size_certificates import ExtendedSpec
from experiments.run_locked_erasure import compare_protocols
from experiments.run_topology_certificate import initials
from integer_snn.model import zero_state
from integer_snn.network import network_reference_step
from integer_snn.timed_erasure import timed_erasure
from integer_snn.topology_certificate import MODEL


SIZES = (128, 256, 512)
REPLICATES = 8
BASE_SEED = 157104931
PRIMARY = 60
RESAMPLES = 10000
RESAMPLE_SEED = 157104932
ARMS = ("row_legacy", "row_tight", "endpoints_legacy", "peeling_legacy",
        "with_release_tail", "timed", "locked")
SOURCE_FILES = (
    "experiments/run_locked_erasure_scale.py",
    "experiments/run_core_scale_comparison.py",
    "experiments/run_deadline_peeling.py",
    "experiments/run_extended_size_certificates.py",
    "experiments/run_locked_erasure.py",
    "experiments/run_topology_certificate.py",
    "experiments/run_core_endpoint_witnesses.py",
    "integer_snn/core_erasure.py",
    "integer_snn/deadline_erasure.py",
    "integer_snn/release_tail.py",
    "integer_snn/timed_erasure.py",
    "integer_snn/topology_certificate.py",
    "integer_snn/model.py",
    "integer_snn/network.py",
    "integer_snn/reference.py",
)
OLD_PANELS = (
    "results/deadline_fresh_20261119_v1/panel.json",
    "results/extended_size_20261119_v1/report.json",
    "results/core_scale_comparison_20261119_v1/report.json",
)


def sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def source_hashes():
    return {name: sha256((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def rank(plan):
    amplitude, pressure, recovery = plan
    return amplitude, pressure + recovery, pressure


def plans_at(spec, deadline):
    plans = compare_protocols(spec, deadline)
    plans["with_release_tail"] = plans["locked"]
    plans["timed"] = timed_erasure(spec, deadline)
    plans["locked"] = min(
        (plan for plan in (plans["with_release_tail"], plans["timed"]) if plan is not None),
        key=rank,
    )
    for plan in plans.values():
        if plan is not None and plan[1] + plan[2] > deadline:
            raise AssertionError((deadline, plan))
    return plans


def audit(spec, plan, seed):
    e, d, h = plan
    starts = initials(spec.module_count, seed)
    for i in (3, 8):
        state = starts[i]
        for tick in range(d + h):
            state = network_reference_step(spec, state, erase=e if tick < d else 0)
        if state != (zero_state(),) * spec.module_count:
            raise AssertionError((spec.module_count, seed, plan, i))
    return dict(trajectories=2, network_steps=2 * (d + h), indices=(3, 8))


def compare(rows, deadline, arm, comparator):
    paired = [(r["plans"][deadline][arm], r["plans"][deadline][comparator]) for r in rows]
    pairs = [(a, b) for a, b in paired if a is not None and b is not None]
    if not pairs:
        return dict(paired=0)
    candidate, reference = zip(*pairs)
    amp = [a[0] - b[0] for a, b in pairs]
    product = [a[0] * a[1] - b[0] * b[1] for a, b in pairs]
    candidate_ed = sum(e * d for e, d, _ in candidate)
    reference_ed = sum(e * d for e, d, _ in reference)
    return dict(
        paired=len(pairs),
        candidate_amplitude_sum=sum(p[0] for p in candidate),
        comparator_amplitude_sum=sum(p[0] for p in reference),
        amplitude_reduction=1 - sum(p[0] for p in candidate) / sum(p[0] for p in reference),
        amplitude_diff_counts=dict(lower=sum(x < 0 for x in amp), equal=amp.count(0),
                                   higher=sum(x > 0 for x in amp)),
        candidate_mean_pressure=mean(p[1] for p in candidate),
        comparator_mean_pressure=mean(p[1] for p in reference),
        candidate_mean_total=mean(p[1] + p[2] for p in candidate),
        comparator_mean_total=mean(p[1] + p[2] for p in reference),
        candidate_amplitude_steps=candidate_ed,
        comparator_amplitude_steps=reference_ed,
        amplitude_step_ratio=candidate_ed / reference_ed,
        amplitude_step_diff_counts=dict(lower=sum(x < 0 for x in product), equal=product.count(0),
                                        higher=sum(x > 0 for x in product)),
    )


def summarize(rows):
    arms = ARMS[1:]
    return {
        deadline: dict(
            vs_row_legacy={arm: compare(rows, deadline, arm, "row_legacy") for arm in arms},
            vs_row_tight=compare(rows, deadline, "locked", "row_tight"),
            vs_with_release_tail=compare(rows, deadline, "locked", "with_release_tail"),
        )
        for deadline in BUDGETS
    }


def paired_bootstrap(rows):
    strata = {
        (n, family): [(r["plans"][PRIMARY]["locked"][0], r["plans"][PRIMARY]["row_tight"][0])
                      for r in rows if r["n"] == n and r["family"] == family]
        for n in SIZES for family in FAMILIES
    }
    rng = random.Random(RESAMPLE_SEED)
    trials = []
    for _ in range(RESAMPLES):
        sample = [rng.choice(pairs) for pairs in strata.values() for _ in pairs]
        candidate, comparator = zip(*sample)
        trials.append(1 - sum(candidate) / sum(comparator))
    cutoffs = quantiles(trials, n=40, method="inclusive")
    return dict(method="stratified paired percentile bootstrap", strata="size x family",
                resamples=RESAMPLES, seed=RESAMPLE_SEED, deadline=PRIMARY,
                statistic="1 - sum E_locked / sum E_row_tight",
                interval_95pct=(cutoffs[0], cutoffs[-1]))


def run(output):
    design = dict(sizes=SIZES, neurons_per_module=2, families=FAMILIES, base_seed=BASE_SEED,
                  replicates=REPLICATES, deadlines=BUDGETS, primary_deadline=PRIMARY,
                  primary_comparator="row_tight", bootstrap_resamples=RESAMPLES,
                  bootstrap_seed=RESAMPLE_SEED, audit_deadline=PRIMARY, audit_trial=0,
                  audit_initial_indices=(3, 8), model=repr(MODEL), source_sha256=source_hashes())
    output.mkdir(exist_ok=False, parents=True)
    write(output / "design.json", design)
    prior = set()
    prior_hashes = {}
    for name in OLD_PANELS:
        raw = (ROOT / name).read_bytes()
        prior_hashes[name] = sha256(raw)
        prior.update(row["matrix_sha256"] for row in json.loads(raw)["rows"])
    rows = []
    for n in SIZES:
        for index, family in enumerate(FAMILIES):
            for trial in range(REPLICATES):
                seed = BASE_SEED + n * 10000 + index * 100 + trial
                matrix = weights(n, family, seed)
                key = sha256(json.dumps(matrix).encode())
                if key in prior:
                    raise AssertionError((n, family, trial, "duplicate graph"))
                prior.add(key)
                spec = ExtendedSpec(MODEL, matrix)
                plans = {deadline: plans_at(spec, deadline) for deadline in BUDGETS}
                audit_result = audit(spec, plans[PRIMARY]["locked"], seed) if trial == 0 else None
                rows.append(dict(n=n, neurons=2 * n, family=family, trial=trial, seed=seed,
                                 matrix_sha256=key, plans=plans, fraction_audit=audit_result))
            print(f"{n} modules, {family}: {REPLICATES} graphs", flush=True)
    if source_hashes() != design["source_sha256"]:
        raise RuntimeError("source changed during the run")
    summary = summarize(rows)
    result = dict(
        design=design,
        prior_data_sha256=prior_hashes,
        summary=summary,
        by_family={family: summarize([r for r in rows if r["family"] == family]) for family in FAMILIES},
        by_size={n: summarize([r for r in rows if r["n"] == n]) for n in SIZES},
        primary_bootstrap=paired_bootstrap(rows),
        fraction_trajectories=sum(r["fraction_audit"]["trajectories"] for r in rows if r["fraction_audit"]),
        fraction_network_steps=sum(r["fraction_audit"]["network_steps"] for r in rows if r["fraction_audit"]),
        rows=rows,
    )
    write(output / "report.json", result)
    print(json.dumps(dict(primary=summary[PRIMARY], bootstrap=result["primary_bootstrap"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
