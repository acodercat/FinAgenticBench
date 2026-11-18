"""Loading, slot indexing and case-resampled bootstrap shared by the analysis scripts."""
from __future__ import annotations

import collections
import csv
import json
import random
from pathlib import Path

DATA = Path("analysis/scratch/final_data.json")
VAR_CATS = Path("analysis/scratch/var_cats.json")
TABLES = Path("analysis/final_20261012/tables")


def load():
    d = json.loads(DATA.read_text())
    return d["cases"], d["turns"], d["vars"]


def load_cats():
    """{(study, case, run_id, turn, var): error-shape category} for wrong, set variables."""
    raw = json.loads(VAR_CATS.read_text())
    out = {}
    for k, v in raw.items():
        s, c, r, t, var = k.split("|")
        out[(s, c, r, int(t), var)] = v
    return out


def slots(vars_, only_set=True):
    """(case, turn, var) -> list of variable rows."""
    by = collections.defaultdict(list)
    for v in vars_:
        if only_set and not v["is_set"]:
            continue
        by[(v["case"], v["turn"], v["var"])].append(v)
    return by


def runs(turns):
    """(study, case, run_id) -> turns sorted by turn index."""
    by = collections.defaultdict(list)
    for t in turns:
        by[(t["study"], t["case"], t["run_id"])].append(t)
    for v in by.values():
        v.sort(key=lambda t: t["turn"])
    return by


def rate(num, den):
    return num / den if den else float("nan")


def boot_ratio_change(per_case_a, per_case_b, n=2000, seed=0):
    """Relative change (b-a)/a of a per-1000 rate, resampling cases. per_case_* : {case: (num, den)}.

    Returns (point, lo, hi) for the 95% interval; cases are the units because runs of one case share a question.
    """
    cases = sorted(set(per_case_a) | set(per_case_b))
    rng = random.Random(seed)

    def est(sample):
        na = sum(per_case_a.get(c, (0, 0))[0] for c in sample); da = sum(per_case_a.get(c, (0, 0))[1] for c in sample)
        nb = sum(per_case_b.get(c, (0, 0))[0] for c in sample); db = sum(per_case_b.get(c, (0, 0))[1] for c in sample)
        ra, rb = rate(na, da), rate(nb, db)
        return (rb - ra) / ra if ra else float("nan")

    point = est(cases)
    draws = sorted(x for x in (est([rng.choice(cases) for _ in cases]) for _ in range(n)) if x == x)
    if not draws:
        return point, float("nan"), float("nan")
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def boot_rate(per_case, n=2000, seed=0):
    """A rate num/den with a case-resampled 95% interval. per_case: {case: (num, den)}. Returns (point, lo, hi)."""
    cases = sorted(per_case)
    rng = random.Random(seed)

    def est(sample):
        return rate(sum(per_case[c][0] for c in sample), sum(per_case[c][1] for c in sample))

    draws = sorted(x for x in (est([rng.choice(cases) for _ in cases]) for _ in range(n)) if x == x)
    if not draws:
        return est(cases), float("nan"), float("nan")
    return est(cases), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def modal_shared(vars_, min_configs=3):
    """{(case, turn, var): modal wrong value} for slots whose most common wrong value was given by >= min_configs
    configurations. The single definition of a shared wrong value, used by every script."""
    by = collections.defaultdict(list)
    for v in vars_:
        if v["is_set"] and not v["correct"]:
            by[(v["case"], v["turn"], v["var"])].append(v)
    out = {}
    for k, xs in by.items():
        val, n = collections.Counter(x["value"] for x in xs).most_common(1)[0]
        if n >= min_configs and len({x["study"] for x in xs if x["value"] == val}) >= min_configs:
            out[k] = val
    return out


def write_csv(name, header, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"-> {TABLES / name}")


def pct(x, nd=1):
    return "nan" if x != x else f"{x * 100:.{nd}f}%"
