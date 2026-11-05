"""Case-resampled 95% intervals for the change in giving the shared wrong value, by knowledge class.

Same rates as knowledge_by_tier.py (attempts on a class's slots that give the slot's >=5-config shared wrong value).
For every class: the relative change from the weak to the strong tier and from non-thinking to thinking within the
six paired models, each with a 95% interval from 2,000 draws resampling the tasks that carry the class; and, for each
class other than K4, the difference between its relative change and that of K4, on resamples of the union of the two
classes' tasks. Every (contrast, class) pair has its own fixed seed, so adding or removing a class leaves the others
unchanged. Writes tables/knowledge_change_ci.csv; the knowledge_by_tier.csv point estimates are unchanged.
"""
import collections
import json
import random

from common import load, write_csv
from groups import PAIRS, STUDIES, task_accuracy, tiers

N_DRAWS = 2000
entries = json.load(open("analysis/final_20261012/knowledge_classification_20261012.json"))
live = {(r["case"], r["var"]): r for r in json.load(open("analysis/scratch/shared_wrong_all_20261012.json"))}
slot_class = {(e["case"], s): e["primary"] for e in entries for s in e["slots"]}
cases, turns, vars_ = load()
TIER = tiers(task_accuracy(turns))
att = collections.defaultdict(lambda: [0, 0])   # (class, case, study) -> [hits, attempts]
for v in vars_:
    k = (v["case"], v["var"])
    if k in slot_class and v["is_set"]:
        r = live[k]; a = att[(slot_class[k], v["case"], v["study"])]
        a[1] += 1; a[0] += (v["turn"] == r["turn"] and v["value"] == r["wrong"])
G = {"weak": [s for s in STUDIES if TIER[s] == "weak"], "strong": [s for s in STUDIES if TIER[s] == "strong"],
     "pairedN": [N for T, N in PAIRS.values()], "pairedT": [T for T, N in PAIRS.values()]}
CLASSES = sorted({c for c, _, _ in att})
TASKS = {c: sorted({x for cc, x, _ in att if cc == c}) for c in CLASSES}


def rate(c, sample, g):
    h = sum(att[(c, x, s)][0] for x in sample for s in G[g]); n = sum(att[(c, x, s)][1] for x in sample for s in G[g])
    return h / n if n else float("nan")


def change(c, sample, a, b):
    ra = rate(c, sample, a)
    return rate(c, sample, b) / ra - 1 if ra else float("nan")


def interval(draws):
    d = sorted(x for x in draws if x == x)
    return (d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]) if d else (float("nan"), float("nan"))


rows = []
for a, b, label in (("weak", "strong", "weak to strong"), ("pairedN", "pairedT", "paired non-thinking to thinking")):
    for c in CLASSES:
        cs = TASKS[c]
        rng = random.Random(f"{label}|{c}")
        point = change(c, cs, a, b)
        lo, hi = interval(change(c, [rng.choice(cs) for _ in cs], a, b) for _ in range(N_DRAWS))
        dpoint = dlo = dhi = float("nan")
        if c != "K4":
            own, k4 = set(cs), set(TASKS["K4"])
            U = sorted(own | k4)
            rng = random.Random(f"{label}|{c}-K4")

            def diff(sample):
                return change(c, [x for x in sample if x in own], a, b) - change("K4", [x for x in sample if x in k4], a, b)

            dpoint = diff(U)
            dlo, dhi = interval(diff([rng.choice(U) for _ in U]) for _ in range(N_DRAWS))
        print(f"{label:32} {c}: {rate(c, cs, a) * 1000:.0f} -> {rate(c, cs, b) * 1000:.0f} per 1000, change {point:+.0%} "
              f"[{lo:+.0%}, {hi:+.0%}] ({len(cs)} tasks)"
              + ("" if c == "K4" else f" | minus K4 {dpoint:+.0%} [{dlo:+.0%}, {dhi:+.0%}]"))
        rows.append([label, c, len(cs), round(rate(c, cs, a) * 1000, 1), round(rate(c, cs, b) * 1000, 1),
                     *(round(x, 4) for x in (point, lo, hi, dpoint, dlo, dhi))])
write_csv("knowledge_change_ci.csv", ["contrast", "class", "tasks", "from_per_1000", "to_per_1000", "change", "change_lo",
                                      "change_hi", "minus_K4", "minus_K4_lo", "minus_K4_hi"], rows)
