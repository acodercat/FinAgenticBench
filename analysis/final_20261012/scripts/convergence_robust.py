"""Robustness of the convergence results to who counts in the consensus, and a slot fixed-effects model.

The consensus error on a slot is the most common wrong value among the other configurations, counted when at least
two of them gave it (convergence.py). Three exclusion rules for "other":
  self     the configuration alone (as in convergence.py);
  sibling  the configuration and the other reasoning mode of the same model;
  vendor   every configuration of the configuration's vendor.
For each rule: the share of each tier's wrong values equal to the consensus error, the Spearman correlation of a
configuration's share with its task accuracy, and the within-slot tier contrasts with task-resampled intervals.

Model-based test (sibling rule): one row per configuration and slot on which the configuration erred; outcome the
share of its wrong values on the slot that equal the consensus error, weighted by the number of those wrong values.
Slot fixed effects absorb how hard a slot is and how many readings it admits, so each coefficient compares
configurations on the same slots. Spec A: task accuracy (per 10 points). Spec B: thinking, with model fixed effects
(the within-model contrast; only the six paired models carry it). Intervals: 1,000 draws resampling tasks.
Writes tables/convergence_robust.csv.
"""
import collections
import random
import statistics

import numpy as np

from common import load, write_csv
from groups import MODE, MODEL, STUDIES, VENDOR, task_accuracy, tiers

cases, turns, vars_ = load()
acc = task_accuracy(turns); T = tiers(acc)
S = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"]:
        S[(v["case"], v["turn"], v["var"])][v["study"]].append(v)
RULES = {
    "self": lambda s: {s},
    "sibling": lambda s: {x for x in STUDIES if MODEL[x] == MODEL[s]},
    "vendor": lambda s: {x for x in STUDIES if VENDOR[x] == VENDOR[s]},
}


def crowd(m, exclude):
    cnt = collections.Counter(); who = collections.defaultdict(set)
    for s, xs in m.items():
        if s in exclude:
            continue
        for x in xs:
            if not x["correct"]:
                cnt[x["value"]] += 1; who[x["value"]].add(s)
    if not cnt:
        return None
    val, _ = cnt.most_common(1)[0]
    return val if len(who[val]) >= 2 else None


def spearman(a, b):
    def rk(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rk(a), rk(b); n = len(a); m = (n - 1) / 2
    return sum((ra[i] - m) * (rb[i] - m) for i in range(n)) / (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** 0.5


def case_boot(by_case, n=2000, seed=0):
    cs = sorted(by_case); rng = random.Random(seed)
    allv = [d for c in cs for d in by_case[c]]
    draws = sorted(statistics.mean(d for c in (rng.choice(cs) for _ in cs) for d in by_case[c]) for _ in range(n))
    return statistics.mean(allv), draws[int(0.025 * n)], draws[int(0.975 * n) - 1], len(allv)


# per rule: (slot, study) -> (matches, wrong values)
cell = {}
for rule, ex in RULES.items():
    c = {}
    for k, m in S.items():
        for s, xs in m.items():
            w = [x for x in xs if not x["correct"]]
            if w:
                val = crowd(m, ex(s))
                c[(k, s)] = (sum(x["value"] == val for x in w), len(w))
    cell[rule] = c

rows = []
order = sorted(STUDIES, key=lambda s: -acc[s])
for rule, c in cell.items():
    per = collections.defaultdict(lambda: [0, 0])
    for (k, s), (h, n) in c.items():
        per[s][0] += h; per[s][1] += n
    rho = spearman([acc[s] for s in order], [per[s][0] / per[s][1] for s in order])
    tier = {t: sum(per[s][0] for s in STUDIES if T[s] == t) / sum(per[s][1] for s in STUDIES if T[s] == t) for t in ("strong", "mid", "weak")}
    print(f"[{rule}] wrong value = consensus error: strong {tier['strong']:.1%}, mid {tier['mid']:.1%}, weak {tier['weak']:.1%}; "
          f"Spearman with accuracy {rho:.2f}")
    rows.append((rule, "tier share", "strong/mid/weak", round(tier["strong"], 6), round(tier["mid"], 6), round(tier["weak"], 6), ""))
    rows.append((rule, "Spearman with accuracy", "", round(rho, 6), "", "", ""))
    for hi, lo in (("strong", "weak"), ("strong", "mid"), ("mid", "weak")):
        by_case = collections.defaultdict(list)
        for k in S:
            r = {}
            for t in (hi, lo):
                hs = [c[(k, s)] for s in STUDIES if T[s] == t and (k, s) in c]
                if hs:
                    r[t] = sum(h for h, _ in hs) / sum(n for _, n in hs)
            if len(r) == 2:
                by_case[k[0]].append(r[hi] - r[lo])
        mean, l, h, n = case_boot(by_case)
        print(f"   within slot {hi} - {lo}: {mean:+.1%} [{l:+.1%}, {h:+.1%}] slots {n}")
        rows.append((rule, f"within slot {hi} - {lo}", n, round(mean, 6), round(l, 6), round(h, 6), ""))

# ---- slot fixed-effects model, sibling rule -----------------------------------------------------------------
MODELS = sorted(set(MODEL.values()))   # all eight; the two thinking-only models then carry no thinking contrast
by_task = collections.defaultdict(list)    # task -> list of (slot, study, y, w)
for (k, s), (h, n) in cell["sibling"].items():
    by_task[k[0]].append((k, s, h / n, n))


def design(rec, spec):
    if spec == "A":
        return [acc[r[1]] * 10 for r in rec], ["accuracy, per 10 points"]
    cols = [[1.0 if MODE[r[1]] == "T" else 0.0 for r in rec]]
    cols += [[1.0 if MODEL[r[1]] == mdl else 0.0 for r in rec] for mdl in MODELS[1:]]
    return list(zip(*cols)), ["thinking"]


def fit(records, spec):
    """Weighted least squares with slot fixed effects by within-slot demeaning (Frisch-Waugh-Lovell)."""
    X, names = design(records, spec)
    X = np.asarray(X, float).reshape(len(records), -1)
    y = np.array([r[2] for r in records]); w = np.array([r[3] for r in records], float)
    ids = {}
    g = np.array([ids.setdefault(r[0], len(ids)) for r in records])
    sw = np.bincount(g, w)
    yd = y - (np.bincount(g, w * y) / sw)[g]
    Xd = X - np.stack([np.bincount(g, w * X[:, j]) / sw for j in range(X.shape[1])], 1)[g]
    keep = np.bincount(g)[g] > 1
    Xd, yd, ww = Xd[keep], yd[keep], w[keep]
    beta = np.linalg.lstsq(Xd * np.sqrt(ww)[:, None], yd * np.sqrt(ww), rcond=None)[0]
    return beta[0], int(keep.sum()), len(set(g[keep]))


tasks = sorted(by_task); rng = random.Random(1)
for spec, label in (("A", "accuracy, per 10 points"), ("B", "thinking, within model")):
    recs = [r for t in tasks for r in by_task[t]]
    b0, nobs, nslots = fit(recs, spec)
    draws = []
    for _ in range(1000):
        sample = [rng.choice(tasks) for _ in tasks]
        recs_b = [((i,) + r[0], r[1], r[2], r[3]) for i, t in enumerate(sample) for r in by_task[t]]   # duplicated tasks get distinct slot ids
        draws.append(fit(recs_b, spec)[0])
    draws.sort()
    lo, hi = draws[25], draws[974]
    print(f"slot fixed effects, sibling rule, {label}: {b0 * 100:+.1f} points [{lo * 100:+.1f}, {hi * 100:+.1f}] "
          f"({nobs} configuration-slot rows on {nslots} slots)")
    rows.append(("sibling", f"slot FE: {label}", nobs, round(b0, 6), round(lo, 6), round(hi, 6), nslots))
write_csv("convergence_robust.csv", ["rule", "measure", "n", "estimate", "lo_or_mid", "hi_or_weak", "n_slots"], rows)
