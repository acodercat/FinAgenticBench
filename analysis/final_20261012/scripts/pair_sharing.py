"""Do two configurations share wrong values more when they share a vendor?

For every pair of the 14 configurations, on each slot where both erred, every wrong attempt of one is paired with
every wrong attempt of the other and the share of pairs with an identical wrong value is recorded (attempt level);
the slot-level share (any identical wrong value on the slot) is kept beside it. Pairs are classed as the two modes
of one model, two models of one vendor, or two vendors, and the class difference is estimated with and without
adjustment for the pair's mean accuracy and accuracy gap (least squares, case-resampled intervals). If shared
wrong values came from shared training material, same-vendor pairs should share more at equal accuracy; if they
came from the task and library defaults every model meets, the three classes should not differ.
Writes tables/pair_sharing.csv (one row per pair) and tables/pair_sharing_summary.csv.
"""
import collections
import itertools
import random

from common import load, write_csv
from groups import MODEL, PAIRS, STUDIES, VENDOR, task_accuracy

cases, turns, vars_ = load()
acc = task_accuracy(turns)
SAME_MODEL = {frozenset(p) for p in PAIRS.values()}


def category(a, b):
    if frozenset((a, b)) in SAME_MODEL:
        return "same model"
    if VENDOR[a] == VENDOR[b]:
        return "same vendor"
    return "cross vendor"


# wrong attempts per slot per study
W = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"] and not v["correct"]:
        W[(v["case"], v["turn"], v["var"])][v["study"]].append(v["value"])

studies = sorted(STUDIES, key=lambda s: -acc[s])
pairs = list(itertools.combinations(studies, 2))
# per pair, per case: (identical attempt pairs, attempt pairs, slots with identical value, slots both wrong)
per_case = {p: collections.defaultdict(lambda: [0, 0, 0, 0]) for p in pairs}
for k, m in W.items():
    for a, b in pairs:
        if a in m and b in m:
            ident = sum(x == y for x in m[a] for y in m[b])
            c = per_case[(a, b)][k[0]]
            c[0] += ident; c[1] += len(m[a]) * len(m[b]); c[2] += ident > 0; c[3] += 1


def pair_stats(p, sample=None):
    d = per_case[p]
    cs = sample if sample is not None else d.keys()
    s = [0, 0, 0, 0]
    for c in cs:
        if c in d:
            for i in range(4):
                s[i] += d[c][i]
    return s


rows = []
for a, b in pairs:
    s = pair_stats((a, b))
    rows.append([a, b, STUDIES[a][0] + (" T" if STUDIES[a][1] == "T" else " N"), STUDIES[b][0] + (" T" if STUDIES[b][1] == "T" else " N"),
                 category(a, b), round((acc[a] + acc[b]) / 2, 6), round(abs(acc[a] - acc[b]), 6),
                 s[1], round(s[0] / s[1], 6) if s[1] else "", s[3], round(s[2] / s[3], 6) if s[3] else ""])
write_csv("pair_sharing.csv", ["study_a", "study_b", "config_a", "config_b", "category", "mean_accuracy", "accuracy_gap",
                               "n_attempt_pairs", "p_identical_attempts", "n_slots_both_wrong", "p_identical_slots"], rows)

# ---- class means and an accuracy-adjusted class difference ---------------------------------------------------
CATS = ("same model", "same vendor", "cross vendor")


def solve(X, y, w):
    """Weighted least squares by normal equations; X rows are [1, same_model, same_vendor, mean_acc, gap]."""
    k = len(X[0])
    A = [[sum(w[i] * X[i][r] * X[i][c] for i in range(len(X))) for c in range(k)] for r in range(k)]
    bvec = [sum(w[i] * X[i][r] * y[i] for i in range(len(X))) for r in range(k)]
    for col in range(k):                      # Gaussian elimination with partial pivoting
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]; bvec[col], bvec[piv] = bvec[piv], bvec[col]
        for r in range(k):
            if r != col and A[col][col]:
                f = A[r][col] / A[col][col]
                A[r] = [A[r][c] - f * A[col][c] for c in range(k)]; bvec[r] -= f * bvec[col]
    return [bvec[r] / A[r][r] for r in range(k)]


def estimates(sample=None):
    """Class rates (pooled attempt pairs) and the adjusted same-model / same-vendor effects relative to cross-vendor."""
    pooled = {c: [0, 0] for c in CATS}
    X, y, w = [], [], []
    for p in pairs:
        s = pair_stats(p, sample)
        if not s[1]:
            continue
        c = category(*p); pooled[c][0] += s[0]; pooled[c][1] += s[1]
        X.append([1.0, float(c == "same model"), float(c == "same vendor"), (acc[p[0]] + acc[p[1]]) / 2, abs(acc[p[0]] - acc[p[1]])])
        y.append(s[0] / s[1]); w.append(float(s[1]))
    beta = solve(X, y, w)
    return {c: pooled[c][0] / pooled[c][1] if pooled[c][1] else float("nan") for c in CATS}, beta


point, beta = estimates()
rng = random.Random(0); cs = sorted(cases)
draws = [estimates([rng.choice(cs) for _ in cs]) for _ in range(2000)]   # a case drawn twice counts twice


def ci_beta(i):
    xs = sorted(d[1][i] for d in draws)
    return xs[50], xs[1949]


summary = []
print("share of wrong-attempt pairs with an identical wrong value, slots where both configurations erred:")
for c in CATS:
    xs = sorted(d[0][c] for d in draws); lo, hi = xs[50], xs[1949]
    n = sum(1 for p in pairs if category(*p) == c)
    print(f"  {c:12} {point[c]:.1%} [{lo:.1%}, {hi:.1%}]  pairs {n}")
    summary.append([c, n, round(point[c], 6), round(lo, 6), round(hi, 6)])
for c in ("same model", "same vendor"):
    xs = sorted(d[0][c] - d[0]["cross vendor"] for d in draws)
    summary.append([f"{c} minus cross vendor, unadjusted", "", round(point[c] - point["cross vendor"], 6), round(xs[50], 6), round(xs[1949], 6)])
    print(f"  {c} minus cross vendor, unadjusted: {point[c] - point['cross vendor']:+.1%} [{xs[50]:+.1%}, {xs[1949]:+.1%}]")
names = ["intercept", "same model", "same vendor", "mean accuracy", "accuracy gap"]
for i, nm in enumerate(names):
    lo, hi = ci_beta(i)
    summary.append([f"adjusted: {nm}", "", round(beta[i], 6), round(lo, 6), round(hi, 6)])
    print(f"  adjusted coefficient {nm:14} {beta[i]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
# each same-vendor pair against the cross-vendor pairs of similar mean accuracy (within 10 points)
print("same-vendor pairs against cross-vendor pairs of similar mean accuracy:")
for a, b in pairs:
    if category(a, b) != "same vendor":
        continue
    s = pair_stats((a, b)); m = (acc[a] + acc[b]) / 2
    peers = [pair_stats(q) for q in pairs if category(*q) == "cross vendor" and abs((acc[q[0]] + acc[q[1]]) / 2 - m) <= 0.1]
    peer = sum(x[0] for x in peers) / sum(x[1] for x in peers)
    print(f"  {STUDIES[a][0]} {STUDIES[a][1]} x {STUDIES[b][0]} {STUDIES[b][1]}: {s[0] / s[1]:.1%} vs {peer:.1%} over {len(peers)} cross-vendor pairs")
    summary.append([f"matched: {STUDIES[a][0]} {STUDIES[a][1]} x {STUDIES[b][0]} {STUDIES[b][1]}", len(peers), round(s[0] / s[1], 6), round(peer, 6), ""])
write_csv("pair_sharing_summary.csv", ["term", "n_pairs", "estimate", "ci_lo", "ci_hi"], summary)
