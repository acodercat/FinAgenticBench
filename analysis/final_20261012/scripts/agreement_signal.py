"""Finding 8: agreement as a correctness signal, same model vs across vendors, on a like-for-like basis.

All comparisons use the thinking configurations only, so that reasoning mode does not enter the contrast.
Same model: two runs of one thinking configuration agree on a slot -> P(correct). Across vendors: the first run
of two thinking configurations from different vendors agree -> P(correct). Three vendors: the best thinking
configuration of each vendor, first run. Each signal reports coverage (share of candidate pairs that agree),
P(correct | agree) and case-resampled 95% intervals; the cross-vendor minus same-model difference in
P(correct | agree) is bootstrapped on the same case resamples. Writes tables/agreement.csv.
"""
import collections
import itertools
import random

from common import boot_rate, load, rate, write_csv
from groups import STUDIES, THINKING, VENDOR

cases, turns, vars_ = load()
S = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"]:
        S[(v["case"], v["turn"], v["var"])][v["study"]].append((v["run_id"], v["value"], bool(v["correct"])))
think = sorted(THINKING)
acc = collections.defaultdict(lambda: [0, 0])
for v in vars_:
    if v["is_set"]:
        acc[v["study"]][0] += bool(v["correct"]); acc[v["study"]][1] += 1
best = {}
for s in think:
    f = VENDOR[s]
    if f not in best or acc[s][0] / acc[s][1] > acc[best[f]][0] / acc[best[f]][1]:
        best[f] = s

# per-case (agree, correct_given_agree, pairs) counters per signal
sig = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))   # signal -> case -> [agree, agree&correct, pairs]
for k, m in S.items():
    case = k[0]
    for s in think:
        if s in m:
            for a, b in itertools.combinations(sorted(m[s]), 2):
                c = sig["same model, 2 runs agree (thinking)"][case]; c[2] += 1
                if a[1] == b[1]:
                    c[0] += 1; c[1] += a[2]
    for s1, s2 in itertools.combinations(think, 2):
        if VENDOR[s1] == VENDOR[s2] or s1 not in m or s2 not in m:
            continue
        a, b = sorted(m[s1])[0], sorted(m[s2])[0]
        c = sig["2 vendors agree (1 run each, thinking)"][case]; c[2] += 1
        if a[1] == b[1]:
            c[0] += 1; c[1] += a[2]
    vals = [sorted(m[s])[0] for s in best.values() if s in m]
    for a, b, d in itertools.combinations(vals, 3):
        c = sig["3 vendors agree"][case]; c[2] += 1
        if a[1] == b[1] == d[1]:
            c[0] += 1; c[1] += a[2]
    for s in think:
        r = m.get(s, [])
        if len(r) == 3:
            c = sig["3 runs identical (thinking)"][case]; c[2] += 1
            if len({x[1] for x in r}) == 1:
                c[0] += 1; c[1] += sum(x[2] for x in r) / 3

rows = []
for name, per in sig.items():
    cov = boot_rate({c: (x[0], x[2]) for c, x in per.items()})
    pc = boot_rate({c: (x[1], x[0]) for c, x in per.items()})
    n = sum(x[2] for x in per.values())
    print(f"{name:44} coverage {cov[0]:.1%} [{cov[1]:.1%},{cov[2]:.1%}] | P(correct|agree) {pc[0]:.2%} [{pc[1]:.2%},{pc[2]:.2%}] | n {n}")
    rows.append((name, round(cov[0], 6), round(cov[1], 6), round(cov[2], 6), round(pc[0], 6), round(pc[1], 6), round(pc[2], 6), n))

# paired difference: cross-vendor minus same-model P(correct|agree), same case resamples
A = sig["same model, 2 runs agree (thinking)"]; B = sig["2 vendors agree (1 run each, thinking)"]
cs = sorted(set(A) | set(B)); rng = random.Random(0)
def diff(sample):
    pa = rate(sum(A[c][1] for c in sample if c in A), sum(A[c][0] for c in sample if c in A))
    pb = rate(sum(B[c][1] for c in sample if c in B), sum(B[c][0] for c in sample if c in B))
    return pb - pa
draws = sorted(diff([rng.choice(cs) for _ in cs]) for _ in range(2000))
d0 = diff(cs)
print(f"cross-vendor minus same-model P(correct|agree): {d0:+.2%} [{draws[50]:+.2%}, {draws[1949]:+.2%}]")
rows.append(("difference: 2 vendors minus same model, P(correct|agree)", "", "", "", round(d0, 6), round(draws[50], 6), round(draws[1949], 6), ""))

# per cross-vendor pair, against the mean of the two members' own-run signal (quality-matched by construction)
print("\nper pair of thinking configurations (different vendors): P(correct|agree) cross vs own-runs mean")
pair_rows = []
for s1, s2 in itertools.combinations(think, 2):
    if VENDOR[s1] == VENDOR[s2]:
        continue
    cross = [0, 0]; own = {s1: [0, 0], s2: [0, 0]}
    for k, m in S.items():
        if s1 in m and s2 in m:
            a, b = sorted(m[s1])[0], sorted(m[s2])[0]
            if a[1] == b[1]:
                cross[0] += 1; cross[1] += a[2]
        for s in (s1, s2):
            for a, b in itertools.combinations(sorted(m.get(s, [])), 2):
                if a[1] == b[1]:
                    own[s][0] += 1; own[s][1] += a[2]
    pc_cross = rate(cross[1], cross[0]); pc_own = (rate(own[s1][1], own[s1][0]) + rate(own[s2][1], own[s2][0])) / 2
    pair_rows.append((STUDIES[s1][0], STUDIES[s2][0], round(pc_cross, 6), round(pc_own, 6), cross[0]))
    print(f"  {STUDIES[s1][0]:18} x {STUDIES[s2][0]:18} cross {pc_cross:.2%} | own {pc_own:.2%} | agreeing slots {cross[0]}")
print(f"pairs where cross > own: {sum(r[2] > r[3] for r in pair_rows)}/{len(pair_rows)}")
write_csv("agreement.csv", ["signal", "coverage", "cov_lo", "cov_hi", "p_correct_given_agree", "pc_lo", "pc_hi", "n"], rows)
write_csv("agreement_pairs.csv", ["study_a", "study_b", "p_correct_cross", "p_correct_own_mean", "n_agree_cross"], pair_rows)
