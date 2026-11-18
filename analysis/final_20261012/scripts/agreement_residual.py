"""What agreement lets through: the wrong values that pass each agreement signal, split into shared and unshared.

Uses the candidate pairs of agreement_signal.py exactly (thinking configurations only; same model = two runs of one
configuration; two vendors = the first run of two thinking configurations from different vendors; three vendors =
the first run of the most accurate thinking configuration of each vendor; three runs = all three runs of one
configuration identical). An agreeing candidate set is wrong when the agreed value fails verification; it is shared
when the agreed value is the slot's shared wrong value (common.modal_shared: the slot's most common wrong value,
given by at least three configurations).

For each signal it reports, with case-resampled 95% intervals:
  - the share of agreeing-but-wrong candidates whose value is the shared wrong value;
  - the residual error rate P(wrong | agree), split into its shared and unshared parts;
and, for two vendors against the same model, the relative change in each part on the same case resamples.
Two circularity guards follow, because the agreeing configurations count toward the three that make a value shared:
the agreed value must also be given by >= 2 configurations outside the agreeing set, or, strictest, by >= 2
configurations of vendors outside it (with a base rate under the same rule).
A second table gives the miss rate of each check: of the wrong values of a first candidate, the share a second
candidate reproduced (a second run of the same configuration, or the first run of a thinking configuration of
another vendor), split by whether >= 2 configurations of vendors outside the pair also gave the value, and by the
modal_shared definition. Both candidates must have stored the output. Writes tables/agreement_miss.csv.
Writes tables/agreement_residual.csv. Rows other than the four signals reuse columns: relative-change rows put the
change and its interval in shared_share / _lo / _hi; convergence and base-rate rows put the denominator in
n_agree_wrong, the numerator in n_agree_wrong_shared and the share in shared_share.
"""
import collections
import itertools
import random

from common import boot_rate, load, modal_shared, rate, write_csv
from groups import STUDIES, THINKING, VENDOR

cases, turns, vars_ = load()
SHARED = modal_shared(vars_)
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

SIGNALS = ["same model, 2 runs agree (thinking)", "2 vendors agree (1 run each, thinking)", "3 vendors agree",
           "3 runs identical (thinking)"]
# signal -> case -> [agree, agree & wrong, agree & wrong & shared]
sig = {name: collections.defaultdict(lambda: [0, 0, 0]) for name in SIGNALS}


def add(name, case, slot, group):
    if len({x[1] for x in group}) != 1:
        return
    c = sig[name][case]; c[0] += 1
    if not group[0][2]:
        c[1] += 1; c[2] += SHARED.get(slot) == group[0][1]


for k, m in S.items():
    case = k[0]
    for s in think:
        for a, b in itertools.combinations(sorted(m.get(s, [])), 2):
            add(SIGNALS[0], case, k, (a, b))
        if len(m.get(s, [])) == 3:
            add(SIGNALS[3], case, k, sorted(m[s]))
    for s1, s2 in itertools.combinations(think, 2):
        if VENDOR[s1] != VENDOR[s2] and s1 in m and s2 in m:
            add(SIGNALS[1], case, k, (sorted(m[s1])[0], sorted(m[s2])[0]))
    vals = [sorted(m[s])[0] for s in best.values() if s in m]
    for trio in itertools.combinations(vals, 3):
        add(SIGNALS[2], case, k, trio)

print(f"three-vendor configurations (most accurate thinking configuration per vendor, chosen on these results): "
      f"{', '.join(STUDIES[s][0] for s in sorted(best.values()))}")
rows = []
for name in SIGNALS:
    per = sig[name]
    agree = sum(x[0] for x in per.values()); wrong = sum(x[1] for x in per.values()); sh = sum(x[2] for x in per.values())
    share = boot_rate({c: (x[2], x[1]) for c, x in per.items() if x[1]})
    err = boot_rate({c: (x[1], x[0]) for c, x in per.items()})
    err_sh = boot_rate({c: (x[2], x[0]) for c, x in per.items()})
    err_un = boot_rate({c: (x[1] - x[2], x[0]) for c, x in per.items()})
    print(f"{name:40} agreeing {agree:6} | wrong {wrong:5} | shared among wrong {share[0]:.1%} [{share[1]:.1%},{share[2]:.1%}]"
          f" | P(wrong|agree) {err[0]:.2%}: shared {err_sh[0]:.2%} [{err_sh[1]:.2%},{err_sh[2]:.2%}],"
          f" unshared {err_un[0]:.2%} [{err_un[1]:.2%},{err_un[2]:.2%}]")
    rows.append((name, agree, wrong, sh, *(round(x, 6) for x in (*share, *err, *err_sh, *err_un))))

# Two vendors against the same model: relative change in each part of the residual error, same case resamples.
A, B = sig[SIGNALS[0]], sig[SIGNALS[1]]
cs = sorted(set(A) | set(B))


def change(sample, part):
    def r(per):
        num = sum(per[c][2] if part == "shared" else per[c][1] - per[c][2] for c in sample if c in per)
        return rate(num, sum(per[c][0] for c in sample if c in per))
    ra, rb = r(A), r(B)
    return (rb - ra) / ra if ra else float("nan")


rng = random.Random(0)
samples = [[rng.choice(cs) for _ in cs] for _ in range(2000)]
for part in ("shared", "unshared"):
    d0 = change(cs, part)
    draws = sorted(x for x in (change(s, part) for s in samples) if x == x)
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
    print(f"2 vendors vs same model, {part:8} part of P(wrong|agree): {d0:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    rows.append((f"relative change, 2 vendors vs same model, {part} part", "", "", "", round(d0, 6), round(lo, 6),
                 round(hi, 6), *[""] * 9))

# Guard against circularity: the agreeing configurations themselves count toward the >= 3 configurations that make
# a value shared. Count instead an agreed wrong value as convergent when at least two configurations OUTSIDE the
# agreeing set gave the same value on the slot (the consensus-error bar of convergence.py, applied to the set).
WRONG_BY = collections.defaultdict(lambda: collections.defaultdict(set))   # slot -> value -> configurations
for v in vars_:
    if v["is_set"] and not v["correct"]:
        WRONG_BY[(v["case"], v["turn"], v["var"])][v["value"]].add(v["study"])
outside = {name: [0, 0] for name in SIGNALS}   # [agree & wrong, of which given by >= 2 outside configurations]


def add_out(name, slot, group, members):
    if len({x[1] for x in group}) == 1 and not group[0][2]:
        o = outside[name]; o[0] += 1
        o[1] += len(WRONG_BY[slot][group[0][1]] - set(members)) >= 2


for k, m in S.items():
    for s in think:
        for a, b in itertools.combinations(sorted(m.get(s, [])), 2):
            add_out(SIGNALS[0], k, (a, b), [s])
        if len(m.get(s, [])) == 3:
            add_out(SIGNALS[3], k, sorted(m[s]), [s])
    for s1, s2 in itertools.combinations(think, 2):
        if VENDOR[s1] != VENDOR[s2] and s1 in m and s2 in m:
            add_out(SIGNALS[1], k, (sorted(m[s1])[0], sorted(m[s2])[0]), [s1, s2])
    present = [s for s in best.values() if s in m]
    for trio in itertools.combinations(present, 3):
        add_out(SIGNALS[2], k, [sorted(m[s])[0] for s in trio], list(trio))
print("\nagreed wrong values also given by >= 2 configurations outside the agreeing set (circularity-free):")
for name in SIGNALS:
    o = outside[name]
    print(f"  {name:40} {o[1]}/{o[0]} = {rate(o[1], o[0]):.1%}")
    rows.append((f"outside-set convergence: {name}", "", o[0], o[1], round(rate(o[1], o[0]), 6), *[""] * 11))

# Strictest guard: only configurations of vendors outside the agreeing set count, so the other mode or the sibling
# model of an agreeing vendor cannot make the value convergent. The base rate uses the same rule for a single wrong
# value of a thinking configuration (configurations of other vendors only).
strict = {SIGNALS[0]: [0, 0], SIGNALS[1]: [0, 0]}


def add_strict(name, slot, group, members):
    if len({x[1] for x in group}) == 1 and not group[0][2]:
        vend = {VENDOR[s] for s in members}
        o = strict[name]; o[0] += 1
        o[1] += sum(VENDOR[s] not in vend for s in WRONG_BY[slot][group[0][1]]) >= 2


for k, m in S.items():
    for s in think:
        for a, b in itertools.combinations(sorted(m.get(s, [])), 2):
            add_strict(SIGNALS[0], k, (a, b), [s])
    for s1, s2 in itertools.combinations(think, 2):
        if VENDOR[s1] != VENDOR[s2] and s1 in m and s2 in m:
            add_strict(SIGNALS[1], k, (sorted(m[s1])[0], sorted(m[s2])[0]), [s1, s2])
base_v = [0, 0]
for v in vars_:
    if v["is_set"] and not v["correct"] and v["study"] in THINKING:
        k = (v["case"], v["turn"], v["var"]); base_v[0] += 1
        base_v[1] += sum(VENDOR[s] != VENDOR[v["study"]] for s in WRONG_BY[k][v["value"]]) >= 2
print("agreed wrong values also given by >= 2 configurations of vendors outside the agreeing set (strictest):")
for name, o in strict.items():
    print(f"  {name:40} {o[1]}/{o[0]} = {rate(o[1], o[0]):.1%}")
    rows.append((f"outside-vendor convergence: {name}", "", o[0], o[1], round(rate(o[1], o[0]), 6), *[""] * 11))
print(f"  base rate, one thinking wrong value, same rule: {base_v[1]}/{base_v[0]} = {rate(base_v[1], base_v[0]):.1%}")
rows.append(("outside-vendor base rate: thinking wrong values", "", base_v[0], base_v[1],
             round(rate(base_v[1], base_v[0]), 6), *[""] * 11))

# Base rate: share of all wrong values of the thinking configurations that are the slot's shared wrong value.
base = [0, 0]
for v in vars_:
    if v["is_set"] and not v["correct"] and v["study"] in THINKING:
        base[0] += 1; base[1] += SHARED.get((v["case"], v["turn"], v["var"])) == v["value"]
print(f"base rate: shared share of all thinking-configuration wrong values {base[1]}/{base[0]} = {rate(base[1], base[0]):.1%}")
rows.append(("base rate: shared share of thinking wrong values", "", base[0], base[1], round(rate(base[1], base[0]), 6),
             *[""] * 11))

# Miss rate: of the wrong values of a first candidate, the share the second candidate reproduced.
# check -> rule -> class -> case -> [reproduced, wrong first candidates]
miss = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
CHECKS = ["second run of the same model", "one run of a second vendor"]
RULES = ["given by >= 2 configurations of other vendors", "shared (>= 3 configurations)"]


def add_miss(check, slot, first, second, vendors):
    if first[2]:
        return
    hit = first[1] == second[1]
    outside = sum(VENDOR[s] not in vendors for s in WRONG_BY[slot][first[1]]) >= 2
    for rule, shared in ((RULES[0], outside), (RULES[1], SHARED.get(slot) == first[1])):
        for cls in ("all", "shared" if shared else "unshared"):
            c = miss[(check, rule, cls)][slot[0]]; c[0] += hit; c[1] += 1


for k, m in S.items():
    for s in think:
        for a, b in itertools.permutations(sorted(m.get(s, [])), 2):
            add_miss(CHECKS[0], k, a, b, {VENDOR[s]})
    for s1, s2 in itertools.permutations(think, 2):
        if VENDOR[s1] != VENDOR[s2] and s1 in m and s2 in m:
            add_miss(CHECKS[1], k, sorted(m[s1])[0], sorted(m[s2])[0], {VENDOR[s1], VENDOR[s2]})
print("\nmiss rate: wrong values of a first candidate that the second candidate reproduced")
miss_rows = []
for check in CHECKS:
    for rule in RULES:
        for cls in ("all", "shared", "unshared"):
            per = miss[(check, rule, cls)]
            p, lo, hi = boot_rate({c: tuple(x) for c, x in per.items()})
            n_hit = sum(x[0] for x in per.values()); n = sum(x[1] for x in per.values())
            print(f"  {check:30} | {rule:46} | {cls:8} {n_hit:5}/{n:5} = {p:.1%} [{lo:.1%}, {hi:.1%}]")
            miss_rows.append((check, rule, cls, n, n_hit, round(p, 6), round(lo, 6), round(hi, 6)))
write_csv("agreement_miss.csv", ["check", "rule", "class", "n_wrong", "n_reproduced", "miss_rate", "lo", "hi"], miss_rows)

write_csv("agreement_residual.csv",
          ["signal", "n_agree", "n_agree_wrong", "n_agree_wrong_shared", "shared_share", "shared_share_lo",
           "shared_share_hi", "p_wrong", "p_wrong_lo", "p_wrong_hi", "p_wrong_shared", "p_wrong_shared_lo",
           "p_wrong_shared_hi", "p_wrong_unshared", "p_wrong_unshared_lo", "p_wrong_unshared_hi"], rows)
