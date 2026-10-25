"""Finding 9: retrying the same model vs switching to another vendor, at the case level.

Per study: single-run accuracy; ORACLE best-of-2 and best-of-3 of its own runs and oracle best-of-2 with one
run from the best other-vendor study (an upper bound: the case counts if any candidate run passes, which no
deployment can know without the answer key); and one EXECUTABLE rule, "accept when two candidates agree on
every output variable of the case, otherwise abstain": coverage and accuracy among accepted cases, for two own
runs vs one own run plus one run of the best other-vendor study. Then, on slots where >=3 vendors share one
wrong value, which vendor is entirely correct. Writes tables/retry.csv.
"""
import collections
import itertools

from common import load, runs, slots, write_csv
from groups import STUDIES, VENDOR, label

cases, turns, vars_ = load()
R = runs(turns)
case_ok = collections.defaultdict(dict)
for (s, c, rid), ts in R.items():
    case_ok[(s, c)][rid] = all(t["ok"] for t in ts)
CASES = sorted({c for _, c in case_ok}); studies = sorted(STUDIES)
single = {s: sum(sum(case_ok[(s, c)].values()) / 3 for c in CASES) / len(CASES) for s in studies}
top = max(studies, key=single.get)
# the executable rule needs each run's full output vector per case
vec = collections.defaultdict(dict)   # (study, case, run_id) -> {(turn, var): value}
for v in vars_:
    vec[(v["study"], v["case"], v["run_id"])][(v["turn"], v["var"])] = v["value"] if v["is_set"] else None
def agree_rule(pairs):
    """pairs: list of ((s1,c,r1),(s2,c,r2)) candidate pairs over all cases. Returns coverage, accuracy among accepted."""
    acc_ = cov = 0
    for a, b in pairs:
        if vec[a] == vec[b]:
            cov += 1; acc_ += case_ok[(a[0], a[1])][a[2]]
    return cov / len(pairs), (acc_ / cov if cov else float("nan"))
rows = []
print(f"{'study':40} {'single':>7} {'oracle b-o-2':>12} {'oracle b-o-3':>12} {'oracle +1 other':>15} | agree-rule own2 cov/acc | own1+other1 cov/acc")
for s in sorted(studies, key=lambda s: -single[s]):
    b2 = sum(sum(any(p) for p in itertools.combinations(case_ok[(s, c)].values(), 2)) / 3 for c in CASES) / len(CASES)
    b3 = sum(any(case_ok[(s, c)].values()) for c in CASES) / len(CASES)
    other = top if VENDOR[s] != VENDOR[top] else max((x for x in studies if VENDOR[x] != VENDOR[s]), key=single.get)
    cross = sum(sum(any((a, b)) for a in case_ok[(s, c)].values() for b in case_ok[(other, c)].values()) / 9 for c in CASES) / len(CASES)
    own_pairs = [((s, c, r1), (s, c, r2)) for c in CASES for r1, r2 in itertools.combinations(sorted(case_ok[(s, c)]), 2)]
    cross_pairs = [((s, c, r1), (other, c, r2)) for c in CASES for r1 in sorted(case_ok[(s, c)]) for r2 in sorted(case_ok[(other, c)])]
    oc, oa = agree_rule(own_pairs); xc, xa = agree_rule(cross_pairs)
    print(f"{label(s):40} {single[s]:7.1%} {b2:12.1%} {b3:12.1%} {cross:15.1%}  (other={STUDIES[other][0]}) | {oc:.1%} / {oa:.1%} | {xc:.1%} / {xa:.1%}")
    rows.append((s, round(single[s], 6), round(b2, 6), round(b3, 6), round(cross, 6), other, round(oc, 6), round(oa, 6), round(xc, 6), round(xa, 6)))
# consensus breakers on shared-wrong slots
S = slots(vars_); breaker = collections.Counter(); present = collections.Counter(); nshared = 0
for k, xs in S.items():
    wrongs = [x for x in xs if not x["correct"]]
    if len(wrongs) < 3:
        continue
    val, n = collections.Counter(x["value"] for x in wrongs).most_common(1)[0]
    fw = {VENDOR[x["study"]] for x in wrongs if x["value"] == val}
    if n < 3 or len(fw) < 3:
        continue
    nshared += 1
    for f in {VENDOR[x["study"]] for x in xs}:
        present[f] += 1
        if f not in fw and all(x["correct"] for x in xs if VENDOR[x["study"]] == f):
            breaker[f] += 1
print(f"\nslots where >=3 vendors share one wrong value: {nshared}; vendor entirely correct there:")
for f in sorted(present, key=lambda f: -breaker[f] / present[f]):
    print(f"  {f:16} {breaker[f]}/{present[f]} = {breaker[f] / present[f]:.1%}")
    rows.append((f"BREAKER:{f}", round(breaker[f] / present[f], 6), breaker[f], present[f], "", "", "", "", "", ""))
write_csv("retry.csv", ["study_or_vendor", "single_or_breaker_rate", "oracle_best_of_2", "oracle_best_of_3", "oracle_plus_one_other", "other_study",
                        "agree_own2_coverage", "agree_own2_accuracy", "agree_cross_coverage", "agree_cross_accuracy"], rows)
