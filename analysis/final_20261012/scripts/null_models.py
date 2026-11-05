"""How much of the concentration of errors on the same slots survives a null stronger than independence, and
how many distinct wrong values a tier produces when it errs.

Part 1, on the slots every configuration attempted (a configuration fails a slot when any of its runs stored a
wrong value there, as in shared_errors.py). The independence baseline in shared_errors.py gives each configuration
one error rate for all slots (null A, recomputed here on the same rates). Null B lets the rate vary by task family
and turn position; null C lets it vary by task, with the slot itself left out of its own task rate. Each null gives
the expected number of slots on which k configurations fail (Poisson-binomial, exact) and simulated 95% bands for
the never-wrong, >=3 and >=6 counts. What null C cannot absorb is concentration within a task.

Part 2, wrong-value diversity at matched counts: on slots where a tier produced exactly n wrong attempts
(n = 2..6), the mean number of distinct wrong values, by tier. Fewer distinct values at the same n means the
tier's erring configurations gave the same value more often; matching on n removes the effect of the strong tier
simply erring less. Writes tables/null_models.csv and tables/wrong_value_diversity.csv.
"""
import collections
import random

from common import load, slots, write_csv
from groups import STUDIES, task_accuracy, tiers

cases, turns, vars_ = load()
S = {k: v for k, v in slots(vars_).items() if len(v) >= 30}
studies = sorted(STUDIES)
T = tiers(task_accuracy(turns))


def fails(xs):
    m = collections.defaultdict(bool)
    for x in xs:
        m[x["study"]] |= not x["correct"]
    return m


FS = {k: fails(xs) for k, xs in S.items()}
F = {k: f for k, f in FS.items() if len(f) == 14}            # slots attempted by every configuration
print(f"slots attempted by all 14 configurations: {len(F)} (rates for null A use all {len(FS)} slots, as shared_errors.py)")
obs_k = collections.Counter(sum(f.values()) for f in F.values())


def poisson_binomial(ps):
    dist = [1.0]
    for p in ps:
        dist = [(dist[i] if i < len(dist) else 0) * (1 - p) + (dist[i - 1] if i > 0 else 0) * p for i in range(len(dist) + 1)]
    return dist


def stratum_rates(key):
    """Per-configuration failure rate within each stratum: key(slot) -> {study: rate}."""
    num = collections.defaultdict(collections.Counter); den = collections.defaultdict(collections.Counter)
    for k, f in F.items():
        st = key(k)
        for s, w in f.items():
            num[st][s] += w; den[st][s] += 1
    return {st: {s: num[st][s] / den[st][s] for s in studies} for st in num}


def expected_k(prob_of):
    """prob_of(slot) -> [p_1..p_14]; expected count per k and simulated 95% bands for three summaries."""
    exp = [0.0] * 15
    P = {k: prob_of(k) for k in F}
    for ps in P.values():
        for i, q in enumerate(poisson_binomial(ps)):
            exp[i] += q
    rng = random.Random(0); sims = {0: [], 3: [], 6: []}
    for _ in range(1000):
        c = collections.Counter(sum(rng.random() < p for p in ps) for ps in P.values())
        sims[0].append(c[0]); sims[3].append(sum(v for k, v in c.items() if k >= 3)); sims[6].append(sum(v for k, v in c.items() if k >= 6))
    return exp, {k: (sorted(v)[25], sorted(v)[974]) for k, v in sims.items()}


num, den = collections.Counter(), collections.Counter()
for f in FS.values():
    for s, w in f.items():
        num[s] += w; den[s] += 1
rateA = {s: num[s] / den[s] for s in studies}
rateB = stratum_rates(lambda k: (cases[k[0]]["family"], k[1]))
by_case = collections.defaultdict(list)
for k in F:
    by_case[k[0]].append(k)


def rateC(k):
    out = []
    for s in studies:
        others = [F[j][s] for j in by_case[k[0]] if j != k]
        out.append(sum(others) / len(others) if others else rateA[s])
    return out


nulls = [("A: one rate per configuration", lambda k: [rateA[s] for s in studies]),
         ("B: rate per configuration within task family and turn", lambda k: [rateB[(cases[k[0]]["family"], k[1])][s] for s in studies]),
         ("C: rate per configuration within the task, slot left out", rateC)]
obs = {0: obs_k[0], 3: sum(v for k, v in obs_k.items() if k >= 3), 6: sum(v for k, v in obs_k.items() if k >= 6)}
print(f"observed: never-wrong slots {obs[0]}, slots with >=3 failing configurations {obs[3]}, with >=6 {obs[6]}")
rows = []
for name, prob_of in nulls:
    exp, band = expected_k(prob_of)
    e = {0: exp[0], 3: sum(exp[3:]), 6: sum(exp[6:])}
    print(f"{name}: expected never-wrong {e[0]:.0f} [{band[0][0]}, {band[0][1]}], >=3 failing {e[3]:.0f} [{band[3][0]}, {band[3][1]}], >=6 failing {e[6]:.0f} [{band[6][0]}, {band[6][1]}]")
    for k, lab in ((0, "never-wrong slots"), (3, "slots with >=3 failing configurations"), (6, "slots with >=6 failing configurations")):
        rows.append([name, lab, obs[k], round(e[k], 1), band[k][0], band[k][1]])
    for k in range(15):
        rows.append([name, f"slots with exactly {k} failing configurations", obs_k[k], round(exp[k], 1), "", ""])
write_csv("null_models.csv", ["null", "statistic", "observed", "expected", "ci_lo", "ci_hi"], rows)

# ---- wrong-value diversity at matched counts ----------------------------------------------------------------
div = collections.defaultdict(list)          # (tier, n wrong attempts) -> distinct values per slot
for xs in S.values():
    by_tier = collections.defaultdict(list)
    for x in xs:
        if not x["correct"]:
            by_tier[T[x["study"]]].append(x["value"])
    for t, vals in by_tier.items():
        if 2 <= len(vals) <= 6:
            div[(t, len(vals))].append(len(set(vals)))
rows = []
print("mean distinct wrong values on slots where the tier produced exactly n wrong attempts (slots in parentheses):")
for t in ("strong", "mid", "weak"):
    line = []
    for n in range(2, 7):
        xs = div[(t, n)]
        rows.append([t, f"{n} attempts", len(xs), round(sum(xs) / len(xs), 4) if xs else ""])
        line.append(f"n={n}: {sum(xs) / len(xs):.2f} ({len(xs)})" if xs else f"n={n}: -")
    print(f"  {t:6} " + "  ".join(line))
# the same with one value per configuration (its modal wrong value), matched on the number of erring configurations,
# so that a configuration repeating its own wrong value across runs does not count as agreement
div = collections.defaultdict(list)
for xs in S.values():
    by_cfg = collections.defaultdict(list)
    for x in xs:
        if not x["correct"]:
            by_cfg[x["study"]].append(x["value"])
    by_tier = collections.defaultdict(list)
    for s, vals in by_cfg.items():
        by_tier[T[s]].append(collections.Counter(vals).most_common(1)[0][0])
    for t, modal in by_tier.items():
        if 2 <= len(modal) <= 4:
            div[(t, len(modal))].append(len(set(modal)))
print("mean distinct modal wrong values on slots where exactly c configurations of the tier erred:")
for t in ("strong", "mid", "weak"):
    line = []
    for c in range(2, 5):
        xs = div[(t, c)]
        rows.append([t, f"{c} configurations", len(xs), round(sum(xs) / len(xs), 4) if xs else ""])
        line.append(f"c={c}: {sum(xs) / len(xs):.2f} ({len(xs)})" if xs else f"c={c}: -")
    print(f"  {t:6} " + "  ".join(line))
write_csv("wrong_value_diversity.csv", ["tier", "matched_on", "n_slots", "mean_distinct_values"], rows)
