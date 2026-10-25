"""Finding 1: errors are correlated across models.

Slot = (case, turn, variable). Prints and writes tables/shared_errors.csv:
- slots never wrong by any config vs the count expected if configs failed independently;
- share of wrong values that equal the slot's single most common wrong value (modal share; counts a value given by
  one configuration too), and the shares under the stricter definitions: modal value given by >=2 / >=3
  configurations, by >=3 vendors, and the >=3-configuration share split by value type;
- slots where one identical wrong value comes from >=3 configs / >=3 vendors;
- R^2 of variable correctness when grouped by model, by slot, and by slot x model cell. These are NESTED
  grouping fits (the cell grouping contains the other two), not additive variance components;
- 14-config plurality vote on first runs vs best single config vs oracle.
"""
import collections
import math
import statistics

from common import load, slots, write_csv
from groups import STUDIES, VENDOR

cases, turns, vars_ = load()
S = slots(vars_)
S = {k: v for k, v in S.items() if len(v) >= 30}          # slots present in essentially every config
studies = sorted(STUDIES)

# 1. modal wrong value share, multi-config and multi-family shared values
wrong_total = modal_hits = shared_cfg = shared_ven = 0
hits = collections.Counter()        # wrong values whose slot's modal wrong value is given by >=k configs / vendors
by_type = collections.Counter()     # the >=3-config shared values, by value type


def value_type(val):
    try:
        float(val); return "numeric"
    except (TypeError, ValueError):
        return "boolean" if val in ("true", "false") else "string"


for xs in S.values():
    wrongs = [x for x in xs if not x["correct"]]
    if not wrongs:
        continue
    wrong_total += len(wrongs)
    val, n = collections.Counter(x["value"] for x in wrongs).most_common(1)[0]
    modal_hits += n
    cfgs = {x["study"] for x in wrongs if x["value"] == val}
    vens = {VENDOR[x["study"]] for x in wrongs if x["value"] == val}
    for k in (2, 3, 5):
        hits[("configs", k)] += n * (len(cfgs) >= k); hits[("vendors", k)] += n * (len(vens) >= k)
    shared_cfg += n >= 3 and len(cfgs) >= 3
    shared_ven += n >= 3 and len(vens) >= 3
    if len(cfgs) >= 3:
        by_type[value_type(val)] += n

# 2. never-wrong slots: observed vs independence
def per_study_fail(xs):
    m = collections.defaultdict(list)
    for x in xs:
        m[x["study"]].append(not x["correct"])
    return {s: any(f) for s, f in m.items()}

marg, cnt = collections.Counter(), collections.Counter()
for xs in S.values():
    for s, f in per_study_fail(xs).items():
        marg[s] += f; cnt[s] += 1
p = {s: marg[s] / cnt[s] for s in studies}
obs_none = exp_none = 0.0; full = 0
for xs in S.values():
    pf = per_study_fail(xs)
    if len(pf) < 14:
        continue
    full += 1
    obs_none += not any(pf.values()); exp_none += math.prod(1 - p[s] for s in studies)

# 3. variance decomposition
rows = [(v["study"], (v["case"], v["turn"], v["var"]), 1.0 if v["correct"] else 0.0) for v in vars_ if v["is_set"]]
mean = statistics.mean(r[2] for r in rows); tot = sum((r[2] - mean) ** 2 for r in rows)

def r2(keyf):
    g = collections.defaultdict(list)
    for s, k, y in rows:
        g[keyf(s, k)].append(y)
    mu = {k: statistics.mean(ys) for k, ys in g.items()}
    return 1 - sum((y - mu[keyf(s, k)]) ** 2 for s, k, y in rows) / tot

r2_model, r2_slot, r2_both = r2(lambda s, k: s), r2(lambda s, k: k), r2(lambda s, k: (s, k))

# 4. plurality vote on first runs
runs_by = collections.defaultdict(dict)
for v in vars_:
    if v["is_set"]:
        runs_by[(v["case"], v["turn"], v["var"])].setdefault(v["study"], []).append((v["run_id"], bool(v["correct"]), v["value"]))
maj_ok = maj_n = any_ok = 0; single = collections.defaultdict(lambda: [0, 0])
for k, m in runs_by.items():
    if len(m) < 14:
        continue
    firsts = {s: sorted(r)[0] for s, r in m.items()}
    top = collections.Counter(x[2] for x in firsts.values()).most_common(1)[0][0]
    correct_vals = {x[2] for x in firsts.values() if x[1]}
    maj_ok += top in correct_vals; maj_n += 1; any_ok += bool(correct_vals)
    for s, x in firsts.items():
        single[s][0] += x[1]; single[s][1] += 1
best_s, (bo, bn) = max(single.items(), key=lambda kv: kv[1][0] / kv[1][1])

print(f"slots analysed {len(S)} | with any error {sum(any(not x['correct'] for x in xs) for xs in S.values())}")
print(f"wrong attempts {wrong_total} | share equal to the slot's modal wrong value {modal_hits / wrong_total:.1%} (any number of configs)")
for k in (2, 3, 5):
    print(f"  modal value given by >={k} configs: {hits[('configs', k)] / wrong_total:.1%} | by >={k} vendors: {hits[('vendors', k)] / wrong_total:.1%}")
print(f"  >=3-config shared values by type: " + ", ".join(f"{k} {v} ({v / max(1, hits[('configs', 3)]):.1%})" for k, v in by_type.most_common()))
print(f"slots with one wrong value from >=3 configs: {shared_cfg} | from >=3 vendors: {shared_ven}")
print(f"slots never wrong (all 14 present, n={full}): observed {obs_none:.0f} vs {exp_none:.1f} expected under independence")
print(f"R^2 of correctness grouped by model {r2_model:.1%} | by slot {r2_slot:.1%} | by slot x model cell {r2_both:.1%} (nested fits, not additive)")
print(f"plurality vote on first runs ({maj_n} slots): {maj_ok / maj_n:.2%} | best single {STUDIES[best_s][0]} {STUDIES[best_s][1]} {bo / bn:.2%} | oracle {any_ok / maj_n:.2%}")
write_csv("shared_errors.csv", ["metric", "value"], [
    ("slots_analysed", len(S)), ("wrong_attempts", wrong_total), ("modal_wrong_share", round(modal_hits / wrong_total, 6)),
    ("shared_share_2configs", round(hits[("configs", 2)] / wrong_total, 6)), ("shared_share_3configs", round(hits[("configs", 3)] / wrong_total, 6)),
    ("shared_share_5configs", round(hits[("configs", 5)] / wrong_total, 6)),
    ("shared_share_2vendors", round(hits[("vendors", 2)] / wrong_total, 6)), ("shared_share_3vendors", round(hits[("vendors", 3)] / wrong_total, 6)),
    ("shared_3configs_numeric", by_type["numeric"]), ("shared_3configs_string", by_type["string"]), ("shared_3configs_boolean", by_type["boolean"]),
    ("slots_shared_3configs", shared_cfg), ("slots_shared_3vendors", shared_ven),
    ("never_wrong_observed", int(obs_none)), ("never_wrong_expected_independent", round(exp_none, 1)),
    ("r2_grouped_by_model", round(r2_model, 6)), ("r2_grouped_by_slot", round(r2_slot, 6)), ("r2_grouped_by_slot_x_model", round(r2_both, 6)),
    ("vote_accuracy", round(maj_ok / maj_n, 6)), ("best_single_accuracy", round(bo / bn, 6)), ("best_single", best_s), ("oracle", round(any_ok / maj_n, 6)),
])

# distribution of "how many configs got this slot wrong at least once", observed vs independence (Poisson-binomial by simulation)
import random
rng = random.Random(0)
obs = collections.Counter(); sim = collections.Counter(); NSIM = 200
for xs in S.values():
    pf = per_study_fail(xs)
    if len(pf) < 14:
        continue
    obs[sum(pf.values())] += 1
    for _ in range(NSIM):
        sim[sum(rng.random() < p[s] for s in studies)] += 1
write_csv("wrong_config_distribution.csv", ["n_configs_wrong", "observed_slots", "expected_independent_slots"],
          [(k, obs.get(k, 0), round(sim.get(k, 0) / NSIM, 2)) for k in range(15)])

# variance decomposition restricted to the strong tier and to the thinking configurations, and the share of strong-tier
# wrong values that equal another strong configuration's wrong value (robustness of the shared-error structure)
from groups import task_accuracy, tiers
T = tiers(task_accuracy(turns))
def decomp(studies):
    rows_ = [(v["study"], (v["case"], v["turn"], v["var"]), 1.0 if v["correct"] else 0.0) for v in vars_ if v["is_set"] and v["study"] in studies]
    mean_ = statistics.mean(r[2] for r in rows_); tot_ = sum((r[2] - mean_) ** 2 for r in rows_)
    def r2_(keyf):
        g = collections.defaultdict(list)
        for s, k, y in rows_:
            g[keyf(s, k)].append(y)
        mu = {k: statistics.mean(ys) for k, ys in g.items()}
        return 1 - sum((y - mu[keyf(s, k)]) ** 2 for s, k, y in rows_) / tot_
    return r2_(lambda s, k: s), r2_(lambda s, k: k), r2_(lambda s, k: (s, k))
sub = []
for name, ss in (("all", set(STUDIES)), ("strong", {s for s in STUDIES if T[s] == "strong"}), ("thinking", {s for s in STUDIES if STUDIES[s][1] == "T"})):
    m, sl, sm = decomp(ss); sub.append((name, len(ss), round(m, 6), round(sl, 6), round(sm, 6)))
    print(f"R^2 grouped, within {name:8} ({len(ss)} configs): by model {m:.1%} | by slot {sl:.1%} | by cell {sm:.1%}")
strong = {s for s in STUDIES if T[s] == "strong"}; tot = sh = 0
for k, xs in S.items():
    w = [x for x in xs if x["study"] in strong and not x["correct"]]
    if not w:
        continue
    val, n = collections.Counter(x["value"] for x in w).most_common(1)[0]
    cf = {x["study"] for x in w if x["value"] == val}
    for x in w:
        tot += 1; sh += x["value"] == val and len(cf) >= 2
print(f"strong-tier wrong values equal to another strong config's wrong value: {sh}/{tot} = {sh / tot:.1%}")
write_csv("variance_by_subset.csv", ["subset", "n_configs", "r2_model", "r2_slot", "r2_slot_x_model"], sub)
