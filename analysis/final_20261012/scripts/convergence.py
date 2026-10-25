"""Convergence of wrong values on the crowd's wrong value, and the determinism of a configuration's errors.

All measures leave the configuration under study out of the "crowd": the crowd value on a slot is the most common
wrong value among the OTHER configurations, counted only when at least two of them gave it.

- by_config: for each configuration, the share of ALL its wrong values equal to the crowd value (leave-one-out; slots on
  which no other configuration erred count in the denominator); the share of its slots wrong in >=2 (or all 3) runs on
  which every wrong value was the same; Spearman with accuracy.
- within_slot: on slots where configurations of two tiers both erred, each tier's convergence to the crowd value,
  paired by slot, with a case-resampled interval on the difference.
- thinking: for each paired model, on slots where both its modes erred, thinking minus non-thinking convergence
  to the crowd (crowd = the other 12 configurations); pooled with a case bootstrap; by knowledge class.
- flips: slots where one mode was correct in all three runs and the other mode erred; share of the erring mode's
  values that equal the crowd value, by direction.
- dose: probability that a configuration's wrong value equals the value most other configurations gave, by the
  number of other configurations that gave it (1..6+), by tier.
Writes tables/convergence_*.csv.
"""
import collections
import json
import random
import statistics

from common import load, write_csv
from groups import PAIRS, STUDIES, task_accuracy, tiers

cases, turns, vars_ = load()
acc = task_accuracy(turns); T = tiers(acc)
S = collections.defaultdict(lambda: collections.defaultdict(list))      # slot -> study -> rows
for v in vars_:
    if v["is_set"]:
        S[(v["case"], v["turn"], v["var"])][v["study"]].append(v)


def crowd(m, exclude):
    """Most common wrong value among configurations not in `exclude`, if >=2 of them gave it."""
    cnt = collections.Counter(); who = collections.defaultdict(set)
    for s, xs in m.items():
        if s in exclude:
            continue
        for x in xs:
            if not x["correct"]:
                cnt[x["value"]] += 1; who[x["value"]].add(s)
    if not cnt:
        return None, 0
    val, _ = cnt.most_common(1)[0]
    return (val if len(who[val]) >= 2 else None), len(who[val])


def spearman(a, b):
    def rk(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rk(a), rk(b); n = len(a); m = (n - 1) / 2
    return sum((ra[i] - m) * (rb[i] - m) for i in range(n)) / (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** 0.5


def case_boot(by_case, n=2000, seed=0):
    """Mean of per-slot values with a case-resampled 95% interval. by_case: {case: [values]}."""
    cs = sorted(by_case); rng = random.Random(seed)
    allv = [d for c in cs for d in by_case[c]]
    draws = sorted(statistics.mean(d for c in (rng.choice(cs) for _ in cs) for d in by_case[c]) for _ in range(n))
    return statistics.mean(allv), draws[int(0.025 * n)], draws[int(0.975 * n) - 1], len(allv)


# ---- by configuration: leave-one-out convergence and determinism -------------------------------------------
loo = collections.Counter(); loo_n = collections.Counter(); ident2 = collections.Counter(); ident2_n = collections.Counter()
ident3 = collections.Counter(); ident3_n = collections.Counter()
for k, m in S.items():
    for s, xs in m.items():
        w = [x for x in xs if not x["correct"]]
        if not w:
            continue
        val, _ = crowd(m, {s})
        for x in w:
            loo_n[s] += 1; loo[s] += x["value"] == val
        if len(w) >= 2:
            ident2_n[s] += 1; ident2[s] += len({x["value"] for x in w}) == 1
        if len(w) == 3:
            ident3_n[s] += 1; ident3[s] += len({x["value"] for x in w}) == 1
order = sorted(STUDIES, key=lambda s: -acc[s])
rows = [(s, STUDIES[s][0], STUDIES[s][1], T[s], round(acc[s], 6), round(loo[s] / loo_n[s], 6), loo_n[s],
         round(ident2[s] / ident2_n[s], 6), ident2_n[s], round(ident3[s] / ident3_n[s], 6), ident3_n[s]) for s in order]
rho_loo = spearman([acc[s] for s in order], [loo[s] / loo_n[s] for s in order])
rho_id2 = spearman([acc[s] for s in order], [ident2[s] / ident2_n[s] for s in order])
rho_id3 = spearman([acc[s] for s in order], [ident3[s] / ident3_n[s] for s in order])
print("leave-one-out convergence and determinism by configuration:")
for r in rows:
    print(f"  {r[1]:18} {r[2]} {r[3]:6} acc {r[4]:.1%} | wrong = crowd value {r[5]:.1%} (n={r[6]}) | identical given >=2 wrong {r[7]:.1%} | identical given 3 wrong {r[9]:.1%}")
tier_loo = {t: (sum(loo[s] for s in STUDIES if T[s] == t), sum(loo_n[s] for s in STUDIES if T[s] == t)) for t in ("strong", "mid", "weak")}
print("  by tier: " + ", ".join(f"{t} {a / b:.1%} (n={b})" for t, (a, b) in tier_loo.items()))
print(f"  Spearman(accuracy, convergence) {rho_loo:.2f} | (accuracy, all identical|>=2 wrong) {rho_id2:.2f} | (accuracy, identical|3 wrong) {rho_id3:.2f}")
tier_id2 = {t: [ident2[s] / ident2_n[s] for s in STUDIES if T[s] == t] for t in ("strong", "mid", "weak")}
print("  all-identical given >=2 wrong, range by tier: " + ", ".join(f"{t} {min(v):.1%}-{max(v):.1%}" for t, v in tier_id2.items()))
rows += [("SPEARMAN_WITH_ACCURACY", "", "", "", "", round(rho_loo, 6), "", round(rho_id2, 6), "", round(rho_id3, 6), "")]
rows += [(f"TIER:{t}", "", "", t, "", round(a / b, 6), b, "", "", "", "") for t, (a, b) in tier_loo.items()]
write_csv("convergence_by_config.csv", ["study", "model", "mode", "tier", "task_accuracy", "wrong_equals_crowd", "n_wrong",
                                        "identical_given_2plus_wrong", "n_slots_2plus", "identical_given_3_wrong", "n_slots_3"], rows)

# ---- within slot: tier pairs --------------------------------------------------------------------------------
rows = []
for hi, lo in (("strong", "weak"), ("strong", "mid"), ("mid", "weak")):
    by_case = collections.defaultdict(list); gt = lt = 0; rh = []; rl = []
    for k, m in S.items():
        rates = {}
        for tier in (hi, lo):
            vals = []
            for s, xs in m.items():
                if T[s] != tier:
                    continue
                w = [x for x in xs if not x["correct"]]
                if not w:
                    continue
                val, _ = crowd(m, {s})
                vals += [x["value"] == val for x in w]
            if vals:
                rates[tier] = statistics.mean(vals)
        if len(rates) == 2:
            d = rates[hi] - rates[lo]; by_case[k[0]].append(d); rh.append(rates[hi]); rl.append(rates[lo]); gt += d > 0; lt += d < 0
    mean, lo_, hi_, n = case_boot(by_case)
    print(f"within slot, {hi} vs {lo}: {hi} {statistics.mean(rh):.1%} vs {lo} {statistics.mean(rl):.1%}; diff {mean:+.1%} [{lo_:+.1%}, {hi_:+.1%}], slots {n}, {hi}>{lo} on {gt}, < on {lt}")
    rows.append((hi, lo, round(statistics.mean(rh), 6), round(statistics.mean(rl), 6), round(mean, 6), round(lo_, 6), round(hi_, 6), n, gt, lt))
write_csv("convergence_within_slot.csv", ["higher_tier", "lower_tier", "higher_rate", "lower_rate", "diff", "ci_lo", "ci_hi", "n_slots", "n_higher_gt", "n_higher_lt"], rows)

# ---- within model: thinking vs non-thinking -------------------------------------------------------------------
kc = json.load(open("analysis/final_20261012/knowledge_classification_20261012.json"))
slot_class = {(e["case"], sl): e["primary"] for e in kc for sl in e["slots"]}
rows = []; pooled = collections.defaultdict(list); byclass = collections.defaultdict(lambda: collections.defaultdict(list))
for model, (Ts, Ns) in PAIRS.items():
    diffs = collections.defaultdict(list)
    for k, m in S.items():
        val, _ = crowd(m, {Ts, Ns})
        if val is None and not any(not x["correct"] for s, xs in m.items() if s not in (Ts, Ns) for x in xs):
            continue
        r = {}
        for s in (Ts, Ns):
            w = [x for x in m.get(s, []) if not x["correct"]]
            if w:
                r[s] = statistics.mean(x["value"] == val for x in w)
        if len(r) == 2:
            d = r[Ts] - r[Ns]; diffs[k[0]].append(d); pooled[k[0]].append(d)
            cls = slot_class.get((k[0], k[2]))
            if cls:
                byclass[cls][k[0]].append(d)
    mean, lo_, hi_, n = case_boot(diffs)
    print(f"thinking minus non-thinking convergence, {model:18} {mean:+.1%} [{lo_:+.1%}, {hi_:+.1%}] slots {n}")
    rows.append((model, round(mean, 6), round(lo_, 6), round(hi_, 6), n))
mean, lo_, hi_, n = case_boot(pooled)
print(f"thinking minus non-thinking convergence, pooled            {mean:+.1%} [{lo_:+.1%}, {hi_:+.1%}] slots {n}")
rows.append(("POOLED", round(mean, 6), round(lo_, 6), round(hi_, 6), n))
for cls in sorted(byclass, key=lambda c: -sum(len(v) for v in byclass[c].values())):
    m_, l_, h_, n_ = case_boot(byclass[cls], seed=1)
    rows.append((f"CLASS:{cls}", round(m_, 6), round(l_, 6), round(h_, 6), n_))
    print(f"   by class {cls}: {m_:+.1%} [{l_:+.1%}, {h_:+.1%}] slots {n_}")
write_csv("convergence_thinking.csv", ["model_or_group", "thinking_minus_nonthinking", "ci_lo", "ci_hi", "n_slots"], rows)

# ---- flips: one mode correct in all three runs, the other mode wrong -------------------------------------------
rows = []; flip_cases = {}
for a_mode, b_mode, lab in (("N", "T", "non-thinking correct, thinking wrong"), ("T", "N", "thinking correct, non-thinking wrong")):
    by_case = collections.defaultdict(list); n_slots = 0; n_crowd = 0; distinct = set()
    for k, m in S.items():
        for model, (Ts, Ns) in PAIRS.items():
            a, b = (Ns, Ts) if a_mode == "N" else (Ts, Ns)
            ra, rb = m.get(a, []), m.get(b, [])
            if len(ra) == 3 and all(x["correct"] for x in ra) and rb and any(not x["correct"] for x in rb):
                n_slots += 1
                val, _ = crowd(m, {Ts, Ns})
                if val is None:
                    continue
                n_crowd += 1; distinct.add(k)
                by_case[k[0]].append(float(any(x["value"] == val for x in rb if not x["correct"])))
    mean, lo_, hi_, n = case_boot(by_case); flip_cases[lab] = by_case
    hits = int(round(mean * n))
    print(f"flips, {lab}: model-slot pairs {n_slots}, with a consensus error {n_crowd} ({len(distinct)} distinct slots), erring mode gave it {hits}/{n_crowd} = {mean:.1%} [{lo_:.1%}, {hi_:.1%}]; over all pairs {hits / n_slots:.1%}")
    rows.append((lab, n_slots, n_crowd, len(distinct), hits, round(mean, 6), round(lo_, 6), round(hi_, 6), round(hits / n_slots, 6)))
# difference between directions, same case resamples
A = flip_cases["non-thinking correct, thinking wrong"]; B = flip_cases["thinking correct, non-thinking wrong"]
cs = sorted(set(A) | set(B)); rng = random.Random(2)
def dmean(sample):
    a = [d for c in sample for d in A.get(c, [])]; b = [d for c in sample for d in B.get(c, [])]
    return statistics.mean(a) - statistics.mean(b)
draws = sorted(dmean([rng.choice(cs) for _ in cs]) for _ in range(2000)); d0 = dmean(cs)
print(f"flips, difference (N->T minus T->N): {d0:+.1%} [{draws[50]:+.1%}, {draws[1949]:+.1%}]")
rows.append(("difference, first minus second", "", "", "", "", round(d0, 6), round(draws[50], 6), round(draws[1949], 6), ""))
write_csv("convergence_flips.csv", ["direction", "n_model_slot_pairs", "n_with_consensus_error", "n_distinct_slots", "n_hits", "share_to_consensus", "ci_lo", "ci_hi", "share_over_all_pairs"], rows)

# ---- dose-response ----------------------------------------------------------------------------------------------
dose = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for k, m in S.items():
    for s, xs in m.items():
        w = [x for x in xs if not x["correct"]]
        if not w:
            continue
        who = collections.defaultdict(set)
        for s2, xs2 in m.items():
            if s2 != s:
                for x in xs2:
                    if not x["correct"]:
                        who[x["value"]].add(s2)
        if not who:
            continue
        val, cfgs = max(who.items(), key=lambda kv: len(kv[1]))
        kb = min(len(cfgs), 6)
        for x in w:
            dose[T[s]][kb][1] += 1; dose[T[s]][kb][0] += x["value"] == val
rows = []
print("dose-response: P(wrong value = value most other configs gave) by number of other configs giving it:")
for tier in ("strong", "mid", "weak"):
    print(f"  {tier:6} " + "  ".join(f"k={k}{'+' if k == 6 else ''}: {dose[tier][k][0] / dose[tier][k][1]:.0%} (n={dose[tier][k][1]})" for k in sorted(dose[tier])))
    for k in sorted(dose[tier]):
        rows.append((tier, f"{k}{'+' if k == 6 else ''}", round(dose[tier][k][0] / dose[tier][k][1], 6), dose[tier][k][1]))
write_csv("convergence_dose.csv", ["tier", "n_other_configs", "p_equal", "n_wrong"], rows)
