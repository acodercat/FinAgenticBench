"""Sensitivity of the headline shared-error results to the analysis choices a reader might question.

Scenarios: the full data; excluding the four cases whose wording was re-pinned on 2026-10-12; the pre-revision
snapshot of 2026-10-11 (same filter applied); excluding the two cases whose shared root cause the reader classed
as residual contract ambiguity (K9); tier thresholds moved by +-0.05; and leaving out each model (both of its
configurations). For each: share of wrong values that are shared (modal value from >=3 configurations), the
same share within the strong tier, and the weak->strong and paired non-thinking->thinking relative changes in the
shared and the other wrong-value rates, with case-bootstrap intervals. Writes tables/sensitivity.csv.
"""
import collections
import json
import random
import statistics
from pathlib import Path

from common import boot_ratio_change, load, modal_shared, rate, write_csv
from groups import MODEL, PAIRS, STUDIES, TIER_THRESHOLDS, task_accuracy

REPINNED = {"bank_year_to_date_flow_derivation", "pension_discount_rate_remeasurement",
            "nfip_compound_poisson_stoploss_capacity", "bis_currency_credit_cointegration_branch"}
K9 = {"leverage_definition_and_goodwill_location", "treasury_delivery_basket_ctd_switch", "nmfp_bridge_facility_state_preservation"}


def tiers_at(acc, strong, mid):
    return {s: "strong" if a >= strong else ("mid" if a >= mid else "weak") for s, a in acc.items()}


def per_case(vars_, studies, pred):
    out = collections.defaultdict(lambda: [0, 0])
    for v in vars_:
        if v["study"] in studies and v["is_set"]:
            out[v["case"]][1] += 1; out[v["case"]][0] += bool(pred(v))
    return {c: tuple(x) for c, x in out.items()}


def evaluate(name, cases, turns, vars_, strong=TIER_THRESHOLDS["strong"], mid=TIER_THRESHOLDS["mid"], drop_model=None):
    studies = [s for s in STUDIES if MODEL[s] != drop_model]
    vars_ = [v for v in vars_ if v["study"] in studies]
    turns = [t for t in turns if t["study"] in studies]
    T = tiers_at(task_accuracy(turns), strong, mid)
    modal = modal_shared(vars_)
    shared = lambda v: v["is_set"] and not v["correct"] and modal.get((v["case"], v["turn"], v["var"])) == v["value"]
    other = lambda v: v["is_set"] and not v["correct"] and not shared(v)
    wrong = [v for v in vars_ if v["is_set"] and not v["correct"]]
    sh_all = rate(sum(shared(v) for v in wrong), len(wrong))
    strong_s = [s for s in studies if T[s] == "strong"]; weak_s = [s for s in studies if T[s] == "weak"]
    ws = [v for v in wrong if v["study"] in strong_s]
    sh_strong = rate(sum(shared(v) for v in ws), len(ws))
    pairs = [(t, n) for t, n in PAIRS.values() if t in studies]
    res = {"scenario": name, "n_configs": len(studies), "n_cases": len({v["case"] for v in vars_}), "n_wrong": len(wrong),
           "shared_share": round(sh_all, 6), "strong_shared_share": round(sh_strong, 6), "n_strong_wrong": len(ws),
           "strong_configs": "|".join(STUDIES[s][0] for s in strong_s)}
    # convergence contrasts (leave-one-out consensus error): strong minus weak on shared slots; thinking minus non-thinking within model
    S_ = collections.defaultdict(lambda: collections.defaultdict(list))
    for v in vars_:
        if v["is_set"]:
            S_[(v["case"], v["turn"], v["var"])][v["study"]].append(v)
    def consensus(m, exclude):
        cnt = collections.Counter(); who = collections.defaultdict(set)
        for s, xs in m.items():
            if s in exclude: continue
            for x in xs:
                if not x["correct"]: cnt[x["value"]] += 1; who[x["value"]].add(s)
        if not cnt: return None
        val, _ = cnt.most_common(1)[0]
        return val if len(who[val]) >= 2 else None
    def boot_mean(by_case, seed=0):
        cs = sorted(by_case); rng = random.Random(seed); allv = [d for c in cs for d in by_case[c]]
        if not allv: return float("nan"), float("nan"), float("nan")
        draws = sorted(statistics.mean(d for c in (rng.choice(cs) for _ in cs) for d in by_case[c]) for _ in range(2000))
        return statistics.mean(allv), draws[50], draws[1949]   # as convergence.py case_boot
    ws_by = collections.defaultdict(list)
    for k, m in S_.items():
        r_ = {}
        for tier, ss_ in (("strong", strong_s), ("weak", weak_s)):
            vals = []
            for s in ss_:
                w_ = [x for x in m.get(s, []) if not x["correct"]]
                if w_:
                    val = consensus(m, {s}); vals += [x["value"] == val for x in w_]
            if vals: r_[tier] = statistics.mean(vals)
        if len(r_) == 2: ws_by[k[0]].append(r_["strong"] - r_["weak"])
    th_by = collections.defaultdict(list)
    for Ts, Ns in pairs:
        for k, m in S_.items():
            val = consensus(m, {Ts, Ns}); r_ = {}
            # as convergence.py: skip slots on which no other configuration erred (no consensus error can exist)
            if val is None and not any(not x["correct"] for s, xs in m.items() if s not in (Ts, Ns) for x in xs):
                continue
            for s in (Ts, Ns):
                w_ = [x for x in m.get(s, []) if not x["correct"]]
                if w_: r_[s] = statistics.mean(x["value"] == val for x in w_)
            if len(r_) == 2: th_by[k[0]].append(r_[Ts] - r_[Ns])
    for lab, by in (("conv_strong_minus_weak", ws_by), ("conv_think_minus_nonthink", th_by)):
        m_, l_, h_ = boot_mean(by)
        res[lab], res[lab + "_lo"], res[lab + "_hi"] = round(m_, 6), round(l_, 6), round(h_, 6)
    for lab, pred in (("shared", shared), ("other", other)):
        ws_ = boot_ratio_change(per_case(vars_, weak_s, pred), per_case(vars_, strong_s, pred)) if weak_s and strong_s else (float("nan"),) * 3
        th = boot_ratio_change(per_case(vars_, [n for _, n in pairs], pred), per_case(vars_, [t for t, _ in pairs], pred))
        res[f"{lab}_weak_to_strong"], res[f"{lab}_ws_lo"], res[f"{lab}_ws_hi"] = (round(x, 6) for x in ws_)
        res[f"{lab}_nonthink_to_think"], res[f"{lab}_th_lo"], res[f"{lab}_th_hi"] = (round(x, 6) for x in th)
    print(f"{name:42} cfgs {len(studies):2} cases {res['n_cases']:3} | shared {sh_all:.1%} | strong-tier shared {sh_strong:.1%} ({len(ws)}) | "
          f"weak->strong shared {res['shared_weak_to_strong']:+.0%} other {res['other_weak_to_strong']:+.0%} | N->T shared {res['shared_nonthink_to_think']:+.0%} other {res['other_nonthink_to_think']:+.0%} | conv S-W {res['conv_strong_minus_weak']:+.1%} T-N {res['conv_think_minus_nonthink']:+.1%}")
    return res


cases, turns, vars_ = load()
rows = [evaluate("full", cases, turns, vars_)]
sub = lambda excl: ([t for t in turns if t["case"] not in excl], [v for v in vars_ if v["case"] not in excl])
rows.append(evaluate("excluding 4 re-pinned cases", cases, *sub(REPINNED)))
rows.append(evaluate("excluding 3 K9 cases", cases, *sub(K9)))
rows.append(evaluate("excluding both sets", cases, *sub(REPINNED | K9)))
pre = Path("analysis/final_20261012/inputs/final_data_prerevision_20261011.json")   # ships with the run archive
if pre.exists():
    d = json.loads(pre.read_text())
    own = {(c, i + 1, v) for c, info in d["cases"].items() for i, st in enumerate(info["turn_stores"]) for v in st}
    pv = [v for v in d["vars"] if (v["case"], v["turn"], v["var"]) in own]
    rows.append(evaluate("pre-revision snapshot 2026-10-11", d["cases"], d["turns"], pv))
for ds, dm in ((0.05, 0.05), (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05)):
    rows.append(evaluate(f"thresholds strong {TIER_THRESHOLDS['strong'] + ds:.2f} mid {TIER_THRESHOLDS['mid'] + dm:.2f}", cases, turns, vars_,
                         TIER_THRESHOLDS["strong"] + ds, TIER_THRESHOLDS["mid"] + dm))
for m in sorted(set(MODEL.values())):
    rows.append(evaluate(f"leave out {m}", cases, turns, vars_, drop_model=m))
hdr = list(rows[0])
write_csv("sensitivity.csv", hdr, [[r[h] for h in hdr] for r in rows])
