"""Core table: defect rates by capability tier x thinking mode, with case-resampled bootstrap intervals.

Rows = defect types; unit = per 1000 set-variable attempts (variable-level defects) or per 1000 turns
(turn-level defects). Columns = strong / mid / weak configuration tiers (groups.TIER_THRESHOLDS), then
the six paired models split top/lower x thinking/non-thinking (groups.model_tiers), then two size pairs.
For each row: weak->strong relative change, paired non-thinking->thinking change over the 6 pairs,
and the "can thinking substitute for capability" cell: lower-tier thinking vs top-tier non-thinking.
Writes tables/defects_by_tier.csv and tables/defects_by_tier_tests.csv.
"""
import collections
import json
import os

from common import boot_ratio_change, load, load_cats, modal_shared, rate, runs, write_csv
from groups import PAIRS, SIZE_PAIRS, STUDIES, label, model_tiers, task_accuracy, tiers

cases, turns, vars_ = load()
cats = load_cats()
acc = task_accuracy(turns); TIER = tiers(acc); MTIER = model_tiers(acc)
flags = json.load(open("analysis/scratch/turn_flags.json")) if os.path.exists("analysis/scratch/turn_flags.json") else None
if flags:
    for t in turns:
        fl = flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}")
        if fl is not None:
            fl["is_wrong"] = t["kind"] == "wrong_value"
print("config tiers:")
for s in sorted(acc, key=lambda s: -acc[s]):
    print(f"  {TIER[s]:6} {acc[s]:.1%} {label(s)}" + (f"   model-tier={MTIER[s]}" if s in MTIER else ""))

# ---- shared-wrong slots (>=3 configs share the modal wrong value) for the shared/own split
modal = modal_shared(vars_)
R = runs(turns)
prev = {}
for key, ts in R.items():
    for a, b in zip(ts, ts[1:]):
        prev[(b["study"], b["case"], b["run_id"], b["turn"])] = a

# ---- defect definitions: each yields {case: (num, den)} for a study ----------------------------
SHAPES = ["far off >10%", "1-10% off", "near miss 0.1-1%", "near miss <0.1% (rounding/convention)",
          "convention factor (12/4/360/252/2)", "power-of-ten scale", "sign flip", "off by one (count)",
          "small count diff (2-5)", "string/token", "string: date"]
V_by = collections.defaultdict(list); T_by = collections.defaultdict(list)
for v in vars_:
    V_by[v["study"]].append(v)
for t in turns:
    T_by[t["study"]].append(t)


def var_defect(pred, den_pred=lambda v: True):
    def f(study):
        out = collections.defaultdict(lambda: [0, 0])
        for v in V_by[study]:
            if den_pred(v):
                out[v["case"]][1] += 1; out[v["case"]][0] += bool(pred(v))
        return {c: tuple(x) for c, x in out.items()}
    return f


def turn_defect(pred, den_pred=lambda t: True):
    def f(study):
        out = collections.defaultdict(lambda: [0, 0])
        for t in T_by[study]:
            if den_pred(t):
                out[t["case"]][1] += 1; out[t["case"]][0] += bool(pred(t))
        return {c: tuple(x) for c, x in out.items()}
    return f


def flag_defect(num_key, den_key=None, den_pred=lambda t: True):
    def f(study):
        out = collections.defaultdict(lambda: [0, 0])
        for t in T_by[study]:
            if not den_pred(t):
                continue
            fl = flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}")
            if fl is None:
                continue
            out[t["case"]][1] += fl[den_key] if den_key else 1; out[t["case"]][0] += int(fl[num_key])
        return {c: tuple(x) for c, x in out.items()}
    return f


wrong = lambda v: v["is_set"] and not v["correct"]
shape = lambda v: cats.get((v["study"], v["case"], v["run_id"], v["turn"], v["var"]))
is_shared = lambda v: (v["case"], v["turn"], v["var"]) in modal and v["value"] == modal[(v["case"], v["turn"], v["var"])]
DEFECTS = [
    ("variable wrong (any)", "per 1000 set vars", var_defect(wrong, lambda v: v["is_set"])),
    ("  shared wrong value (>=3 configs)", "per 1000 set vars", var_defect(lambda v: wrong(v) and is_shared(v), lambda v: v["is_set"])),
    ("  other wrong value (not shared by >=3 configs)", "per 1000 set vars", var_defect(lambda v: wrong(v) and not is_shared(v), lambda v: v["is_set"])),
] + [(f"  shape: {s}", "per 1000 set vars", var_defect((lambda s: lambda v: wrong(v) and shape(v) == s)(s), lambda v: v["is_set"])) for s in SHAPES] + [
    ("variable not set", "per 1000 vars", var_defect(lambda v: not v["is_set"])),
    ("turn: variables_not_set", "per 1000 turns", turn_defect(lambda t: t["kind"] == "variables_not_set")),
    ("turn: budget exhausted (max_steps)", "per 1000 turns", turn_defect(lambda t: t["stop"] == "max_steps")),
    ("turn: output_limit", "per 1000 turns", turn_defect(lambda t: t["kind"] == "output_limit")),
    ("turn: lossy_snapshot", "per 1000 turns", turn_defect(lambda t: t["kind"] == "lossy_snapshot")),
    ("turn: wrong after previous turn wrong", "per 1000 turns w/ prev wrong", turn_defect(lambda t: not t["ok"], lambda t: (k := prev.get((t["study"], t["case"], t["run_id"], t["turn"]))) is not None and k["kind"] == "wrong_value")),
    ("turn: wrong after previous turn correct", "per 1000 turns w/ prev pass", turn_defect(lambda t: not t["ok"], lambda t: (k := prev.get((t["study"], t["case"], t["run_id"], t["turn"]))) is not None and k["ok"])),
]
if flags:
    DEFECTS += [
        ("delivery: computed but not delivered", "per 1000 wrong numeric vars", flag_defect("n_computed_not_delivered", "n_wrong_numeric")),
        ("delivery: near-name variable instead of contract name", "per 1000 vars", flag_defect("n_near_name", None)),  # den fixed below
        ("delivery: phantom completion", "per 1000 turns", flag_defect("phantom")),
        ("self-check present, among wrong turns", "per 1000 wrong turns", flag_defect("has_check", None, lambda t: t["kind"] == "wrong_value")),
        ("turn wrong, given an explicit self-check", "per 1000 checked turns", flag_defect("is_wrong", None, lambda t: t["kind"] in ("pass", "wrong_value") and flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}", {}).get("has_check"))),
        ("turn wrong, given no self-check", "per 1000 unchecked turns", flag_defect("is_wrong", None, lambda t: t["kind"] in ("pass", "wrong_value") and not flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}", {}).get("has_check", True))),
        ("self-check visibly failed, among wrong checked turns", "per 1000 wrong turns with checks", None),
        ("no visible check failure, among wrong checked turns", "per 1000 wrong turns with checks", None),
    ]
    # near-name denominator = all variables; self-check-failed needs its own function
    def near_name(study):
        out = collections.defaultdict(lambda: [0, 0])
        for t in T_by[study]:
            fl = flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}")
            if fl is None:
                continue
            out[t["case"]][1] += len(cases[t["case"]]["turn_stores"][t["turn"] - 1]); out[t["case"]][0] += fl["n_near_name"]
        return {c: tuple(x) for c, x in out.items()}
    def check_outcome(failed):
        def g(study):
            out = collections.defaultdict(lambda: [0, 0])
            for t in T_by[study]:
                fl = flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}")
                if fl is None or t["kind"] != "wrong_value" or not fl["has_check"]:
                    continue
                out[t["case"]][1] += 1; out[t["case"]][0] += (fl["check_failed"] if failed else not fl["check_failed"])
            return {c: tuple(x) for c, x in out.items()}
        return g
    DEFECTS = [(n, u, near_name if n.startswith("delivery: near-name") else (check_outcome(True) if n.startswith("self-check visibly") else (check_outcome(False) if n.startswith("no visible check failure") else f))) for n, u, f in DEFECTS]


def merge(fs, studies):
    out = collections.defaultdict(lambda: [0, 0])
    for s in studies:
        for c, (a, b) in fs(s).items():
            out[c][0] += a; out[c][1] += b
    return {c: tuple(x) for c, x in out.items()}


def tot(pc):
    a = sum(x[0] for x in pc.values()); b = sum(x[1] for x in pc.values())
    return a, b, rate(a, b) * 1000


GROUPS = {
    "strong": [s for s in STUDIES if TIER[s] == "strong"], "mid": [s for s in STUDIES if TIER[s] == "mid"], "weak": [s for s in STUDIES if TIER[s] == "weak"],
    "topT": [T for T, N in PAIRS.values() if MTIER[T] == "top"], "topN": [N for T, N in PAIRS.values() if MTIER[T] == "top"],
    "lowT": [T for T, N in PAIRS.values() if MTIER[T] == "lower"], "lowN": [N for T, N in PAIRS.values() if MTIER[T] == "lower"],
}
print({k: [STUDIES[s][0] + " " + STUDIES[s][1] for s in v] for k, v in GROUPS.items()})
hdr = ["defect", "unit"] + [f"{g}_rate" for g in GROUPS] + [f"{g}_n" for g in GROUPS] + [f"{k}_from" for k in SIZE_PAIRS] + [f"{k}_to" for k in SIZE_PAIRS]
rows = []; tests = []
for name, unit, f in DEFECTS:
    pc = {g: merge(f, ss) for g, ss in GROUPS.items()}
    sp = {k: (tot(f(a)), tot(f(b))) for k, (a, b) in SIZE_PAIRS.items()}
    rows.append([name, unit] + [round(tot(pc[g])[2], 6) for g in GROUPS] + [tot(pc[g])[1] for g in GROUPS]
                + [round(sp[k][0][2], 6) for k in SIZE_PAIRS] + [round(sp[k][1][2], 6) for k in SIZE_PAIRS])
    # tests: weak->strong ; paired N->T ; lowT vs topN ; size pairs pooled
    ws = boot_ratio_change(pc["weak"], pc["strong"])
    pn = merge(f, [N for T, N in PAIRS.values()]); pt = merge(f, [T for T, N in PAIRS.values()])
    th = boot_ratio_change(pn, pt)
    sub = boot_ratio_change(pc["topN"], pc["lowT"])
    sz = boot_ratio_change(merge(f, [a for a, b in SIZE_PAIRS.values()]), merge(f, [b for a, b in SIZE_PAIRS.values()]))
    tests.append([name, unit] + [round(x, 6) for x in ws] + [round(x, 6) for x in th] + [round(x, 6) for x in sub] + [round(x, 6) for x in sz])
    print(f"{name:52} {unit:28} S {tot(pc['strong'])[2]:6.1f} M {tot(pc['mid'])[2]:6.1f} W {tot(pc['weak'])[2]:6.1f} | weak->strong {ws[0]:+.0%} [{ws[1]:+.0%},{ws[2]:+.0%}] | N->T {th[0]:+.0%} [{th[1]:+.0%},{th[2]:+.0%}] | lowT vs topN {sub[0]:+.0%} [{sub[1]:+.0%},{sub[2]:+.0%}] | small->large {sz[0]:+.0%} [{sz[1]:+.0%},{sz[2]:+.0%}]")
write_csv("defects_by_tier.csv", hdr, rows)
write_csv("defects_by_tier_tests.csv", ["defect", "unit", "weak_to_strong", "ws_lo", "ws_hi", "nonthink_to_think_paired", "th_lo", "th_hi",
                                        "topN_to_lowT", "sub_lo", "sub_hi", "small_to_large_size_pairs", "sz_lo", "sz_hi"], tests)
