"""What the rewording of four tasks on 2026-10-12 did to their shared wrong values.

The four tasks had their query or variable descriptions made explicit (answer keys unchanged) and all 14 x 3 runs
were re-executed; every other task's runs are the same files before and after, which the script asserts. For each
slot of the four tasks: wrong attempts, the modal wrong value's configuration count and whether it is shared (>=3
configurations), before and after; per tier, the shared-wrong attempts before and after; and task passes.
Reads the 2026-10-11 snapshot beside the final data. Writes tables/rewording_effect.csv.
"""
import collections
import json
from pathlib import Path

from common import load, write_csv
from groups import STUDIES, task_accuracy, tiers

REWORDED = ("bank_year_to_date_flow_derivation", "pension_discount_rate_remeasurement",
            "nfip_compound_poisson_stoploss_capacity", "bis_currency_credit_cointegration_branch")

cases, turns, vars_ = load()
pre = json.loads(Path("analysis/final_20261012/inputs/final_data_prerevision_20261011.json").read_text())
own = {(c, i + 1, v) for c, info in pre["cases"].items() for i, st in enumerate(info["turn_stores"]) for v in st}
pre_vars = [v for v in pre["vars"] if (v["case"], v["turn"], v["var"]) in own]
pre_turns = pre["turns"]

# control: every other task is byte-for-byte the same set of verdicts, except one task re-scored on the same runs
RESCORED = ("bdc_fiscal_label_cross_section_alignment",)
key = lambda v: (v["study"], v["case"], v["run_id"], v["turn"], v["var"], v["is_set"], v["correct"], str(v["value"]))
skip = REWORDED + RESCORED
assert {key(v) for v in vars_ if v["case"] not in skip} == {key(v) for v in pre_vars if v["case"] not in skip}
T = tiers(task_accuracy(turns))


def slot_table(vs):
    out = {}
    by = collections.defaultdict(list)
    for v in vs:
        if v["case"] in REWORDED and v["is_set"] and not v["correct"]:
            by[(v["case"], v["turn"], v["var"])].append(v)
    for k, xs in by.items():
        val, n = collections.Counter(x["value"] for x in xs).most_common(1)[0]
        cfg = {x["study"] for x in xs if x["value"] == val}
        out[k] = (len(xs), n, len(cfg), n >= 3 and len(cfg) >= 3, collections.Counter(T[x["study"]] for x in xs if x["value"] == val and n >= 3 and len(cfg) >= 3))
    return out


def passes(ts):
    runs = collections.defaultdict(list)
    for t in ts:
        if t["case"] in REWORDED:
            runs[(t["case"], t["study"], t["run_id"])].append(t["ok"])
    return collections.Counter(c for (c, _, _), oks in runs.items() if all(oks))


before, after = slot_table(pre_vars), slot_table(vars_)
pb, pa = passes(pre_turns), passes(turns)
rows = []
print("slot                                                                  wrong attempts   modal value configs   shared   (before -> after)")
for k in sorted(set(before) | set(after)):
    b = before.get(k, (0, 0, 0, False, collections.Counter())); a = after.get(k, (0, 0, 0, False, collections.Counter()))
    rows.append([k[0], k[1], k[2], b[0], a[0], b[1], a[1], b[2], a[2], int(b[3]), int(a[3])])
    if b[3] or a[3]:
        print(f"  {k[0]}/{k[1]}/{k[2]:40} {b[0]:3} -> {a[0]:3}   {b[2]:2} -> {a[2]:2}   {int(b[3])} -> {int(a[3])}")
tot = collections.Counter()
for name, tab in (("before", before), ("after", after)):
    for n, _, _, sh, by_tier in tab.values():
        tot[(name, "wrong attempts")] += n
        if sh:
            tot[(name, "shared-wrong attempts")] += sum(by_tier.values())
            for t, c in by_tier.items():
                tot[(name, f"shared-wrong attempts, {t} tier")] += c
            tot[(name, "shared slots")] += 1
for stat in ("wrong attempts", "shared-wrong attempts", "shared-wrong attempts, strong tier", "shared-wrong attempts, mid tier",
             "shared-wrong attempts, weak tier", "shared slots"):
    print(f"four tasks, {stat}: {tot[('before', stat)]} -> {tot[('after', stat)]}")
    rows.append(["ALL", "", stat, tot[("before", stat)], tot[("after", stat)], "", "", "", "", "", ""])
for c in REWORDED:
    print(f"task passes out of 42, {c}: {pb[c]} -> {pa[c]}")
    rows.append([c, "", "task passes of 42", pb[c], pa[c], "", "", "", "", "", ""])
print(f"task passes, four tasks: {sum(pb.values())} -> {sum(pa.values())}")
rows.append(["ALL", "", "task passes of 168", sum(pb.values()), sum(pa.values()), "", "", "", "", "", ""])
write_csv("rewording_effect.csv", ["case", "turn", "var", "wrong_before", "wrong_after", "modal_n_before", "modal_n_after",
                                   "modal_configs_before", "modal_configs_after", "shared_before", "shared_after"], rows)
