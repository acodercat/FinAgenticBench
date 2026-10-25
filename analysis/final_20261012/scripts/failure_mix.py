"""Failure composition by tier: among failed turns, the share that is wrong_value vs unanswered
(variables_not_set / max_steps / output_limit / lossy_snapshot), and among wrong set variables the
share that is shared vs model-specific. Writes tables/failure_mix.csv."""
import collections

from common import load, modal_shared, write_csv
from groups import PAIRS, STUDIES, label, model_tiers, task_accuracy, tiers

cases, turns, vars_ = load()
acc = task_accuracy(turns); TIER = tiers(acc); MT = model_tiers(acc)
G = {"strong": [s for s in STUDIES if TIER[s] == "strong"], "mid": [s for s in STUDIES if TIER[s] == "mid"], "weak": [s for s in STUDIES if TIER[s] == "weak"],
     "topT": [T for T, N in PAIRS.values() if MT[T] == "top"], "topN": [N for T, N in PAIRS.values() if MT[T] == "top"],
     "lowT": [T for T, N in PAIRS.values() if MT[T] == "lower"], "lowN": [N for T, N in PAIRS.values() if MT[T] == "lower"]}
G.update({s: [s] for s in STUDIES})
modal = modal_shared(vars_)
rows = []
print(f"{'group':40} {'failed turns':>12} {'wrong_value':>11} {'vars_not_set':>12} {'max_steps':>9} {'other':>6} | {'wrong vars':>10} {'shared':>7} {'task acc':>8}")
for g, ss in G.items():
    ss = set(ss)
    kinds = collections.Counter(t["kind"] for t in turns if t["study"] in ss and not t["ok"])
    nf = sum(kinds.values())
    wv = [v for v in vars_ if v["study"] in ss and v["is_set"] and not v["correct"]]
    sh = sum(1 for v in wv if (v["case"], v["turn"], v["var"]) in modal and v["value"] == modal[(v["case"], v["turn"], v["var"])])
    ta = sum(acc[s] for s in ss) / len(ss)
    name = label(g) if g in STUDIES else g
    r = (name, nf, kinds["wrong_value"] / nf, kinds["variables_not_set"] / nf, kinds["max_steps"] / nf, (kinds["output_limit"] + kinds["lossy_snapshot"]) / nf, len(wv), sh / len(wv), ta)
    print(f"{r[0]:40} {r[1]:12} {r[2]:11.1%} {r[3]:12.1%} {r[4]:9.1%} {r[5]:6.1%} | {r[6]:10} {r[7]:7.1%} {r[8]:8.1%}")
    rows.append((g, nf, round(r[2], 6), round(r[3], 6), round(r[4], 6), round(r[5], 6), len(wv), round(r[7], 6), round(ta, 6)))
write_csv("failure_mix.csv", ["group", "failed_turns", "share_wrong_value", "share_variables_not_set", "share_max_steps", "share_other", "wrong_vars", "share_shared_wrong", "task_accuracy"], rows)
