"""Evaluate the prospective prediction for GPT-6.1 Sol (thinking) as fixed in PREDICTION.md.

The fourteen configurations come from analysis/scratch/final_data.json unchanged; the Sol study is read
from experiments_prospective/ with the same extraction rules as analysis/final_20261012/scripts/extract.py (only the
outputs a turn itself stores; value = the stored output, JSON-encoded) and is never added to final_data.json,
so no number of the paper changes. Run from the repository root.
"""
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "analysis/final_20261012/scripts")
from common import load  # noqa: E402
from groups import STUDIES, task_accuracy, tiers  # noqa: E402

SOL = "gpt-6.1-sol-thinking-20261115"
cases, turns, vars_ = load()
turns = [t for t in turns if t["study"] in STUDIES]
vars_ = [v for v in vars_ if v["study"] in STUDIES]


def extract(study):
    t_rows, v_rows = [], []
    for d in sorted((Path("experiments_prospective") / study / "runs").glob("*/*/*")):
        case, run_id = d.parts[-3], d.name
        r = json.loads((d / "run.json").read_text())
        p = json.loads((d / "programmatic_verification.json").read_text())
        pvt = [t for c in p["conversations"] for t in c["turns"]]
        rt = [t for c in r["conversations"] for t in c["turns"]]
        for i, (t, v) in enumerate(zip(rt, pvt)):
            pv = v.get("programmatic_verification") or {}
            t_rows.append(dict(study=study, case=case, run_id=run_id, turn=i + 1,
                               ok=v.get("status") == "scored" and bool(pv.get("success"))))
            own = set(cases[case]["turn_stores"][i])
            for name, item in ((pv.get("variable_verification") or {}).get("items") or {}).items():
                if name in own:
                    v_rows.append(dict(study=study, case=case, run_id=run_id, turn=i + 1, var=name,
                                       is_set=item.get("is_set"), correct=item.get("correct"),
                                       value=json.dumps((t.get("outputs") or {}).get(name), default=str)[:120]))
    return t_rows, v_rows


sol_turns, sol_vars = extract(SOL)
n_runs = len({(t["case"], t["run_id"]) for t in sol_turns})
A = task_accuracy(sol_turns)[SOL]
acc = task_accuracy(turns); T = tiers(acc)
strong = [s for s in STUDIES if T[s] == "strong"]
print(f"Sol runs {n_runs}, tasks {len({t['case'] for t in sol_turns})}; accuracy A = {A:.4f}")

# consensus error of each slot from the fourteen only: modal wrong value, given by >= 2 configurations
S = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"]:
        S[(v["case"], v["turn"], v["var"])][v["study"]].append(v)


def consensus(m, exclude=()):
    cnt = collections.Counter(); who = collections.defaultdict(set)
    for s, xs in m.items():
        if s in exclude:
            continue
        for x in xs:
            if not x["correct"]:
                cnt[x["value"]] += 1; who[x["value"]].add(s)
    if not cnt:
        return None
    val, _ = cnt.most_common(1)[0]
    return val if len(who[val]) >= 2 else None


sol_wrong = collections.defaultdict(list)
for v in sol_vars:
    if v["is_set"] and not v["correct"]:
        sol_wrong[(v["case"], v["turn"], v["var"])].append(v["value"])
hits = sum(x == consensus(S[k]) for k, xs in sol_wrong.items() for x in xs)
n_wrong = sum(len(xs) for xs in sol_wrong.values())
C = hits / n_wrong
print(f"Sol wrong values {n_wrong} on {len(sol_wrong)} slots in {len({k[0] for k in sol_wrong})} tasks; "
      f"equal to the consensus error C = {C:.4f}")

# P1: prediction interval of the line recorded in PREDICTION.md (C = -0.3152 + 1.1221 A, residual s.d. 0.0743,
# n = 14), fitted to the fourteen configurations as they stood on 2026-11-15. The line stays as recorded: re-scoring
# one task afterwards moved the fourteen slightly, and refitting would change the prediction after the outcome
# is known. mx and sxx are the mean and sum of squares of those fourteen accuracies.
n, a, b, s = 14, -0.31524183359043834, 1.1221199683943546, 0.07431933742815243
mx, sxx = 0.6712495714285714, 0.19770998075342858
pred = a + b * A; h = 2.179 * s * math.sqrt(1 + 1 / n + (A - mx) ** 2 / sxx)
print(f"P1: predicted C = {pred:.3f}, 95% prediction interval [{pred - h:.3f}, {pred + h:.3f}] -> "
      f"{'inside' if pred - h <= C <= pred + h else ('above' if C > pred + h else 'below')}")

# P2: within-slot contrast, Sol minus the strong tier pooled, same consensus
by_task = collections.defaultdict(list)
for k, xs_ in sol_wrong.items():
    val = consensus(S[k])
    st = [x for s_ in strong for x in S[k].get(s_, []) if not x["correct"]]
    if st:
        by_task[k[0]].append(sum(x == val for x in xs_) / len(xs_) - sum(x["value"] == val for x in st) / len(st))
tasks = sorted(by_task); allv = [d for t in tasks for d in by_task[t]]
if allv:
    rng = random.Random(0)
    draws = sorted(statistics.mean(d for t in (rng.choice(tasks) for _ in tasks) for d in by_task[t]) for _ in range(2000))
    D, lo, hi = statistics.mean(allv), draws[50], draws[1949]
    strong_acc = statistics.mean(acc[s_] for s_ in strong)
    print(f"P2: D = {D:+.3f} [{lo:+.3f}, {hi:+.3f}] on {len(allv)} slots in {len(tasks)} tasks; "
          f"A - strong mean = {A - strong_acc:+.3f}; expected about {0.60 * (A - strong_acc):+.3f}")

# Tables for the paper: the summary, and Sol's wrong values by task.
import csv  # noqa: E402

OUT = Path("analysis/prospective_20261115/tables"); OUT.mkdir(exist_ok=True)


def given_by_two(m, value):
    return len({s_ for s_, xs_ in m.items() for x in xs_ if not x["correct"] and x["value"] == value}) >= 2


any_two = sum(given_by_two(S[k], x) for k, xs_ in sol_wrong.items() for x in xs_)
per_task = collections.defaultdict(lambda: [0, 0])
for k, xs_ in sol_wrong.items():
    val = consensus(S[k])
    per_task[k[0]][0] += len(xs_); per_task[k[0]][1] += sum(x == val for x in xs_)
top, (top_n, top_hits) = max(per_task.items(), key=lambda kv: kv[1][0])
print(f"Sol wrong values given by >= 2 of the fourteen: {any_two} of {n_wrong} = {any_two / n_wrong:.1%}")
print(f"largest task {top}: {top_n} wrong values; without it C = {(hits - top_hits) / (n_wrong - top_n):.3f} "
      f"({hits - top_hits} of {n_wrong - top_n})")
with open(OUT / "prospective.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["measure", "value"])
    w.writerows([["runs", n_runs], ["task_accuracy_pct", f"{A * 100:.1f}"], ["wrong_values", n_wrong],
                 ["slots", len(sol_wrong)], ["tasks", len(per_task)], ["equal_consensus", hits],
                 ["equal_consensus_pct", f"{C * 100:.1f}"], ["predicted_pct", f"{pred * 100:.1f}"],
                 ["interval_lo_pct", f"{(pred - h) * 100:.1f}"], ["interval_hi_pct", f"{(pred + h) * 100:.1f}"],
                 ["given_by_two", any_two], ["given_by_two_pct", f"{any_two / n_wrong * 100:.1f}"],
                 ["largest_task", top], ["largest_task_wrong", top_n],
                 ["without_largest_pct", f"{(hits - top_hits) / (n_wrong - top_n) * 100:.1f}"]])
with open(OUT / "prospective_tasks.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "wrong_values", "equal_consensus"])
    w.writerows([t, n_, h_] for t, (n_, h_) in sorted(per_task.items(), key=lambda kv: -kv[1][0]))
