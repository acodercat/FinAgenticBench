"""Verification study: what a verifier does to another configuration's stored answers (measures of PREDICTION.md).

For every output of the answering run (run 1 of the answering configuration, from analysis/scratch/final_data.json)
on the 149 tasks of tasks.json, the verifier's output on the same task, turn and name is read from
experiments_verify/ with the extraction rules of analysis/final_20261012/scripts/extract.py (only the outputs a
turn itself stores; value = the stored output, JSON-encoded, cut at 120 characters).

  wrong before   the answering run stored a value the validator rejected
                 -> fixed (verifier's value accepted), kept (the same wrong value), changed (another wrong value),
                    unstored (the verifier stored nothing)
  correct before -> broken (verifier's output not accepted, stored or not)
  shared         the wrong value was given by >= 2 configurations of vendors other than the answering one's
  verifier gives the verifier's own configuration stored this wrong value on this output in >= 1 of its 3 runs
  relation       self (GPT-6 Luna on its own answers), same vendor (DeepSeek V4 Flash on DeepSeek V4 Pro, GPT-6.1 Sol
                 on GPT-6 Luna), other vendor

Rates pool outputs over tasks; 95% intervals from 2,000 task resamples (differences on the same resamples).
Run from the repository root. Writes tables/*.csv; the terminal output is logs/analyse.log.
"""
import collections
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "analysis/final_20261012/scripts")
from common import load  # noqa: E402
from groups import STUDIES, VENDOR  # noqa: E402

HERE = Path("analysis/verification_20261116")
ANSWERS = {"deepseek-v4-pro-0813-thinking-20261006": "DeepSeek V4 Pro", "gemini-3.8-flash-thinking-20261007": "Gemini 3.8 Flash",
           "gpt-6-luna-thinking-20261006": "GPT-6 Luna"}
# verifier section -> (display name, its own study of three independent runs, vendor)
VERIFIERS = {"deepseek-v4-flash-api1-thinking": ("DeepSeek V4 Flash", "experiments/deepseek-v4-flash-thinking-20261006", "deepseek"),
             "gpt-6-luna-thinking": ("GPT-6 Luna", "experiments/gpt-6-luna-thinking-20261006", "openai"),
             "gpt-6.1-sol-thinking": ("GPT-6.1 Sol", "experiments_prospective/gpt-6.1-sol-thinking-20261115", "openai")}

TASKS = json.loads((HERE / "tasks.json").read_text())
TASK_SET = set(TASKS["shared"]) | set(TASKS["other"])
cases, turns, vars_ = load()
vars_ = [v for v in vars_ if v["study"] in STUDIES]


def read_runs(study_dir: Path):
    """(task, run id) -> {(turn, name): (is_set, correct, value)} over the outputs each turn itself stores."""
    out = {}
    for d in sorted(study_dir.glob("runs/*/*/*")):
        if not (d / "programmatic_verification.json").exists():
            continue
        case = d.parts[-3]
        r = json.loads((d / "run.json").read_text())
        p = json.loads((d / "programmatic_verification.json").read_text())
        rt = [t for c in r["conversations"] for t in c["turns"]]
        pt = [t for c in p["conversations"] for t in c["turns"]]
        slots = {}
        for i, (t, v) in enumerate(zip(rt, pt)):
            items = ((v.get("programmatic_verification") or {}).get("variable_verification") or {}).get("items") or {}
            for name in cases[case]["turn_stores"][i]:
                item = items.get(name) or {}
                value = (t.get("outputs") or {}).get(name)
                slots[(i + 1, name)] = (bool(item.get("is_set")) and value is not None, bool(item.get("correct")),
                                        json.dumps(value, default=str)[:120])
        out[(case, d.name)] = slots
    return out


# the answering runs: run 1 (first run id) of each answering configuration, from final_data
run_ids = collections.defaultdict(set)
for v in vars_:
    run_ids[(v["study"], v["case"])].add(v["run_id"])
ANS = collections.defaultdict(dict)        # (answering study, task) -> slot -> (is_set, correct, value)
for v in vars_:
    if v["study"] in ANSWERS and v["case"] in TASK_SET and v["run_id"] == min(run_ids[(v["study"], v["case"])]):
        ANS[(v["study"], v["case"])][(v["turn"], v["var"])] = (bool(v["is_set"]), bool(v["correct"]), v["value"])
WRONG_BY = collections.defaultdict(lambda: collections.defaultdict(set))   # (task, slot) -> wrong value -> configurations
for v in vars_:
    if v["is_set"] and not v["correct"]:
        WRONG_BY[(v["case"], (v["turn"], v["var"]))][v["value"]].add(v["study"])

OWN = {}                                    # verifier -> (task, slot) -> wrong values in its own three runs
for ver, (_, path, _) in VERIFIERS.items():
    own = collections.defaultdict(set)
    for (case, _), slots in read_runs(Path(path)).items():
        for slot, (is_set, ok, value) in slots.items():
            if is_set and not ok:
                own[(case, slot)].add(value)
    OWN[ver] = own

rows = []                                   # one row per output of an answering run, per verifier
task_rows = []
for ans, ans_name in ANSWERS.items():
    for ver, (ver_name, _, ver_vendor) in VERIFIERS.items():
        study = Path("experiments_verify") / f"{ans.rsplit('-', 1)[0]}__{ver}-20261116"
        got = read_runs(study)
        by_task = {case: slots for (case, _), slots in got.items()}
        assert len(got) == len(by_task) == len(TASK_SET), (study, len(got), len(by_task))
        relation = "self" if ANSWERS[ans] == ver_name else ("same vendor" if VENDOR[ans] == ver_vendor else "other vendor")
        for case in sorted(TASK_SET):
            before, after = ANS[(ans, case)], by_task[case]
            all_slots = [(i + 1, n) for i, st in enumerate(cases[case]["turn_stores"]) for n in st]
            task_rows.append(dict(answers=ans_name, verifier=ver_name, task=case,
                                  before=all(before.get(s, (0, 0))[1] for s in all_slots),
                                  after=all(after[s][1] for s in all_slots)))
            for slot in all_slots:
                b_set, b_ok, b_val = before.get(slot, (False, False, None))
                a_set, a_ok, a_val = after[slot]
                if not b_set:
                    state = "unstored before"
                elif b_ok:
                    state = "correct before"
                else:
                    state = "wrong before"
                if state == "wrong before":
                    outcome = "fixed" if a_ok else "kept" if a_set and a_val == b_val else "changed" if a_set else "unstored"
                    others = WRONG_BY[(case, slot)][b_val]
                    shared = sum(VENDOR[c] != VENDOR[ans] for c in others) >= 2
                    gives = b_val in OWN[ver][(case, slot)]
                else:
                    outcome = ("kept" if a_ok else "broken" if a_set else "broken, unstored") if state == "correct before" else \
                              ("fixed" if a_ok else "not fixed")
                    shared = gives = None
                rows.append(dict(answers=ans_name, verifier=ver_name, relation=relation, task=case, turn=slot[0],
                                 output=slot[1], state=state, outcome=outcome, shared=shared, verifier_gives=gives,
                                 value_before=b_val, value_after=a_val if a_set else ""))

out_dir = HERE / "tables"
out_dir.mkdir(exist_ok=True)
with open(out_dir / "verify_outputs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def boot(select, num, n=2000, seed=0):
    """Rate num/den over rows in select, with a task-resampled 95% interval. num: row -> bool."""
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if select(r):
            per[r["task"]][0] += num(r); per[r["task"]][1] += 1
    tasks = sorted(TASK_SET); rng = random.Random(seed)
    point = sum(per[t][0] for t in tasks) / max(1, sum(per[t][1] for t in tasks))
    draws = []
    for _ in range(n):
        s = [rng.choice(tasks) for _ in tasks]; d = sum(per[t][1] for t in s)
        if d:
            draws.append(sum(per[t][0] for t in s) / d)
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1], sum(per[t][1] for t in tasks)


def boot_diff(sel_a, sel_b, num, n=2000, seed=0):
    """Rate in a minus rate in b, on the same task resamples."""
    pa, pb = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for sel, per in ((sel_a, pa), (sel_b, pb)):
            if sel(r):
                per[r["task"]][0] += num(r); per[r["task"]][1] += 1
    tasks = sorted(TASK_SET); rng = random.Random(seed)

    def est(s):
        da, db = sum(pa[t][1] for t in s), sum(pb[t][1] for t in s)
        return sum(pa[t][0] for t in s) / da - sum(pb[t][0] for t in s) / db if da and db else None
    draws = sorted(x for x in (est([rng.choice(tasks) for _ in tasks]) for _ in range(n)) if x is not None)
    return est(tasks), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def fmt(p):
    return f"{p[0]:.1%} [{p[1]:.1%}, {p[2]:.1%}] (n={p[3]})"


def fmtd(p):
    return f"{p[0]:+.1%} [{p[1]:+.1%}, {p[2]:+.1%}]"


fixed = lambda r: r["outcome"] == "fixed"           # noqa: E731
wrong = lambda r: r["state"] == "wrong before"      # noqa: E731
correct = lambda r: r["state"] == "correct before"  # noqa: E731
broken = lambda r: r["outcome"].startswith("broken")  # noqa: E731

print(f"answering outputs per answer set: "
      + ", ".join(f"{a}: {sum(1 for r in rows if r['answers'] == a and r['verifier'] == 'GPT-6.1 Sol')}" for a in ANSWERS.values()))
print(f"wrong before per answer set: "
      + ", ".join(f"{a}: {sum(1 for r in rows if r['answers'] == a and r['verifier'] == 'GPT-6.1 Sol' and wrong(r))}" for a in ANSWERS.values()))

# table 1: per answer set x verifier
t1 = []
print("\nanswers / verifier: task accuracy before -> after on the 149 tasks; wrong outputs fixed / kept / changed / unstored; correct outputs broken")
for a in ANSWERS.values():
    for v in (x[0] for x in VERIFIERS.values()):
        sel = [r for r in rows if r["answers"] == a and r["verifier"] == v]
        tr = [t for t in task_rows if t["answers"] == a and t["verifier"] == v]
        nb, na = sum(t["before"] for t in tr), sum(t["after"] for t in tr)
        w = [r for r in sel if wrong(r)]; c = [r for r in sel if correct(r)]
        cnt = collections.Counter(r["outcome"] for r in w)
        br = sum(broken(r) for r in c); bru = sum(r["outcome"] == "broken, unstored" for r in c)
        print(f"  {a:17} / {v:17}  tasks {nb:3d} -> {na:3d}   wrong {len(w):3d}: fixed {cnt['fixed']:3d} kept {cnt['kept']:3d} "
              f"changed {cnt['changed']:3d} unstored {cnt['unstored']:3d}   correct {len(c):4d}: broken {br:3d} (unstored {bru})")
        sh = [r for r in w if r["shared"]]; un = [r for r in w if not r["shared"]]
        t1.append([a, v, sel[0]["relation"], len(tr), nb, na, len(w), cnt["fixed"], cnt["kept"], cnt["changed"], cnt["unstored"], len(c), br, bru,
                   len(sh), sum(fixed(r) for r in sh), sum(r["outcome"] == "kept" for r in sh), len(un), sum(fixed(r) for r in un),
                   sum(r["outcome"] == "kept" for r in un)])
with open(out_dir / "verify_by_pair.csv", "w", newline="") as f:
    w_ = csv.writer(f)
    w_.writerow(["answers", "verifier", "relation", "tasks", "tasks_passed_before", "tasks_passed_after", "wrong_before",
                 "fixed", "kept", "changed", "unstored", "correct_before", "broken", "broken_unstored",
                 "shared", "shared_fixed", "shared_kept", "unshared", "unshared_fixed", "unshared_kept"])
    w_.writerows(t1)

# table 2: rates with intervals
t2 = []


def line(label, p):
    print(f"  {label:58} {fmt(p)}")
    t2.append([label, round(p[0], 6), round(p[1], 6), round(p[2], 6), p[3]])


print("\nfix rate of wrong outputs")
line("all", boot(wrong, fixed))
for v in (x[0] for x in VERIFIERS.values()):
    line(f"verifier {v}", boot(lambda r, v=v: wrong(r) and r["verifier"] == v, fixed))
for rel in ("self", "same vendor", "other vendor"):
    line(f"relation {rel}", boot(lambda r, rel=rel: wrong(r) and r["relation"] == rel, fixed))
for sh in (True, False):
    line(f"{'shared' if sh else 'unshared'}", boot(lambda r, sh=sh: wrong(r) and r["shared"] is sh, fixed))
    for v in (x[0] for x in VERIFIERS.values()):
        line(f"{'shared' if sh else 'unshared'}, verifier {v}", boot(lambda r, sh=sh, v=v: wrong(r) and r["shared"] is sh and r["verifier"] == v, fixed))
    for rel in ("self", "same vendor", "other vendor"):
        line(f"{'shared' if sh else 'unshared'}, relation {rel}", boot(lambda r, sh=sh, rel=rel: wrong(r) and r["shared"] is sh and r["relation"] == rel, fixed))
for g in (True, False):
    line(f"verifier {'gives' if g else 'does not give'} it", boot(lambda r, g=g: wrong(r) and r["verifier_gives"] is g, fixed))
    for v in (x[0] for x in VERIFIERS.values()):
        line(f"verifier {'gives' if g else 'does not give'} it, {v}", boot(lambda r, g=g, v=v: wrong(r) and r["verifier_gives"] is g and r["verifier"] == v, fixed))
print("\nkept (the same wrong value) among wrong outputs")
for sh in (True, False):
    line(f"kept, {'shared' if sh else 'unshared'}", boot(lambda r, sh=sh: wrong(r) and r["shared"] is sh, lambda r: r["outcome"] == "kept"))
for g in (True, False):
    line(f"kept, verifier {'gives' if g else 'does not give'} it", boot(lambda r, g=g: wrong(r) and r["verifier_gives"] is g, lambda r: r["outcome"] == "kept"))
print("\nbreak rate of correct outputs")
for v in (x[0] for x in VERIFIERS.values()):
    line(f"broken, verifier {v}", boot(lambda r, v=v: correct(r) and r["verifier"] == v, broken))
    line(f"broken with a stored wrong value, verifier {v}", boot(lambda r, v=v: correct(r) and r["verifier"] == v, lambda r: r["outcome"] == "broken"))
with open(out_dir / "verify_rates.csv", "w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["measure", "rate", "lo", "hi", "n"]); w_.writerows(t2)

# predictions
print("\npredictions (PREDICTION.md)")
d1 = boot_diff(lambda r: wrong(r) and r["verifier_gives"] is False, lambda r: wrong(r) and r["verifier_gives"] is True, fixed)
print(f"  1. fix rate, verifier does not give it minus gives it: {fmtd(d1)} -> {'holds' if d1[1] > 0 else 'does not hold'}")
d2 = boot_diff(lambda r: wrong(r) and r["shared"] is False, lambda r: wrong(r) and r["shared"] is True, fixed)
print(f"  2. fix rate, unshared minus shared: {fmtd(d2)} -> {'holds' if d2[1] > 0 else 'does not hold'}")
sol = "GPT-6.1 Sol"
res3 = []
for v in (x[0] for x in VERIFIERS.values()):
    if v != sol:
        d = boot_diff(lambda r: wrong(r) and r["shared"] is True and r["verifier"] == sol,
                      lambda r, v=v: wrong(r) and r["shared"] is True and r["verifier"] == v, fixed)
        res3.append(d[0] > 0)
        print(f"  3. shared fix rate, Sol minus {v}: {fmtd(d)}")
print(f"  3. -> {'holds' if all(res3) else 'does not hold'} (point estimates)")
rel_rates = {rel: boot(lambda r, rel=rel: wrong(r) and r["shared"] is True and r["relation"] == rel, fixed)[0]
             for rel in ("self", "same vendor", "other vendor")}
print(f"  4. shared fix rate by relation: " + ", ".join(f"{k} {v:.1%}" for k, v in rel_rates.items())
      + f" -> {'holds' if min(rel_rates, key=rel_rates.get) == 'self' else 'does not hold'}")
d5 = boot_diff(lambda r: correct(r) and r["verifier"] == "DeepSeek V4 Flash", lambda r: correct(r) and r["verifier"] == sol, broken)
print(f"  5. break rate, DeepSeek V4 Flash minus Sol: {fmtd(d5)} -> {'holds' if d5[0] > 0 else 'does not hold'}")

# ---- further analyses, not among the predictions (written after the results were seen) ----
print("\nfurther analyses (after the results were seen)")
# prediction 1 without the self relation, where the answering run is one of the verifier's own three runs
d1s = boot_diff(lambda r: wrong(r) and r["verifier_gives"] is False and r["relation"] != "self",
                lambda r: wrong(r) and r["verifier_gives"] is True and r["relation"] != "self", fixed)
print(f"  1 without self: fix rate, verifier does not give it minus gives it: {fmtd(d1s)}")
for sh in (True, False):
    for g in (True, False):
        line(f"{'shared' if sh else 'unshared'}, verifier {'gives' if g else 'does not give'} it",
             boot(lambda r, sh=sh, g=g: wrong(r) and r["shared"] is sh and r["verifier_gives"] is g, fixed))
# GPT-6 Luna on its own answers against GPT-6 Luna on the other two answer sets, shared wrong outputs
d4 = boot_diff(lambda r: wrong(r) and r["shared"] is True and r["verifier"] == "GPT-6 Luna" and r["relation"] != "self",
               lambda r: wrong(r) and r["shared"] is True and r["relation"] == "self", fixed)
print(f"  GPT-6 Luna, shared fix rate on other configurations' answers minus on its own: {fmtd(d4)}")

# shown values against answering alone: on the same outputs, the verifier's fix rate against the share of its own
# three independent runs that stored the correct value
ALONE = {}
for ver, (name, path, _) in VERIFIERS.items():
    per = collections.defaultdict(list)
    for (case, _), slots in read_runs(Path(path)).items():
        per[case].append(slots)
    ALONE[name] = per
for r in rows:
    runs_ = ALONE[r["verifier"]][r["task"]]
    r["alone_correct"] = sum(s[(r["turn"], r["output"])][1] for s in runs_) / len(runs_)


def boot_mean(select, val, n=2000, seed=0):
    per = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        if select(r):
            per[r["task"]][0] += val(r); per[r["task"]][1] += 1
    tasks = sorted(TASK_SET); rng = random.Random(seed)
    est = lambda s: sum(per[t][0] for t in s) / sum(per[t][1] for t in s) if sum(per[t][1] for t in s) else None  # noqa: E731
    draws = sorted(x for x in (est([rng.choice(tasks) for _ in tasks]) for _ in range(n)) if x is not None)
    return est(tasks), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


t3 = []
print("  outputs wrong before: verifier's fix rate when shown the values, its correct rate answering alone, difference")
for v in [x[0] for x in VERIFIERS.values()] + [None]:
    for sh in (True, False):
        sel = lambda r, v=v, sh=sh: wrong(r) and r["shared"] is sh and (v is None or r["verifier"] == v)  # noqa: E731
        shown = boot(sel, fixed); alone = boot_mean(sel, lambda r: r["alone_correct"])
        diff = boot_mean(sel, lambda r: fixed(r) - r["alone_correct"])
        label = f"{v or 'pooled'}, {'shared' if sh else 'unshared'}"
        print(f"    {label:30} shown {shown[0]:.1%}  alone {alone[0]:.1%}  difference {fmtd(diff)} (n={shown[3]})")
        t3.append([v or "pooled", "shared" if sh else "unshared", "wrong before", shown[3], round(shown[0], 6), round(alone[0], 6),
                   round(diff[0], 6), round(diff[1], 6), round(diff[2], 6)])
print("  outputs correct before: share kept correct when shown the values, correct rate answering alone, difference")
for v in [x[0] for x in VERIFIERS.values()] + [None]:
    sel = lambda r, v=v: correct(r) and (v is None or r["verifier"] == v)  # noqa: E731
    kept = boot_mean(sel, lambda r: not broken(r)); alone = boot_mean(sel, lambda r: r["alone_correct"])
    diff = boot_mean(sel, lambda r: (not broken(r)) - r["alone_correct"])
    print(f"    {v or 'pooled':30} shown {kept[0]:.1%}  alone {alone[0]:.1%}  difference {fmtd(diff)}")
    t3.append([v or "pooled", "", "correct before", sum(1 for r in rows if sel(r)), round(kept[0], 6), round(alone[0], 6),
               round(diff[0], 6), round(diff[1], 6), round(diff[2], 6)])
with open(out_dir / "verify_vs_alone.csv", "w", newline="") as f:
    w_ = csv.writer(f)
    w_.writerow(["verifier", "class", "outputs", "n", "correct_when_shown", "correct_alone", "difference", "lo", "hi"])
    w_.writerows(t3)

# task accuracy on the 149 tasks: the answering run, the verifier, and the verifier answering alone (mean of its runs)
print("  task accuracy on the 149 tasks: answering run -> verifier; verifier alone")
for v in (x[0] for x in VERIFIERS.values()):
    alone = sum(sum(all(s[x][1] for x in s) for s in ALONE[v][c]) / len(ALONE[v][c]) for c in TASK_SET)
    print(f"    {v:17} alone {alone:5.1f}; " + "; ".join(
        f"{a} {sum(t['before'] for t in task_rows if t['answers'] == a and t['verifier'] == v)} -> "
        f"{sum(t['after'] for t in task_rows if t['answers'] == a and t['verifier'] == v)}" for a in ANSWERS.values()))

# number of other vendors (of the six among the fourteen configurations) that gave the wrong value
ANS_STUDY = {name: study for study, name in ANSWERS.items()}
for r in rows:
    if wrong(r):
        r["other_vendors"] = len({VENDOR[c] for c in WRONG_BY[(r["task"], (r["turn"], r["output"]))][r["value_before"]]
                                  if VENDOR[c] != VENDOR[ANS_STUDY[r["answers"]]]})
kept = lambda r: r["outcome"] == "kept"  # noqa: E731
t4 = []
print("  wrong outputs by the number of other vendors that gave the value: fixed, kept")
for k in range(6):
    sel = lambda r, k=k: wrong(r) and r["other_vendors"] == k  # noqa: E731
    f_, k_ = boot(sel, fixed), boot(sel, kept)
    print(f"    {k} other vendors: fixed {fmt(f_)}  kept {k_[0]:.1%} [{k_[1]:.1%}, {k_[2]:.1%}]")
    t4.append([k, f_[3], *(round(x, 6) for x in f_[:3]), *(round(x, 6) for x in k_[:3])])
with open(out_dir / "verify_by_vendors.csv", "w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["other_vendors", "n", "fixed", "fixed_lo", "fixed_hi", "kept", "kept_lo", "kept_hi"]); w_.writerows(t4)

# the verifier's agreement with the shown value as a check: what it passes
print("  wrong outputs the verifier stored unchanged, and their share among all outputs it stored unchanged")
for v in [x[0] for x in VERIFIERS.values()] + [None]:
    w = [r for r in rows if wrong(r) and (v is None or r["verifier"] == v)]
    passed = [r for r in w if kept(r)]
    same_correct = sum(1 for r in rows if correct(r) and not broken(r) and (v is None or r["verifier"] == v))
    print(f"    {v or 'pooled':18} shared {sum(kept(r) for r in w if r['shared'])}/{sum(1 for r in w if r['shared'])}, "
          f"unshared {sum(kept(r) for r in w if not r['shared'])}/{sum(1 for r in w if not r['shared'])}; "
          f"of the {len(passed)} passed, shared {sum(1 for r in passed if r['shared'])} = {sum(1 for r in passed if r['shared']) / len(passed):.1%}; "
          f"passed wrong among all unchanged or accepted outputs {len(passed) / (len(passed) + same_correct):.2%}")
print(f"  correct outputs that remained correct: {fmt(boot(correct, lambda r: not broken(r)))}")

# outputs every verifier left wrong, and cases a verifier had right in all three of its own runs
by_out = collections.defaultdict(list)
for r in rows:
    if wrong(r):
        by_out[(r["answers"], r["task"], r["turn"], r["output"])].append(r)
never = [v for v in by_out.values() if not any(fixed(r) for r in v)]
print(f"  wrong outputs no verifier fixed: {len(never)} of {len(by_out)} in {len({v[0]['task'] for v in never})} tasks, shared {sum(v[0]['shared'] for v in never)}; "
      f"fixed by at least one {len(by_out) - len(never)} = {(len(by_out) - len(never)) / len(by_out):.1%}")
clean = [r for r in rows if wrong(r) and r["alone_correct"] == 1 and kept(r)]
print(f"  shown wrong values kept by a verifier correct in all three of its own runs: {len(clean)}, shared {sum(r['shared'] for r in clean)}; "
      + ", ".join(f"{v} {sum(1 for r in clean if r['verifier'] == v)}" for v in (x[0] for x in VERIFIERS.values())))


# effort: model calls per task when verifying and when answering alone
def steps(study_dir):
    out = collections.defaultdict(list)
    for d in sorted(Path(study_dir).glob("runs/*/*/*")):
        if (d / "run.json").exists() and d.parts[-3] in TASK_SET:
            r = json.loads((d / "run.json").read_text())
            out[d.parts[-3]].append(sum(t.get("steps") or 0 for c in r["conversations"] for t in c["turns"]))
    return sum(sum(v) / len(v) for v in out.values()) / len(out)


print("  model calls per task, answering alone and verifying (mean over the three answer sets)")
for sec, (name, path, _) in VERIFIERS.items():
    ver = [steps(f"experiments_verify/{a.rsplit('-', 1)[0]}__{sec}-20261116") for a in ANSWERS]
    print(f"    {name:18} alone {steps(path):.1f}  verifying {sum(ver) / len(ver):.1f}")

# sensitivity: without the tasks whose shared error was classed as ambiguity in the task (K9)
K9 = {e["case"] for e in json.loads(Path("analysis/final_20261012/knowledge_classification_20261012.json").read_text()) if e["primary"] == "K9"}
rows[:] = [r for r in rows if r["task"] not in K9]
TASK_SET -= K9
print(f"  without the {len(K9 & (set(TASKS['shared']) | set(TASKS['other'])))} ambiguous tasks ({sum(1 for r in rows if wrong(r)) // 3} wrong outputs):")
line("without ambiguous tasks, shared", boot(lambda r: wrong(r) and r["shared"] is True, fixed))
line("without ambiguous tasks, unshared", boot(lambda r: wrong(r) and r["shared"] is False, fixed))
d_cls = boot_diff(lambda r: wrong(r) and r["shared"] is False, lambda r: wrong(r) and r["shared"] is True, fixed)
d_alone = boot_mean(lambda r: wrong(r) and r["shared"] is True, lambda r: fixed(r) - r["alone_correct"])
print(f"    unshared minus shared {fmtd(d_cls)}; shared, shown minus alone {fmtd(d_alone)}")
with open(out_dir / "verify_sensitivity.csv", "w", newline="") as f:
    w_ = csv.writer(f); w_.writerow(["measure", "rate", "lo", "hi", "n"])
    w_.writerows(t2[-2:] + [["unshared minus shared", *(round(x, 6) for x in d_cls), ""], ["shared, shown minus alone", *(round(x, 6) for x in d_alone), ""]])
