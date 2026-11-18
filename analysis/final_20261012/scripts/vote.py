"""Answering by vote: how accurate a task becomes when several candidates vote on every output, and what is left.

A vote takes, on every output of a task, the value stored by the most candidates; a tie goes to the value of the
more accurate candidate (task accuracy). Candidates without a stored value do not vote, and an output no candidate
stored is wrong. The task passes when the voted value of every output is correct.

Votes:
  own runs        the three runs of one thinking configuration;
  k vendors       every combination of k of the six vendors, each represented by its most accurate thinking
                  configuration (a choice made on these results, as for three-vendor agreement in
                  agreement_signal.py), voting with run r of each, averaged over r = 1, 2, 3.
Task accuracy is averaged over the votes of a kind; 95% intervals from 2,000 task resamples. What is left: of the
outputs on which the vote was wrong, the share whose voted value at least two configurations of vendors outside the
vote also gave (for own runs, vendors other than the configuration's). The rule needs vendors outside the vote, so it
is reported only where at least three of the six remain outside (one to three vendors voting), and pooled over the
thinking configurations for one run against their own three runs. Wrong outputs are given per vote, over all 3,845
outputs, and their change against one run of a vendor's representative for that representative's three runs and for
three vendors.

A check first confirms that a run passes a task exactly when each of its stored outputs is correct, so that a vote
of one candidate reproduces that candidate's task accuracy. Writes tables/vote.csv.
"""
import collections
import itertools
import random
import statistics

from common import load, write_csv
from groups import STUDIES, THINKING, VENDOR, task_accuracy

cases, turns, vars_ = load()
turns = [t for t in turns if t["study"] in STUDIES]
vars_ = [v for v in vars_ if v["study"] in STUDIES]
acc = task_accuracy(turns)
SLOTS = {c: [(i + 1, n) for i, st in enumerate(d["turn_stores"]) for n in st] for c, d in cases.items()}
TASKS = sorted(SLOTS)

run_ids = collections.defaultdict(set)
for t in turns:
    run_ids[(t["study"], t["case"])].add(t["run_id"])
RUN = {k: sorted(v) for k, v in run_ids.items()}                     # (study, task) -> run ids, run index order
OUT = collections.defaultdict(dict)                                  # (study, task, run index) -> slot -> (value, correct)
for v in vars_:
    if v["is_set"]:
        OUT[(v["study"], v["case"], RUN[(v["study"], v["case"])].index(v["run_id"]))][(v["turn"], v["var"])] = (v["value"], bool(v["correct"]))
WRONG_BY = collections.defaultdict(lambda: collections.defaultdict(set))   # (task, slot) -> wrong value -> configurations
for v in vars_:
    if v["is_set"] and not v["correct"]:
        WRONG_BY[(v["case"], (v["turn"], v["var"]))][v["value"]].add(v["study"])

# check: a run passes a task iff every output of the task is stored and correct
passed = collections.defaultdict(list)
for t in turns:
    passed[(t["study"], t["case"], RUN[(t["study"], t["case"])].index(t["run_id"]))].append(t["ok"])
mismatch = sum(all(ok) != all(OUT[k].get(s, (None, False))[1] for s in SLOTS[k[1]]) for k, ok in passed.items())
print(f"check: runs whose task verdict differs from their outputs' verdicts: {mismatch} of {len(passed)}")


def vote(task, members):
    """members: (study, run index), most accurate first. Returns (passed, [(slot, voted value) wrong])."""
    wrong = []
    for slot in SLOTS[task]:
        votes = [OUT[(s, task, r)][slot] for s, r in members if slot in OUT[(s, task, r)]]
        if not votes:
            wrong.append((slot, None)); continue
        n = collections.Counter(value for value, _ in votes); top = max(n.values())
        value, ok = next(x for x in votes if n[x[0]] == top)
        if not ok:
            wrong.append((slot, value))
    return not wrong, wrong


def k_outside(groups):
    return min(6 - len(vendors) for vendors, _ in groups)


def summarise(label, k, groups):
    """groups: list of (vendors of the vote, member lists per run index). Per task, the mean pass over every vote."""
    per_task = collections.defaultdict(list); left = [0, 0]
    for vendors, member_sets in groups:
        for members in member_sets:
            for task in TASKS:
                ok, wrong = vote(task, members)
                per_task[task].append(ok)
                for slot, value in wrong:
                    if value is None:
                        continue
                    left[1] += 1
                    left[0] += sum(VENDOR[c] not in vendors for c in WRONG_BY[(task, slot)][value]) >= 2
    means = [statistics.mean(per_task[t]) for t in TASKS]
    rng = random.Random(0)
    draws = sorted(statistics.mean(rng.choice(means) for _ in means) for _ in range(2000))
    point = statistics.mean(means)
    share = left[0] / left[1] if left[1] and k_outside(groups) >= 3 else float("nan")
    n_votes = sum(len(member_sets) for _, member_sets in groups)
    per_vote = left[1] / n_votes
    print(f"{label:34} k={k}  task accuracy {point:.1%} [{draws[50]:.1%}, {draws[1949]:.1%}]  "
          f"wrong outputs per vote {per_vote:6.1f}, shared {share:.1%}")
    return [label, k, n_votes, round(point, 6), round(draws[50], 6), round(draws[1949], 6), round(per_vote, 1),
            round(share, 6) if share == share else "", round(per_vote * share, 1) if share == share else "",
            round(per_vote * (1 - share), 1) if share == share else ""]


rows = []
order = sorted(THINKING, key=lambda s: -acc[s])
for s in order:
    rows.append(summarise(f"one run: {STUDIES[s][0]}", 1, [({VENDOR[s]}, [[(s, r)] for r in range(3)])]))
    rows.append(summarise(f"own runs: {STUDIES[s][0]}", 3, [({VENDOR[s]}, [[(s, 0), (s, 1), (s, 2)]])]))
for label, k, members in (("one run, thinking configurations", 1, lambda s: [[(s, r)] for r in range(3)]),
                          ("own runs, thinking configurations", 3, lambda s: [[(s, 0), (s, 1), (s, 2)]])):
    rows.append(summarise(label, k, [({VENDOR[s]}, members(s)) for s in order]))
best = {}
for s in order:
    best.setdefault(VENDOR[s], s)
reps = sorted(best.values(), key=lambda s: -acc[s])
print("vendors' representatives:", ", ".join(f"{STUDIES[s][0]} ({acc[s]:.1%})" for s in reps))
rows.append(summarise("own runs, vendor representatives", 3, [({VENDOR[s]}, [[(s, 0), (s, 1), (s, 2)]]) for s in reps]))
for k in range(1, len(reps) + 1):
    groups = [({VENDOR[s] for s in combo}, [[(s, r) for s in combo] for r in range(3)]) for combo in itertools.combinations(reps, k)]
    rows.append(summarise("vendors", k, groups))
# change in wrong outputs per vote against one run of a vendor's representative, shared and unshared
by = {(r[0], r[1]): r for r in rows}
base = by[("vendors", 1)]
for key, label in ((("own runs, vendor representatives", 3), "three own runs"), (("vendors", 3), "three vendors")):
    r = by[key]
    ch = [(r[i] - base[i]) / base[i] for i in (8, 9)]
    print(f"{label}, against one run of a vendor's representative: shared wrong outputs {ch[0]:+.1%}, unshared {ch[1]:+.1%}")
    r += [round(ch[0], 6), round(ch[1], 6)]
write_csv("vote.csv", ["vote", "candidates", "n_votes", "task_accuracy", "lo", "hi", "wrong_outputs_per_vote",
                       "shared_share", "shared_per_vote", "unshared_per_vote", "shared_change", "unshared_change"], rows)
