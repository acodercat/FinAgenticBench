"""Per-configuration task accuracy with a case-resampled 95% bootstrap interval, plus turn accuracy,
three-run stability and mean tokens per case. Writes tables/headline.csv (the source for Table 1 and Fig. 1c)."""
import collections
import random

from common import load, runs, write_csv
from groups import MODE, STUDIES, task_accuracy, tiers

cases, turns, vars_ = load()
R = runs(turns)
per = collections.defaultdict(lambda: collections.defaultdict(list))   # study -> case -> [ok per run]
tok = collections.defaultdict(list); steps = collections.defaultdict(list)
for (s, c, rid), ts in R.items():
    per[s][c].append(all(t["ok"] for t in ts))
    tok[s].append(sum(t["prompt_tokens"] + t["completion_tokens"] for t in ts)); steps[s].append(sum(t["steps"] or 0 for t in ts))
acc = task_accuracy(turns); T = tiers(acc)
rng = random.Random(0); CASES = sorted(cases)
rows = []
for s in sorted(STUDIES, key=lambda s: -acc[s]):
    m = {c: sum(v) / len(v) for c, v in per[s].items()}
    draws = sorted(sum(m[c] for c in (rng.choice(CASES) for _ in CASES)) / len(CASES) for _ in range(4000))
    lo, hi = draws[int(0.025 * 4000)], draws[int(0.975 * 4000) - 1]
    all3 = sum(all(v) for v in per[s].values()); any1 = sum(any(v) for v in per[s].values())
    turn_acc = sum(t["ok"] for t in turns if t["study"] == s) / sum(1 for t in turns if t["study"] == s)
    rows.append((s, STUDIES[s][0], MODE[s], T[s], round(acc[s], 6), round(lo, 6), round(hi, 6), all3, any1, round(turn_acc, 6),
                 round(sum(tok[s]) / len(tok[s])), round(sum(steps[s]) / len(steps[s]), 1)))
    print(f"{STUDIES[s][0]:18} {MODE[s]} {T[s]:6} {acc[s]:.1%} [{lo:.1%}, {hi:.1%}] all3={all3} any={any1} turn={turn_acc:.1%} tok={rows[-1][10]:,} steps={rows[-1][11]}")
write_csv("headline.csv", ["study", "model", "mode", "tier", "task_accuracy", "ci_lo", "ci_hi", "all_three", "at_least_one", "turn_accuracy", "mean_tokens_per_case", "mean_steps_per_case"], rows)
