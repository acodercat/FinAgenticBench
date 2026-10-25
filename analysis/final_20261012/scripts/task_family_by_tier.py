"""Task accuracy by task family and difficulty, per tier and per thinking mode. Writes tables/task_family_by_tier.csv."""
import collections

from common import load, runs, write_csv
from groups import PAIRS, STUDIES, model_tiers, task_accuracy, tiers

cases, turns, vars_ = load()
acc = task_accuracy(turns); TIER = tiers(acc); MT = model_tiers(acc)
G = {"strong": [s for s in STUDIES if TIER[s] == "strong"], "mid": [s for s in STUDIES if TIER[s] == "mid"], "weak": [s for s in STUDIES if TIER[s] == "weak"],
     "pairedN": [N for T, N in PAIRS.values()], "pairedT": [T for T, N in PAIRS.values()],
     "topN": [N for T, N in PAIRS.values() if MT[T] == "top"], "lowT": [T for T, N in PAIRS.values() if MT[T] == "lower"]}
R = runs(turns)
ok = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for (s, c, rid), ts in R.items():
    for key in (("family", cases[c]["family"]), ("difficulty", cases[c]["difficulty"]), ("n_turns", str(cases[c]["n_turns"])), ("sources", "multi" if len(cases[c]["sources"]) > 1 else "single"), ("all", "all")):
        ok[key][s][0] += all(t["ok"] for t in ts); ok[key][s][1] += 1
rows = []
print(f"{'axis':10} {'level':32} " + " ".join(f"{g:>8}" for g in G) + "   weak->strong pts  N->T pts")
for key in sorted(ok):
    r = {g: sum(ok[key][s][0] for s in ss) / sum(ok[key][s][1] for s in ss) for g, ss in G.items()}
    ncase = sum(ok[key][s][1] for s in G["strong"]) // 3 // len(G["strong"])
    print(f"{key[0]:10} {key[1]:32} " + " ".join(f"{r[g]:8.1%}" for g in G) + f"   {100 * (r['strong'] - r['weak']):+6.1f}       {100 * (r['pairedT'] - r['pairedN']):+6.1f}   (cases={ncase})")
    rows.append([key[0], key[1], ncase] + [round(r[g], 6) for g in G] + [round(r["strong"] - r["weak"], 6), round(r["pairedT"] - r["pairedN"], 6)])
write_csv("task_family_by_tier.csv", ["axis", "level", "cases"] + list(G) + ["weak_to_strong_pts", "paired_think_pts"], rows)
