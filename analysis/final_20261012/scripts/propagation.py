"""Finding 4: errors propagate across turns.

Pass rate of turn t given turn t-1 passed / was wrong / was unanswered, per study and pooled;
then the within-(case, turn) contrast, which removes task difficulty. Writes tables/propagation.csv.
"""
import collections

from common import load, runs, write_csv
from groups import label

cases, turns, vars_ = load()
R = runs(turns)
cond = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
slotcond = collections.defaultdict(lambda: {"pp": [0, 0], "wp": [0, 0]})
origin = collections.Counter(); recover = [0, 0]
for (s, case, rid), ts in R.items():
    fails = [t for t in ts if not t["ok"]]
    if fails:
        f0 = fails[0]["turn"]; origin[f0] += 1
        later = [t for t in ts if t["turn"] > f0]
        if later:
            recover[0] += all(t["ok"] for t in later); recover[1] += 1
    for a, b in zip(ts, ts[1:]):
        key = "prev pass" if a["ok"] else ("prev wrong_value" if a["kind"] == "wrong_value" else "prev unanswered")
        cond[s][key][0] += b["ok"]; cond[s][key][1] += 1
        if a["kind"] in ("pass", "wrong_value"):
            k = "pp" if a["ok"] else "wp"
            slotcond[(case, b["turn"])][k][0] += b["ok"]; slotcond[(case, b["turn"])][k][1] += 1
pool = collections.defaultdict(lambda: [0, 0]); rows = []
print(f"{'study':40} {'prev pass':>12} {'prev wrong':>12} {'prev unans':>12}")
for s in sorted(cond, key=lambda s: -cond[s]["prev pass"][0] / max(1, cond[s]["prev pass"][1])):
    cells = []
    for key in ("prev pass", "prev wrong_value", "prev unanswered"):
        c = cond[s][key]; pool[key][0] += c[0]; pool[key][1] += c[1]
        cells.append(c[0] / c[1] if c[1] else float("nan"))
        rows.append((s, key, c[0], c[1], round(c[0] / c[1], 6) if c[1] else ""))
    print(f"{label(s):40} " + " ".join(f"{x:12.1%}" for x in cells))
print("pooled:", {k: f"{v[0] / v[1]:.1%} (n={v[1]})" for k, v in pool.items()})
gaps = [(c["pp"][0] / c["pp"][1] - c["wp"][0] / c["wp"][1], c["wp"][1]) for c in slotcond.values() if c["pp"][1] >= 3 and c["wp"][1] >= 3]
w_gap = sum(g * w for g, w in gaps) / sum(w for _, w in gaps)
print(f"within-task: {len(gaps)} (case,turn) slots; next-turn pass is {w_gap:+.1%} higher after a correct turn than after a wrong one; "
      f"{sum(g > 0 for g, _ in gaps) / len(gaps):.0%} of slots agree in sign")
tot_failed = sum(origin.values())
print(f"failed runs {tot_failed}; first failure at turn: {dict(sorted(origin.items()))}; later turns all recover in {recover[0]}/{recover[1]}")
for key, v in pool.items():
    rows.append(("POOLED", key, v[0], v[1], round(v[0] / v[1], 6)))
rows.append(("WITHIN_TASK", "pass_gap_after_correct_vs_wrong", round(w_gap, 6), len(gaps), round(sum(g > 0 for g, _ in gaps) / len(gaps), 6)))
write_csv("propagation.csv", ["study", "condition", "passes", "n", "rate"], rows)
