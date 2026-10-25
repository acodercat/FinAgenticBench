"""Finding 7: four intuitions, raw vs within-task (case, turn) controlled.

Step count, number of required outputs, use of every declared table, and hedging wording
(the last re-reads run.json responses). Writes tables/intuitions.csv.
"""
import collections
import glob
import json
import re
import statistics

from common import load, runs, write_csv

cases, turns, vars_ = load()
rows = []
scored = [t for t in turns if t["status"] == "scored"]

# 1. steps: raw buckets, then within (case, turn) above/below median among completed turns
b = collections.defaultdict(lambda: [0, 0])
for t in scored:
    k = min((t["steps"] or 0) // 10 * 10, 30); b[k][0] += t["ok"]; b[k][1] += 1
print("steps (raw):", {f"{k}-{k + 9}" if k < 30 else "30+": f"{v[0] / v[1]:.1%}" for k, v in sorted(b.items())})
by = collections.defaultdict(list)
for t in scored:
    if t["stop"] != "max_steps":
        by[(t["case"], t["turn"])].append(t)
hi, lo = [0, 0], [0, 0]
for ts in by.values():
    med = statistics.median(t["steps"] or 0 for t in ts)
    for t in ts:
        (hi if (t["steps"] or 0) > med else lo)[0] += t["ok"]; (hi if (t["steps"] or 0) > med else lo)[1] += 1
print(f"steps within task: above-median {hi[0] / hi[1]:.1%} vs at/below {lo[0] / lo[1]:.1%} -> {lo[0] / lo[1] - hi[0] / hi[1]:+.1%}")
rows += [("steps", "raw 30+ vs <10", round(b[30][0] / b[30][1], 6), round(b[0][0] / b[0][1], 6)),
         ("steps", "within-task above vs at/below median", round(hi[0] / hi[1], 6), round(lo[0] / lo[1], 6))]

# 2. number of outputs required
nv = collections.defaultdict(lambda: [0, 0])
for t in turns:
    k = min(len(cases[t["case"]]["turn_stores"][t["turn"] - 1]), 7); nv[k][0] += t["ok"]; nv[k][1] += 1
print("outputs required -> pass:", {k: f"{v[0] / v[1]:.1%}" for k, v in sorted(nv.items())})
for k, v in sorted(nv.items()):
    rows.append(("n_outputs", str(k), round(v[0] / v[1], 6), v[1]))

# 3. every declared table used (run level, within case)
R = runs(turns); by_case = collections.defaultdict(lambda: {True: [0, 0], False: [0, 0]})
for (s, case, rid), ts in R.items():
    unref_all = set.intersection(*[set(t["unref"]) for t in ts]) if ts else set()
    ok = all(t["ok"] for t in ts); by_case[case][len(unref_all) > 0][0] += ok; by_case[case][len(unref_all) > 0][1] += 1
within = [m[False][0] / m[False][1] - m[True][0] / m[True][1] for m in by_case.values() if m[True][1] >= 2 and m[False][1] >= 2]
print(f"every declared table used vs >=1 never loaded, within case ({len(within)} cases): {statistics.mean(within):+.1%}")
rows.append(("tables", "within-case all-used minus some-ignored", round(statistics.mean(within), 6), len(within)))

# 4. hedging cues within (case, turn)
pat = {"assumption": re.compile(r"\b(assum\w*|treated as|interpret\w* (?:this|the) as)\b", re.I),
       "verified": re.compile(r"\b(verif\w+|cross-check\w*|reconcil\w+|sanity check|double-check\w*|validated?)\b", re.I),
       "uncertain": re.compile(r"\b(cannot (?:be )?determin\w*|not (?:directly )?available|unavailable|could not (?:find|locate|identify)|uncertain|unclear|ambigu\w*|caveat|limitation)\b", re.I)}
T = {(t["study"], t["case"], t["run_id"], t["turn"]): t for t in turns}
slot = {n: collections.defaultdict(lambda: {True: [0, 0], False: [0, 0]}) for n in pat}
raw = {n: {True: [0, 0], False: [0, 0]} for n in pat}
for f in glob.glob("experiments/*/runs/*/*/*/run.json"):
    r = json.load(open(f)); study = f.split("/")[1]
    for c in r["conversations"]:
        for t in c["turns"]:
            tt = T.get((study, r["case"], r["run"]["run_id"], t["turn"]))
            if not tt or tt["status"] != "scored" or tt["kind"] == "variables_not_set":
                continue
            resp = t.get("response") or ""
            for n, p in pat.items():
                hit = bool(p.search(resp))
                raw[n][hit][0] += tt["ok"]; raw[n][hit][1] += 1
                slot[n][(r["case"], t["turn"])][hit][0] += tt["ok"]; slot[n][(r["case"], t["turn"])][hit][1] += 1
for n in pat:
    rr = raw[n]; raw_gap = rr[False][0] / rr[False][1] - rr[True][0] / rr[True][1]
    gaps = [(c[False][0] / c[False][1] - c[True][0] / c[True][1], min(c[True][1], c[False][1])) for c in slot[n].values() if c[True][1] >= 3 and c[False][1] >= 3]
    wg = sum(g * w for g, w in gaps) / sum(w for _, w in gaps)
    print(f"cue '{n}': raw gap {raw_gap:+.1%} (no-cue minus cue) | within-task {wg:+.1%} over {len(gaps)} slots, {sum(g > 0 for g, _ in gaps) / len(gaps):.0%} positive")
    rows.append((f"cue:{n}", "raw no-cue minus cue", round(raw_gap, 6), rr[True][1]))
    rows.append((f"cue:{n}", "within-task no-cue minus cue", round(wg, 6), len(gaps)))
write_csv("intuitions.csv", ["intuition", "contrast", "value_a", "value_b_or_n"], rows)
