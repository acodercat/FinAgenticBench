"""Finding 3: what thinking removes — a model's own errors or the errors shared with other models.

For each of the 6 same-model pairs, error rate per 1000 set-variable attempts, split into
shared (the slot's modal wrong value, produced by >=3 configs) vs other, and by error shape.
Writes tables/thinking_by_error_class.csv.
"""
import collections

from common import load, load_cats, slots, write_csv
from groups import PAIRS

cases, turns, vars_ = load()
cats = load_cats()
S = slots(vars_)
modal = {}
for k, xs in S.items():
    wrongs = [x for x in xs if not x["correct"]]
    if len(wrongs) < 3:
        continue
    val, n = collections.Counter(x["value"] for x in wrongs).most_common(1)[0]
    if n >= 3 and len({x["study"] for x in wrongs if x["value"] == val}) >= 3:
        modal[k] = val
att = collections.Counter(v["study"] for v in vars_)
ORDER = ["far off >10%", "1-10% off", "near miss 0.1-1%", "near miss <0.1% (rounding/convention)",
         "convention factor (12/4/360/252/2)", "power-of-ten scale", "sign flip", "off by one (count)",
         "small count diff (2-5)", "string/token", "string: date"]
rows = []
pool = collections.defaultdict(lambda: [0, 0, 0, 0])   # key -> [n_nonthink, att_nonthink, n_think, att_think]
for model, (T, N) in PAIRS.items():
    c = {T: collections.Counter(), N: collections.Counter()}
    for k, xs in S.items():
        for x in xs:
            if x["study"] in c and not x["correct"]:
                shared = k in modal and x["value"] == modal[k]
                c[x["study"]]["shared" if shared else "other"] += 1
                c[x["study"]]["shape:" + cats.get((x["study"], x["case"], x["run_id"], x["turn"], x["var"]), "?")] += 1
    for s in (N, T):
        c[s]["unset"] = sum(1 for v in vars_ if v["study"] == s and not v["is_set"])
    for key in ["shared", "other", "unset"] + ["shape:" + o for o in ORDER]:
        rn, rt = c[N][key] / att[N] * 1000, c[T][key] / att[T] * 1000
        pool[key][0] += c[N][key]; pool[key][1] += att[N]; pool[key][2] += c[T][key]; pool[key][3] += att[T]
        rows.append((model, key, round(rn, 6), round(rt, 6), round((rt - rn) / rn, 6) if rn else ""))
print(f"{'error class':44} {'non-think /1000':>15} {'think /1000':>12} {'change':>8}   (pooled over 6 pairs)")
for key, (a, b, c_, d) in pool.items():
    rn, rt = a / b * 1000, c_ / d * 1000
    print(f"{key:44} {rn:15.1f} {rt:12.1f} {(rt - rn) / rn:+8.0%}")
    rows.append(("POOLED", key, round(rn, 6), round(rt, 6), round((rt - rn) / rn, 6)))
write_csv("thinking_by_error_class.csv", ["model", "error_class", "nonthinking_per_1000", "thinking_per_1000", "relative_change"], rows)
