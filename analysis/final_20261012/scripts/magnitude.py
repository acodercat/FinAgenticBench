import json, re, collections
d = json.load(open("analysis/scratch/final_data.json")); V = d["vars"]; C = d["cases"]
def parse(msg):
    m = re.match(r"^(\S+?)=(.*?),? expected (.*)$", msg)
    if not m: return None
    got, exp = m.group(2).strip(), m.group(3).strip()
    return got, exp
def num(s):
    try: return float(s.replace(",", ""))
    except: return None
cats = collections.Counter(); by_model = collections.defaultdict(collections.Counter); examples = collections.defaultdict(list)
per_var_cat = {}
for v in V:
    if not (v["is_set"] and v["correct"] is False): continue
    p = parse(v["message"])
    if not p: cats["unparsed"] += 1; continue
    got, exp = p; g, e = num(got), num(exp)
    if g is None or e is None:
        cat = "string/token"
        if got.strip("'\"").lower() == exp.strip("'\"").lower() and got != exp: cat = "string: case only"
        elif re.fullmatch(r"'?\d{4}-\d{2}-\d{2}'?", exp.strip()): cat = "string: date"
    elif e == 0: cat = "ref zero"
    elif g == e: cat = "equal?? (precision/format)"
    else:
        r = g / e if e else None
        if r is not None and r < 0 and abs(abs(r) - 1) < 0.01: cat = "sign flip"
        elif r is not None and any(abs(r / 10**k - 1) < 0.01 for k in (-6,-4,-3,-2,-1,1,2,3,4,6)): cat = "power-of-ten scale"
        elif r is not None and any(abs(r / f - 1) < 0.005 for f in (12, 1/12, 4, 1/4, 100/36000, 360/365, 365/360, 252/365, 365/252, 2, 0.5)): cat = "convention factor (12/4/360/252/2)"
        elif abs(g - e) == 1 and float(e).is_integer(): cat = "off by one (count)"
        elif float(e).is_integer() and float(g).is_integer() and abs(g - e) <= 5: cat = "small count diff (2-5)"
        elif abs(g - e) / abs(e) < 0.001: cat = "near miss <0.1% (rounding/convention)"
        elif abs(g - e) / abs(e) < 0.01: cat = "near miss 0.1-1%"
        elif abs(g - e) / abs(e) < 0.1: cat = "1-10% off"
        else: cat = "far off >10%"
    cats[cat] += 1; by_model[v["study"]][cat] += 1
    per_var_cat[(v["study"], v["case"], v["run_id"], v["turn"], v["var"])] = cat
    if len(examples[cat]) < 3: examples[cat].append((v["case"], v["var"], got[:30], exp[:30]))
tot = sum(cats.values())
print("== wrong set variables by magnitude category (n=%d)" % tot)
for c, n in cats.most_common(): print(f"  {n:6} {n/tot:6.1%}  {c}")
print("\n== examples"); [print(" ", c, ex) for c, exs in examples.items() for ex in exs[:2]]
print("\n== share by model (top categories)")
top = [c for c, _ in cats.most_common(8)]
print(f"{'':42}" + "".join(f"{c[:14]:>15}" for c in top))
for s in sorted(by_model, key=lambda s: -sum(by_model[s].values())):
    n = sum(by_model[s].values()); print(f"{s[:42]:42}" + "".join(f"{by_model[s][c]/n:15.1%}" for c in top) + f"  n={n}")
json.dump({"|".join(map(str, k)): v for k, v in per_var_cat.items()}, open("analysis/scratch/var_cats.json", "w"))
