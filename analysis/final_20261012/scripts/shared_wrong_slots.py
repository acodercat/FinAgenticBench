"""Shared-wrong slots: (case, turn, variable) where >=5 configurations gave one identical wrong value.

Rebuilds analysis/scratch/shared_wrong_all_20261012.json with the fields the reader batches used,
and diffs it against the pre-refix file behind knowledge_classification.json (308 slots).
The same rule applied to inputs/final_data_prerevision_20261011.json reproduces that file exactly.
"""
import collections
import glob
import json
import re

from common import load

OUT = "analysis/scratch/shared_wrong_all_20261012.json"
OLD = "analysis/final_20261012/inputs/shared_wrong_prerevision_20261011.json"
MIN_CFGS = 5

cases, turns, vars_ = load()
S = collections.defaultdict(list)
for v in vars_:
    if v["is_set"] and not v["correct"]:
        S[(v["case"], v["turn"], v["var"])].append(v)
rows = []
for (case, turn, var), xs in sorted(S.items()):
    val, n = collections.Counter(x["value"] for x in xs).most_common(1)[0]
    hits = [x for x in xs if x["value"] == val]
    cfgs = {x["study"] for x in hits}
    if len(cfgs) < MIN_CFGS:
        continue
    x = sorted(hits, key=lambda x: (x["study"], x["run_id"]))[0]
    f = glob.glob(f"experiments/{x['study']}/runs/{case}/*/{x['run_id']}/run.json")[0]
    r = json.load(open(f)); src = [t for c in r["conversations"] for t in c["turns"] if t["turn"] == turn][0]
    code = [c for c in (src.get("code_snippets") or []) if var in c]
    mod = open(glob.glob(f"evals/*/{case}.py")[0]).read()
    m = re.search(r'^"""(.*?)"""', mod, re.S)
    rows.append(dict(case=case, turn=turn, var=var, wrong=val, n=n, n_cfgs=len(cfgs), family=cases[case]["family"],
                     case_type=cases[case]["case_type"], message=x["message"], var_desc=cases[case]["var_desc"].get(var, ""),
                     query=src.get("query", ""), docstring=m.group(1) if m else "", code=code,
                     response=(src.get("response") or "")[:1200], study=x["study"]))
json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"{OUT}: {len(rows)} slots, {sum(r['n'] for r in rows)} attempts, {len({r['case'] for r in rows})} cases")
old = json.load(open(OLD))
ok = {(r["case"], r["turn"], r["var"]): r for r in old}; nk = {(r["case"], r["turn"], r["var"]): r for r in rows}
added = sorted(k for k in nk if k not in ok); gone = sorted(k for k in ok if k not in nk)
changed = sorted(k for k in nk if k in ok and nk[k]["wrong"] != ok[k]["wrong"])
print(f"vs pre-refix: kept {len(nk) - len(added) - len(changed)} | added {len(added)} | gone {len(gone)} | same slot, different modal value {len(changed)}")
for tag, ks in (("ADDED", added), ("GONE", gone), ("CHANGED", changed)):
    for k in ks:
        src = nk.get(k) or ok[k]
        print(f"  {tag:7} {k[0]} t{k[1]} {k[2]}  wrong={src['wrong']} n={src['n']} cfgs={src['n_cfgs']}" + (f" (was {ok[k]['wrong']})" if tag == "CHANGED" else ""))
json.dump({"added": added, "gone": gone, "changed": changed}, open("analysis/final_20261012/tables/shared_wrong_diff.json", "w"), indent=1)
