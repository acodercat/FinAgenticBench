"""Numbers quoted in the paper that no table carries: Gemini's share of strong-tier budget exhaustion,
share of max_steps turns that never assigned a target variable, pooled identical-wrong rate."""
import collections
import glob
import json
import re

from common import load
from groups import task_accuracy, tiers

cases, turns, vars_ = load()
T = tiers(task_accuracy(turns))
ms = [t for t in turns if t["stop"] == "max_steps"]; strong = [t for t in ms if T[t["study"]] == "strong"]
gem = sum(t["study"].startswith("gemini") for t in strong)
print(f"max_steps turns {len(ms)}; strong tier {len(strong)}, of which Gemini {gem} ({gem / len(strong):.1%})")
never = 0
for t in ms:
    r = json.load(open(glob.glob(f"experiments/{t['study']}/runs/{t['case']}/*/{t['run_id']}/run.json")[0]))
    src = [x for c in r["conversations"] for x in c["turns"] if x["turn"] == t["turn"]][0]
    stores = cases[t["case"]]["turn_stores"][t["turn"] - 1]
    never += not any(re.search(rf"^\s*{re.escape(s)}\s*=", c, re.M) for c in (src.get("code_snippets") or []) for s in stores)
print(f"never assigned any target variable: {never}/{len(ms)} = {never / len(ms):.0%}")
S = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"]:
        S[(v["case"], v["turn"], v["var"])][v["study"]].append((v["correct"], v["value"]))
a = b = 0
for m in S.values():
    for r in m.values():
        if len(r) == 3 and not any(c for c, _ in r):
            b += 1; a += len({v for _, v in r}) == 1
print(f"config wrong in all 3 runs with one identical value: {a}/{b} = {a / b:.0%}")
k = json.load(open("analysis/final_20261012/knowledge_classification_20261012.json"))
print("knows_but_fails=yes by class:", dict(collections.Counter(e["primary"] for e in k if e["knows_but_fails"] == "yes")),
      "| decidable:", sum(e["knows_but_fails"] in ("yes", "no") for e in k))

# share of code snippets that are explicit checks: budget-exhausted turns vs a sample of passing turns (defects F6 definition)
import random, statistics
random.seed(0)
chk = re.compile(r"\b(assert|abs\(.*-.*\)\s*<|np\.isclose|allclose|check|verify|sanity|recompute|re-?run)\b", re.I)
def snippet_share(ts):
    shares = []
    for t in ts:
        r = json.load(open(glob.glob(f"experiments/{t['study']}/runs/{t['case']}/*/{t['run_id']}/run.json")[0]))
        code = [x for c in r["conversations"] for x in c["turns"] if x["turn"] == t["turn"]][0].get("code_snippets") or []
        if code:
            shares.append(sum(1 for c in code if chk.search(c)) / len(code))
    return statistics.mean(shares)
passing = random.sample([t for t in turns if t["ok"] and (t["steps"] or 0) >= 5], 600)
print(f"explicit-check share of snippets: max_steps turns {snippet_share(ms):.1%} | passing turns (n=600 sample) {snippet_share(passing):.1%}")
