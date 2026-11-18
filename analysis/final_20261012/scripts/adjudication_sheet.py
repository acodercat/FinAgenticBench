"""Blinded adjudication sheet for the 86 root causes behind the most widely shared wrong values.

For each root cause (knowledge_classification_20261012.json) it takes the slot on which the most configurations gave
the shared wrong value, and writes one item: the queries of the conversation up to that turn, the output's name and
description, the task's data sources, and two candidate values in random order, one the recomputed reference and one
the shared wrong value. An adjudicator who has not seen the results marks which candidate the task requires, or that
the wording supports both. Writes adjudication/items.md (what the adjudicator reads), adjudication/sheet.csv (what
they fill in) and adjudication/key.csv (which candidate is the reference; keep it from the adjudicators).
Needs analysis/scratch/final_data.json and shared_wrong_all_20261012.json. Run from the repository root.
"""
import collections
import csv
import importlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "analysis" / "final_20261012" / "adjudication"
causes = json.loads((ROOT / "analysis/final_20261012/knowledge_classification_20261012.json").read_text())
shared = json.loads((ROOT / "analysis/scratch/shared_wrong_all_20261012.json").read_text())
vars_ = json.loads((ROOT / "analysis/scratch/final_data.json").read_text())["vars"]
registry = json.loads((ROOT / "benchmarks.json").read_text())
registry = registry.get("cases", registry)

correct = collections.defaultdict(collections.Counter)
for v in vars_:
    if v["is_set"] and v["correct"]:
        correct[(v["case"], v["var"])][v["value"]] += 1
by_slot = collections.defaultdict(list)
for r in shared:
    by_slot[(r["case"], r["var"])].append(r)

rng = random.Random(20261114)
order = list(range(len(causes))); rng.shuffle(order)
items, sheet, key, skipped = [], [], [], []
for n, i in enumerate(order, 1):
    c = causes[i]
    cand = [r for var in c["slots"] for r in by_slot.get((c["case"], var), [])]
    if not cand:
        skipped.append((c["case"], c["slots"])); continue
    r = max(cand, key=lambda r: (r["n_cfgs"], r["n"]))
    ref = correct[(c["case"], r["var"])].most_common(1)
    if not ref:
        skipped.append((c["case"], r["var"])); continue
    spec = json.loads((ROOT / registry[c["case"]]).read_text())
    desc = {v.name: v.description for v in importlib.import_module(spec["module"]).variables}
    queries = None
    for conv in spec["conversations"]:
        for t, turn in enumerate(conv["turns"]):
            if r["var"] in turn["stores"]:
                queries = [x["query"] for x in conv["turns"][:t + 1]]
    pair = [("reference", ref[0][0]), ("shared wrong value", r["wrong"])]
    rng.shuffle(pair)
    iid = f"Q{n:02d}"
    items.append(f"## {iid}\n\n**Data sources:** {', '.join(spec['data_sources'])}\n\n" +
                 "\n\n".join(f"**Turn {k}.** {q}" for k, q in enumerate(queries, 1)) +
                 f"\n\n**Output to adjudicate (turn {len(queries)}):** `{r['var']}`: {desc.get(r['var'], '')}\n\n"
                 f"**Candidate A:** `{pair[0][1]}`\n\n**Candidate B:** `{pair[1][1]}`\n")
    sheet.append((iid, r["var"], pair[0][1], pair[1][1], "", "", ""))
    key.append((iid, c["case"], r["var"], "A" if pair[0][0] == "reference" else "B", c["primary"], r["n_cfgs"]))

OUT.mkdir(exist_ok=True)
(OUT / "items.md").write_text(
    "# Adjudication items\n\nFor each item, decide which candidate value the task as worded requires for the named output. "
    "You may use the data sources. Mark A, B, or 'both' if the wording supports both readings, and say in one line why.\n\n" + "\n".join(items))
for name, header, rows in (("sheet.csv", ["item", "output", "candidate_A", "candidate_B", "choice (A/B/both)", "confidence (1-3)", "reason"], sheet),
                           ("key.csv", ["item", "case", "output", "reference_is", "knowledge_class", "configurations_giving_shared_value"], key)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
print(f"{len(sheet)} items written to {OUT}; skipped {len(skipped)}: {skipped}")
