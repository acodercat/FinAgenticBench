"""Adjudication sheet that asks which of two computations the task requires, not which of two numbers.

adjudication_sheet.py shows two candidate values; most pairs cannot be told apart without recomputing the task. This
script attaches to each candidate the computation that yields it (adjudication/readings.json, written from the
mechanisms in knowledge_classification_20261012.json), in the candidate order already fixed by key.csv. Items whose
shared wrong value comes from an implementation or upstream departure rather than a second reading are listed in
readings.json under `_implementation` and left out. Writes adjudication/items_readings.md and sheet_readings.csv;
key.csv stays the key. Run from the repository root after adjudication_sheet.py.
"""
import csv
import json
import re
from pathlib import Path

OUT = Path("analysis/final_20261012/adjudication")
readings = json.loads((OUT / "readings.json").read_text())
skip = set(readings["_implementation"])
key = {r["item"]: r for r in csv.DictReader(open(OUT / "key.csv"))}
sheet = {r["item"]: r for r in csv.DictReader(open(OUT / "sheet.csv"))}
blocks = re.split(r"(?m)^## (Q\d+)\n", (OUT / "items.md").read_text())[1:]

items, rows = [], []
for iid, body in zip(blocks[::2], blocks[1::2]):
    if iid in skip:
        continue
    text = {key[iid]["reference_is"]: readings[iid]["ref"]}
    text["B" if "A" in text else "A"] = readings[iid]["alt"]
    head = body.split("**Candidate A:**")[0].rstrip()
    s = sheet[iid]
    items.append(f"## {iid}\n{head}\n\n**Approach A:** {text['A']} Result: `{s['candidate_A']}`\n\n"
                 f"**Approach B:** {text['B']} Result: `{s['candidate_B']}`\n")
    rows.append((iid, s["output"], text["A"], s["candidate_A"], text["B"], s["candidate_B"], "", "", ""))

(OUT / "items_readings.md").write_text(
    "# Adjudication items\n\nEach item gives a task, one of its outputs, and two ways of computing that output with "
    "the value each gives. Decide which approach the task as worded requires. Mark A, B, or 'both' if the wording "
    "supports both, give your confidence (1 low, 3 high) and one line of reason. You may consult the data sources.\n\n"
    + "\n".join(items))
with open(OUT / "sheet_readings.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "output", "approach_A", "value_A", "approach_B", "value_B", "choice (A/B/both)",
                "confidence (1-3)", "reason"])
    w.writerows(rows)
print(f"{len(rows)} items written; {len(skip)} implementation items left out")
