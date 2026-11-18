"""Agreement of the two readers' adjudications with the references and with each other.

Reads adjudication/sheet_readings_1.csv and sheet_readings_2.csv (filled independently by the two readers, who did
not see key.csv) and maps each choice to the reference or the shared reading through the candidate values in
sheet.csv and key.csv. Writes tables/adjudication.csv (summary) and tables/adjudication_items.csv (per item).
Run from the repository root.
"""
import collections
import csv
from pathlib import Path

D = Path("analysis/final_20261012/adjudication")
T = Path("analysis/final_20261012/tables")
key = {r["item"]: r for r in csv.DictReader(open(D / "key.csv"))}
sheet = {r["item"]: r for r in csv.DictReader(open(D / "sheet.csv"))}
readers = [{r["item"]: r for r in csv.DictReader(open(D / f"sheet_readings_{i}.csv"))} for i in (1, 2)]
items = sorted(readers[0])
assert items == sorted(readers[1])


def norm(v):
    v = v.strip().strip('"')
    try:
        return float(v)
    except ValueError:
        return v


rows = []
for q in items:
    ref_side = key[q]["reference_is"]
    ref = norm(sheet[q]["candidate_" + ref_side])
    alt = norm(sheet[q]["candidate_" + ("B" if ref_side == "A" else "A")])
    out = [q, key[q]["case"], key[q]["output"], key[q]["knowledge_class"]]
    for r in readers:
        x = r[q]
        va, vb = norm(x["value_A"]), norm(x["value_B"])
        assert {va, vb} == {ref, alt}, q
        c = x["choice (A/B/both)"].strip()
        out += ["both" if c == "both" else ("reference" if (va if c == "A" else vb) == ref else "shared"),
                x["confidence (1-3)"]]
    rows.append(out)

n = len(rows)
c1 = collections.Counter(r[4] for r in rows)
c2 = collections.Counter(r[6] for r in rows)
both_ref = sum(r[4] == r[6] == "reference" for r in rows)
both_shared = sum(r[4] == r[6] == "shared" for r in rows)
agree = sum(r[4] == r[6] for r in rows)
p = (c1["reference"] + c2["reference"]) / (2 * n)
pe = 2 * p * (1 - p)
po = agree / n
summary = [
    ("root_causes", 86), ("implementation_items", 86 - n), ("items", n),
    ("r1_reference", c1["reference"]), ("r1_shared", c1["shared"]), ("r1_both", c1["both"]),
    ("r2_reference", c2["reference"]), ("r2_shared", c2["shared"]), ("r2_both", c2["both"]),
    ("both_reference", both_ref), ("both_reference_pct", round(100 * both_ref / n, 1)),
    ("both_shared", both_shared), ("readers_agree", agree), ("readers_agree_pct", round(100 * po, 1)),
    ("gwet_ac1", round((po - pe) / (1 - pe), 2)),
]
T.mkdir(parents=True, exist_ok=True)
with open(T / "adjudication.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["measure", "value"]); w.writerows(summary)
with open(T / "adjudication_items.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "case", "output", "knowledge_class", "reader1", "reader1_confidence", "reader2",
                "reader2_confidence"])
    w.writerows(rows)
for k, v in summary:
    print(f"{k:22s} {v}")
print("items where a reader did not choose the reference:")
for r in rows:
    if r[4] != "reference" or r[6] != "reference":
        print("  " + " ".join(r))
