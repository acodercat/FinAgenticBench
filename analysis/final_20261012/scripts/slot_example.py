"""One slot in full: every configuration's stored value on each of its three runs, for the slot shown in Fig. 1c.

Reads the run artifacts directly (run.json for the stored value, programmatic_verification.json for the verdict and the
reference) and writes tables/slot_example.csv, the source of Supplementary Table 19."""
import csv
import glob
import json
import re
from pathlib import Path

from groups import MODE, STUDIES

CASE, TURN, VAR = "thirteen_f_option_underlier_exposure_boundary", 1, "option_underlier_q4_accessions"
rows = []
for study in STUDIES:
    for d in sorted(glob.glob(f"experiments/{study}/runs/{CASE}/*/*/")):
        run = json.loads((Path(d) / "run.json").read_text())
        pv = json.loads((Path(d) / "programmatic_verification.json").read_text())
        item = pv["conversations"][0]["turns"][TURN - 1]["programmatic_verification"]["variable_verification"]["items"][VAR]
        stored = run["conversations"][0]["turns"][TURN - 1]["outputs"][VAR]
        m = re.search(r"expected ([\d.]+)", item["message"])
        rows.append({"study": study, "model": STUDIES[study][0], "mode": MODE[study], "run_id": Path(d).name,
                     "stored": stored, "correct": int(item["correct"]), "reference": m.group(1) if m else ""})
ref = {r["reference"] for r in rows if r["reference"]}
assert len(ref) == 1, ref
for r in rows:
    r["reference"] = ref.pop() if not r["reference"] else r["reference"]
    ref = {r["reference"]}
with open("analysis/final_20261012/tables/slot_example.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
wrong = [r for r in rows if not r["correct"]]
modal = max({r["stored"] for r in wrong}, key=lambda v: sum(x["stored"] == v for x in wrong))
print(f"{CASE} turn {TURN} {VAR}: {len(rows)} runs, {len(wrong)} wrong, {sum(r['stored'] == modal for r in wrong)} equal to {modal}, reference {rows[0]['reference']}")
