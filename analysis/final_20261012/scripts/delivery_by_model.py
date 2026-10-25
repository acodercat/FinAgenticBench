"""Per-configuration delivery-layer and self-check numbers (from turn_flags.json), plus
deterministic-misreading share (3 runs wrong with one identical value). Writes tables/delivery_by_model.csv."""
import collections
import json

from common import load, write_csv
from groups import STUDIES, label, task_accuracy

cases, turns, vars_ = load()
flags = json.load(open("analysis/scratch/turn_flags.json"))
acc = task_accuracy(turns)
agg = collections.defaultdict(collections.Counter)
for t in turns:
    fl = flags.get(f"{t['study']}|{t['case']}|{t['run_id']}|{t['turn']}")
    if fl is None:
        continue
    a = agg[t["study"]]
    a["wrong_numeric"] += fl["n_wrong_numeric"]; a["cnd"] += fl["n_computed_not_delivered"]
    a["missing"] += fl["n_missing"] if t["kind"] == "variables_not_set" else 0; a["near"] += fl["n_near_name"]
    a["unans_completed"] += t["kind"] == "variables_not_set" and t["stop"] == "completed"; a["phantom"] += fl["phantom"]
    if t["kind"] in ("pass", "wrong_value"):
        a["checked"] += fl["has_check"]; a["checked_wrong"] += fl["has_check"] and t["kind"] == "wrong_value"
        a["checked_wrong_failed"] += fl["has_check"] and t["kind"] == "wrong_value" and fl["check_failed"]
        a["unchecked"] += not fl["has_check"]; a["unchecked_wrong"] += (not fl["has_check"]) and t["kind"] == "wrong_value"
    a["max_steps"] += t["stop"] == "max_steps"
# deterministic misreading: slots where a config is wrong in all 3 runs
S = collections.defaultdict(lambda: collections.defaultdict(list))
for v in vars_:
    if v["is_set"]:
        S[(v["case"], v["turn"], v["var"])][v["study"]].append((v["correct"], v["value"]))
det = collections.defaultdict(lambda: [0, 0])
for k, m in S.items():
    for s, r in m.items():
        if len(r) == 3 and not any(c for c, _ in r):
            det[s][1] += 1; det[s][0] += len({v for _, v in r}) == 1
rows = []
print(f"{'config':34} {'acc':>6} {'correct value seen in own tool output':>36} {'near-name share of missing':>26} {'phantom/completed-unanswered':>28} {'checked turns wrong':>19} {'unchecked wrong':>15} {'check failed|wrong':>18} {'3x wrong identical':>18}")
for s in sorted(STUDIES, key=lambda s: -acc[s]):
    a = agg[s]
    r = (label(s), acc[s], a["cnd"] / a["wrong_numeric"], a["near"] / a["missing"] if a["missing"] else float("nan"), a["phantom"] / a["unans_completed"] if a["unans_completed"] else float("nan"),
         a["checked_wrong"] / a["checked"], a["unchecked_wrong"] / a["unchecked"], a["checked_wrong_failed"] / a["checked_wrong"] if a["checked_wrong"] else float("nan"), det[s][0] / det[s][1] if det[s][1] else float("nan"))
    print(f"{r[0]:34} {r[1]:6.1%} {r[2]:>22.1%} ({a['cnd']}/{a['wrong_numeric']}) {r[3]:>14.1%} ({a['near']}/{a['missing']}) {r[4]:>14.1%} ({a['phantom']}/{a['unans_completed']}) {r[5]:>12.1%} ({a['checked']}) {r[6]:>14.1%} {r[7]:>18.1%} {r[8]:>12.1%} ({det[s][1]})")
    rows.append((s,) + tuple(round(x, 6) for x in r[1:]) + (a["wrong_numeric"], a["missing"], a["unans_completed"], a["checked"], a["checked_wrong"], det[s][1]))
write_csv("delivery_by_model.csv", ["study", "task_accuracy", "computed_not_delivered_share", "near_name_share_of_missing", "phantom_share_of_completed_unanswered",
                                    "wrong_rate_checked_turns", "wrong_rate_unchecked_turns", "check_failed_share_of_wrong_checked", "three_wrong_identical_share",
                                    "n_wrong_numeric", "n_missing", "n_completed_unanswered", "n_checked_turns", "n_checked_wrong", "n_three_wrong_slots"], rows)
