"""Knowledge classification on the final data, and the nine classes by tier x thinking.

Builds knowledge_classification_20261012.json = pre-refix entries minus slots that left the
>=5-config shared-wrong set (the four re-pinned slots, all K9) plus the reader-classified new
slots (knowledge_added_20261012.json). Weights each root cause by the attempts on its slots
(shared_wrong_all_20261012.json). Then per class: share of shared-wrong attempts, author
anticipation, knows-but-fails, error rate per 1000 attempts on the class's slots by tier and
mode, and the paired thinking change. Writes tables/knowledge_by_tier.csv.
"""
import collections
import json

from common import load, write_csv
from groups import MODE, PAIRS, STUDIES, model_tiers, task_accuracy, tiers

K = {"K1": "accounting/standards semantics", "K2": "market microstructure & calendars", "K3": "instrument & contract mechanics",
     "K4": "statistical method conventions", "K5": "data-model & identifier semantics", "K6": "economic scope selection",
     "K7": "unit/scale/sign", "K8": "output protocol", "K9": "contract ambiguity"}
OUT = "analysis/final_20261012/knowledge_classification_20261012.json"
old = json.load(open("analysis/final_20261012/inputs/knowledge_classification_prerefix.json"))
added = json.load(open("analysis/final_20261012/knowledge_added_20261012.json"))
shared = json.load(open("analysis/scratch/shared_wrong_all_20261012.json"))
live = {(r["case"], r["var"]): r for r in shared}
entries = []; dropped = []
for e in old:
    keep = [s for s in e["slots"] if (e["case"], s) in live]
    gone = [s for s in e["slots"] if (e["case"], s) not in live]
    if gone:
        dropped.append((e["case"], gone, e["primary"]))
    if keep:
        entries.append({**e, "slots": keep})
entries += added
json.dump(entries, open(OUT, "w"), indent=1, ensure_ascii=False)
cls_slots = {(e["case"], s) for e in entries for s in e["slots"]}
unclassified = [k for k in live if k not in cls_slots]
print(f"pre-refix root causes {len(old)} ({sum(len(e['slots']) for e in old)} slots); dropped slots {dropped}; added {len(added)} root cause(s) covering {sum(len(e['slots']) for e in added)} slots")
print(f"final: {len(entries)} root causes, {len(cls_slots)} slots classified of {len(live)} shared-wrong slots; unclassified: {unclassified}")

cases, turns, vars_ = load()
acc = task_accuracy(turns); TIER = tiers(acc); MT = model_tiers(acc)
# per class: slots -> wrong value; attempts per study on those slots (set vars), hits = value == shared wrong
slot_class = {}
for e in entries:
    for s in e["slots"]:
        slot_class[(e["case"], s)] = e["primary"]
att = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # class -> study -> [hits, attempts]
for v in vars_:
    k = (v["case"], v["var"])
    if k in slot_class and v["is_set"]:
        c = slot_class[k]; r = live[k]
        att[c][v["study"]][1] += 1; att[c][v["study"]][0] += (v["turn"] == r["turn"] and v["value"] == r["wrong"])


def pooled(c, studies):
    h = sum(att[c][s][0] for s in studies); n = sum(att[c][s][1] for s in studies)
    return h / n * 1000 if n else float("nan")


G = {"strong": [s for s in STUDIES if TIER[s] == "strong"], "mid": [s for s in STUDIES if TIER[s] == "mid"], "weak": [s for s in STUDIES if TIER[s] == "weak"],
     "think": [s for s in STUDIES if MODE[s] == "T"], "nonthink": [s for s in STUDIES if MODE[s] == "N"],
     "pairedN": [N for T, N in PAIRS.values()], "pairedT": [T for T, N in PAIRS.values()],
     "topN": [N for T, N in PAIRS.values() if MT[T] == "top"], "lowT": [T for T, N in PAIRS.values() if MT[T] == "lower"]}
tot_att = sum(r["n"] for r in shared); rows = []
print(f"\n{'class':40} {'roots':>5} {'slots':>5} {'attempts':>8} {'share':>6} {'antic':>6} {'kbf':>6} | per 1000 attempts on these slots: " + " ".join(f"{g:>8}" for g in G))
for c in sorted(K, key=lambda c: -sum(live[(e['case'], s)]['n'] for e in entries if e['primary'] == c for s in e['slots'])):
    es = [e for e in entries if e["primary"] == c]
    n_att = sum(live[(e["case"], s)]["n"] for e in es for s in e["slots"])
    antic = sum(e["anticipated_by_author"] == "yes" for e in es)
    kbf_known = [e for e in es if e["knows_but_fails"] in ("yes", "no")]; kbf = sum(e["knows_but_fails"] == "yes" for e in kbf_known)
    rates = {g: pooled(c, ss) for g, ss in G.items()}
    print(f"{c} {K[c]:37} {len(es):5} {sum(len(e['slots']) for e in es):5} {n_att:8} {n_att / tot_att:6.1%} {antic:>3}/{len(es):<2} {kbf:>3}/{len(kbf_known):<2} | " + " ".join(f"{rates[g]:8.0f}" for g in G))
    rows.append([c, K[c], len(es), sum(len(e["slots"]) for e in es), n_att, round(n_att / tot_att, 6), antic, kbf, len(kbf_known)] + [round(rates[g], 6) for g in G]
                + [round((rates["pairedT"] - rates["pairedN"]) / rates["pairedN"], 3), round((rates["strong"] - rates["weak"]) / rates["weak"], 3), round((rates["lowT"] - rates["topN"]) / rates["topN"], 3)])
print(f"total root causes {len(entries)}, attempts {tot_att}")
write_csv("knowledge_by_tier.csv", ["class", "name", "root_causes", "slots", "attempts", "share", "author_anticipated", "knows_but_fails_yes", "knows_but_fails_known"]
          + [f"{g}_per_1000" for g in G] + ["paired_think_change", "weak_to_strong_change", "topN_to_lowT_change"], rows)
