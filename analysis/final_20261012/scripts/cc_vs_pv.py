"""Claim coverage (CC) against programmatic verification (PV), per turn, for the three single judges.

The CC judge sees the conversation, the pre-registered claims and the agent's final prose, never the reference
values. A turn is CC-passed when no required claim is missed. For each judge: CC pass rate by PV outcome of the
same turn (pass / wrong value / wrong value that is a shared wrong value / unanswered), the AUC of the turn's mean
required-claim coverage as a predictor of PV pass, and the rank correlation between configuration-level CC and PV
task accuracy. Also the three judges' agreement on turn-level CC pass. Writes tables/cc_vs_pv.csv.
"""
import collections
import glob
from pathlib import Path
import json

from common import load, modal_shared, write_csv
from groups import task_accuracy

# each run keeps one claim-coverage file per judge panel: runs/<case>/<model>/<run_id>/claim_coverage/<panel>.json
JUDGES = {"DeepSeek V4 Flash": "deepseek-v4-flash-oneapi", "GPT-6 Luna": "gpt-6-luna-nonthinking", "GLM-5.3 Flash": "glm-5.3-flash"}
cases, turns, vars_ = load()
T = {(t["study"], t["case"], t["run_id"], t["turn"]): t for t in turns}
modal = modal_shared(vars_)
shared_turn = set()
for v in vars_:
    if v["is_set"] and not v["correct"] and modal.get((v["case"], v["turn"], v["var"])) == v["value"]:
        shared_turn.add((v["study"], v["case"], v["run_id"], v["turn"]))


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0] * len(xs)
    for k, i in enumerate(o):
        r[i] = k
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def auc(pos, neg):
    allv = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    # Mann-Whitney with ties
    r = 0.0; i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            r_k = (i + j + 1) / 2
            if allv[k][1]:
                r += r_k
        i = j
    n1, n0 = len(pos), len(neg)
    return (r - n1 * (n1 + 1) / 2) / (n1 * n0)


acc = task_accuracy(turns)
rows = []; per_judge_pass = {}; cfg_rows = collections.defaultdict(dict)
for jname, panel in JUDGES.items():
    cc = {}
    for f in glob.glob(f"experiments/*/runs/*/*/*/claim_coverage/{panel}.json"):
        study, _, case, _, rid = Path(f).relative_to("experiments").parts[:5]
        d = json.load(open(f))
        for c in d["conversations"]:
            for t in c["turns"]:
                if t.get("status") != "scored":
                    continue
                c_ = t["claim_coverage"]; rates = c_.get("coverage_rate") or {}
                if not rates:
                    continue
                cc[(study, case, rid, t["turn"])] = (not c_["required_claims_missed_or_tied"], sum(rates.values()) / len(rates))
    per_judge_pass[jname] = {k: v[0] for k, v in cc.items()}
    groups = collections.defaultdict(list)
    pos, neg = [], []
    cfg = collections.defaultdict(list); cfg_pass = collections.defaultdict(list)
    for k, (ok, mean) in cc.items():
        t = T.get(k)
        if t is None:
            continue
        g = "PV pass" if t["ok"] else ("PV wrong value, shared" if t["kind"] == "wrong_value" and k in shared_turn else
                                       "PV wrong value, other" if t["kind"] == "wrong_value" else "PV unanswered or other failure")
        groups[g].append(ok)
        (pos if t["ok"] else neg).append(mean)
        cfg[k[0]].append(mean); cfg_pass[k[0]].append(ok)
    a = auc(pos, neg)
    studies = sorted(cfg)
    rho = spearman([sum(cfg[s]) / len(cfg[s]) for s in studies], [acc[s] for s in studies])
    for s in studies:
        cfg_rows[s][jname] = sum(cfg_pass[s]) / len(cfg_pass[s])
    print(f"\n{jname}: scored turns {len(cc)} | AUC(mean coverage -> PV pass) {a:.3f} | config rank corr CC vs PV task accuracy {rho:.2f}")
    for g in ("PV pass", "PV wrong value, other", "PV wrong value, shared", "PV unanswered or other failure"):
        xs = groups[g]
        print(f"   CC pass among {g:32} {sum(xs) / len(xs):6.1%}  (n={len(xs)})")
        rows.append((jname, g, round(sum(xs) / len(xs), 6), len(xs), round(a, 6), round(rho, 6)))
# judge agreement on turn-level CC pass
names = list(per_judge_pass); common = set.intersection(*(set(per_judge_pass[n]) for n in names))
agree = sum(len({per_judge_pass[n][k] for n in names}) == 1 for k in common)
print(f"\nthree judges agree on turn-level CC pass: {agree / len(common):.1%} of {len(common)} turns")
rows.append(("all three", "turn-level agreement", round(agree / len(common), 6), len(common), "", ""))
write_csv("cc_by_config.csv", ["study", "pv_task_accuracy"] + [f"cc_turn_pass_{j}" for j in JUDGES],
          [(s, round(acc[s], 6)) + tuple(round(cfg_rows[s][j], 6) for j in JUDGES) for s in sorted(cfg_rows, key=lambda s: -acc[s])])
write_csv("cc_vs_pv.csv", ["judge", "group", "cc_pass_rate", "n_turns", "auc_mean_coverage_pv_pass", "config_rank_corr_cc_pv"], rows)
