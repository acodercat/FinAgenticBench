"""The judge comparisons without self-judging: each judge's verdicts on its own vendor's configurations removed.

Every judge comes from a vendor whose models are also tested (DeepSeek V4 Flash judges DeepSeek configurations,
GPT-6 Luna judges GPT-6 Luna configurations, GLM-5.3 Flash judges GLM-5.3), in both judged channels. For every judge and channel, all configurations and then only the configurations of other
vendors: the area under the curve for a verified pass (claim coverage: the turn's mean coverage over its claims, as
cc_vs_pv.py; response quality: the overall score, as rq_vs_pv.py) against all other turns and against wrong-value
turns, each with a 95% interval from 1,000 task resamples; for claim coverage also the pass rate of verified and of
shared-wrong turns, the Spearman correlation of configuration-level coverage with task accuracy, and the pass rate of
the least and the most accurate remaining configuration. Writes tables/judge_self_exclusion.csv.
"""
import collections
import glob
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from common import load, modal_shared, write_csv
from groups import STUDIES, VENDOR, task_accuracy

JUDGE_VENDOR = {"deepseek-v4-flash-oneapi": "deepseek", "gpt-6-luna-nonthinking": "openai", "glm-5.3-flash": "zhipu"}
cases, turns, vars_ = load()
acc = task_accuracy(turns)
kind = {(t["study"], t["case"], t["run_id"], t["turn"]): t["kind"] for t in turns}
modal = modal_shared(vars_)
shared_turn = set()
for v in vars_:
    if v["is_set"] and not v["correct"] and modal.get((v["case"], v["turn"], v["var"])) == v["value"]:
        shared_turn.add((v["study"], v["case"], v["run_id"], v["turn"]))
reg = json.load(open("benchmarks.json")); reg = reg.get("cases", reg)
REQUIRED = set()
for case, entry in reg.items():
    spec = json.load(open(entry if isinstance(entry, str) else entry["path"]))
    convs = spec.get("conversations") or [{"turns": [{"claims": spec.get("claims", [])}]}]
    for conv in convs:
        for i, turn in enumerate(conv["turns"]):
            if any(c.get("mode") == "required" for c in turn.get("claims", [])):
                REQUIRED.add((case, i + 1))

# records: (study, case, score, cc_pass or None, verification kind, shared)
data = collections.defaultdict(list)       # (channel, judge) -> records
for f in glob.glob("experiments/*/runs/*/*/*/claim_coverage/*.json"):
    study, _, case, _, rid = Path(f).relative_to("experiments").parts[:5]
    judge = Path(f).stem
    for c in json.load(open(f))["conversations"]:
        for t in c["turns"]:
            if t.get("status") != "scored" or (case, t["turn"]) not in REQUIRED:
                continue
            cc = t["claim_coverage"]; rates = cc.get("coverage_rate") or {}
            key = (study, case, rid, t["turn"])
            if rates and key in kind:
                data[("claim coverage", judge)].append((study, case, sum(rates.values()) / len(rates),
                                                        not cc["required_claims_missed_or_tied"], kind[key], key in shared_turn))
for study in STUDIES:
    for f in Path(f"experiments/{study}/runs").glob("*/*/*/response_quality/*.json"):
        d = json.loads(f.read_text()); rid = f.parents[1].name
        for conv in d["conversations"]:
            for t in conv["turns"]:
                key = (study, d["case"], rid, t["turn"])
                if t.get("status") == "scored" and t.get("response_quality") and key in kind:
                    data[("response quality", f.stem)].append((study, d["case"], float(t["response_quality"]["overall_score"]),
                                                               None, kind[key], key in shared_turn))


def auc(scores, labels):
    if labels.all() or not labels.any():
        return float("nan")
    r = rankdata(scores); n1 = labels.sum(); n0 = len(labels) - n1
    return (r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(recs, wrong_only, rng):
    if wrong_only:
        recs = [r for r in recs if r[4] in ("pass", "wrong_value")]
    by_case = collections.defaultdict(list)
    for r in recs:
        by_case[r[1]].append((r[2], r[4] == "pass"))
    cs = sorted(by_case)
    arr = {c: np.array(by_case[c]) for c in cs}
    def est(sample):
        a = np.concatenate([arr[c] for c in sample]); return auc(a[:, 0], a[:, 1].astype(bool))
    point = est(cs)
    draws = sorted(x for x in (est([rng.choice(cs) for _ in cs]) for _ in range(1000)) if x == x)
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


rows = []
for (channel, judge), recs in sorted(data.items()):
    jv = JUDGE_VENDOR[judge]
    for subset, keep in (("all configurations", lambda s: True), (f"excluding {jv}", lambda s: VENDOR[s] != jv)):
        rs = [r for r in recs if keep(r[0])]
        rng = random.Random(f"{channel}|{judge}|{subset}")
        a_all = auc_ci(rs, False, rng); a_wrong = auc_ci(rs, True, rng)
        line = (f"{channel:16} {judge:26} {subset:20} turns {len(rs):6} | AUC pass vs rest {a_all[0]:.3f} [{a_all[1]:.3f}, {a_all[2]:.3f}]"
                f" | pass vs wrong value {a_wrong[0]:.3f} [{a_wrong[1]:.3f}, {a_wrong[2]:.3f}]")
        extra = ["", "", "", "", ""]
        if channel == "claim coverage":
            pv_pass = [r[3] for r in rs if r[4] == "pass"]; sh = [r[3] for r in rs if r[4] == "wrong_value" and r[5]]
            cfg = collections.defaultdict(list); cfg_pass = collections.defaultdict(list)
            for r in rs:
                cfg[r[0]].append(r[2]); cfg_pass[r[0]].append(r[3])
            st = sorted(cfg, key=lambda s: acc[s])
            xs = [sum(cfg[s]) / len(cfg[s]) for s in st]; ys = [acc[s] for s in st]
            rho = np.corrcoef(rankdata(xs), rankdata(ys))[0, 1]
            lo_s, hi_s = st[0], st[-1]
            p_lo, p_hi = sum(cfg_pass[lo_s]) / len(cfg_pass[lo_s]), sum(cfg_pass[hi_s]) / len(cfg_pass[hi_s])
            line += (f"\n{'':64} CC pass: verified {sum(pv_pass) / len(pv_pass):.1%}, shared wrong {sum(sh) / len(sh):.1%} | Spearman {rho:.2f}"
                     f" | least accurate {STUDIES[lo_s][0]} {STUDIES[lo_s][1]} {p_lo:.1%} vs most accurate {STUDIES[hi_s][0]} {STUDIES[hi_s][1]} {p_hi:.1%}")
            extra = [round(sum(pv_pass) / len(pv_pass), 6), round(sum(sh) / len(sh), 6), round(rho, 6), round(p_lo, 6), round(p_hi, 6)]
        print(line)
        rows.append([channel, judge, subset, len(rs), *(round(x, 6) for x in (*a_all, *a_wrong)), *extra])
write_csv("judge_self_exclusion.csv", ["channel", "judge", "subset", "turns", "auc_pass_vs_rest", "lo", "hi", "auc_pass_vs_wrong",
                                       "lo_wrong", "hi_wrong", "cc_pass_verified", "cc_pass_shared_wrong", "spearman_cc_accuracy",
                                       "cc_pass_least_accurate", "cc_pass_most_accurate"], rows)
