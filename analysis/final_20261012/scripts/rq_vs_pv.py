"""Does a judge that sees the reference values separate verified turns from wrong ones?

The response-quality channel scored every turn on a five-dimension 1-5 rubric with the recomputed reference outputs
shown to the judge (claim coverage, by contrast, withheld them). Reads the per-run response_quality files of all
studies, joins each turn to its verification outcome from final_data, and reports per judge: mean overall score by
verification outcome, the area under the curve of the overall score as a predictor of a verified pass, and the
Spearman correlation between configuration-level mean score and task accuracy. Writes tables/rq_vs_pv.csv.
"""
import collections
import json
import statistics
from pathlib import Path

from common import load, modal_shared, write_csv
from groups import STUDIES, task_accuracy

cases, turns, vars_ = load()
acc = task_accuracy(turns)
kind = {(t["study"], t["case"], t["run_id"], t["turn"]): t["kind"] for t in turns}
modal = modal_shared(vars_)                                          # the paper's shared definition (>=3 configurations)
shared_turn = collections.defaultdict(bool)
for v in vars_:
    k = (v["case"], v["turn"], v["var"])
    if v["is_set"] and not v["correct"] and modal.get(k) == v["value"]:
        shared_turn[(v["study"], v["case"], v["run_id"], v["turn"])] = True


def group(key):
    k = kind.get(key)
    if k is None:
        return None
    if k == "pass":
        return "PV pass"
    if k == "wrong_value":
        return "PV wrong value, shared" if shared_turn[key] else "PV wrong value, other"
    return "PV unanswered or other failure"


rows_by_judge = collections.defaultdict(list)       # judge -> [(study, group, score_capped, score_raw)]
runs_scored = collections.defaultdict(set)
for study in STUDIES:
    for f in Path(f"experiments/{study}/runs").glob("*/*/*/response_quality/*.json"):
        d = json.loads(f.read_text()); judge = f.stem; run_id = f.parents[1].name
        runs_scored[judge].add((study, d["case"], run_id))
        for conv in d["conversations"]:
            for t in conv["turns"]:
                if t.get("status") != "scored" or not t.get("response_quality"):
                    continue
                g = group((study, d["case"], run_id, t["turn"]))
                if g is None:
                    continue
                rq = t["response_quality"]
                rows_by_judge[judge].append((study, g, float(rq["overall_score"]), float(rq["overall_score_raw"])))


def auc(pos, neg):
    """Probability that a random positive outscores a random negative, ties counting one half."""
    pos, neg = sorted(pos), sorted(neg)
    import bisect
    wins = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def spearman(a, b):
    def rk(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rk(a), rk(b); n = len(a); m = (n - 1) / 2
    return sum((ra[i] - m) * (rb[i] - m) for i in range(n)) / (sum((ra[i] - m) ** 2 for i in range(n)) * sum((rb[i] - m) ** 2 for i in range(n))) ** 0.5


print("runs with a response-quality file: " + ", ".join(f"{j} {len(s):,}" for j, s in sorted(runs_scored.items())))
GROUPS = ("PV pass", "PV wrong value, other", "PV wrong value, shared", "PV unanswered or other failure")
out = []
for judge, rows in sorted(rows_by_judge.items()):
    by_g = collections.defaultdict(list)
    for _, g, s, _ in rows:
        by_g[g].append(s)
    pos = by_g["PV pass"]; neg = [s for g, xs in by_g.items() if g != "PV pass" for s in xs]
    neg_wrong = by_g["PV wrong value, other"] + by_g["PV wrong value, shared"]
    a_all, a_wrong = auc(pos, neg), auc(pos, neg_wrong)
    per_cfg = collections.defaultdict(list)
    for s, _, sc, _ in rows:
        per_cfg[s].append(sc)
    cfgs = sorted(per_cfg); rho = spearman([acc[s] for s in cfgs], [statistics.mean(per_cfg[s]) for s in cfgs])
    print(f"{judge}: turns {len(rows)}; mean overall score " + ", ".join(f"{g} {statistics.mean(by_g[g]):.2f} (n={len(by_g[g])})" for g in GROUPS)
          + f"; AUC for a verified pass {a_all:.3f} (vs wrong values only {a_wrong:.3f}); config rank correlation with task accuracy {rho:.2f}")
    for g in GROUPS:
        out.append([judge, g, round(statistics.mean(by_g[g]), 4), len(by_g[g]), round(a_all, 4), round(a_wrong, 4), round(rho, 4)])
write_csv("rq_vs_pv.csv", ["judge", "group", "mean_overall_score", "n_turns", "auc_pv_pass", "auc_pv_pass_vs_wrong_value", "config_rank_corr_rq_pv"], out)
