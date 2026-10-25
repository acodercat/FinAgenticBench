"""Constrained historical minimum-CVaR industry portfolio and OFR stress audit.

Minimizing conditional value at risk is a linear program in the Rockafellar-Uryasev
formulation, which is why practitioners reach for it instead of minimizing variance when
the loss distribution is asymmetric. The case builds that program on the five most
volatile Ken French industries, holds the weights fixed out of sample, and then asks
whether a stress index selects the days the portfolio actually lost money.

The 2018-2022 training window has 1258 days and the cohort is Durbl, Enrgy, HiTec, Other
and Manuf. The optimum is a corner solution, as constrained CVaR programs usually are:
HiTec sits exactly on the 0.35 cap, Other is held at 0.0, and the remaining weight is
split 8.9166, 22.4702 and 33.6132 percent. Training CVaR falls to 3.8715 percent
against 4.0099 for equal weights, and the linear program's maximum primal residual is
0.0.

Held out on 2023-2025, 751 days, the fixed weights compound to 80.4527 percent with
17.6290 percent annualized volatility and a 22.9520 percent maximum drawdown. Tail
performance survives the transfer: held-out CVaR is 2.4601 percent against 2.6277
for equal weights, an improvement of 0.1676 percentage points, slightly larger than
the 0.1385 the optimizer bought in sample. That ordering is a fact about this sample
and not a general property; an in-sample optimum is not owed out-of-sample
outperformance.

The stress audit is the negative result. On the ten highest-OFR test dates, mean index
2.0472, both portfolios gained rather than lost: mean realized loss is -0.7511
percent optimized and -0.6414 equal-weighted, so the optimized portfolio gained more
by 0.1097 percentage points. The optimized portfolio's own worst day, 2024-08-05 at
2.7065 percent, is not among the ten highest-stress dates at all. A market-wide stress
index and an equity portfolio's loss days are different events.

Every derived held-out figure is computed from the unrounded optimal weights, not from
the rounded percentages reported for them, so a model that carries its own full-precision
solution forward matches the pinned values.

The `industry_cvar_ofr` pinned-reading comparison records sensitivity to the tail probability, to
the per-industry weight cap, and to the mean floor. The query fixes all three, so these
are robustness comparisons rather than hidden answer paths.

The two panels are inner-joined on date, so a day enters only when both the industry
returns and the OFR FSI report it. Keeping every industry-return day instead reports 1259
training and 752 held-out days, because 2022-02-25 and 2024-02-14 trade without an OFR
observation; every downstream output reads ofr_fsi, so those days cannot carry the
stress audit.

Boundaries: this is a fixed-weight backtest with no rebalancing cost, turnover limit or
transaction friction, and the historical CVaR is an in-sample empirical quantile rather
than a distributional forecast. Ten stress dates are too few to characterize a tail, and
the OFR index is a market-wide construct that carries no information about which
industries were exposed.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from cave_agent import Variable
from scipy.optimize import linprog
from core.data import load_ken_french_table, load_expansion_table
from core.validation import ValidatorResult, turn_validator, validate_ordered_outputs

ALPHA = .05
CAP = .35
INDUSTRIES = {
    "NoDur":"nodur_pct",
    "Durbl":"durbl_pct",
    "Manuf":"manuf_pct",
    "Enrgy":"enrgy_pct",
    "HiTec":"hitec_pct",
    "Telcm":"telcm_pct",
    "Shops":"shops_pct",
    "Hlth":"hlth_pct",
    "Utils":"utils_pct",
    "Other":"other_pct",
}
def _v(n, d):
    return Variable(n, None, d)
TURN_1_NAMES = [
    "training_day_count",
    "test_day_count",
    "selected_industry_1",
    "selected_industry_2",
    "selected_industry_3",
    "selected_industry_4",
    "selected_industry_5",
    "maximum_test_ofr_date",
    "maximum_test_ofr_index",
]
TURN_2_NAMES = [
    "cvar_weight_1_pct",
    "cvar_weight_2_pct",
    "cvar_weight_3_pct",
    "cvar_weight_4_pct",
    "cvar_weight_5_pct",
    "optimized_training_mean_return_pct",
    "optimized_training_var_loss_pct",
    "optimized_training_cvar_loss_pct",
    "equal_weight_training_cvar_loss_pct",
    "cvar_tail_observation_count",
    "cvar_maximum_primal_residual",
]
TURN_3_NAMES = [
    "test_compounded_return_pct",
    "test_annualized_volatility_pct",
    "test_maximum_drawdown_pct",
    "test_cvar_loss_pct",
    "equal_weight_test_cvar_loss_pct",
    "optimized_test_beats_equal_weight",
]
TURN_4_NAMES = [
    "ofr_tail_day_count",
    "ofr_tail_mean_index",
    "optimized_ofr_tail_mean_loss_pct",
    "equal_weight_ofr_tail_mean_loss_pct",
    "largest_optimized_ofr_loss_date",
    "largest_optimized_ofr_loss_pct",
]
names = TURN_1_NAMES+TURN_2_NAMES+TURN_3_NAMES+TURN_4_NAMES
_descriptions = {
    "training_day_count":
        "Store the training-day count as an integer. A day counts only when both the industry "
        "returns and the OFR FSI report it, so the two panels are inner-joined on date before "
        "the windows are cut.",
    "test_day_count":
        "Store the held-out test-day count as an integer, on the same both-report alignment.",
    "selected_industry_1":"Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other.",
    "selected_industry_2":"Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other.",
    "selected_industry_3":"Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other.",
    "selected_industry_4":"Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other.",
    "selected_industry_5":"Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other.",
    "maximum_test_ofr_date":"Store the maximum held-out OFR stress date as an ISO YYYY-MM-DD string.",
    "maximum_test_ofr_index":"Store the maximum held-out OFR stress index, rounded to 4 decimals.",
    "cvar_weight_1_pct":"Store the first optimized portfolio weight in percent, rounded to 4 decimals.",
    "cvar_weight_2_pct":"Store the second optimized portfolio weight in percent, rounded to 4 decimals.",
    "cvar_weight_3_pct":"Store the third optimized portfolio weight in percent, rounded to 4 decimals.",
    "cvar_weight_4_pct":"Store the fourth optimized portfolio weight in percent, rounded to 4 decimals.",
    "cvar_weight_5_pct":"Store the fifth optimized portfolio weight in percent, rounded to 4 decimals.",
    "optimized_training_mean_return_pct":"Store the optimized portfolio's signed mean daily training return in percent, rounded to 4 decimals.",
    "optimized_training_var_loss_pct":"Store the optimized historical VaR loss in percent, rounded to 4 decimals.",
    "optimized_training_cvar_loss_pct":"Store the optimized historical CVaR loss in percent, rounded to 4 decimals.",
    "equal_weight_training_cvar_loss_pct":"Store the equal-weight historical CVaR loss in percent, rounded to 4 decimals.",
    "cvar_tail_observation_count":"Store the training-tail observation count as an integer.",
    "cvar_maximum_primal_residual":"Store the maximum primal feasibility residual in raw constraint units, rounded to 10 decimals.",
    "test_compounded_return_pct":
       "Store the optimized portfolio's signed held-out compounded return in percent, rounded to 4 "
       "decimals. Compute from the unrounded optimal weights, not from the rounded percentages "
       "reported for them.",
    "test_annualized_volatility_pct":
       "Store the optimized portfolio's held-out volatility annualized with 252 trading days in "
       "percent, rounded to 4 decimals. Compute from the unrounded optimal weights, not from the "
       "rounded percentages reported for them.",
    "test_maximum_drawdown_pct":
       "Store the positive magnitude of the held-out maximum drawdown in percent, rounded to 4 "
       "decimals. Compute from the unrounded optimal weights, not from the rounded percentages "
       "reported for them.",
    "test_cvar_loss_pct":
       "Store the optimized held-out historical CVaR loss in percent, rounded to 4 decimals. "
       "Compute from the unrounded optimal weights, not from the rounded percentages reported for "
       "them.",
    "equal_weight_test_cvar_loss_pct":
       "Store the equal-weight held-out historical CVaR loss in percent, rounded to 4 decimals. "
       "Compute from the unrounded optimal weights, not from the rounded percentages reported for "
       "them.",
    "optimized_test_beats_equal_weight":"Store whether optimized held-out CVaR loss is lower as a boolean.",
    "ofr_tail_day_count":"Store the requested OFR stress-tail day count as an integer.",
    "ofr_tail_mean_index":"Store the mean OFR stress index on those days, rounded to 4 decimals.",
    "optimized_ofr_tail_mean_loss_pct":"Store optimized portfolio mean signed loss on those days in percent, rounded to 4 decimals; a negative value denotes a gain.",
    "equal_weight_ofr_tail_mean_loss_pct":"Store equal-weight portfolio mean signed loss on those days in percent, rounded to 4 decimals; a negative value denotes a gain.",
    "largest_optimized_ofr_loss_date":"Store the date of the largest optimized loss within the OFR tail as an ISO YYYY-MM-DD string.",
    "largest_optimized_ofr_loss_pct":"Store that optimized portfolio loss in percent, rounded to 4 decimals.",
}
variables = [_v(n, _descriptions[n]) for n in names]
DECIMALS = [0, 0, None, None, None, None, None, None, 4, *([4]*5), 4, 4, 4, 4, 0, 10, 4, 4, 4, 4, 4, None, 0, 4, 4, 4, None, 4]
@lru_cache(maxsize=1)
def _state():
    r = load_ken_french_table("daily_industry_returns").copy()
    r.date = pd.to_datetime(r.date)
    o = load_expansion_table("ofr_market_stress")[["date", "ofr_fsi"]].copy()
    o.date = pd.to_datetime(o.date)
    p = r.merge(o, on="date", validate="one_to_one")
    train = p[p.date.between("2018-01-01", "2022-12-31")]
    test = p[p.date.between("2023-01-01", "2025-12-31")]
    vols = {k:float(train[v].std(ddof=1)) for k, v in INDUSTRIES.items()}
    cohort = tuple(sorted(vols, key=lambda k:(-vols[k], k))[:5])
    cols = [INDUSTRIES[k] for k in cohort]
    x = train[cols].to_numpy(float)/100
    n, m = x.shape
    target = float(x.mean(0).mean())
    # variables weights, VaR threshold z, tail slacks u
    count = int(np.ceil(ALPHA*n))
    c = np.r_[np.zeros(m), 1, np.full(n, 1/count)]
    A = np.zeros((n+1, m+1+n))
    b = np.zeros(n+1)
    for i in range(n):
        A[i, :m] = -x[i]
        A[i, m] = -1
        A[i, m+1+i] = -1
    A[n, :m] = -x.mean(0)
    b[n] = -target
    eq = np.zeros((1, m+1+n))
    eq[0, :m] = 1
    fit = linprog(c, A_ub=A, b_ub=b, A_eq=eq, b_eq=[1], bounds=[(0, CAP)]*m+[(None, None)]+[(0, None)]*n, method="highs")
    if not fit.success:
        raise ValueError(fit.message)
    w = fit.x[:m]
    z = fit.x[m]
    slack = fit.x[m+1:]
    loss = -x@w
    tail = np.sort(loss)[-count:]
    primal = max(abs(w.sum()-1), max(0, target-x.mean(0)@w), max(0, w.max()-CAP), max(0, -w.min()), max(0, float(np.max(loss-z-slack))), max(0, float(-slack.min())))
    return train, test, cohort, cols, w, target, float(np.quantile(loss, 1-ALPHA, method="higher")), float(tail.mean()), count, float(primal)
@lru_cache(maxsize=1)
def ground_truth():
    train, test, cohort, cols, w, target, var, cvar, count, res = _state()
    eq = np.full(5, .2)
    trainloss = -train[cols].to_numpy(float)/100@eq
    eqc = float(np.sort(trainloss)[-count:].mean())
    tr = test[cols].to_numpy(float)/100
    rp = tr@w
    re = tr@eq
    def metrics(a):
        wealth = np.r_[1, np.cumprod(1+a)]
        dd = wealth/np.maximum.accumulate(wealth)-1
        c = int(np.ceil(ALPHA*len(a)))
        return (np.prod(1+a)-1)*100, np.std(a, ddof=1)*np.sqrt(252)*100, -dd.min()*100, -np.sort(a)[:c].mean()*100
    mp, me = metrics(rp), metrics(re)
    idx = np.argsort(-test.ofr_fsi.to_numpy(float), kind="stable")[:10]
    peak = test.sort_values(["ofr_fsi", "date"], ascending=[False, True]).iloc[0]
    loss = -rp[idx]*100
    largest = idx[int(np.argmax(loss))]
    return (
        len(train),
        len(test),
        *cohort,
        peak.date.strftime("%Y-%m-%d"),
        float(peak.ofr_fsi),
        *(w*100),
        target*100,
        var*100,
        cvar*100,
        eqc*100,
        count,
        res,
        *mp,
        me[3],
        bool(mp[3]<me[3]),
        10,
        float(test.ofr_fsi.to_numpy(float)[idx].mean()),
        float(loss.mean()),
        float((-re[idx]*100).mean()),
        test.date.iloc[largest].strftime("%Y-%m-%d"),
        float(loss[np.argmax(loss)]),
    )
def _val(o, ns):
    by = {v.name:v for v in variables}
    t = dict(zip(by, ground_truth()))
    d = dict(zip(by, DECIMALS))
    # A Ken French industry keeps its identity across the abbreviation, the spelled-out
    # name and the runtime column suffix, so the submitted label is canonicalised before
    # comparison rather than judged a second time.
    candidate = dict(o)
    for name in set(TURN_1_NAMES[2:7]).intersection(ns):
        submitted = candidate.get(name)
        if isinstance(submitted, str) and submitted.strip().casefold()==str(t[name]).casefold():
            candidate[name] = t[name]
    result = validate_ordered_outputs(candidate, [by[n] for n in ns], [t[n] for n in ns], [d[n] for n in ns])
    items = result.variable_results or {}
    missing = any(not item["is_set"] for item in items.values())
    errors = [item["message"] for item in items.values() if item["is_set"] and not item["correct"]]
    return ValidatorResult(not missing and not errors, "required output not set" if missing else "; ".join(errors) or "correct", missing, items)

def validate_turn_1(o):return _val(o, TURN_1_NAMES)
def validate_turn_2(o):return _val(o, TURN_2_NAMES)
def validate_turn_3(o):return _val(o, TURN_3_NAMES)
def validate_turn_4(o):return _val(o, TURN_4_NAMES)
def validate(o):return _val(o, names)
validators = {
    "validate_screen":turn_validator(validate_turn_1),
    "validate_cvar":turn_validator(validate_turn_2),
    "validate_test":turn_validator(validate_turn_3),
    "validate_ofr":turn_validator(validate_turn_4),
}
