"""Rank-one error-correction model for the ten-year yield and speculative positioning.

The ten-year yield, the two-year yield and SOFR move together because they are all
anchored to the same policy path, so a regression in levels between them is
near-spurious while a regression in differences throws the anchor away. An
error-correction model keeps both: a cointegrating relation in levels plus an adjustment
term that pulls the yield back toward it.

Over 261 weekly observations from 2021-01-05 to 2025-12-30 the deepest inversion is
2023-07-03 at -1.08 percentage points, with SOFR at 5.06 percent and leveraged-money net
share at -30.448258 percent of open interest. The fitted cointegrating relation puts a
0.5681 coefficient on the two-year yield and only 0.0202 on SOFR, with an
equilibrium error standard deviation of 0.2069 percentage points. The augmented
Dickey-Fuller coefficient on that error is -0.1103 with a t statistic of -3.1434,
so mean reversion is present but the evidence for it is not overwhelming.

The error-correction equation adjusts at -0.0978 per week, roughly a ten-week
half-life, and explains 62.3818 percent of weekly changes with a maximum normal moment
of 0.0. Out of sample over 105 weeks it beats a random walk, 0.0948 against
0.1118 percentage points of root-mean-square error, an improvement of 0.0170.
That margin is small: the model wins by about fifteen percent of the random walk's error,
which is a real but modest edge, and its own worst week, 2025-04-08 at 0.2609
percentage points, is more than twice its average error. Leveraged-money net share on
that date is -38.307589 percent, a more negative position than at the deepest inversion.

The `treasury_rank1_vecm` pinned-reading comparison records sensitivity to the positioning report
scope, to the training-window end, and to whether the regressors are dated
contemporaneously or lagged. The query fixes all three, so these are robustness
comparisons rather than hidden answer paths.

Boundaries: the cointegrating vector is estimated rather than tested for rank, so a
single relation is imposed by the question and not established by a trace or maximum
eigenvalue statistic. The Dickey-Fuller statistic is reported on a residual from an
estimated relation, which changes its distribution, so it is a diagnostic rather than a
size-correct test. Positioning is a weekly reported aggregate across all leveraged-money
accounts and is not a directional forecast; nothing here identifies causality in either
direction between positioning and yields.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs
def _v(n, d):
    return Variable(n, None, d)
TURN_1_NAMES = [
    "aligned_week_count", "aligned_start_date", "aligned_end_date",
    "minimum_curve_spread_date", "minimum_ten_minus_two_spread_pp",
    "same_date_sofr_pct", "same_date_leveraged_net_share_pct",
]
TURN_2_NAMES = [
    "rank1_training_week_count",
    "cointegrating_intercept",
    "cointegrating_two_year_coefficient",
    "cointegrating_sofr_coefficient",
    "equilibrium_error_std_pp",
    "equilibrium_error_adf_coefficient",
    "equilibrium_error_adf_t_stat",
]
TURN_3_NAMES = [
    "ecm_observation_count", "ecm_intercept_pp", "ecm_adjustment_coefficient",
    "ecm_two_year_change_coefficient", "ecm_sofr_change_coefficient",
    "ecm_r_squared", "ecm_maximum_normal_moment",
]
TURN_4_NAMES = [
    "ecm_test_week_count",
    "ecm_test_rmse_pp",
    "random_walk_test_rmse_pp",
    "ecm_beats_random_walk",
    "largest_error_date",
    "largest_error_pp",
    "same_date_leveraged_net_share_pct_test",
]
names = TURN_1_NAMES+TURN_2_NAMES+TURN_3_NAMES+TURN_4_NAMES
_descriptions = {
    "aligned_week_count":"Store the aligned weekly observation count as an integer.",
    "aligned_start_date":"Store the first aligned report date as an ISO YYYY-MM-DD string.",
    "aligned_end_date":"Store the last aligned report date as an ISO YYYY-MM-DD string.",
    "minimum_curve_spread_date":"Store the minimum 10-year-minus-2-year spread date as an ISO YYYY-MM-DD string.",
    "minimum_ten_minus_two_spread_pp":"Store the signed 10-year-minus-2-year Treasury spread in percentage points, rounded to 4 decimals.",
    "same_date_sofr_pct":"Store same-date SOFR in percent, rounded to 2 decimals.",
    "same_date_leveraged_net_share_pct":"Store the same-date signed leveraged-funds net share of open interest in percent, rounded to 6 decimals.",
    "rank1_training_week_count":"Store the rank-one training-week count as an integer.",
    "cointegrating_intercept":"Store the cointegrating regression intercept in percentage-point units, rounded to 4 decimals.",
    "cointegrating_two_year_coefficient":"Store the coefficient on the 2-year Treasury yield, rounded to 4 decimals.",
    "cointegrating_sofr_coefficient":"Store the coefficient on SOFR, rounded to 4 decimals.",
    "equilibrium_error_std_pp":"Store the sample standard deviation of the equilibrium error in percentage points, rounded to 4 decimals.",
    "equilibrium_error_adf_coefficient":"Store the zero-lag ADF coefficient on the lagged equilibrium error, rounded to 4 decimals.",
    "equilibrium_error_adf_t_stat":"Store the t-statistic for that ADF coefficient, rounded to 4 decimals.",
    "ecm_observation_count":"Store the ECM regression observation count as an integer.",
    "ecm_intercept_pp":"Store the ECM intercept in percentage points, rounded to 6 decimals.",
    "ecm_adjustment_coefficient":"Store the coefficient on the lagged equilibrium error, rounded to 4 decimals.",
    "ecm_two_year_change_coefficient":"Store the coefficient on the contemporaneous 2-year yield change, rounded to 4 decimals.",
    "ecm_sofr_change_coefficient":"Store the coefficient on the contemporaneous SOFR change, rounded to 4 decimals.",
    "ecm_r_squared":"Store the ECM R-squared, rounded to 4 decimals.",
    "ecm_maximum_normal_moment":"Store the maximum absolute normal-equation moment, rounded to 10 decimals.",
    "ecm_test_week_count":"Store the held-out ECM test-week count as an integer.",
    "ecm_test_rmse_pp":"Store the ECM held-out RMSE in percentage points, rounded to 4 decimals.",
    "random_walk_test_rmse_pp":"Store the no-change random-walk held-out RMSE in percentage points, rounded to 4 decimals.",
    "ecm_beats_random_walk":"Store whether the ECM RMSE is lower as a boolean.",
    "largest_error_date":"Store the largest absolute ECM error date as an ISO YYYY-MM-DD string.",
    "largest_error_pp":"Store the signed actual-minus-predicted ECM error in percentage points, rounded to 4 decimals.",
    "same_date_leveraged_net_share_pct_test":"Store that date's signed leveraged-funds net share of open interest in percent, rounded to 6 decimals.",
}
variables = [_v(n, _descriptions[n]) for n in names]
DECIMALS = [0, None, None, None, 4, 2, 6, 0, 4, 4, 4, 4, 4, 4, 0, 6, 4, 4, 4, 4, 10, 0, 4, 4, None, None, 4, 6]
@lru_cache(maxsize=1)
def _panel():
    c = load_expansion_table("treasury_yield_curve").copy()
    c["date"] = pd.to_datetime(c.Date)
    c["two"] = pd.to_numeric(c["2 Yr"])
    c["ten"] = pd.to_numeric(c["10 Yr"])
    s = load_expansion_table("nyfed_reference_rates")
    s = s[s["Rate Type"].eq("SOFR")][["Effective Date", "Rate (%)"]].copy()
    s["date"] = pd.to_datetime(s["Effective Date"])
    s["sofr"] = pd.to_numeric(s["Rate (%)"])
    f = load_expansion_table("cftc_cot")
    f = f[(f.report_scope=="futures_only")&(f.CFTC_Contract_Market_Code.astype(str)=="043602")].copy()
    f["date"] = pd.to_datetime(f["Report_Date_as_YYYY-MM-DD"])
    f["net"] = (f.Lev_Money_Positions_Long_All-f.Lev_Money_Positions_Short_All)/f.Open_Interest_All*100
    p = f[["date", "net"]].merge(c[["date", "two", "ten"]], on="date", validate="one_to_one").merge(s[["date", "sofr"]], on="date", validate="one_to_one")
    p = p[p.date.between("2021-01-01", "2025-12-31")].sort_values("date").reset_index(drop=True)
    p["spread"] = p.ten-p.two
    return p
@lru_cache(maxsize=1)
def _model():
    p = _panel()
    tr = p[p.date.le("2023-12-31")]
    y = tr.ten.to_numpy(float)
    X = np.c_[np.ones(len(tr)), tr[["two", "sofr"]].to_numpy(float)]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y-X@b
    ad = np.c_[np.ones(len(e)-1), e[:-1]]
    de = np.diff(e)
    a = np.linalg.lstsq(ad, de, rcond=None)[0]
    err = de-ad@a
    v = err@err/(len(err)-2)
    t = a[1]/np.sqrt(v*np.linalg.inv(ad.T@ad)[1, 1])
    dy = np.diff(y)
    dx = np.diff(tr[["two", "sofr"]].to_numpy(float), axis=0)
    Z = np.c_[np.ones(len(dy)), e[:-1], dx]
    q = np.linalg.lstsq(Z, dy, rcond=None)[0]
    u = dy-Z@q
    r2 = 1-u@u/((dy-dy.mean())@(dy-dy.mean()))
    return p, tr, b, e, a, float(t), q, float(r2), float(np.max(np.abs(Z.T@u)))
@lru_cache(maxsize=1)
def ground_truth():
    p, tr, b, e, a, t, q, r2, mom = _model()
    sel = p.sort_values(["spread", "date"], ascending=[True, True]).iloc[0]
    test = p[p.date.gt("2023-12-31")]
    errs = []
    rw = []
    for i in test.index:
        prior = p.loc[i-1]
        row = p.loc[i]
        eq = prior.ten-(b[0]+b[1]*prior.two+b[2]*prior.sofr)
        pred = q[0]+q[1]*eq+q[2]*(row.two-prior.two)+q[3]*(row.sofr-prior.sofr)
        act = row.ten-prior.ten
        errs.append(act-pred)
        rw.append(act)
    errs = np.array(errs)
    rw = np.array(rw)
    rm = np.sqrt(np.mean(errs**2))
    rr = np.sqrt(np.mean(rw**2))
    j = int(np.argmax(np.abs(errs)))
    row = test.iloc[j]
    return(
        len(p),
        p.date.iloc[0].strftime("%Y-%m-%d"),
        p.date.iloc[-1].strftime("%Y-%m-%d"),
        sel.date.strftime("%Y-%m-%d"),
        float(sel.spread),
        float(sel.sofr),
        float(sel.net),
        len(tr),
        *map(float, b),
        float(np.std(e, ddof=1)),
        float(a[1]),
        t,
        len(tr)-1,
        *map(float, q),
        r2,
        mom,
        len(test),
        float(rm),
        float(rr),
        bool(rm<rr),
        row.date.strftime("%Y-%m-%d"),
        float(errs[j]),
        float(row.net),
    )
def _val(o, ns):
    by = {v.name:v for v in variables}
    t = dict(zip(by, ground_truth()))
    d = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(o, [by[n] for n in ns], [t[n] for n in ns], [d[n] for n in ns])
def validate_turn_1(o):return _val(o, TURN_1_NAMES)
def validate_turn_2(o):return _val(o, TURN_2_NAMES)
def validate_turn_3(o):return _val(o, TURN_3_NAMES)
def validate_turn_4(o):return _val(o, TURN_4_NAMES)
def validate(o):return _val(o, names)
validators = {
    "validate_alignment":turn_validator(validate_turn_1),
    "validate_rank1":turn_validator(validate_turn_2),
    "validate_ecm":turn_validator(validate_turn_3),
    "validate_test":turn_validator(validate_turn_4),
}
