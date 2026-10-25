"""Oil-conditioned Gaussian GARCH tail diagnostics for energy-industry returns.

Energy-industry excess returns are heteroskedastic and partly driven by oil, so a
constant-variance value-at-risk understates risk in turbulent months and overstates it
in calm ones. The case fits a GARCH(1,1) whose conditional mean carries the
contemporaneous WTI log return, then asks whether the resulting conditional tail
forecasts are calibrated out of sample.

The 2018-2022 fit gives an oil mean beta of 0.43729 and a variance recursion with
alpha 0.13192 and beta 0.85231, so persistence is 0.984227 and unconditional
volatility 2.033 percent per day. The training log likelihood is -2347.796258 and
the variance recursion reproduces itself exactly, with a maximum recursion residual of
0.0.

Over the 745 held-out days the one-percent conditional Gaussian bound is breached 7
times against 7.45 expected, a 0.939597 percent rate, and the Kupiec unconditional
coverage test cannot reject calibration with a likelihood ratio of 0.0280 and a
p-value of 0.8671. The interesting result is what happens inside the stress
subsample: on the ten highest-OFR-stress days conditional volatility nearly doubles to
2.1358 percent against 1.195352 on the other days, yet not one of those days breaches
the bound and the realized mean loss is -0.097 percent, a gain. Average expected
shortfall on those days is 5.8522 percent, so the model overshoots realized loss by
5.9492 percentage points. A model that passes an unconditional coverage test can still
be badly calibrated on the subsample a risk manager cares about.

The query fixes the estimator completely, including the two-start L-BFGS-B optimizer,
its relative objective tolerance and iteration cap, the parameterization and the
starting values, so the fitted parameters are determined by the question rather than by
the solver a model happens to reach for. The five parameters are pinned at 5 decimals
and every downstream quantity is computed from unrounded parameters.

Precision is set by what the optimizer reproduces, not by what the arithmetic could
print. Running this module on two machines that differ only in their BLAS moves the
fitted alpha in its sixth decimal, and the quantities derived from it move further:
persistence by 2e-8, implied unconditional volatility by 2e-6 and the stress-tail
Gaussian ES by 2e-6. Persistence, the perturbation loss, the conditional Gaussian VaR
and ES averages, and the three stress-tail volatility and ES figures are therefore
pinned at 4 to 6 decimals -- each with at least twenty-five times the observed
cross-machine spread in its tolerance -- so the shared rounding-boundary tolerance
absorbs solver dispersion between conforming implementations without admitting a second
estimator.

The `energy_garch_ofr_tail` pinned-reading comparison records sensitivity to using simple rather
than log WTI returns, to the tail probability, and to the number of stress days in the
subsample. The query fixes all three, so these are robustness comparisons rather than
hidden answer paths.

Boundaries: a Gaussian innovation is assumed rather than tested, so the tail forecasts
inherit that assumption and expected shortfall in particular is sensitive to it. The
conditional mean uses the contemporaneous oil return, which is a same-day conditioning
variable and not a forecast a risk manager would have had. The maximum stress day in the
sample, 2020-03-19 at an index of 10.266, sits inside the training window, so the
held-out stress days are milder, with a mean index of 2.0472.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from core.data import load_expansion_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"
TAIL_PROBABILITY = 0.01
STRESS_DAYS = 10


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "aligned_observation_count", "training_observation_count", "test_observation_count",
    "maximum_ofr_stress_date", "maximum_ofr_stress_index", "same_day_energy_excess_return_pct",
    "same_day_wti_return_pct", "minimum_wti_price_usd_per_barrel",
]
TURN_2_NAMES = [
    "garch_mean_intercept_pct", "garch_oil_mean_beta", "garch_omega_pct2",
    "garch_alpha", "garch_beta", "garch_persistence", "garch_unconditional_volatility_pct",
    "garch_training_log_likelihood", "garch_variance_recursion_max_residual_pct2",
    "garch_minimum_perturbation_log_likelihood_loss",
]
TURN_3_NAMES = [
    "conditional_test_start_date", "conditional_test_end_date", "conditional_var_breach_count",
    "conditional_var_expected_breach_count", "conditional_var_breach_rate_pct",
    "conditional_gaussian_var_average_loss_pct", "conditional_gaussian_es_average_loss_pct",
    "kupiec_unconditional_coverage_lr", "kupiec_unconditional_coverage_p_value",
]
TURN_4_NAMES = [
    "ofr_stress_tail_day_count", "ofr_stress_tail_mean_index",
    "stress_tail_mean_conditional_volatility_pct", "other_days_mean_conditional_volatility_pct",
    "stress_tail_var_breach_count", "stress_tail_realized_mean_loss_pct",
    "stress_tail_gaussian_es_mean_loss_pct",
]

variables = [_v(name, description) for name, description in [
    (TURN_1_NAMES[0], "Store the full aligned daily observation count."),
    (TURN_1_NAMES[1], "Store the 2018-2022 training count."),
    (TURN_1_NAMES[2], "Store the 2023-2025 conditional-test count."),
    (TURN_1_NAMES[3], "Store the maximum OFR stress date as a YYYY-MM-DD string."),
    (TURN_1_NAMES[4], "Store its OFR stress index, rounded to 4 decimals."),
    (TURN_1_NAMES[5], "Store same-day energy excess return in percent, rounded to 4 decimals."),
    (TURN_1_NAMES[6], "Store same-day WTI log return in percent, rounded to 4 decimals."),
    (TURN_1_NAMES[7], "Store the minimum aligned WTI price in USD/barrel, rounded to 2 decimals."),
    (TURN_2_NAMES[0], "Store the conditional-mean intercept in percent, rounded to 5 decimals."),
    (TURN_2_NAMES[1], "Store the WTI-return mean coefficient, rounded to 5 decimals."),
    (TURN_2_NAMES[2], "Store GARCH omega in squared-percent units, rounded to 5 decimals."),
    (TURN_2_NAMES[3], "Store GARCH alpha, rounded to 5 decimals."),
    (TURN_2_NAMES[4], "Store GARCH beta, rounded to 5 decimals."),
    (TURN_2_NAMES[5], "Store alpha plus beta, rounded to 6 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_2_NAMES[6],
       "Store the fitted model's implied daily unconditional volatility in percent, rounded to 4 "
       "decimals. Compute from the unrounded fitted parameters, not from the rounded figures "
       "reported for them."),
    (TURN_2_NAMES[7], "Store maximized Gaussian training log likelihood, rounded to 6 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_2_NAMES[8], "Store max absolute unscaled variance-recursion residual, rounded to 10 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_2_NAMES[9],
       "Store the minimum fitted-minus-perturbed log likelihood over the declared local "
       "perturbations, rounded to 5 decimals. Compute from the unrounded fitted parameters, not "
       "from the rounded figures reported for them."),
    (TURN_3_NAMES[0], "Store first conditional-test date as an ISO YYYY-MM-DD string."),
    (TURN_3_NAMES[1], "Store last conditional-test date as an ISO YYYY-MM-DD string."),
    (TURN_3_NAMES[2], "Store 1% VaR breach count. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_3_NAMES[3], "Store expected 1% breach count, rounded to 2 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_3_NAMES[4], "Store realized breach rate in percent, rounded to 6 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_3_NAMES[5],
       "Store average conditional Gaussian 1% VaR loss in percent, rounded to 5 decimals. Compute "
       "from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_3_NAMES[6], "Store average conditional Gaussian 1% ES loss in percent, rounded to 5 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_3_NAMES[7], "Store Kupiec unconditional-coverage LR, rounded to 4 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_3_NAMES[8], "Store its chi-square(1) p-value, rounded to 4 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_4_NAMES[0], "Store the requested OFR stress-tail day count."),
    (TURN_4_NAMES[1], "Store mean OFR index on those days, rounded to 4 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_4_NAMES[2], "Store their mean daily conditional volatility in percent, rounded to 4 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_4_NAMES[3],
       "Store other test days' mean daily conditional volatility in percent, rounded to 6 decimals. "
       "Compute from the unrounded fitted parameters, not from the rounded figures reported for "
       "them."),
    (TURN_4_NAMES[4], "Store stress-tail VaR breach count. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_4_NAMES[5], "Store stress-tail realized mean loss in percent, rounded to 4 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
    (TURN_4_NAMES[6], "Store stress-tail mean Gaussian ES loss in percent, rounded to 4 decimals. Compute from the unrounded fitted parameters, not from the rounded figures reported for them."),
]]

DECIMALS = [0, 0, 0, None, 4, 4, 4, 2, 5, 5, 5, 5, 5, 6, 4, 6, 10, 5, None, None, 0, 2, 6, 5, 5, 4, 4, 0, 4, 4, 6, 0, 4, 4]


@lru_cache(maxsize=1)
def _panel():
    energy = load_ken_french_table("daily_industry_returns")[["date", "enrgy_pct"]]
    factors = load_ken_french_table("daily_factors")[["date", "rf_pct"]]
    wti = load_expansion_table("eia_bulk_wti_daily").copy()
    wti["date"] = pd.to_datetime(wti.observation_date)
    wti["price"] = pd.to_numeric(
        wti.wti_spot_price_usd_per_barrel, errors="coerce"
    ).astype(float)
    wti = wti.sort_values("date")
    valid = wti.price.gt(0) & wti.price.shift().gt(0)
    wti["oil_return"] = np.nan
    wti.loc[valid, "oil_return"] = (
        np.log(wti.loc[valid, "price"].to_numpy() /
               wti.price.shift().loc[valid].to_numpy()) * 100
    )
    ofr = load_expansion_table("ofr_market_stress")[["date", "ofr_fsi"]].copy()
    for frame in (energy, factors, ofr):
        frame["date"] = pd.to_datetime(frame.date)
    panel = energy.merge(factors, on="date", validate="one_to_one").merge(
        wti[["date", "price", "oil_return"]], on="date", validate="one_to_one"
    ).merge(ofr, on="date", validate="one_to_one").dropna()
    panel["energy_excess"] = panel.enrgy_pct - panel.rf_pct
    return panel.loc[panel.date.between(TRAIN_START, TEST_END)].reset_index(drop=True)


def _unpack(theta):
    omega = np.exp(theta[2])
    ea, eb = np.exp(theta[3]), np.exp(theta[4])
    scale = 0.999 / (1 + ea + eb)
    return theta[0], theta[1], omega, ea * scale, eb * scale


def _variance_path(residual, omega, alpha, beta, initial):
    variance = np.empty(len(residual))
    variance[0] = initial
    for index in range(1, len(residual)):
        variance[index] = omega + alpha * residual[index-1] ** 2 + beta * variance[index-1]
    return variance


@lru_cache(maxsize=1)
def _fit():
    panel = _panel()
    train = panel.loc[panel.date.le(TRAIN_END)]
    y = train.energy_excess.to_numpy(float)
    oil = train.oil_return.to_numpy(float)
    base = np.linalg.lstsq(np.column_stack([np.ones(len(y)), oil]), y, rcond=None)[0]
    raw = y - base[0] - base[1] * oil
    initial = float(np.var(raw, ddof=1))
    def objective(theta):
        mean, oil_beta, omega, alpha, beta = _unpack(theta)
        residual = y - mean - oil_beta * oil
        variance = _variance_path(residual, omega, alpha, beta, initial)
        return float(.5 * np.sum(np.log(2*np.pi) + np.log(variance) + residual**2/variance))
    starts = [
        np.array([base[0], base[1], np.log(initial*.03), -2.0, 2.0]),
        np.array([base[0], base[1], np.log(initial*.08), -1.0, 1.0]),
    ]
    fits = [minimize(objective, start, method="L-BFGS-B", options={"maxiter":3000, "ftol":1e-13}) for start in starts]
    fit = min((item for item in fits if item.success), key=lambda item: item.fun)
    parameters = _unpack(fit.x)
    residual = y - parameters[0] - parameters[1] * oil
    variance = _variance_path(residual, *parameters[2:], initial)
    return train, parameters, residual, variance, -float(fit.fun)


@lru_cache(maxsize=1)
def ground_truth():
    panel = _panel()
    train, p, residual, variance, loglike = _fit()
    mean, oil_beta, omega, alpha, beta = p
    peak = panel.sort_values(["ofr_fsi", "date"], ascending=[False, True]).iloc[0]
    test = panel.loc[panel.date.gt(TRAIN_END)].copy()
    all_residual = panel.energy_excess.to_numpy(float) - mean - oil_beta * panel.oil_return.to_numpy(float)
    all_variance = np.empty(len(panel))
    all_variance[:len(train)] = variance
    for index in range(len(train), len(panel)):
        all_variance[index] = omega + alpha*all_residual[index-1]**2 + beta*all_variance[index-1]
    test_variance = all_variance[len(train):]
    sigma = np.sqrt(test_variance)
    conditional_mean = mean + oil_beta*test.oil_return.to_numpy(float)
    q = norm.ppf(TAIL_PROBABILITY)
    threshold = conditional_mean + sigma*q
    breaches = test.energy_excess.to_numpy(float) < threshold
    var_loss = -threshold
    es_loss = -(conditional_mean - sigma*norm.pdf(q)/TAIL_PROBABILITY)
    n, x = len(test), int(breaches.sum())
    phat = x/n
    if x in (0, n):
        lr = np.inf
    else:
        lr = -2*((n-x)*np.log(1-TAIL_PROBABILITY)+x*np.log(TAIL_PROBABILITY)-(n-x)*np.log(1-phat)-x*np.log(phat))
    stress_index = np.argsort(-test.ofr_fsi.to_numpy(float), kind="stable")[:STRESS_DAYS]
    mask = np.zeros(n, bool)
    mask[stress_index] = True
    realized_loss = -test.energy_excess.to_numpy(float)
    recursion = variance[1:] - (omega + alpha*residual[:-1]**2 + beta*variance[:-1])
    initial = float(variance[0])
    def fixed_mean_loglike(candidate):
        path = _variance_path(residual, *candidate, initial)
        return float(-.5*np.sum(np.log(2*np.pi)+np.log(path)+residual**2/path))
    perturbed = [
        (omega*1.05, alpha, beta), (omega*.95, alpha, beta),
        (omega, alpha*1.01, beta), (omega, alpha, beta*0.999),
    ]
    perturbation_loss = min(loglike-fixed_mean_loglike(candidate) for candidate in perturbed)
    return (
        len(panel), len(train), len(test), peak.date.strftime("%Y-%m-%d"), float(peak.ofr_fsi),
        float(peak.energy_excess), float(peak.oil_return), float(panel.price.min()),
        mean, oil_beta, omega, alpha, beta, alpha+beta, float(np.sqrt(omega/(1-alpha-beta))),
        loglike, float(np.max(np.abs(recursion))), float(perturbation_loss), test.date.iloc[0].strftime("%Y-%m-%d"),
        test.date.iloc[-1].strftime("%Y-%m-%d"), x, n*TAIL_PROBABILITY, x/n*100,
        float(np.mean(var_loss)), float(np.mean(es_loss)), float(lr), float(chi2.sf(lr, 1)),
        STRESS_DAYS, float(test.ofr_fsi.to_numpy(float)[mask].mean()), float(sigma[mask].mean()),
        float(sigma[~mask].mean()), int(breaches[mask].sum()), float(realized_loss[mask].mean()),
        float(es_loss[mask].mean()),
    )


def _validate_subset(outputs, names):
    by = {v.name:v for v in variables}
    truth = dict(zip(by, ground_truth()))
    decimals = dict(zip(by, DECIMALS))
    return validate_ordered_outputs(outputs, [by[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(o): return _validate_subset(o, TURN_1_NAMES)
def validate_turn_2(o): return _validate_subset(o, TURN_2_NAMES)
def validate_turn_3(o): return _validate_subset(o, TURN_3_NAMES)
def validate_turn_4(o): return _validate_subset(o, TURN_4_NAMES)
def validate(o): return _validate_subset(o, [v.name for v in variables])
validators = {
    "validate_alignment": turn_validator(validate_turn_1),
    "validate_garch": turn_validator(validate_turn_2),
    "validate_coverage": turn_validator(validate_turn_3),
    "validate_stress_tail": turn_validator(validate_turn_4),
}
