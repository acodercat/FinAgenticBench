"""Build an equal-risk-contribution industry portfolio before an OFR stress peak.

The later current-revised OFR history selects the largest 2019--2025 FSI level.
That date fixes a 504-common-date training sample ending twenty observations
before the peak and a disjoint sixty-date stress window.  The five industries
with the highest training volatility form the carried cohort; no stress-window
return enters selection or estimation.

The nearest active case, ``industry_beta_crisis_prediction_failure``, shares
the Ken French sources and 22 dates in the COVID stress episode.  It tests the
breakdown of pre-crisis single-factor predictions and an oil-factor omission;
this case instead solves a covariance-based risk budget and allocates portfolio
expected shortfall.  The shared event is therefore a portfolio-concentration
watch item, not the case's financial construct or terminal diagnostic.

The case then estimates a decimal-return sample covariance and Fama--French
three-factor loadings.  Equal-risk-contribution weights are the unique positive,
fully invested solution for equal risk budgets.  The canonical implementation
uses cyclic coordinate descent.  An independent damped-Newton implementation in
the case-local tests agrees to machine precision.  Equal portfolio weights are
not equal risk contributions and produce a measured alternative stress result.
The reported maximum contribution-share deviation is pinned at 5 decimals, so the
exact equal-risk-contribution solution reports zero and the shared rounding-boundary
tolerance absorbs the residual any properly converged solver leaves behind.

The final historical expected shortfall uses daily-rebalanced portfolio returns
and the two worst portfolio-return days in the sixty-date window.  Component
contributions are evaluated on exactly those tail dates and add to portfolio ES,
which is the explicit invariant.  The exercise is a retrospective stress test
using current-revised OFR and Ken French files; it is not a real-time trading
strategy, causal attribution or evidence of future performance.

The `industry_erc_stress` pinned-reading comparison records sensitivity to selecting the cohort
on the stress window rather than the training window, to a population rather than
sample covariance denominator, to allocating equal weight rather than equal risk
contribution, to raw rather than excess factor returns, and to taking the floor rather
than the ceiling of the tail count. The risk-allocation axis is the one worth naming:
equal portfolio weights are not equal risk contributions, and substituting them
produces a measured alternative stress result. The query fixes all five, so these are
robustness comparisons rather than hidden answer paths.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_ken_french_table
from core.validation import (
    ValidatorResult,
    turn_validator,
    validate_ordered_outputs,
)


INDUSTRIES = {
    "NoDur": "nodur_pct", "Durbl": "durbl_pct", "Manuf": "manuf_pct",
    "Enrgy": "enrgy_pct", "HiTec": "hitec_pct", "Telcm": "telcm_pct",
    "Shops": "shops_pct", "Hlth": "hlth_pct", "Utils": "utils_pct",
    "Other": "other_pct",
}
PEAK_START = "2019-01-01"
PEAK_END = "2025-12-31"
TRAINING_OBSERVATIONS = 504
PRE_PEAK_GAP = 20
POST_PEAK_OBSERVATIONS = 40
COHORT_SIZE = 5
ANNUAL_DAYS = 252


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "selected_stress_peak_date", "selected_stress_peak_fsi",
    "training_start_date", "training_end_date", "stress_start_date",
    "stress_end_date", "stress_observation_count",
    "volatility_rank_1_industry", "volatility_rank_2_industry",
    "volatility_rank_3_industry", "volatility_rank_4_industry",
    "volatility_rank_5_industry",
]
TURN_2_NAMES = [
    "minimum_covariance_eigenvalue", "covariance_condition_number",
    "highest_market_beta_industry", "highest_market_beta",
    "lowest_market_beta_industry", "lowest_market_beta",
]
TURN_3_NAMES = [
    "erc_weight_rank_1_pct", "erc_weight_rank_2_pct", "erc_weight_rank_3_pct",
    "erc_weight_rank_4_pct", "erc_weight_rank_5_pct",
    "erc_annualized_training_volatility_pct",
    "maximum_erc_share_deviation_pp", "erc_market_beta",
    "erc_smb_beta", "erc_hml_beta",
]
TURN_4_NAMES = [
    "stress_tail_observation_count", "erc_stress_cumulative_return_pct",
    "erc_stress_annualized_volatility_pct", "erc_stress_maximum_drawdown_pct",
    "erc_stress_expected_shortfall_pct", "equal_weight_stress_cumulative_return_pct",
    "equal_weight_stress_expected_shortfall_pct", "largest_erc_es_contributor_industry",
    "largest_erc_es_contribution_pct", "erc_es_additivity_residual_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the selected OFR stress-peak date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[1], "Store the selected OFR FSI level, rounded to 3 decimals."),
    _v(TURN_1_NAMES[2], "Store the training-window start date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[3], "Store the training-window end date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[4], "Store the stress-window start date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[5], "Store the stress-window end date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[6], "Store the stress-window common-date count as an integer."),
    _v(TURN_1_NAMES[7], "Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other."),
    _v(TURN_1_NAMES[8], "Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other."),
    _v(TURN_1_NAMES[9], "Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other."),
    _v(TURN_1_NAMES[10], "Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other."),
    _v(TURN_1_NAMES[11], "Store exactly one canonical Ken French industry token from NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, or Other."),
    _v(TURN_2_NAMES[0], "Store the minimum covariance eigenvalue in decimal-return-squared units, rounded to 10 decimals."),
    _v(TURN_2_NAMES[1], "Store the covariance spectral condition number, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the highest-beta industry as its exact canonical Ken French token from the declared ten-token vocabulary."),
    _v(TURN_2_NAMES[3], "Store the highest fitted market beta, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the lowest-beta industry as its exact canonical Ken French token from the declared ten-token vocabulary."),
    _v(TURN_2_NAMES[5], "Store the lowest fitted market beta, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the ERC weight of the first retained industry in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the ERC weight of the second retained industry in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the ERC weight of the third retained industry in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the ERC weight of the fourth retained industry in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the ERC weight of the fifth retained industry in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store the ERC portfolio annualized training volatility in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6],
       "Store the maximum absolute risk-contribution-share deviation from 20 percent in percentage "
       "points, rounded to 5 decimals; an exact equal-risk-contribution solution drives it to zero."),
    _v(TURN_3_NAMES[7], "Store the ERC portfolio fitted market beta, rounded to 4 decimals."),
    _v(TURN_3_NAMES[8], "Store the ERC portfolio fitted SMB beta, rounded to 4 decimals."),
    _v(TURN_3_NAMES[9], "Store the ERC portfolio fitted HML beta, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the historical-ES tail observation count as an integer."),
    _v(TURN_4_NAMES[1], "Store the ERC stress-window compounded return in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the ERC stress-window annualized realized volatility in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the ERC stress-window maximum drawdown as a positive percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the ERC 97.5-percent historical expected shortfall as a positive loss percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the equal-weight stress-window compounded return in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store the equal-weight 97.5-percent historical expected shortfall as a positive loss percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[7], "Store the largest ERC expected-shortfall contributor as its exact canonical Ken French token from the declared ten-token vocabulary."),
    _v(TURN_4_NAMES[8], "Store its component expected-shortfall contribution in percentage points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[9], "Store component-sum minus portfolio expected shortfall in percentage points, rounded to 10 decimals."),
]

DECIMALS = [
    None, 3, None, None, None, None, 0, None, None, None, None, None,
    10, 4, None, 4, None, 4,
    4, 4, 4, 4, 4, 4, 5, 4, 4, 4,
    0, 4, 4, 4, 4, 4, 4, None, 4, 10,
]


@lru_cache(maxsize=1)
def _windows_and_cohort():
    stress = load_expansion_table("ofr_market_stress").copy()
    industries = load_ken_french_table("daily_industry_returns").copy()
    factors = load_ken_french_table("daily_factors").copy()
    panel = stress.merge(industries, on="date", how="inner", validate="one_to_one")
    panel = panel.merge(factors, on="date", how="inner", validate="one_to_one")
    panel["date"] = pd.to_datetime(panel.date)
    panel = panel.sort_values("date").reset_index(drop=True)
    candidates = panel.loc[panel.date.between(PEAK_START, PEAK_END)]
    maximum = candidates.ofr_fsi.max()
    peak_index = int(candidates.index[candidates.ofr_fsi.eq(maximum)][0])
    training_end_index = peak_index - PRE_PEAK_GAP
    training_start_index = training_end_index - TRAINING_OBSERVATIONS + 1
    stress_start_index = training_end_index + 1
    stress_end_index = peak_index + POST_PEAK_OBSERVATIONS
    if training_start_index < 0 or stress_end_index >= len(panel):
        raise ValueError("selected stress peak lacks the required windows")
    training = panel.iloc[training_start_index : training_end_index + 1].copy()
    stress_window = panel.iloc[stress_start_index : stress_end_index + 1].copy()
    if len(training) != TRAINING_OBSERVATIONS:
        raise ValueError("unexpected training observation count")
    volatilities = {
        label: float(training[column].std(ddof=1) * np.sqrt(ANNUAL_DAYS))
        for label, column in INDUSTRIES.items()
    }
    cohort = tuple(sorted(volatilities, key=lambda label: (-volatilities[label], label))[:COHORT_SIZE])
    return panel.iloc[peak_index], training, stress_window, cohort


@lru_cache(maxsize=1)
def _estimation_state():
    _, training, _, cohort = _windows_and_cohort()
    columns = [INDUSTRIES[label] for label in cohort]
    returns = training[columns].to_numpy(dtype=float) / 100.0
    covariance = np.cov(returns, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        raise ValueError("selected covariance is not positive definite")

    design = np.column_stack([
        np.ones(len(training)),
        training[["mkt_rf_pct", "smb_pct", "hml_pct"]].to_numpy(dtype=float),
    ])
    excess_returns = (
        training[columns].to_numpy(dtype=float)
        - training[["rf_pct"]].to_numpy(dtype=float)
    )
    coefficients = np.linalg.lstsq(design, excess_returns, rcond=None)[0]
    factor_betas = coefficients[1:, :].T
    return covariance, eigenvalues, factor_betas


def _erc_coordinate_descent(covariance):
    count = len(covariance)
    budgets = np.full(count, 1.0 / count)
    solution = np.ones(count)
    for _ in range(100_000):
        prior = solution.copy()
        for index in range(count):
            cross = covariance[index] @ solution - covariance[index, index] * solution[index]
            discriminant = cross * cross + 4.0 * covariance[index, index] * budgets[index]
            solution[index] = (
                -cross + np.sqrt(discriminant)
            ) / (2.0 * covariance[index, index])
        if np.max(np.abs(solution - prior)) < 1e-13:
            break
    else:
        raise ValueError("ERC coordinate descent did not converge")
    weights = solution / solution.sum()
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    risk_shares = weights * marginal / variance
    return weights, variance, risk_shares


def _stress_metrics(returns):
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1.0
    tail_count = int(np.ceil(0.025 * len(returns)))
    tail_indices = np.argsort(returns, kind="stable")[:tail_count]
    return {
        "cumulative_pct": float((np.prod(1.0 + returns) - 1.0) * 100.0),
        "volatility_pct": float(np.std(returns, ddof=1) * np.sqrt(ANNUAL_DAYS) * 100.0),
        "drawdown_pct": float(-np.min(drawdowns) * 100.0),
        "expected_shortfall_pct": float(-np.mean(returns[tail_indices]) * 100.0),
        "tail_count": tail_count,
        "tail_indices": tail_indices,
    }


@lru_cache(maxsize=1)
def ground_truth():
    peak, training, stress_window, cohort = _windows_and_cohort()
    covariance, eigenvalues, factor_betas = _estimation_state()
    market_betas = dict(zip(cohort, factor_betas[:, 0]))
    highest = sorted(market_betas, key=lambda label: (-market_betas[label], label))[0]
    lowest = sorted(market_betas, key=lambda label: (market_betas[label], label))[0]

    weights, variance, risk_shares = _erc_coordinate_descent(covariance)
    factor_exposures = weights @ factor_betas
    columns = [INDUSTRIES[label] for label in cohort]
    stress_returns = stress_window[columns].to_numpy(dtype=float) / 100.0
    erc_returns = stress_returns @ weights
    equal_returns = stress_returns @ np.full(COHORT_SIZE, 1.0 / COHORT_SIZE)
    erc_metrics = _stress_metrics(erc_returns)
    equal_metrics = _stress_metrics(equal_returns)
    tail_indices = erc_metrics["tail_indices"]
    contributions = -np.mean(
        stress_returns[tail_indices] * weights,
        axis=0,
    ) * 100.0
    largest_contributor_index = sorted(
        range(COHORT_SIZE),
        key=lambda index: (-contributions[index], cohort[index]),
    )[0]
    additivity_residual = float(
        contributions.sum() - erc_metrics["expected_shortfall_pct"]
    )

    return (
        peak.date.strftime("%Y-%m-%d"), float(peak.ofr_fsi),
        training.date.iloc[0].strftime("%Y-%m-%d"),
        training.date.iloc[-1].strftime("%Y-%m-%d"),
        stress_window.date.iloc[0].strftime("%Y-%m-%d"),
        stress_window.date.iloc[-1].strftime("%Y-%m-%d"), int(len(stress_window)),
        *cohort,
        float(eigenvalues[0]), float(np.linalg.cond(covariance)),
        highest, float(market_betas[highest]), lowest, float(market_betas[lowest]),
        *(float(weight * 100.0) for weight in weights),
        float(np.sqrt(variance) * np.sqrt(ANNUAL_DAYS) * 100.0),
        float(np.max(np.abs(risk_shares - 1.0 / COHORT_SIZE)) * 100.0),
        float(factor_exposures[0]), float(factor_exposures[1]), float(factor_exposures[2]),
        int(erc_metrics["tail_count"]), float(erc_metrics["cumulative_pct"]),
        float(erc_metrics["volatility_pct"]), float(erc_metrics["drawdown_pct"]),
        float(erc_metrics["expected_shortfall_pct"]),
        float(equal_metrics["cumulative_pct"]), float(equal_metrics["expected_shortfall_pct"]),
        cohort[largest_contributor_index], float(contributions[largest_contributor_index]),
        additivity_residual,
    )


LABEL_OUTPUTS = {
    *TURN_1_NAMES[7:12], TURN_2_NAMES[2], TURN_2_NAMES[4], TURN_4_NAMES[7],
}


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    for name in LABEL_OUTPUTS.intersection(names):
        # The contract is the exact canonical token; only its casing is display
        # variance, so a case-folded match is canonicalised before comparison.
        submitted = candidate.get(name)
        if isinstance(submitted, str) and submitted.strip().casefold() == str(truth[name]).casefold():
            candidate[name] = truth[name]
    result = validate_ordered_outputs(
        candidate,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )
    items = result.variable_results or {}
    missing = any(not current["is_set"] for current in items.values())
    errors = [
        current["message"]
        for current in items.values()
        if current["is_set"] and not current["correct"]
    ]
    return ValidatorResult(
        not missing and not errors,
        "required output not set" if missing else "; ".join(errors) or "correct",
        missing,
        items,
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_stress_windows": turn_validator(validate_turn_1),
    "validate_covariance_and_factors": turn_validator(validate_turn_2),
    "validate_equal_risk_contribution": turn_validator(validate_turn_3),
    "validate_tail_stress": turn_validator(validate_turn_4),
}
