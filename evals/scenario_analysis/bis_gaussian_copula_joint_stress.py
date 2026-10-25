"""Estimate a rank-Gaussian common-stress direction across three BIS panels.

Three BIS stress features move together: a credit-to-GDP gap, a private-sector debt
service ratio and a real effective exchange-rate depreciation. Scoring an economy on
each one separately ignores that dependence. A Gaussian copula converts each feature to
a standard-normal score through its own empirical rank, then combines them under the
fitted correlation, so an economy that is moderately stressed on three correlated
measures is not counted three times.

The balanced pooled training panel holds 640 observations across 32 economies. The
Gaussian-copula correlations are -0.1161 between credit gap and DSR, 0.1682 between
gap and depreciation and 0.0828 between DSR and depreciation, with a correlation
condition number of 1.5605, so the combination is not read off a near-singular matrix.

Across the 32 scored economies the leader is JP, with a credit gap of 6.7837 percentage
points, a private DSR of 15.5 percent and real effective depreciation of 5.4420
percent. Its scores are 0.9906, 0.0490 and 1.1428, giving a common stress score of
1.130678 and an upper-tail probability of 0.12909537, ahead of the runner-up by 0.34715.
The three contributions, 0.590661, 0.030872 and 0.509145, reconcile to the score with a
residual of 0.0.

The conditional step is where the dependence would show, and here it barely moves.
Conditioning DSR on the other two features gives a mean of -0.0121 and a standard
deviation of 0.9878, so the leader's standardized residual is 0.061867 against an
unconditional score of 0.0490, and its conditional upper-tail probability is
0.4753. That is the arithmetic consequence of weak dependence: with pairwise
correlations of -0.1161, 0.1682 and 0.0828, knowing two features says almost
nothing about the third. The leader's debt service is unremarkable either way, and the
conditional machinery is what establishes that rather than assuming it.

The stated scenario raises the credit gap to 19.21154 percentage points, DSR to 30.11
percent and depreciation to 10.9049 percent. That lifts the common stress score to
2.750395 and cuts the tail probability to 0.00297618, a score increase of 1.619717 and a
probability ratio of 0.023054.

The case deliberately requires the solver to supply the Gaussian-copula and
conditional-normal formulas.  The query fixes rank, tie, clipping, covariance, and
quantile conventions without printing those formulas.

Boundaries: the Gaussian copula is assumed rather than tested, and it has no tail
dependence, so joint-tail probabilities are a property of that assumption rather than an
observed frequency. The tail probabilities are model quantities on 640 pooled
observations, not event rates. The scenario is a stated hypothetical, not a forecast or a
supervisory scenario, and the correlation matrix identifies no causal channel among the
three features.

The `bis_gaussian_copula` pinned-reading comparison records sensitivity to the training-window
start, to breaking rank ties by the minimum rather than the average, and to replacing
the fitted dependence with an identity correlation. The dependence choice is the one
worth naming: with an identity correlation the conditional step collapses, and the
leader's conditional DSR residual of 0.061867 could no longer distinguish a feature
that is unremarkable once the other two are known from one that is not. The query fixes
all three, so these are robustness comparisons rather than hidden answer paths.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable
from scipy.stats import norm

from core.data import load_expansion_table
from core.validation import ValidatorResult, turn_validator, validate_ordered_outputs


COUNTRY = "BORROWERS_CTY:Borrowers' country"
COUNTRY_CODE = "borrower_code"
PERIOD = "TIME_PERIOD:Time period or range"
VALUE = "OBS_VALUE:Observation Value"
PRIVATE = "P: Private non-financial sector"
GAP = "C: Credit-to-GDP gaps (actual-trend)"
REAL = "R: Real"
FEATURES = ("credit_gap", "private_dsr", "real_eer_depreciation")


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "copula_training_observation_count", "copula_training_country_count",
    "gap_dsr_gaussian_correlation", "gap_depreciation_gaussian_correlation",
    "dsr_depreciation_gaussian_correlation", "copula_correlation_condition_number",
]
TURN_2_NAMES = [
    "joint_stress_country_count", "joint_stress_leader", "leader_credit_gap_pp",
    "leader_private_dsr_pct", "leader_real_eer_depreciation_pct", "leader_gap_z",
    "leader_dsr_z", "leader_depreciation_z", "leader_common_stress_score",
    "leader_common_stress_upper_tail_probability", "leader_score_margin",
]
TURN_3_NAMES = [
    "leader_gap_score_contribution", "leader_dsr_score_contribution",
    "leader_depreciation_score_contribution", "leader_contribution_sum_residual",
    "leader_conditional_dsr_z_mean", "leader_conditional_dsr_z_std",
    "leader_conditional_dsr_standardized_residual", "leader_conditional_dsr_upper_tail_probability",
]
TURN_4_NAMES = [
    "scenario_credit_gap_pp", "scenario_private_dsr_pct", "scenario_real_eer_depreciation_pct",
    "scenario_common_stress_score", "scenario_common_stress_upper_tail_probability",
    "scenario_tail_probability_ratio",
]

variables = [
    _v("copula_training_observation_count", "Store the balanced pooled training row count as an integer."),
    _v("copula_training_country_count", "Store the balanced training-country count as an integer."),
    _v("gap_dsr_gaussian_correlation", "Store the Gaussian-score correlation between credit gap and private DSR, rounded to 4 decimals."),
    _v("gap_depreciation_gaussian_correlation", "Store the Gaussian-score correlation between credit gap and real-EER depreciation, rounded to 4 decimals."),
    _v("dsr_depreciation_gaussian_correlation", "Store the Gaussian-score correlation between private DSR and real-EER depreciation, rounded to 4 decimals."),
    _v("copula_correlation_condition_number", "Store the correlation matrix 2-norm condition number, rounded to 4 decimals."),
    _v("joint_stress_country_count", "Store the 2025-Q4 scored-country count as an integer."),
    _v("joint_stress_leader", "Store the exact two-letter BIS borrower-area code of the economy with the largest common-stress score, as text."),
    _v("leader_credit_gap_pp", "Store its credit gap in percentage points, rounded to 4 decimals."),
    _v("leader_private_dsr_pct", "Store its private-sector DSR in percent, rounded to 1 decimal."),
    _v("leader_real_eer_depreciation_pct", "Store its year-over-year real-EER depreciation in percent, rounded to 4 decimals."),
    _v("leader_gap_z", "Store its empirical Gaussian credit-gap score, rounded to 4 decimals."),
    _v("leader_dsr_z", "Store its empirical Gaussian DSR score, rounded to 4 decimals."),
    _v("leader_depreciation_z", "Store its empirical Gaussian depreciation score, rounded to 4 decimals."),
    _v("leader_common_stress_score", "Store its correlation-adjusted common-stress score, rounded to 6 decimals."),
    _v("leader_common_stress_upper_tail_probability", "Store the standard-normal upper-tail probability of that score, rounded to 8 decimals."),
    _v("leader_score_margin", "Store leader score minus runner-up score, rounded to 6 decimals."),
    _v("leader_gap_score_contribution", "Store the credit-gap additive contribution to the leader score, rounded to 6 decimals."),
    _v("leader_dsr_score_contribution", "Store the DSR additive contribution to the leader score, rounded to 6 decimals."),
    _v("leader_depreciation_score_contribution", "Store the depreciation additive contribution to the leader score, rounded to 6 decimals."),
    _v("leader_contribution_sum_residual", "Store the absolute contribution-sum residual, rounded to 12 decimals."),
    _v("leader_conditional_dsr_z_mean", "Store conditional mean DSR Gaussian score given the other two scores, rounded to 4 decimals."),
    _v("leader_conditional_dsr_z_std", "Store the corresponding conditional standard deviation, rounded to 4 decimals."),
    _v("leader_conditional_dsr_standardized_residual", "Store the conditional standardized DSR residual, rounded to 6 decimals."),
    _v("leader_conditional_dsr_upper_tail_probability",
       "Store its conditional standard-normal upper-tail probability, rounded to 4 decimals. "
       "Compute from the unrounded standardized residual, not from the rounded figure reported for "
       "it."),
    _v("scenario_credit_gap_pp", "Store the pooled-training linear 95th-percentile credit gap, rounded to 4 decimals."),
    _v("scenario_private_dsr_pct", "Store the pooled-training linear 95th-percentile private DSR, rounded to 4 decimals."),
    _v("scenario_real_eer_depreciation_pct", "Store the pooled-training linear 95th-percentile real-EER depreciation, rounded to 4 decimals."),
    _v("scenario_common_stress_score", "Store the copula common-stress score at those three marginal shocks, rounded to 6 decimals."),
    _v("scenario_common_stress_upper_tail_probability", "Store its standard-normal upper-tail probability, rounded to 8 decimals."),
    _v("scenario_tail_probability_ratio", "Store scenario tail probability divided by observed-leader tail probability, rounded to 6 decimals."),
]

DECIMALS = [
    0, 0, 4, 4, 4, 4,
    0, None, 4, 1, 4, 4, 4, 4, 6, 8, 6,
    6, 6, 6, 12, 4, 4, 6, 4,
    4, 4, 4, 6, 8, 6,
]


def _display(value):
    text = " ".join(str(value).split())
    prefix, separator, remainder = text.partition(": ")
    return remainder if separator and len(prefix) == 2 else text


def _panel():
    credit = load_expansion_table("bis_credit_gap")
    credit = credit.loc[
        credit["TC_BORROWERS:Borrowing sector"].eq(PRIVATE)
        & credit["TC_LENDERS:Lending sector"].str.startswith("A:")
        & credit["CG_DTYPE:Credit gap data type"].eq(GAP),
        [COUNTRY, PERIOD, VALUE],
    ].rename(columns={VALUE: FEATURES[0]})
    credit[COUNTRY_CODE] = credit[COUNTRY].astype(str).str.split(":", n=1).str[0]

    dsr = load_expansion_table("bis_debt_service")
    dsr = dsr.loc[dsr["DSR_BORROWERS:Borrowers"].eq(PRIVATE), [COUNTRY, PERIOD, VALUE]].rename(
        columns={VALUE: FEATURES[1]}
    )

    eer = load_expansion_table("bis_effective_exchange_rates")
    eer = eer.loc[eer["EER_TYPE:Type"].eq(REAL), ["REF_AREA:Reference area", PERIOD, VALUE]].copy()
    eer = eer.sort_values(["REF_AREA:Reference area", PERIOD])
    eer[FEATURES[2]] = -100 * eer.groupby("REF_AREA:Reference area")[VALUE].pct_change(12)
    eer = eer.loc[eer[PERIOD].str[-2:].isin(["03", "06", "09", "12"])]
    eer[PERIOD] = pd.PeriodIndex(eer[PERIOD], freq="M").asfreq("Q").astype(str).str.replace("Q", "-Q")
    eer = eer.rename(columns={"REF_AREA:Reference area": COUNTRY})[[COUNTRY, PERIOD, FEATURES[2]]]

    for frame in (credit, dsr, eer):
        frame[COUNTRY] = frame[COUNTRY].map(_display)
    return credit.merge(dsr, on=[COUNTRY, PERIOD]).merge(eer, on=[COUNTRY, PERIOD]).dropna()


def _transform(training, values):
    n = len(training)
    transformed = np.empty((len(values), len(FEATURES)))
    for j, feature in enumerate(FEATURES):
        ordered = np.sort(training[feature].to_numpy(float))
        raw = values[feature].to_numpy(float)
        left = np.searchsorted(ordered, raw, side="left")
        right = np.searchsorted(ordered, raw, side="right")
        probabilities = (left + 0.5 * (right - left)) / n
        probabilities = np.clip(probabilities, 0.5 / n, 1 - 0.5 / n)
        transformed[:, j] = norm.ppf(probabilities)
    return transformed


@lru_cache(maxsize=1)
def _state():
    panel = _panel()
    training = panel.loc[panel[PERIOD].between("2020-Q1", "2024-Q4")].copy()
    test = panel.loc[panel[PERIOD].eq("2025-Q4")].copy()
    counts = training.groupby(COUNTRY)[PERIOD].nunique()
    countries = sorted(set(counts[counts.eq(20)].index) & set(test[COUNTRY]))
    training = training.loc[training[COUNTRY].isin(countries)].sort_values([PERIOD, COUNTRY]).reset_index(drop=True)
    test = test.loc[test[COUNTRY].isin(countries)].sort_values(COUNTRY).reset_index(drop=True)
    if len(training) != 640 or len(test) != 32:
        raise ValueError("unexpected balanced BIS copula panel")

    n = len(training)
    training_z = np.column_stack([
        norm.ppf((training[feature].rank(method="average").to_numpy(float) - 0.5) / n)
        for feature in FEATURES
    ])
    correlation = np.corrcoef(training_z, rowvar=False)
    inverse_one = np.linalg.solve(correlation, np.ones(3))
    denominator = float(np.sqrt(np.ones(3) @ inverse_one))
    coefficients = inverse_one / denominator

    test_z = _transform(training, test)
    scores = test_z @ coefficients
    order = np.lexsort((test[COUNTRY].to_numpy(), -scores))
    leader_index, runner_index = order[:2]
    return training, test, training_z, correlation, coefficients, test_z, scores, leader_index, runner_index


@lru_cache(maxsize=1)
def ground_truth():
    training, test, _, correlation, coefficients, test_z, scores, leader_index, runner_index = _state()
    leader = test.iloc[leader_index]
    z = test_z[leader_index]
    score = float(scores[leader_index])
    contributions = coefficients * z

    x_indices = [0, 2]
    r_yx = correlation[1, x_indices]
    r_xx = correlation[np.ix_(x_indices, x_indices)]
    conditional_mean = float(r_yx @ np.linalg.solve(r_xx, z[x_indices]))
    conditional_variance = float(1 - r_yx @ np.linalg.solve(r_xx, r_yx))
    conditional_std = float(np.sqrt(conditional_variance))
    conditional_residual = float((z[1] - conditional_mean) / conditional_std)

    scenario_raw = np.array([np.quantile(training[f].to_numpy(float), 0.95, method="linear") for f in FEATURES])
    scenario_frame = pd.DataFrame([dict(zip(FEATURES, scenario_raw))])
    scenario_z = _transform(training, scenario_frame)[0]
    scenario_score = float(scenario_z @ coefficients)
    leader_tail = float(norm.sf(score))
    scenario_tail = float(norm.sf(scenario_score))

    return (
        len(training),
        training[COUNTRY].nunique(),
        correlation[0, 1],
        correlation[0, 2],
        correlation[1, 2],
        np.linalg.cond(correlation),
        len(test),
        str(leader[COUNTRY_CODE]),
        float(leader[FEATURES[0]]),
        float(leader[FEATURES[1]]),
        float(leader[FEATURES[2]]),
        float(z[0]),
        float(z[1]),
        float(z[2]),
        score,
        leader_tail,
        score - float(scores[runner_index]),
        float(contributions[0]),
        float(contributions[1]),
        float(contributions[2]),
        abs(float(contributions.sum() - score)),
        conditional_mean,
        conditional_std,
        conditional_residual,
        float(norm.sf(conditional_residual)),
        float(scenario_raw[0]),
        float(scenario_raw[1]),
        float(scenario_raw[2]),
        scenario_score,
        scenario_tail,
        scenario_tail / leader_tail,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    if "joint_stress_leader" in candidate:
        # The query asks for the exact two-letter code; only its casing is display
        # variance, so a case-folded match is canonicalised before comparison.
        submitted = candidate["joint_stress_leader"]
        if isinstance(submitted, str) and submitted.strip().casefold() == str(truth["joint_stress_leader"]).casefold():
            candidate["joint_stress_leader"] = truth["joint_stress_leader"]
    result = validate_ordered_outputs(candidate, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])
    items = result.variable_results or {}
    missing = any(not item["is_set"] for item in items.values())
    errors = [item["message"] for item in items.values() if item["is_set"] and not item["correct"]]
    return ValidatorResult(not missing and not errors, "required output not set" if missing else "; ".join(errors) or "correct", missing, items)


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_copula_fit": turn_validator(validate_turn_1),
    "validate_joint_stress": turn_validator(validate_turn_2),
    "validate_conditional_decomposition": turn_validator(validate_turn_3),
    "validate_joint_scenario": turn_validator(validate_turn_4),
}
