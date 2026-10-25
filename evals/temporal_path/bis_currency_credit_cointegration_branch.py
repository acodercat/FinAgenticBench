"""Branch from a real-EER stress screen into a disciplined credit time-series model.

The case maps USD, EUR and JPY denominations to the corresponding BIS broad real
effective exchange-rate reference areas.  It selects the currency with the
largest absolute 2019-Q1-to-2025-Q4 real-EER change, then carries that currency
into the BIS EME non-bank credit decomposition.

An Engle--Granger residual ADF diagnostic is estimated on 2000-Q1--2022-Q4.  The
public contract uses a fixed finite-sample critical boundary.  The frozen JPY
sample does not reject a unit root, so the live branch estimates a stationary
first-difference ARDL rather than claiming an error-correction equilibrium.  The
terminal 2023--2025 rolling ex-post conditional-prediction exercise uses the
realized contemporaneous loan change, compares the fitted relation with a
no-change benchmark, and overlays the selected quarter's real-EER move.  It is
not an ex-ante forecast comparison.  Normal-equation
moments and component-versus-total credit identities are explicit invariants.

The `bis_currency_cointegration` pinned-reading comparison records sensitivity to screening on
the nominal rather than the real effective exchange rate, to forcing an
error-correction specification instead of letting the residual diagnostic choose the
branch, and to benchmarking the conditional forecast against the training-mean change
rather than no change. The branch rule is the one worth naming: the frozen JPY residual
gives an ADF t statistic of -1.345669, which does not reject a unit root, so forcing
error correction would report an adjustment coefficient the data does not support. The
query fixes all three, so these are robustness comparisons rather than hidden answer
paths.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import ValidatorResult, turn_validator, validate_ordered_outputs


CURRENCY_AREA = {
    "USD": ("USD: US dollar", "US: United States"),
    "EUR": ("EUR: Euro", "XM: Euro area"),
    "JPY": ("JPY: Yen", "JP: Japan"),
}
TRAIN_END = "2022-Q4"
ADF_CRITICAL = -3.37


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "currency_screen_count", "selected_currency_code", "selected_reference_area",
    "selected_real_eer_start", "selected_real_eer_end", "selected_real_eer_change_pct",
    "selected_eme_credit_start_mn", "selected_eme_credit_end_mn",
    "selected_eme_credit_change_pct",
]
TURN_2_NAMES = [
    "cointegration_training_start_quarter", "cointegration_training_end_quarter",
    "cointegration_training_observation_count", "long_run_intercept",
    "long_run_log_loan_slope", "long_run_residual_std", "residual_adf_coefficient",
    "residual_adf_t_stat", "residual_adf_rejects_unit_root",
    "maximum_credit_component_identity_residual_mn",
]
TURN_3_NAMES = [
    "selected_dynamic_model", "dynamic_model_observation_count",
    "dynamic_intercept", "dynamic_lag_state_coefficient",
    "dynamic_current_loan_change_coefficient", "dynamic_lagged_loan_change_coefficient",
    "dynamic_model_r_squared", "dynamic_maximum_normal_equation_moment",
]
TURN_4_NAMES = [
    "conditional_test_start_quarter", "conditional_test_end_quarter",
    "conditional_test_observation_count", "dynamic_conditional_prediction_rmse_log_points",
    "no_change_benchmark_rmse_log_points", "conditional_prediction_beats_no_change",
    "largest_conditional_prediction_error_quarter",
    "largest_conditional_prediction_error_log_points", "same_quarter_real_eer_change_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the screened currency count as an integer."),
    _v(TURN_1_NAMES[1], "Store the selected exact uppercase three-letter currency token: USD, EUR, or JPY."),
    _v(TURN_1_NAMES[2], "Store the selected BIS reference-area label verbatim, including its two-letter prefix, colon, spacing, and area name."),
    _v(TURN_1_NAMES[3], "Store the selected currency's 2019-Q1-end broad real EER index level (2020=100), rounded to 2 decimals."),
    _v(TURN_1_NAMES[4], "Store the selected currency's 2025-Q4-end broad real EER index level (2020=100), rounded to 2 decimals."),
    _v(TURN_1_NAMES[5], "Store the selected real-EER percentage change, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store the selected currency's 2019-Q1 total EME non-bank credit in currency millions, rounded to 3 decimals."),
    _v(TURN_1_NAMES[7], "Store the selected currency's 2025-Q4 total EME non-bank credit in currency millions, rounded to 3 decimals."),
    _v(TURN_1_NAMES[8], "Store the selected total-credit percentage change, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the long-run estimation start quarter as a YYYY-Qn string."),
    _v(TURN_2_NAMES[1], "Store the long-run estimation end quarter as a YYYY-Qn string."),
    _v(TURN_2_NAMES[2], "Store the long-run estimation observation count as an integer."),
    _v(TURN_2_NAMES[3], "Store the Engle-Granger log-level intercept, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the Engle-Granger log-loan slope, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store the sample standard deviation of the long-run residual, rounded to 8 decimals."),
    _v(TURN_2_NAMES[6], "Store the no-lag residual ADF level coefficient, rounded to 8 decimals."),
    _v(TURN_2_NAMES[7], "Store the residual ADF t statistic, rounded to 6 decimals."),
    _v(TURN_2_NAMES[8], "Store whether the residual ADF t statistic is below the declared critical value as a Boolean."),
    _v(TURN_2_NAMES[9], "Store the maximum absolute total-credit-minus-components identity residual in currency millions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[0], "Store exactly one canonical branch token: first_difference_ardl or error_correction."),
    _v(TURN_3_NAMES[1], "Store the selected dynamic-model observation count as an integer."),
    _v(TURN_3_NAMES[2], "Store the selected dynamic-model intercept, rounded to 6 decimals."),
    _v(TURN_3_NAMES[3], "Store the selected branch's lag-state coefficient, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the contemporaneous loan-change coefficient, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store the lagged loan-change coefficient, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the selected dynamic-model R-squared, rounded to 8 decimals."),
    _v(TURN_3_NAMES[7], "Store the maximum absolute unscaled OLS normal-equation moment max|X' residual|, rounded to 10 decimals."),
    _v(TURN_4_NAMES[0], "Store the conditional evaluation start quarter as a YYYY-Qn string."),
    _v(TURN_4_NAMES[1], "Store the conditional evaluation end quarter as a YYYY-Qn string."),
    _v(TURN_4_NAMES[2], "Store the conditional evaluation observation count as an integer."),
    _v(TURN_4_NAMES[3], "Store the dynamic ex-post conditional-prediction RMSE in log points, meaning natural-log units not multiplied by 100, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the no-change benchmark RMSE in log points, meaning natural-log units not multiplied by 100, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store whether the dynamic conditional prediction has lower RMSE than the no-change benchmark as a Boolean."),
    _v(TURN_4_NAMES[6], "Store the quarter with the largest absolute conditional-prediction error as a YYYY-Qn string."),
    _v(TURN_4_NAMES[7], "Store the signed largest conditional-prediction error, actual minus predicted debt-security log change, rounded to 4 decimals."),
    _v(TURN_4_NAMES[8], "Store the broad real EER change from the prior quarter's last month to this quarter's last month, rounded to 4 decimals."),
]

DECIMALS = [
    0, None, None, 2, 2, 4, 3, 3, 4,
    None, None, 0, 4, 4, 8, 8, 6, None, 6,
    None, 0, 6, 4, 4, 4, 8, 10,
    None, None, 0, 4, 4, None, None, 4, 4,
]


@lru_cache(maxsize=1)
def _panels():
    eer = load_expansion_table("bis_effective_exchange_rates").copy()
    eer = eer.loc[eer["EER_TYPE:Type"].eq("R: Real")].copy()
    eer["value"] = pd.to_numeric(eer["OBS_VALUE:Observation Value"])
    liquidity = load_expansion_table("bis_global_liquidity").copy()
    liquidity["value"] = pd.to_numeric(liquidity["OBS_VALUE:Observation Value"])
    credit = liquidity.pivot(
        index="TIME_PERIOD:Time period or range",
        columns=["CURR_DENOM:Currency of denomination", "L_INSTR:Type of instruments"],
        values="value",
    ).sort_index()
    return eer, credit


@lru_cache(maxsize=1)
def _selection():
    eer, credit = _panels()
    rows = []
    for code, (currency, area) in CURRENCY_AREA.items():
        series = eer.loc[eer["REF_AREA:Reference area"].eq(area)].set_index(
            "TIME_PERIOD:Time period or range"
        )["value"]
        start = float(series.loc["2019-03"])
        end = float(series.loc["2025-12"])
        change = (end / start - 1.0) * 100.0
        total = credit.loc[:, (currency, "B: Credit (loans & debt securities)")]
        rows.append((code, currency, area, start, end, change, float(total.loc["2019-Q1"]), float(total.loc["2025-Q4"])))
    selected = sorted(rows, key=lambda row: (-abs(row[5]), row[0]))[0]
    return rows, selected


@lru_cache(maxsize=1)
def _model_state():
    _, credit = _panels()
    _, selected = _selection()
    currency = selected[1]
    panel = credit.loc[:, currency].copy()
    training = panel.loc[:TRAIN_END]
    log_debt = np.log(training["D: Debt securities"].to_numpy(dtype=float))
    log_loans = np.log(training["G: Loans and deposits"].to_numpy(dtype=float))
    long_design = np.column_stack([np.ones(len(training)), log_loans])
    long_coefficients = np.linalg.lstsq(long_design, log_debt, rcond=None)[0]
    residual = log_debt - long_design @ long_coefficients
    adf_design = np.column_stack([np.ones(len(residual) - 1), residual[:-1]])
    adf_change = np.diff(residual)
    adf_coefficients = np.linalg.lstsq(adf_design, adf_change, rcond=None)[0]
    adf_error = adf_change - adf_design @ adf_coefficients
    variance = float(adf_error @ adf_error / (len(adf_error) - adf_design.shape[1]))
    covariance = variance * np.linalg.inv(adf_design.T @ adf_design)
    adf_t = float(adf_coefficients[1] / np.sqrt(covariance[1, 1]))
    rejects = bool(adf_t < ADF_CRITICAL)

    debt_change = np.diff(log_debt)
    loan_change = np.diff(log_loans)
    if rejects:
        dynamic_label = "error_correction"
        response = debt_change
        dynamic_design = np.column_stack([
            np.ones(len(response)), residual[:-1], loan_change, np.zeros(len(response)),
        ])
        coefficients = np.linalg.lstsq(dynamic_design, response, rcond=None)[0]
    else:
        dynamic_label = "first_difference_ardl"
        response = debt_change[1:]
        dynamic_design = np.column_stack([
            np.ones(len(response)), debt_change[:-1], loan_change[1:], loan_change[:-1],
        ])
        coefficients = np.linalg.lstsq(dynamic_design, response, rcond=None)[0]
    fitted = dynamic_design @ coefficients
    dynamic_error = response - fitted
    r_squared = 1.0 - float(dynamic_error @ dynamic_error) / float((response - response.mean()) @ (response - response.mean()))
    moment = float(np.max(np.abs(dynamic_design.T @ dynamic_error)))
    return panel, training, long_coefficients, residual, adf_coefficients, adf_t, rejects, dynamic_label, coefficients, r_squared, moment, len(response)


@lru_cache(maxsize=1)
def ground_truth():
    eer, credit = _panels()
    rows, selected = _selection()
    code, currency, area, eer_start, eer_end, eer_change, credit_start, credit_end = selected
    panel, training, long_coef, residual, adf_coef, adf_t, rejects, dynamic_label, coef, r2, moment, dynamic_n = _model_state()
    identity = panel["B: Credit (loans & debt securities)"] - panel["D: Debt securities"] - panel["G: Loans and deposits"]

    full_debt = np.log(panel["D: Debt securities"].to_numpy(dtype=float))
    full_loans = np.log(panel["G: Loans and deposits"].to_numpy(dtype=float))
    quarters = list(panel.index)
    cutoff = quarters.index(TRAIN_END)
    errors = []
    actual_changes = []
    for index in range(cutoff + 1, len(panel)):
        actual = full_debt[index] - full_debt[index - 1]
        if rejects:
            lag_residual = full_debt[index - 1] - (long_coef[0] + long_coef[1] * full_loans[index - 1])
            prediction = coef[0] + coef[1] * lag_residual + coef[2] * (full_loans[index] - full_loans[index - 1])
        else:
            prediction = (
                coef[0]
                + coef[1] * (full_debt[index - 1] - full_debt[index - 2])
                + coef[2] * (full_loans[index] - full_loans[index - 1])
                + coef[3] * (full_loans[index - 1] - full_loans[index - 2])
            )
        actual_changes.append(actual)
        errors.append(actual - prediction)
    errors = np.asarray(errors)
    actual_changes = np.asarray(actual_changes)
    test_quarters = quarters[cutoff + 1:]
    dynamic_rmse = float(np.sqrt(np.mean(errors**2)))
    no_change_rmse = float(np.sqrt(np.mean(actual_changes**2)))
    largest_index = sorted(range(len(errors)), key=lambda index: (-abs(errors[index]), test_quarters[index]))[0]
    largest_quarter = test_quarters[largest_index]
    year, quarter_number = largest_quarter.split("-Q")
    month = {"1": "03", "2": "06", "3": "09", "4": "12"}[quarter_number]
    current_month = f"{year}-{month}"
    prior_month = {
        "03": f"{int(year) - 1}-12", "06": f"{year}-03",
        "09": f"{year}-06", "12": f"{year}-09",
    }[month]
    eer_series = eer.loc[eer["REF_AREA:Reference area"].eq(area)].set_index(
        "TIME_PERIOD:Time period or range"
    )["value"]
    same_quarter_eer_change = (float(eer_series.loc[current_month]) / float(eer_series.loc[prior_month]) - 1.0) * 100.0

    return (
        int(len(rows)),
        code,
        area,
        eer_start,
        eer_end,
        eer_change,
        credit_start,
        credit_end,
        (credit_end / credit_start - 1.0) * 100.0,
        str(training.index[0]),
        str(training.index[-1]),
        int(len(training)),
        float(long_coef[0]),
        float(long_coef[1]),
        float(np.std(residual, ddof=1)),
        float(adf_coef[1]),
        adf_t,
        rejects,
        float(np.max(np.abs(identity))),
        dynamic_label,
        int(dynamic_n),
        float(coef[0]),
        float(coef[1]),
        float(coef[2]),
        float(coef[3]),
        float(r2),
        moment,
        test_quarters[0],
        test_quarters[-1],
        int(len(test_quarters)),
        dynamic_rmse,
        no_change_rmse,
        bool(dynamic_rmse < no_change_rmse),
        largest_quarter,
        float(errors[largest_index]),
        float(same_quarter_eer_change),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    for name in ("selected_currency_code", "selected_reference_area", "selected_dynamic_model"):
        if name in names and isinstance(candidate.get(name), str):
            candidate[name] = " ".join(candidate[name].split()).casefold()
            truth[name] = " ".join(str(truth[name]).split()).casefold()
    if "selected_dynamic_model" in candidate:
        # The contract is the exact branch token; only its casing is display
        # variance, so a case-folded match is canonicalised before comparison.
        submitted = candidate["selected_dynamic_model"]
        if isinstance(submitted, str) and submitted.strip().casefold() == str(truth["selected_dynamic_model"]).casefold():
            candidate["selected_dynamic_model"] = truth["selected_dynamic_model"]
    result = validate_ordered_outputs(
        candidate, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )
    items = result.variable_results or {}
    missing = any(not item["is_set"] for item in items.values())
    errors = [
        item["message"] for item in items.values()
        if item["is_set"] and not item["correct"]
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
    "validate_currency_screen": turn_validator(validate_turn_1),
    "validate_cointegration_diagnostic": turn_validator(validate_turn_2),
    "validate_dynamic_branch": turn_validator(validate_turn_3),
    "validate_forecast_and_eer": turn_validator(validate_turn_4),
}
