"""Infer latent final values from correlated real-time release measurements.

Successive statistical vintages are not independent estimates of the same truth: they
share source data and method, so their revision errors are strongly correlated. Treating
three vintages as three independent draws and averaging them is therefore wrong. The best
linear unbiased estimator uses the error covariance, and the question is whether the
covariance structure is worth estimating.

On 219 training quarters of real GDP growth the vintage error means are -0.4695,
-0.3465 and -0.2977 percentage points, so every vintage understates on average and the
bias shrinks with each revision. The errors are almost collinear, with correlations of
0.9505, 0.9306 and 0.9825 and a covariance condition number of 193.52438. That
near-collinearity is why the choice barely matters here: on 12 validation quarters the full
covariance scores 0.6193 against 0.6253 for a diagonal approximation, so full is
selected by a margin of six thousandths of a percentage point. The BLUE weights are
0.3255, 0.2977 and 0.3768, summing to one within the scored precision, and the estimator
cuts the third-release absolute error from 0.600000 to 0.291808 percentage points, an
improvement of 0.308192.

The Z1 household-wealth branch is where the case earns its keep. Transferring the
covariance model selected on GDP gives a validation error of 44.190544 basis points while
the alternative gives 28.908576, so the diagonal model wins and the selection does not
agree with GDP. A covariance structure validated on one series is not a property of vintage
data in general. The Z1 BLUE estimate of 155,999,408.166 million dollars improves on the
third vintage by 622,054.166 million, but its 95 percent interval, 155,703,180.957 to
156,296,198.952, does not contain the newest vintage of 156,536,718: the estimator is
closer than the third release and still excludes the eventual figure.

Every BLUE quantity is computed from unrounded inputs, not from the rounded figures
reported for the weights, means and standard error, so a model carrying its own
full-precision intermediates matches the pinned values.

The `correlated_release_blue` pinned-reading comparison records sensitivity to the covariance policy
and to whether the vintage bias correction is applied. The query fixes both, so these are
robustness comparisons rather than hidden answer paths.

Boundaries: the error covariance is estimated on a training window and applied out of
sample as if stationary, which revision practice does not guarantee; both series changed
methodology inside the samples used here. The Z1 branch has 31 training and 4 validation
quarters, far too few to establish a covariance structure, which is why its disagreement
with GDP is reported as a measured outcome rather than as evidence that diagonal is
correct. Unbiasedness is asserted through the weight constraint, not tested.
"""

from functools import lru_cache

import numpy as np
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


RTDSM_SERIES = "real_gdp_annualized_pct"
RTDSM_TARGET = "2024:Q2"
RTDSM_TRAIN_END = "2020:Q4"
RTDSM_VALIDATION_END = "2023:Q4"
RELEASE_COLUMNS = ("first_release", "second_release", "third_release")
Z1_SERIES = "FL152090005.Q"
Z1_TABLE = "b101"
Z1_TARGET = "2023-12-31"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "gdp_training_quarter_count", "gdp_first_release_error_mean_pp",
    "gdp_second_release_error_mean_pp", "gdp_third_release_error_mean_pp",
    "gdp_first_second_error_correlation", "gdp_first_third_error_correlation",
    "gdp_second_third_error_correlation", "gdp_error_covariance_condition_number",
]
TURN_2_NAMES = [
    "gdp_validation_quarter_count", "gdp_full_covariance_validation_rmse_pp",
    "gdp_diagonal_covariance_validation_rmse_pp", "gdp_selected_covariance_model",
    "gdp_blue_first_weight", "gdp_blue_second_weight", "gdp_blue_third_weight",
    "gdp_blue_weight_sum_residual", "gdp_blue_estimate_pct", "gdp_blue_standard_error_pp",
    "gdp_most_recent_pct", "gdp_third_release_error_pp",
]
TURN_3_NAMES = [
    "z1_training_quarter_count", "z1_validation_quarter_count",
    "z1_transferred_covariance_model", "z1_transferred_model_validation_rmse_bp",
    "z1_alternative_model_validation_rmse_bp", "z1_selected_covariance_model",
    "z1_selection_agrees_with_gdp", "z1_first_vintage_log_error_mean_bp",
    "z1_second_vintage_log_error_mean_bp", "z1_third_vintage_log_error_mean_bp",
    "z1_first_second_error_correlation", "z1_first_third_error_correlation",
    "z1_second_third_error_correlation", "z1_blue_first_weight",
    "z1_blue_second_weight", "z1_blue_third_weight", "z1_blue_log_standard_error_bp",
]
TURN_4_NAMES = [
    "z1_target_first_vintage_usd_millions", "z1_target_second_vintage_usd_millions",
    "z1_target_third_vintage_usd_millions", "z1_blue_terminal_estimate_usd_millions",
    "z1_blue_lower_95_usd_millions", "z1_blue_upper_95_usd_millions",
    "z1_newest_vintage_usd_millions", "z1_third_vintage_error_usd_millions",
]

variables = [
    _v("gdp_training_quarter_count", "Store the complete real-GDP training-quarter count as an integer."),
    _v("gdp_first_release_error_mean_pp", "Store mean first-release minus most-recent error in percentage points, rounded to 4 decimals."),
    _v("gdp_second_release_error_mean_pp", "Store mean second-release minus most-recent error in percentage points, rounded to 4 decimals."),
    _v("gdp_third_release_error_mean_pp", "Store mean third-release minus most-recent error in percentage points, rounded to 4 decimals."),
    _v("gdp_first_second_error_correlation", "Store correlation of first- and second-release errors, rounded to 4 decimals."),
    _v("gdp_first_third_error_correlation", "Store correlation of first- and third-release errors, rounded to 4 decimals."),
    _v("gdp_second_third_error_correlation", "Store correlation of second- and third-release errors, rounded to 4 decimals."),
    _v("gdp_error_covariance_condition_number", "Store the GDP error-covariance 2-norm condition number, rounded to 4 decimals."),
    _v("gdp_validation_quarter_count", "Store the complete held-out covariance-selection quarter count as an integer."),
    _v("gdp_full_covariance_validation_rmse_pp", "Store full-covariance BLUE validation RMSE in percentage points, rounded to 4 decimals."),
    _v("gdp_diagonal_covariance_validation_rmse_pp", "Store diagonal-covariance BLUE validation RMSE in percentage points, rounded to 4 decimals."),
    _v("gdp_selected_covariance_model", "Store the selected covariance architecture as full or diagonal."),
    _v("gdp_blue_first_weight", "Store the first-release BLUE weight, rounded to 4 decimals."),
    _v("gdp_blue_second_weight", "Store the second-release BLUE weight, rounded to 4 decimals."),
    _v("gdp_blue_third_weight", "Store the third-release BLUE weight, rounded to 4 decimals."),
    _v("gdp_blue_weight_sum_residual", "Store absolute weight-sum residual, rounded to 12 decimals."),
    _v("gdp_blue_estimate_pct", "Store the bias-corrected BLUE estimate of target real-GDP growth in percent, rounded to 4 decimals."),
    _v("gdp_blue_standard_error_pp", "Store its correlated-measurement standard error in percentage points, rounded to 4 decimals."),
    _v("gdp_most_recent_pct", "Store target most-recent real-GDP growth in percent, rounded to 4 decimals."),
    _v("gdp_third_release_error_pp", "Store third release minus most-recent value in percentage points, rounded to 6 decimals."),
    _v("z1_training_quarter_count", "Store the complete pre-validation household-net-worth training-quarter count as an integer."),
    _v("z1_validation_quarter_count", "Store the complete Z.1 covariance-selection validation-quarter count as an integer."),
    _v("z1_transferred_covariance_model", "Store the GDP-selected transfer-candidate covariance architecture as exactly full or diagonal."),
    _v("z1_transferred_model_validation_rmse_bp", "Store the transferred architecture's Z.1 validation RMSE in log basis points, rounded to 6 decimals."),
    _v("z1_alternative_model_validation_rmse_bp", "Store the other architecture's Z.1 validation RMSE in log basis points, rounded to 6 decimals."),
    _v("z1_selected_covariance_model", "Store the covariance architecture selected by Z.1 validation as exactly full or diagonal."),
    _v("z1_selection_agrees_with_gdp", "Store whether the Z.1-selected architecture equals the GDP-selected architecture as a boolean."),
    _v("z1_first_vintage_log_error_mean_bp", "Store mean first-vintage log error relative to the seventh vintage in basis points, rounded to 6 decimals."),
    _v("z1_second_vintage_log_error_mean_bp", "Store mean second-vintage log error in basis points, rounded to 6 decimals."),
    _v("z1_third_vintage_log_error_mean_bp", "Store mean third-vintage log error in basis points, rounded to 6 decimals."),
    _v("z1_first_second_error_correlation", "Store correlation of first- and second-vintage log errors, rounded to 4 decimals."),
    _v("z1_first_third_error_correlation", "Store correlation of first- and third-vintage log errors, rounded to 4 decimals."),
    _v("z1_second_third_error_correlation", "Store correlation of second- and third-vintage log errors, rounded to 4 decimals."),
    _v("z1_blue_first_weight", "Store the first-vintage log-BLUE weight, rounded to 4 decimals."),
    _v("z1_blue_second_weight", "Store the second-vintage log-BLUE weight, rounded to 4 decimals."),
    _v("z1_blue_third_weight", "Store the third-vintage log-BLUE weight, rounded to 4 decimals."),
    _v("z1_blue_log_standard_error_bp", "Store the log-BLUE standard error in basis points, rounded to 6 decimals."),
    _v("z1_target_first_vintage_usd_millions", "Store target first-vintage level in millions of USD, rounded to 3 decimals."),
    _v("z1_target_second_vintage_usd_millions", "Store target second-vintage level in millions of USD, rounded to 3 decimals."),
    _v("z1_target_third_vintage_usd_millions", "Store target third-vintage level in millions of USD, rounded to 3 decimals."),
    _v("z1_blue_terminal_estimate_usd_millions",
       "Store exponentiated bias-corrected log-BLUE terminal estimate in millions of USD, rounded "
       "to 3 decimals. Compute from the unrounded inputs, not from the rounded figures reported for "
       "them."),
    _v("z1_blue_lower_95_usd_millions",
       "Store exponentiated symmetric-log 95% lower endpoint in millions of USD, rounded to 3 "
       "decimals. Compute from the unrounded inputs, not from the rounded figures reported for "
       "them."),
    _v("z1_blue_upper_95_usd_millions",
       "Store exponentiated symmetric-log 95% upper endpoint in millions of USD, rounded to 3 "
       "decimals. Compute from the unrounded inputs, not from the rounded figures reported for "
       "them."),
    _v("z1_newest_vintage_usd_millions", "Store target seventh-vintage level in millions of USD, rounded to 3 decimals."),
    _v("z1_third_vintage_error_usd_millions", "Store third-vintage minus seventh-vintage level in millions of USD, rounded to 3 decimals."),
]

DECIMALS = [
    0, 4, 4, 4, 4, 4, 4, 4,
    0, 4, 4, None, 4, 4, 4, 12, 4, 4, 4, 6,
    0, 0, None, 6, 6, None, None, 6, 6, 6, 4, 4, 4, 4, 4, 4, 6,
    3, 3, 3, 3, 3, 3, 3, 3,
]


def _blue(error_matrix):
    means = error_matrix.mean(axis=0)
    covariance = np.cov(error_matrix, rowvar=False, ddof=1)
    inverse_one = np.linalg.solve(covariance, np.ones(3))
    weights = inverse_one / (np.ones(3) @ inverse_one)
    standard_error = float(np.sqrt(1 / (np.ones(3) @ np.linalg.solve(covariance, np.ones(3)))))
    return means, covariance, weights, standard_error


@lru_cache(maxsize=1)
def _state():
    rtdsm = load_expansion_table("philadelphia_fed_rtdsm")
    gdp = rtdsm.loc[rtdsm.series.astype(str).eq(RTDSM_SERIES)].copy()
    complete = gdp.dropna(subset=[*RELEASE_COLUMNS, "most_recent"])
    training = complete.loc[complete.period.astype(str).le(RTDSM_TRAIN_END)]
    validation = complete.loc[
        complete.period.astype(str).gt(RTDSM_TRAIN_END)
        & complete.period.astype(str).le(RTDSM_VALIDATION_END)
    ]
    gdp_errors = training[list(RELEASE_COLUMNS)].to_numpy(float) - training[["most_recent"]].to_numpy(float)
    gdp_means, gdp_covariance, gdp_weights, gdp_se = _blue(gdp_errors)
    diagonal = np.diag(np.diag(gdp_covariance))
    one = np.ones(3)
    diagonal_weights = np.linalg.solve(diagonal, one)
    diagonal_weights /= one @ diagonal_weights
    validation_observations = validation[list(RELEASE_COLUMNS)].to_numpy(float) - gdp_means
    validation_terminal = validation.most_recent.to_numpy(float)
    full_rmse = float(np.sqrt(np.mean((validation_observations @ gdp_weights - validation_terminal) ** 2)))
    diagonal_rmse = float(np.sqrt(np.mean((validation_observations @ diagonal_weights - validation_terminal) ** 2)))
    selected_model = "full" if full_rmse <= diagonal_rmse else "diagonal"
    selected_weights = gdp_weights if selected_model == "full" else diagonal_weights
    selected_covariance = gdp_covariance if selected_model == "full" else diagonal
    gdp_se = float(np.sqrt(1 / (one @ np.linalg.solve(selected_covariance, one))))
    target = gdp.loc[gdp.period.astype(str).eq(RTDSM_TARGET)].iloc[0]
    gdp_estimate = float(selected_weights @ (target[list(RELEASE_COLUMNS)].to_numpy(float) - gdp_means))

    z1 = load_expansion_table("fed_z1_vintages")
    z1 = z1.loc[
        z1.table.astype(str).eq(Z1_TABLE) & z1.frequency.astype(str).eq("Q")
        & z1.series.astype(str).eq(Z1_SERIES)
    ].copy()
    vintages = sorted(z1.vintage.astype(str).unique())
    panel = z1.pivot_table(index=["series", "date"], columns="vintage", values="value", aggfunc="first")
    dates = panel.index.get_level_values("date").astype(str)
    history = panel.loc[dates <= "2022-09-30"].dropna(subset=vintages)
    z1_validation = panel.loc[(dates > "2022-09-30") & (dates <= "2023-09-30")].dropna(subset=vintages)
    if (history[vintages] <= 0).any(axis=None) or (z1_validation[vintages] <= 0).any(axis=None):
        raise ValueError("log-BLUE training levels must be positive")
    z1_errors = np.log(history[vintages[:3]].to_numpy(float)) - np.log(history[[vintages[-1]]].to_numpy(float))
    z1_means, z1_covariance, z1_full_weights, z1_full_se = _blue(z1_errors)
    z1_diagonal = np.diag(np.diag(z1_covariance))
    z1_diagonal_weights = np.linalg.solve(z1_diagonal, one)
    z1_diagonal_weights /= one @ z1_diagonal_weights
    z1_diagonal_se = float(np.sqrt(1 / (one @ np.linalg.solve(z1_diagonal, one))))
    z1_validation_observations = np.log(z1_validation[vintages[:3]].to_numpy(float)) - z1_means
    z1_validation_terminal = np.log(z1_validation[vintages[-1]].to_numpy(float))
    z1_rmse = {
        "full": float(np.sqrt(np.mean(((z1_validation_observations @ z1_full_weights - z1_validation_terminal) * 10000) ** 2))),
        "diagonal": float(np.sqrt(np.mean(((z1_validation_observations @ z1_diagonal_weights - z1_validation_terminal) * 10000) ** 2))),
    }
    alternative_model = "diagonal" if selected_model == "full" else "full"
    z1_selected_model = selected_model if z1_rmse[selected_model] <= z1_rmse[alternative_model] else alternative_model
    if z1_selected_model == "full":
        z1_weights, z1_se = z1_full_weights, z1_full_se
    else:
        z1_weights, z1_se = z1_diagonal_weights, z1_diagonal_se
    z1_target = panel.loc[(Z1_SERIES, Z1_TARGET), vintages]
    return (training, validation, gdp_means, gdp_covariance, selected_weights, gdp_se,
            full_rmse, diagonal_rmse, selected_model, target, gdp_estimate, history,
            z1_validation, z1_means, z1_covariance, z1_weights, z1_se, z1_rmse,
            z1_selected_model, z1_target, vintages)


@lru_cache(maxsize=1)
def ground_truth():
    (training, validation, gm, gc, gw, gse, full_rmse, diagonal_rmse, selected_model,
     target, gest, history, zvalidation, zm, zc, zw, zse, zrmse, zselected,
     ztarget, vintages) = _state()
    gerrors = training[list(RELEASE_COLUMNS)].to_numpy(float) - training[["most_recent"]].to_numpy(float)
    gcorr = np.corrcoef(gerrors, rowvar=False)
    zerrors = np.log(history[vintages[:3]].to_numpy(float)) - np.log(history[[vintages[-1]]].to_numpy(float))
    zcorr = np.corrcoef(zerrors, rowvar=False)
    most_recent = float(target.most_recent)
    third_error = float(target.third_release) - most_recent
    gerror = gest - most_recent
    theta = float(zw @ (np.log(ztarget[vintages[:3]].to_numpy(float)) - zm))
    zest = float(np.exp(theta))
    lower, upper = float(np.exp(theta - 1.96 * zse)), float(np.exp(theta + 1.96 * zse))
    newest = float(ztarget[vintages[-1]])
    zthird_error = float(ztarget[vintages[2]]) - newest
    zerror = zest - newest
    return (
        len(training),
        gm[0],
        gm[1],
        gm[2],
        gcorr[0, 1],
        gcorr[0, 2],
        gcorr[1, 2],
        np.linalg.cond(gc),
        len(validation),
        full_rmse,
        diagonal_rmse,
        selected_model,
        gw[0],
        gw[1],
        gw[2],
        abs(float(gw.sum() - 1)),
        gest,
        gse,
        most_recent,
        third_error,
        len(history),
        len(zvalidation),
        selected_model,
        zrmse[selected_model],
        zrmse["diagonal" if selected_model == "full" else "full"],
        zselected,
        zselected == selected_model,
        zm[0] * 10000,
        zm[1] * 10000,
        zm[2] * 10000,
        zcorr[0, 1],
        zcorr[0, 2],
        zcorr[1, 2],
        zw[0],
        zw[1],
        zw[2],
        zse * 10000,
        float(ztarget[vintages[0]]),
        float(ztarget[vintages[1]]),
        float(ztarget[vintages[2]]),
        zest,
        lower,
        upper,
        newest,
        zthird_error,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_gdp_error_model": turn_validator(validate_turn_1),
    "validate_gdp_blue": turn_validator(validate_turn_2),
    "validate_z1_error_model": turn_validator(validate_turn_3),
    "validate_z1_blue": turn_validator(validate_turn_4),
}
