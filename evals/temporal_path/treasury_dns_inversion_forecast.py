"""Dynamic Nelson--Siegel forecast at the deepest SOFR-era inversion.

The case selects the common Treasury/SOFR date with the most negative 10-year
minus 2-year CMT spread, bootstraps every trailing curve into continuously
compounded zero yields, and estimates one shared Diebold--Li decay parameter by
panel least squares.  The parameter is identified by the global minimum on the
declared compact interval; date-specific level, slope and curvature factors are
linear conditional on that decay.

The terminal comparison estimates a full VAR(1) for the carried factor history
and forecasts the next Treasury curve date.  It compares the resulting curve
forecast with a factor random walk.  This is a retrospective analytical
benchmark, not a no-arbitrage production curve or evidence that inversion alone
predicts returns.  Repricing, conditional OLS orthogonality and VAR stability are
the numerical invariants.  Intermediate calculations retain full precision.

The `treasury_dns_forecast` pinned-reading comparison records sensitivity to treating reported
par yields as zero yields instead of bootstrapping them, to a 250-curve rather than
252-curve training window, and to estimating separate AR(1) factor dynamics rather than
a full VAR(1). The curve basis is the one worth naming: reading par as spot
skips the bootstrap entirely, so it changes every zero yield the factors are fitted to
rather than only the forecast. The query fixes all three, so these are
robustness comparisons rather than hidden answer paths.

Boundaries: the shared decay is identified by a global minimum on a declared compact
interval, so it is a panel fit rather than a market-implied parameter. The VAR(1)
spectral radius of 0.9848 sits close to the unit circle on 252 observations, so its
stability is a diagnostic rather than an established property, and one next-date
comparison cannot establish forecast skill.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable
from scipy.optimize import minimize_scalar

from core.data import load_expansion_table
from core.validation import (
    turn_validator, validate_ordered_outputs,
)


NODE_MATURITIES = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
NODE_COLUMNS = ("6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr")
GRID = np.arange(0.5, 30.0 + 0.25, 0.5)
WINDOW = 252
LAMBDA_BOUNDS = (0.01, 5.0)


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "common_date_count", "selected_inversion_date", "selected_two_year_yield_pct",
    "selected_ten_year_yield_pct", "selected_sofr_pct",
]
TURN_2_NAMES = [
    "dns_training_start_date", "dns_training_end_date", "dns_training_observation_count",
    "dns_decay_lambda_per_year", "dns_panel_rmse_pp", "dns_loading_condition_number",
    "maximum_bootstrap_repricing_residual_per100",
]
TURN_3_NAMES = [
    "selected_dns_level_pct", "selected_dns_slope_pct", "selected_dns_curvature_pct",
    "selected_dns_fitted_two_year_pct", "selected_dns_fitted_ten_year_pct",
    "selected_dns_fitted_thirty_year_pct", "selected_dns_minimum_forward_maturity_years",
    "selected_dns_minimum_forward_rate_pct", "selected_dns_maximum_ols_moment",
]
TURN_4_NAMES = [
    "dns_next_curve_date", "dns_forecast_level_pct", "dns_forecast_slope_pct",
    "dns_forecast_curvature_pct", "dns_var_spectral_radius", "dns_forecast_curve_rmse_pp",
    "dns_random_walk_curve_rmse_pp", "dns_forecast_beats_random_walk",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the eligible common Treasury/SOFR date count as an integer."),
    _v(TURN_1_NAMES[1], "Store the selected inversion date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[2], "Store the selected two-year Treasury par yield in percent, rounded to 2 decimals."),
    _v(TURN_1_NAMES[3], "Store the selected ten-year Treasury par yield in percent, rounded to 2 decimals."),
    _v(TURN_1_NAMES[4], "Store the selected-date SOFR in percent, rounded to 2 decimals."),
    _v(TURN_2_NAMES[0], "Store the first date in the carried 252-curve training window as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1], "Store the last date in the carried 252-curve training window as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[2], "Store the carried DNS training observation count as an integer."),
    _v(TURN_2_NAMES[3], "Store the fitted shared DNS decay parameter per year, rounded to 5 decimals."),
    _v(TURN_2_NAMES[4],
       "Store the DNS panel root-mean-square zero-yield fitting error in percentage points, rounded "
       "to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure "
       "reported for it."),
    _v(TURN_2_NAMES[5],
       "Store the selected DNS loading-matrix spectral (2-norm) condition number, rounded to 4 "
       "decimals. Compute from the unrounded shared decay, not from the rounded figure reported for "
       "it."),
    _v(TURN_2_NAMES[6], "Store the maximum absolute par-instrument bootstrap repricing residual per 100 face across the training panel, rounded to 10 decimals."),
    _v(TURN_3_NAMES[0], "Store the selected-date DNS level factor in percent, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_3_NAMES[1], "Store the selected-date DNS slope factor in percent, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_3_NAMES[2], "Store the selected-date DNS curvature factor in percent, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_3_NAMES[3],
       "Store the selected-date DNS fitted two-year continuous zero yield in percent, rounded to 4 "
       "decimals. Compute from the unrounded shared decay, not from the rounded figure reported for "
       "it."),
    _v(TURN_3_NAMES[4],
       "Store the selected-date DNS fitted ten-year continuous zero yield in percent, rounded to 4 "
       "decimals. Compute from the unrounded shared decay, not from the rounded figure reported for "
       "it."),
    _v(TURN_3_NAMES[5],
       "Store the selected-date DNS fitted thirty-year continuous zero yield in percent, rounded to "
       "4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported "
       "for it."),
    _v(TURN_3_NAMES[6],
       "Store the maturity of the minimum fitted instantaneous forward rate in years, rounded to 2 "
       "decimals. Compute from the unrounded shared decay, not from the rounded figure reported for "
       "it."),
    _v(TURN_3_NAMES[7], "Store the minimum fitted instantaneous forward rate in percent, rounded to 6 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_3_NAMES[8],
       "Store the maximum absolute unscaled selected-date DNS OLS normal-equation moment max|X' "
       "residual|, rounded to 10 decimals. Compute from the unrounded shared decay, not from the "
       "rounded figure reported for it."),
    _v(TURN_4_NAMES[0], "Store the next available Treasury curve date as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[1], "Store the one-step VAR forecast DNS level factor in percent, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_4_NAMES[2], "Store the one-step VAR forecast DNS slope factor in percent, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_4_NAMES[3],
       "Store the one-step VAR forecast DNS curvature factor in percent, rounded to 4 decimals. "
       "Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_4_NAMES[4], "Store the fitted VAR transition matrix spectral radius, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_4_NAMES[5],
       "Store the one-step DNS forecast curve RMSE in percentage points, rounded to 4 decimals. "
       "Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_4_NAMES[6], "Store the factor-random-walk curve RMSE in percentage points, rounded to 4 decimals. Compute from the unrounded shared decay, not from the rounded figure reported for it."),
    _v(TURN_4_NAMES[7], "Store whether the DNS VAR forecast has lower curve RMSE than the factor random walk as a Boolean."),
]

DECIMALS = [
    0, None, 2, 2, 2,
    None, None, 0, 5, 4, 4, 10,
    4, 4, 4, 4, 4, 4, 2, 6, 10,
    None, 4, 4, 4, 4, 4, 4, None,
]


@lru_cache(maxsize=1)
def _selection():
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    sofr = load_expansion_table("nyfed_reference_rates").copy()
    sofr = sofr.loc[sofr["Rate Type"].eq("SOFR")].copy()
    sofr["Effective Date"] = pd.to_datetime(sofr["Effective Date"])
    sofr["sofr"] = pd.to_numeric(sofr["Rate (%)"])
    common = curve.merge(
        sofr[["Effective Date", "sofr"]], left_on="Date", right_on="Effective Date",
        how="inner", validate="one_to_one",
    )
    common["spread"] = pd.to_numeric(common["10 Yr"]) - pd.to_numeric(common["2 Yr"])
    selected = common.sort_values(["spread", "Date"], ascending=[True, True]).iloc[0]
    return curve.sort_values("Date").reset_index(drop=True), common, selected


def _bootstrap(row):
    nodes = np.array([float(pd.to_numeric(row[column])) for column in NODE_COLUMNS])
    par = np.interp(GRID, NODE_MATURITIES, nodes) / 100.0
    dfs = []
    residuals = []
    for rate in par:
        coupon = rate / 2.0
        df = (1.0 - coupon * sum(dfs)) / (1.0 + coupon)
        dfs.append(df)
        residuals.append(coupon * 100.0 * sum(dfs[:-1]) + (100.0 + coupon * 100.0) * df - 100.0)
    dfs = np.asarray(dfs)
    zeros = -np.log(dfs) / GRID * 100.0
    return zeros, float(np.max(np.abs(residuals)))


def _loadings(decay):
    scaled = decay * GRID
    slope = (1.0 - np.exp(-scaled)) / scaled
    curvature = slope - np.exp(-scaled)
    return np.column_stack([np.ones(len(GRID)), slope, curvature])


@lru_cache(maxsize=1)
def _dns_state():
    curve, _, selected = _selection()
    training = curve.loc[curve.Date.le(selected.Date)].tail(WINDOW).copy()
    zero_panel = np.vstack([_bootstrap(row)[0] for _, row in training.iterrows()])

    def objective(decay):
        loadings = _loadings(float(decay))
        factors = np.linalg.lstsq(loadings, zero_panel.T, rcond=None)[0]
        residual = zero_panel.T - loadings @ factors
        return float(np.mean(residual**2))

    fitted = minimize_scalar(
        objective, bounds=LAMBDA_BOUNDS, method="bounded",
        options={"xatol": 1e-13, "maxiter": 1000},
    )
    if not fitted.success:
        raise ValueError("DNS decay optimization failed")
    decay = float(fitted.x)
    loadings = _loadings(decay)
    factors = np.linalg.lstsq(loadings, zero_panel.T, rcond=None)[0].T
    fitted_panel = factors @ loadings.T
    bootstrap_residual = max(_bootstrap(row)[1] for _, row in training.iterrows())
    return training, zero_panel, decay, loadings, factors, fitted_panel, bootstrap_residual


@lru_cache(maxsize=1)
def ground_truth():
    curve, common, selected = _selection()
    training, zero_panel, decay, loadings, factors, fitted_panel, bootstrap_residual = _dns_state()
    selected_factors = factors[-1]
    selected_fit = fitted_panel[-1]
    moment = np.max(np.abs(loadings.T @ (zero_panel[-1] - selected_fit)))
    forward_grid = np.arange(0.25, 30.0 + 0.125, 0.25)
    forward = (
        selected_factors[0]
        + selected_factors[1] * np.exp(-decay * forward_grid)
        + selected_factors[2] * decay * forward_grid * np.exp(-decay * forward_grid)
    )
    minimum_index = int(np.argmin(forward))

    lagged_design = np.column_stack([np.ones(len(factors) - 1), factors[:-1]])
    var_coefficients = np.linalg.lstsq(lagged_design, factors[1:], rcond=None)[0]
    forecast_factors = np.r_[1.0, factors[-1]] @ var_coefficients
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(var_coefficients[1:]))))
    next_row = curve.loc[curve.Date.gt(selected.Date)].iloc[0]
    observed_next = _bootstrap(next_row)[0]
    forecast_curve = loadings @ forecast_factors
    random_walk_curve = loadings @ factors[-1]
    forecast_rmse = float(np.sqrt(np.mean((observed_next - forecast_curve) ** 2)))
    random_walk_rmse = float(np.sqrt(np.mean((observed_next - random_walk_curve) ** 2)))

    grid_index = {float(value): index for index, value in enumerate(GRID)}
    two = float(pd.to_numeric(selected["2 Yr"]))
    ten = float(pd.to_numeric(selected["10 Yr"]))
    sofr = float(selected.sofr)
    return (
        int(len(common)),
        selected.Date.strftime("%Y-%m-%d"),
        two,
        ten,
        sofr,
        training.Date.iloc[0].strftime("%Y-%m-%d"),
        training.Date.iloc[-1].strftime("%Y-%m-%d"),
        int(len(training)),
        decay,
        float(np.sqrt(np.mean((zero_panel - fitted_panel) ** 2))),
        float(np.linalg.cond(loadings)),
        float(bootstrap_residual),
        float(selected_factors[0]),
        float(selected_factors[1]),
        float(selected_factors[2]),
        float(selected_fit[grid_index[2.0]]),
        float(selected_fit[grid_index[10.0]]),
        float(selected_fit[grid_index[30.0]]),
        float(forward_grid[minimum_index]),
        float(forward[minimum_index]),
        float(moment),
        next_row.Date.strftime("%Y-%m-%d"),
        float(forecast_factors[0]),
        float(forecast_factors[1]),
        float(forecast_factors[2]),
        spectral_radius,
        forecast_rmse,
        random_walk_rmse,
        bool(forecast_rmse < random_walk_rmse),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_inversion_selection": turn_validator(validate_turn_1),
    "validate_dns_panel_fit": turn_validator(validate_turn_2),
    "validate_dns_selected_curve": turn_validator(validate_turn_3),
    "validate_dns_forecast": turn_validator(validate_turn_4),
}
