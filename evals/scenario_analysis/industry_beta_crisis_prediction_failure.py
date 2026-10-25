"""Predict a crash from estimated betas, then measure how badly the betas held.

A market beta is the multiplier a linear factor model puts on a systematic move, so
it is the natural instrument for asking what a market decline does to a portfolio.
Estimated on the 1,253 trading days from 2015 through 2019 on which the industry
returns, the research factors and the daily oil price all report, the ten
value-weighted industry portfolios span a wide range: technology at 1.1750 and
utilities at 0.4311.

Between 2020-02-19 and 2020-03-23, 24 trading days, the market excess return summed
to -38.3400 percent. Because the model is linear in daily returns, the prediction it
implies is the intercept times the number of days plus beta times that sum, and the
outcome it should be compared with is the same arithmetic sum of daily excess
returns rather than a compounded return. On that footing the predictions miss by a
mean absolute 12.9895 percentage points, and 7 of the 10 industries fell further
than their beta said they would.

Scaling each industry's estimation-period residual standard deviation to a 24-day
window gives a diagnostic distance for the cumulative residual, and 9 of the 10
industries land more than three such units from their prediction; telecommunications
is at -1.0506. These ratios are not calibrated hypothesis-test probabilities.
Utilities is the sharpest failure: a beta of 0.4311 predicted -16.1914 percent and
it delivered -44.4900, a miss of -28.2986 at -7.7540 standard errors. The low-beta
defensive sector did not defend. Technology missed as far in the other direction,
predicted at -44.5995 and delivering -32.6600, a positive 11.9395 at 6.0525.
Ranking by raw miss and by standard errors does not agree: energy has the largest
miss at -29.4706 but utilities the most extreme one, because energy's residuals were
always wider.

Oil collapsed over the same window from 53.31 to 23.33 dollars a barrel, a fall of
56.2371 percent and a daily sum of -63.9247. Re-estimating each industry on the market and the daily oil
return together, energy carries an oil loading of 0.2731 and its miss narrows from
-29.4706 to -20.0608, a 31.9294 percent numerical reduction in absolute miss.
Utilities carries an oil loading of 0.0161 and its miss barely moves, to -27.7425.
This is a fitted-prediction comparison, not a causal decomposition.

The `industry_beta_crisis` pinned-reading comparison records sensitivity to geometric
cumulation, omitting the intercept, changing the estimation or crisis window, using
raw rather than excess industry returns, and reporting the widest miss without its
sign after selecting the industry on absolute error. The query fixes the reported
linear-model convention, so these are robustness comparisons rather than hidden
answer paths.

Interpretive boundary: beta describes an estimated average sensitivity in the
chosen period, not a promise about a crisis. The residual scaling is a model-distance
diagnostic, and a linear daily-return prediction is evaluated against an arithmetic
sum on the same basis.

Boundaries: this is one episode, and an episode selected because it was severe.
The standard error used here is the estimation-period residual standard
deviation scaled by the square root of the day count, which treats those
residuals as independent and identically distributed, something a crisis is
precisely the occasion to doubt, and which omits the uncertainty in the fitted
coefficients themselves. That omission is not large but it is not nothing:
adding it inflates the standard errors by a median factor of 1.0433, which
leaves utilities at -7.4319 and energy at -5.7586 but moves health care from
3.0306 to 2.9047 and so takes the count beyond three from 9 to 8. The figures
reported here are distances from the model rather than calibrated probabilities.
The oil test uses a spot price rather than the futures curve the industry
actually hedges on.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and industry labels match exactly.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


ESTIMATION_WINDOW = ("2015-01-01", "2019-12-31")
CRISIS_WINDOW = ("2020-02-19", "2020-03-23")
MARKET_COLUMN = "mkt_rf_pct"
RISK_FREE_COLUMN = "rf_pct"
SIGMA_THRESHOLD = 3.0
UTILITIES, TECHNOLOGY, ENERGY, TELECOM = "utils", "hitec", "enrgy", "telcm"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["estimation_days", "crisis_days", "market_excess_sum_pct",
                "highest_beta_industry", "highest_beta", "lowest_beta"]
TURN_2_NAMES = ["utilities_predicted_pct", "utilities_actual_pct", "technology_predicted_pct",
                "technology_actual_pct", "mean_absolute_error_pp", "industries_worse_than_predicted"]
TURN_3_NAMES = ["utilities_standard_errors", "technology_standard_errors", "telecom_standard_errors",
                "industries_beyond_three_errors", "largest_miss_industry", "largest_miss_pp"]
TURN_4_NAMES = ["oil_start_usd", "oil_end_usd", "oil_change_pct", "oil_daily_sum_pct",
                "energy_oil_loading", "energy_miss_with_oil_pp", "utilities_miss_with_oil_pp"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many estimation days all three series share as an integer."),
    _v(TURN_1_NAMES[1], "Store how many crisis-window days they share as an integer."),
    _v(TURN_1_NAMES[2], "Store the summed market excess return over the crisis window in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3],
       "Store the label of the highest-beta industry as the canonical Ken French token: one of "
       "nodur, durbl, manuf, enrgy, hitec, telcm, shops, hlth, utils, other. Report the token "
       "alone, without the table's _pct column suffix and without expanding it to a descriptive "
       "name."),
    _v(TURN_1_NAMES[4], "Store its beta, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the lowest beta in the cross-section, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the predicted cumulative excess return for utilities in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store its realized cumulative excess return in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the predicted figure for technology in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store its realized figure in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the mean absolute prediction error across the ten industries in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store how many industries fell further than predicted as an integer."),
    _v(TURN_3_NAMES[0], "Store the utilities error divided by its window standard error, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the same figure for technology, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the same figure for telecommunications, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store how many industries lie more than three standard errors from their prediction as an integer."),
    _v(TURN_3_NAMES[4],
       "Store the label of the industry with the largest absolute error as the same canonical Ken "
       "French token used in turn 1."),
    _v(TURN_3_NAMES[5], "Store that industry's signed error, actual minus predicted, in percentage points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the oil price on the first crisis day in USD, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store it on the last crisis day in USD, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the percentage change between them, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the summed daily oil return over the window in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the energy portfolio's oil loading from the two-factor fit, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the energy error under the two-factor prediction in percentage points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store the utilities error under the two-factor prediction in percentage points, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 4, None, 4, 4,
            4, 4, 4, 4, 4, 0,
            4, 4, 4, 0, None, 4,
            4, 4, 4, 4, 4, 4, 4]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def _aligned():
    industries = load_ken_french_table("daily_industry_returns").copy()
    factors = load_ken_french_table("daily_factors").copy()
    oil = load_expansion_table("eia_bulk_wti_daily").copy()
    industries["day"] = pd.to_datetime(industries.date, errors="coerce")
    factors["day"] = pd.to_datetime(factors.date, errors="coerce")
    oil["day"] = pd.to_datetime(oil.observation_date, errors="coerce")
    oil["price"] = pd.to_numeric(oil.wti_spot_price_usd_per_barrel, errors="coerce")
    oil = oil.sort_values("day")
    oil["oil_return"] = oil.price.pct_change() * 100
    frame = industries.merge(factors, on="day", how="inner").merge(
        oil[["day", "price", "oil_return"]], on="day", how="inner")
    frame = frame.dropna(subset=["oil_return"])
    for column in (*_industry_columns(industries), MARKET_COLUMN, RISK_FREE_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _industry_columns(frame):
    return tuple(column for column in frame.columns if column.endswith("_pct"))


@lru_cache(maxsize=1)
def _fits():
    frame = _aligned()
    columns = tuple(
        column for column in frame.columns
        if column.endswith("_pct") and column not in (MARKET_COLUMN, RISK_FREE_COLUMN,
                                                      "smb_pct", "hml_pct")
    )
    estimation = frame.loc[frame.day.between(*ESTIMATION_WINDOW)]
    crisis = frame.loc[frame.day.between(*CRISIS_WINDOW)]
    single = np.column_stack([np.ones(len(estimation)), estimation[MARKET_COLUMN].values])
    double = np.column_stack([
        np.ones(len(estimation)), estimation[MARKET_COLUMN].values, estimation.oil_return.values,
    ])
    market_sum = float(crisis[MARKET_COLUMN].sum())
    oil_sum = float(crisis.oil_return.sum())

    rows = []
    for column in columns:
        excess = (estimation[column] - estimation[RISK_FREE_COLUMN]).values
        one = np.linalg.lstsq(single, excess, rcond=None)[0]
        two = np.linalg.lstsq(double, excess, rcond=None)[0]
        residual = excess - single @ one
        actual = float((crisis[column] - crisis[RISK_FREE_COLUMN]).sum())
        predicted = one[0] * len(crisis) + one[1] * market_sum
        predicted_two = two[0] * len(crisis) + two[1] * market_sum + two[2] * oil_sum
        window_error = float(residual.std(ddof=1)) * np.sqrt(len(crisis))
        rows.append({
            "industry": column[:-4], "beta": float(one[1]), "predicted": float(predicted),
            "actual": actual, "error": actual - float(predicted),
            "standard_errors": (actual - float(predicted)) / window_error,
            "oil_loading": float(two[2]), "error_two_factor": actual - float(predicted_two),
        })
    return pd.DataFrame(rows), estimation, crisis, market_sum, oil_sum


@lru_cache(maxsize=1)
def ground_truth():
    fits, estimation, crisis, market_sum, oil_sum = _fits()
    indexed = fits.set_index("industry")
    ranked = fits.sort_values(["beta", "industry"], ascending=[False, True]).reset_index(drop=True)
    widest = fits.assign(size=fits.error.abs()).sort_values(
        ["size", "industry"], ascending=[False, True]).reset_index(drop=True).iloc[0]

    return (
        int(len(estimation)), int(len(crisis)), market_sum,
        str(ranked.industry.iloc[0]), float(ranked.beta.iloc[0]), float(fits.beta.min()),
        float(indexed.loc[UTILITIES, "predicted"]), float(indexed.loc[UTILITIES, "actual"]),
        float(indexed.loc[TECHNOLOGY, "predicted"]), float(indexed.loc[TECHNOLOGY, "actual"]),
        float(fits.error.abs().mean()), int((fits.error < 0).sum()),
        float(indexed.loc[UTILITIES, "standard_errors"]),
        float(indexed.loc[TECHNOLOGY, "standard_errors"]),
        float(indexed.loc[TELECOM, "standard_errors"]),
        int(fits.standard_errors.abs().gt(SIGMA_THRESHOLD).sum()),
        str(widest.industry), float(widest.error),
        float(crisis.price.iloc[0]), float(crisis.price.iloc[-1]),
        float(crisis.price.iloc[-1] / crisis.price.iloc[0] - 1) * 100, oil_sum,
        float(indexed.loc[ENERGY, "oil_loading"]), float(indexed.loc[ENERGY, "error_two_factor"]),
        float(indexed.loc[UTILITIES, "error_two_factor"]),
    )


NAME_OUTPUTS = ("highest_beta_industry", "largest_miss_industry")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_beta_estimation": turn_validator(validate_turn_1),
    "validate_scenario_prediction": turn_validator(validate_turn_2),
    "validate_error_significance": turn_validator(validate_turn_3),
    "validate_omitted_factor": turn_validator(validate_turn_4),
}
