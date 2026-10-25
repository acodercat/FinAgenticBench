"""Baseline for a fully specified three-factor regression.

The query pins the dependent variable, regressors, units and annualization formula.
Omitting the final percentage conversion gives 0.03% instead of 3.36%, but the
conversion is expressly part of the requested calculation.
"""

import numpy as np
from cave_agent import Variable

from core.data import load_ken_french_table
from core.validation import ValidatorResult, numeric_equal, turn_validator


variables = [
    Variable("aligned_observation_count", None, "Store the aligned daily observation count as an integer."),
    Variable("daily_alpha_pct", None, "Store the OLS intercept in daily percentage points, rounded to 4 decimals."),
    Variable("market_beta", None, "Store the Mkt-RF coefficient, rounded to 4 decimals."),
    Variable("smb_beta", None, "Store the SMB coefficient, rounded to 4 decimals."),
    Variable("hml_beta", None, "Store the HML coefficient, rounded to 4 decimals."),
    Variable("r_squared", None, "Store ordinary in-sample R-squared, rounded to 4 decimals."),
    Variable("compounded_annualized_alpha_pct", None, "Store (1 + daily alpha/100)^252 - 1 as a percentage, rounded to 2 decimals."),
]


def ground_truth() -> tuple[int, float, float, float, float, float, float]:
    industries = load_ken_french_table("daily_industry_returns")[["date", "hitec_pct"]]
    factors = load_ken_french_table("daily_factors")
    rows = industries.merge(factors, on="date")
    rows = rows.loc[rows["date"].between("2018-01-01", "2022-12-31")]
    y = (rows["hitec_pct"] - rows["rf_pct"]).to_numpy(dtype=float)
    x = np.column_stack([
        np.ones(len(rows)),
        rows[["mkt_rf_pct", "smb_pct", "hml_pct"]].to_numpy(dtype=float),
    ])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    r_squared = 1 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))
    annualized_alpha = ((1 + coefficients[0] / 100) ** 252 - 1) * 100
    return len(rows), *(float(value) for value in coefficients), float(r_squared), float(annualized_alpha)


def validate(outputs: dict) -> ValidatorResult:
    if any(value is None for value in outputs.values()):
        return ValidatorResult(False, "required output not set", True)
    expected = ground_truth()
    decimals = [0, 4, 4, 4, 4, 4, 2]
    errors = []
    for variable, target, places in zip(variables, expected, decimals):
        actual = outputs[variable.name]
        if not numeric_equal(actual, target, decimals=places):
            errors.append(f"{variable.name}={actual!r}, expected {target:.{places}f}")
    return ValidatorResult(not errors, "; ".join(errors) or "correct")



TURN_1_NAMES = [
    "aligned_observation_count",
    "daily_alpha_pct",
    "market_beta",
    "smb_beta",
    "hml_beta",
    "r_squared",
]
TURN_2_NAMES = [
    "compounded_annualized_alpha_pct",
]


def _turn_subset(names):
    """Score only `names`, filling the other outputs with their own answers."""
    def run(outputs):
        merged = dict(zip((variable.name for variable in variables), ground_truth()))
        merged.update({name: outputs.get(name) for name in names})
        return validate(merged)

    return run


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_regression_fit": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_annualized_alpha": turn_validator(_turn_subset(TURN_2_NAMES)),
}
