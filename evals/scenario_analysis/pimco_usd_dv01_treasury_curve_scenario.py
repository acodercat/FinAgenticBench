"""Apply an observed tenor-aligned Treasury move to reported USD DV01.

The PIMCO Income Fund has one USD interest-rate-risk row in N-PORT accession
0001099263-25-004485 for 2025-09-30. From that date through 2025-12-31, the
Treasury par yields at 3 months, 1 year, 5 years, 10 years and 30 years change
by -35, -20, -1, +2 and +11 basis points. Under the pinned convention that a
positive reported DV01 is a gain for a one-basis-point yield decline, the
tenor contributions sum to +443.015889 million USD; their gross absolute sum
is 496.749617 million USD. The 1-year bucket is largest at +405.324463 million
USD, and the signed total is 0.218699% of reported net assets.

This is a retrospective first-order proxy, not observed fund performance. The
Treasury par curve is also a public benchmark proxy rather than proof of the
fund's exact valuation risk factors. Reversing the DV01 direction convention
changes the signed result to -443.015889 million USD; treating percentage-point
changes as basis points shrinks it to 4.430159 million USD; applying the 10-year
change in parallel to every bucket gives -176.545224 million USD. The query pins
the scenario choices, so these are regression probes for a medium baseline.

Numeric validation uses 0.6 x 10^-N rounding-boundary tolerance for N requested
decimals. The bucket label is one of the five tokens the variable description lists;
case and spacing variants of those tokens are normalized, but a field-name spelling
such as 1year is not.
"""

import re

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


ACCESSION = "0001099263-25-004485"
START_DATE = "2025-09-30"
END_DATE = "2025-12-31"
TENORS = [
    ("3 month", "dv01_3month", "3 Mo"),
    ("1 year", "dv01_1year", "1 Yr"),
    ("5 year", "dv01_5year", "5 Yr"),
    ("10 year", "dv01_10year", "10 Yr"),
    ("30 year", "dv01_30year", "30 Yr"),
]

variables = [
    Variable("three_month_yield_change_basis_points", None, "Store ending minus starting three-month Treasury par yield in basis points, rounded to 1 decimal."),
    Variable("one_year_yield_change_basis_points", None, "Store ending minus starting one-year Treasury par yield in basis points, rounded to 1 decimal."),
    Variable("five_year_yield_change_basis_points", None, "Store ending minus starting five-year Treasury par yield in basis points, rounded to 1 decimal."),
    Variable("ten_year_yield_change_basis_points", None, "Store ending minus starting ten-year Treasury par yield in basis points, rounded to 1 decimal."),
    Variable("thirty_year_yield_change_basis_points", None, "Store ending minus starting thirty-year Treasury par yield in basis points, rounded to 1 decimal."),
    Variable("linear_value_change_usd_millions", None, "Store the signed tenor-summed linear value-change estimate in USD millions, rounded to 3 decimals."),
    Variable("gross_absolute_tenor_contribution_usd_millions", None, "Store the sum of absolute tenor contributions in USD millions, rounded to 3 decimals."),
    Variable("largest_absolute_contribution_tenor", None, "Store the maturity bucket with the largest absolute contribution as exactly one of 3 month, 1 year, 5 year, 10 year or 30 year."),
    Variable("largest_tenor_signed_contribution_usd_millions", None, "Store that maturity bucket's signed contribution in USD millions, rounded to 3 decimals."),
    Variable("linear_value_change_pct_net_assets", None, "Store the signed linear estimate divided by reported net assets in percent, rounded to 4 decimals."),
]


def _one(frame, description):
    if len(frame) != 1:
        raise ValueError(f"expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def _normalized_tenor(value):
    if not isinstance(value, str):
        return value
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    aliases = {"1 yr": "1 year", "one year": "1 year"}
    return aliases.get(normalized, normalized)


def ground_truth():
    risk = load_expansion_table("sec_nport_interest_rate_risk")
    row = _one(
        risk.loc[
            risk.accession.eq(ACCESSION)
            & risk.report_date.eq(START_DATE)
            & risk.currency_code.eq("USD")
        ],
        "PIMCO USD N-PORT interest-rate-risk",
    )

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    start = _one(curve.loc[curve.Date.eq(START_DATE)], "starting Treasury curve")
    end = _one(curve.loc[curve.Date.eq(END_DATE)], "ending Treasury curve")

    yield_changes = []
    contributions = []
    for label, dv01_column, curve_column in TENORS:
        change_basis_points = (float(end[curve_column]) - float(start[curve_column])) * 100
        contribution = float(row[dv01_column]) * -change_basis_points
        yield_changes.append(change_basis_points)
        contributions.append((label, contribution))

    net = sum(value for _, value in contributions)
    gross = sum(abs(value) for _, value in contributions)
    largest_label, largest_value = sorted(
        contributions, key=lambda item: (-abs(item[1]), item[0])
    )[0]
    return (
        *yield_changes,
        net / 1e6,
        gross / 1e6,
        largest_label,
        largest_value / 1e6,
        net / float(row.net_assets_usd) * 100,
    )


def validate(outputs):
    normalized_outputs = dict(outputs)
    normalized_outputs["largest_absolute_contribution_tenor"] = _normalized_tenor(
        outputs.get("largest_absolute_contribution_tenor")
    )
    expected = list(ground_truth())
    expected[7] = _normalized_tenor(expected[7])
    return validate_ordered_outputs(
        normalized_outputs,
        variables,
        expected,
        [1, 1, 1, 1, 1, 3, 3, None, 3, 4],
    )



TURN_1_NAMES = [
    "three_month_yield_change_basis_points",
    "one_year_yield_change_basis_points",
    "five_year_yield_change_basis_points",
    "ten_year_yield_change_basis_points",
    "thirty_year_yield_change_basis_points",
]
TURN_2_NAMES = [
    "linear_value_change_usd_millions",
    "gross_absolute_tenor_contribution_usd_millions",
]
TURN_3_NAMES = [
    "largest_absolute_contribution_tenor",
    "largest_tenor_signed_contribution_usd_millions",
    "linear_value_change_pct_net_assets",
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
    "validate_curve_moves": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_scenario_totals": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_tenor_attribution": turn_validator(_turn_subset(TURN_3_NAMES)),
}
