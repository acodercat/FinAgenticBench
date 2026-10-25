"""Separate OTC derivative notional, replacement value and risk-class overlap.

At 2025-S2 the outstanding notional of OTC contracts is 844.577771 trillion USD
while gross market value is 22.802351 trillion, or 2.6999 percent of notional.
Notional is a reference quantity that scales contractual payments. Gross market
value sums absolute replacement values at current prices before bilateral netting
and collateral; it measures the scale of financial risk transfer, not net
counterparty credit exposure.

The published risk categories overlap. Foreign exchange plus gold reproduces
foreign exchange including gold, while gold is also inside commodities. Adding
the six named headline classes therefore counts gold twice, producing 845.714576
trillion against the published 844.577771 total, an excess of 1.136804 trillion.

Market-value-to-notional ratios differ across categories, but these aggregates do
not isolate why. Across the five headline categories that partition the total,
interest rate is lowest at 2.4384 percent and other derivatives highest at 7.2946,
0.18 percentage points above equity. Including a containing aggregate would compare
a parent category with its members. The commodity sub-breakdown is also outside the
ranking: gold at 9.7753 percent, other precious metals at 17.2852 and other
commodities at 12.6880 all exceed every headline category, but they sit one level
below the five and inside the commodity aggregate, so ranking them beside interest
rate compares a component with a market.

The aggregate ratio moves from 3.3582 percent at 2022-S2 to 2.3091 percent at
2024-S1 and 2.6999 percent at 2025-S2. This shows that gross market value and
notional need not move proportionally. Three ten-year Treasury yield snapshots
provide concurrent rate context, not an identified cause of the all-category
ratio.

Measured rejected alternatives include
summing overlapping classes, substituting the exclusive foreign-exchange class,
ranking every published class, changing a ratio denominator and selecting another
period. The ranking scope is the one worth naming: a scope described by a property
rather than enumerated admits the commodity sub-breakdown, and that moves the highest
ratio from other derivatives at 7.2946 percent to other precious metals at 17.2852,
which is why the query now lists the five ranked categories. Adding the containing
aggregates on top of the sub-breakdown changes nothing further, because none of them
holds an extreme ratio, so both readings are one measured alternative. The query pins
every one of these choices, so they are baseline instruction-following probes rather
than a live hidden-method trap. This is a medium baseline case.

Boundaries: gross credit exposure, not declared here, adjusts gross market value
for legally enforceable bilateral netting and is the closer counterparty-credit-
risk measure before collateral. Category ratios can reflect contract mix,
maturity, moneyness and market moves; the two declared tables do not decompose
those drivers. The identity excess is computed in trillions to absorb sub-million
publication rounding. Numeric validation uses the shared rounding tolerance.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


PERIOD = "2025-S2"
STRESS_PERIOD = "2022-S2"
TROUGH_PERIOD = "2024-S1"
NOTIONAL_MEASURE = "A"
MARKET_VALUE_MEASURE = "D"
TOTAL_CLASS = "A"
HEADLINE_CLASSES = ("D", "C", "E", "J", "T", "U")
NON_OVERLAPPING_CLASSES = ("D", "B", "E", "T", "U")
MILLIONS_PER_TRILLION = 1e6
PERIOD_END_DATES = {"2022-S2": "2022-12-30", "2024-S1": "2024-06-28", "2025-S2": "2025-12-31"}
CURVE_TENOR = "10 Yr"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "period_label", "total_notional_usd_trillions", "total_market_value_usd_trillions",
    "market_value_share_of_notional_pct", "notional_to_market_value_multiple",
]
TURN_2_NAMES = [
    "foreign_exchange_notional_usd_trillions", "gold_notional_usd_trillions",
    "foreign_exchange_including_gold_usd_trillions", "commodities_notional_usd_trillions",
    "headline_class_sum_usd_trillions",
]
TURN_3_NAMES = [
    "interest_rate_ratio_pct", "foreign_exchange_ratio_pct", "equity_ratio_pct",
    "credit_ratio_pct", "highest_ratio_class_code", "highest_ratio_pct",
    "lowest_ratio_class_code", "lowest_ratio_pct",
]
TURN_4_NAMES = [
    "stress_period_ratio_pct", "trough_period_ratio_pct",
    "stress_period_notional_usd_trillions", "stress_period_ten_year_yield_pct",
    "trough_period_ten_year_yield_pct", "latest_period_ten_year_yield_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the label of the latest reported period as a YYYY-Sn string."),
    _v(TURN_1_NAMES[1], "Store total outstanding notional in USD trillions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[2], "Store total gross market value in USD trillions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[3], "Store gross market value as a percent of notional, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store notional divided by gross market value, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the foreign-exchange notional in USD trillions, rounded to 6 "
                        "decimals, taken from its own unrounded source row rather than from the "
                        "other two foreign-exchange figures reported here."),
    _v(TURN_2_NAMES[1], "Store the gold notional in USD trillions, rounded to 6 decimals, taken "
                        "from its own unrounded source row rather than from the other two "
                        "foreign-exchange figures reported here."),
    _v(TURN_2_NAMES[2], "Store the foreign-exchange-including-gold notional in USD trillions, "
                        "rounded to 6 decimals, taken from its own unrounded source row rather "
                        "than from the other two foreign-exchange figures reported here."),
    _v(TURN_2_NAMES[3], "Store the commodities notional in USD trillions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[4], "Store the sum of the six headline classes in USD trillions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[0], "Store the interest-rate class ratio of market value to notional in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the foreign-exchange class ratio in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the equity class ratio in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the credit class ratio in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the single-letter code of the class with the highest ratio as text."),
    _v(TURN_3_NAMES[5], "Store that ratio in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the single-letter code of the class with the lowest ratio as text."),
    _v(TURN_3_NAMES[7], "Store that ratio in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the aggregate ratio at the stress period in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store the aggregate ratio at the trough period in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store total notional at the stress period in USD trillions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[3], "Store the ten-year par yield at the stress period end in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the ten-year par yield at the trough period end in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the ten-year par yield at the latest period end in percent, rounded to 4 decimals."),
]

DECIMALS = [
    None, 6, 6, 4, 4,
    6, 6, 6, 6, 6,
    4, 4, 4, 4, None, 4, None, 4,
    4, 4, 6, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _table():
    frame = load_expansion_table("bis_otc_derivatives").copy()
    frame.columns = [column.split(":")[0] for column in frame.columns]
    frame["value"] = pd.to_numeric(frame.OBS_VALUE, errors="coerce")
    frame["measure"] = frame.DER_TYPE.astype(str).str.slice(0, 1)
    frame["risk"] = frame.DER_RISK.astype(str).str.slice(0, 1)
    frame["period"] = frame.TIME_PERIOD.astype(str)
    return frame


def _series(period, measure):
    frame = _table()
    rows = frame.loc[frame.period.eq(period) & frame.measure.eq(measure)]
    if rows.empty:
        raise ValueError(f"no rows for {period} measure {measure}")
    if rows.risk.duplicated().any():
        raise ValueError(f"duplicate risk classes for {period} measure {measure}")
    return rows.set_index("risk").value / MILLIONS_PER_TRILLION


@lru_cache(maxsize=1)
def ground_truth():
    notional = _series(PERIOD, NOTIONAL_MEASURE)
    market_value = _series(PERIOD, MARKET_VALUE_MEASURE)
    total_notional = float(notional[TOTAL_CLASS])
    total_market_value = float(market_value[TOTAL_CLASS])

    headline_sum = float(sum(notional[code] for code in HEADLINE_CLASSES))
    ratios = {
        code: float(market_value[code]) / float(notional[code]) * 100
        for code in NON_OVERLAPPING_CLASSES
    }
    highest = max(ratios, key=lambda code: ratios[code])
    lowest = min(ratios, key=lambda code: ratios[code])

    stress_notional = _series(STRESS_PERIOD, NOTIONAL_MEASURE)
    stress_market_value = _series(STRESS_PERIOD, MARKET_VALUE_MEASURE)
    trough_notional = _series(TROUGH_PERIOD, NOTIONAL_MEASURE)
    trough_market_value = _series(TROUGH_PERIOD, MARKET_VALUE_MEASURE)
    stress_ratio = float(stress_market_value[TOTAL_CLASS]) / float(stress_notional[TOTAL_CLASS]) * 100
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["date"] = curve.Date.astype(str)

    def ten_year(period):
        row = curve.loc[curve.date.eq(PERIOD_END_DATES[period]), CURVE_TENOR]
        if row.empty:
            raise ValueError(f"no curve row for {period}")
        return float(pd.to_numeric(row, errors="coerce").iloc[0])

    trough_ratio = float(trough_market_value[TOTAL_CLASS]) / float(trough_notional[TOTAL_CLASS]) * 100
    current_ratio = total_market_value / total_notional * 100

    return (
        PERIOD,
        total_notional,
        total_market_value,
        current_ratio,
        total_notional / total_market_value,
        float(notional["B"]),
        float(notional["L"]),
        float(notional["C"]),
        float(notional["J"]),
        headline_sum,
        ratios["D"],
        ratios["B"],
        ratios["E"],
        ratios["T"],
        highest,
        ratios[highest],
        lowest,
        ratios[lowest],
        stress_ratio,
        trough_ratio,
        float(stress_notional[TOTAL_CLASS]),
        ten_year(STRESS_PERIOD),
        ten_year(TROUGH_PERIOD),
        ten_year(PERIOD),
    )


NAME_OUTPUTS = ("period_label", "highest_ratio_class_code", "lowest_ratio_class_code")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = str(normalized[name]).strip().upper()
            truth[name] = str(truth[name]).strip().upper()
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
    "validate_notional_versus_value": turn_validator(validate_turn_1),
    "validate_class_overlap": turn_validator(validate_turn_2),
    "validate_ratio_by_class": turn_validator(validate_turn_3),
    "validate_ratio_path": turn_validator(validate_turn_4),
}
