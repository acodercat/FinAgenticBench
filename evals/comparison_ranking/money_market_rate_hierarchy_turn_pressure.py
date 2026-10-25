"""Compare overnight benchmark segments, target-range placement and date effects.

TGCR, BGCR and SOFR are nested but distinct Treasury-repo benchmarks; OBFR and
EFFR measure unsecured overnight bank-funding segments.  The New York Fed
publishes each as a volume-weighted median.  Cross-benchmark ordering and a
SOFR-minus-EFFR spread are descriptive comparisons of different transaction
populations, not an identification of credit risk, dealer constraints or any
other causal driver.

The FOMC target range is for the federal funds rate.  EFFR is a market statistic
steered within that range through administered tools; SOFR is not the policy
target and has no compliance boundary at the range edges.  Its location inside
the range on one day does not by itself diagnose funding pressure.

The case measures SOFR deviations on four named quarter-end or last-business-day
observations relative to the prior five complete-panel dates.  Heterogeneous
deviations establish date sensitivity in the sample only; the rates table does
not observe balance-sheet management or prove a reporting-date mechanism.

SOFR's 99th percentile and published rate are volume-weighted distribution
statistics, not borrower identities.  Treasury's one- and three-month values are
interpolated constant-maturity par yields derived from indicative bid-side market
quotes, not observed bill transaction yields.  Subtracting them from SOFR is a
cross-instrument, cross-horizon comparison and cannot by itself define a single
money-market yield-curve inversion.

The pinned-reading comparison measures the turn reference rate, the spread statistic, the
turn baseline, the tail measure and the tail threshold.  Measuring the quarter-end
deviations on BGCR instead of SOFR reads the first, third and fourth as 1, 5 and 6
basis points rather than 3, 12 and 9, and moves the largest-deviation date from
2024-09-30 to 2024-12-31; taking the mean secured-minus-unsecured spread instead
of the median reports +0.144 basis points rather than -1 basis point; baselining
on the previous business day instead of the prior five complete-panel dates reads
the first, second and fourth as 1, -1 and 12 rather than 3, 0 and 9; measuring
the tail as the target range's full width instead of upper bound minus SOFR
reports 12 wide days and a 62-basis-point maximum on 2024-09-30 rather than 6
days and 52 basis points on 2024-09-19; and a 20-basis-point
threshold reports 8 wide days rather than 6.  The query pins all five choices.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and dates match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


YEAR = "2024"
RATE_TYPES = ("TGCR", "BGCR", "SOFR", "OBFR", "EFFR")
SECURED_RATE = "SOFR"
UNSECURED_RATE = "EFFR"
QUARTER_END_DATES = ("2024-03-28", "2024-06-28", "2024-09-30", "2024-12-31")
YEAR_END_DATE = "2024-12-31"
LOOKBACK_DAYS = 5
TAIL_THRESHOLD_BP = 25.0


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "complete_business_day_count", "median_tri_party_rate_pct", "median_broad_collateral_rate_pct",
    "median_secured_overnight_rate_pct", "median_bank_funding_rate_pct", "median_fed_funds_rate_pct",
    "median_secured_minus_unsecured_bp", "widest_secured_minus_unsecured_date",
    "widest_secured_minus_unsecured_bp",
]
TURN_2_NAMES = [
    "year_end_target_floor_pct", "year_end_target_ceiling_pct", "year_end_secured_rate_pct",
    "year_end_secured_above_floor_bp", "year_end_fed_funds_above_floor_bp",
]
TURN_3_NAMES = [
    "quarter_one_deviation_bp", "quarter_two_deviation_bp", "quarter_three_deviation_bp",
    "quarter_four_deviation_bp", "largest_deviation_quarter_end_date", "largest_deviation_bp",
]
TURN_4_NAMES = [
    "wide_tail_day_count", "wide_tail_december_day_count", "widest_tail_date",
    "widest_tail_bp", "year_end_one_month_treasury_par_yield_pct",
    "year_end_three_month_treasury_par_yield_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many business days carry a published rate for all five benchmarks as an integer."),
    _v(TURN_1_NAMES[1], "Store the median tri-party general collateral rate in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2], "Store the median broad general collateral rate in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store the median secured overnight financing rate in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the median overnight bank funding rate in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the median effective federal funds rate in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store the median of the daily secured-minus-unsecured spread in basis points, rounded to 4 decimals."),
    _v(TURN_1_NAMES[7], "Store the date of the widest such spread as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[8], "Store that spread in basis points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the lower bound of the target range in effect at the year-end date, in percent rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store its upper bound in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the secured overnight financing rate on that date in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store how far that rate sits above the lower bound in basis points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store how far the effective federal funds rate sits above the lower bound in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the first named date's SOFR deviation from its prior-five-date median in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the second named date's deviation in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the third named date's deviation in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the fourth named date's deviation in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the named date with the largest signed deviation as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[5], "Store that signed deviation in basis points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many business days in the year carry a tail at or above the stated threshold as an integer."),
    _v(TURN_4_NAMES[1], "Store how many of those fall in December as an integer."),
    _v(TURN_4_NAMES[2], "Store the date of the widest tail as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[3], "Store that tail in basis points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the one-month Treasury constant-maturity par yield at year end in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the three-month Treasury constant-maturity par yield at year end in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, 4, 4, 4, 4, 4, 4, None, 4,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, None, 4,
    0, 0, None, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _rates():
    rates = load_expansion_table("nyfed_reference_rates").copy()
    rates["date"] = rates["Effective Date"].astype(str)
    for column in ("Rate (%)", "99th Percentile (%)", "Target Rate From (%)", "Target Rate To (%)"):
        rates[column] = pd.to_numeric(rates[column], errors="coerce")
    return rates.loc[rates.date.str.startswith(YEAR)]


@lru_cache(maxsize=1)
def _panel():
    rates = _rates()
    panel = rates.pivot_table(index="date", columns="Rate Type", values="Rate (%)")
    missing = [name for name in RATE_TYPES if name not in panel.columns]
    if missing:
        raise ValueError(f"the reference-rate table is missing {missing}")
    return panel.dropna(subset=list(RATE_TYPES)).sort_index()


@lru_cache(maxsize=1)
def ground_truth():
    panel = _panel()
    spread = (panel[SECURED_RATE] - panel[UNSECURED_RATE]) * 100
    widest = spread.sort_values(ascending=False)
    if (spread == widest.iloc[0]).sum() != 1:
        raise ValueError("the widest secured-minus-unsecured spread is not unique")

    rates = _rates()
    band = rates.loc[rates["Rate Type"].astype(str).eq(UNSECURED_RATE) & rates.date.eq(YEAR_END_DATE)]
    floor = float(band["Target Rate From (%)"].iloc[0])
    ceiling = float(band["Target Rate To (%)"].iloc[0])

    dates = list(panel.index)
    turns = {}
    for quarter_end in QUARTER_END_DATES:
        if quarter_end not in panel.index:
            raise ValueError(f"{quarter_end} is not a business day in the panel")
        position = dates.index(quarter_end)
        prior = panel[SECURED_RATE].iloc[position - LOOKBACK_DAYS:position]
        turns[quarter_end] = (float(panel.loc[quarter_end, SECURED_RATE]) - float(prior.median())) * 100
    largest_turn = max(turns, key=lambda key: turns[key])

    secured = rates.loc[rates["Rate Type"].astype(str).eq(SECURED_RATE)].copy()
    # "tail" would shadow DataFrame.tail, so the column is named explicitly.
    secured["tail_bp"] = (secured["99th Percentile (%)"] - secured["Rate (%)"]) * 100
    wide = secured.loc[secured["tail_bp"].ge(TAIL_THRESHOLD_BP)].sort_values(
        ["tail_bp", "date"], ascending=[False, True]
    )

    curve = load_expansion_table("treasury_yield_curve").copy()
    row = curve.loc[curve.Date.astype(str).eq(YEAR_END_DATE)]
    if row.empty:
        raise ValueError("the yield curve has no row for the year-end date")
    one_month = float(pd.to_numeric(row["1 Mo"], errors="coerce").iloc[0])
    three_month = float(pd.to_numeric(row["3 Mo"], errors="coerce").iloc[0])
    year_end_secured = float(panel.loc[YEAR_END_DATE, SECURED_RATE])

    return (
        int(len(panel)),
        float(panel["TGCR"].median()),
        float(panel["BGCR"].median()),
        float(panel["SOFR"].median()),
        float(panel["OBFR"].median()),
        float(panel["EFFR"].median()),
        float(spread.median()),
        str(widest.index[0]),
        float(widest.iloc[0]),
        floor,
        ceiling,
        year_end_secured,
        (year_end_secured - floor) * 100,
        (float(panel.loc[YEAR_END_DATE, UNSECURED_RATE]) - floor) * 100,
        turns[QUARTER_END_DATES[0]],
        turns[QUARTER_END_DATES[1]],
        turns[QUARTER_END_DATES[2]],
        turns[QUARTER_END_DATES[3]],
        largest_turn,
        turns[largest_turn],
        int(len(wide)),
        int(wide.date.str.startswith(f"{YEAR}-12").sum()),
        str(wide.date.iloc[0]),
        float(wide["tail_bp"].iloc[0]),
        one_month,
        three_month,
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
    "validate_rate_hierarchy": turn_validator(validate_turn_1),
    "validate_target_band_placement": turn_validator(validate_turn_2),
    "validate_quarter_end_pressure": turn_validator(validate_turn_3),
    "validate_tail_and_front_curve": turn_validator(validate_turn_4),
}
