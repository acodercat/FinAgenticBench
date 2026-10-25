"""Price a real curve move against the bucket exposures bond funds report.

Registered funds report the dollar value of a basis point separately for the
three-month, one-year, five-year, ten-year and thirty-year points, so a portfolio
is described by where its rate risk sits and not only by how much of it there is.
At the 2025-06-30 report date 2,052 filings report in US dollars, of which 1,132
carry positive total exposure. Dividing that total by net assets and by one basis
point gives an implied duration whose median is 2.6672 years and whose ninetieth
percentile is 7.9217. The longest runs to 27.6627 years, a zero coupon Treasury
index fund holding all of its exposure at ten years and beyond.

Over the following half year the curve did not move in parallel. Between
2025-06-30 and 2025-12-31 the three-month point fell 74.0000 basis points and the
one-year point 48.0000, while the five- and ten-year points fell 6.0000 each and
the thirty-year point rose 6.0000. Applying each fund's bucket exposures to the
matching change gives an estimated mark of 8,329.0658 million USD across the
population, a median of 612,870.50 per fund and a median of 0.3405 percent of net
assets, with outcomes running from -1.2820 to 5.1937 percent. The floor of that
range belongs to the 27.6627 year fund: the longest duration in the population
produces its worst outcome, because the only point that rose was the one it sits on.

The two largest funds by net assets show why duration alone does not order the
outcomes. The larger carries 4.4291 years of duration with 61.2416 percent of its
exposure at the five-year point and gains 0.6571 percent of net assets; the second
carries 6.2583 years, nearly two years more, but holds 70.0742 percent of its
exposure at ten years and beyond, where yields fell least and then rose, and gains
0.3093 percent. A ranking by duration and a ranking by outcome disagree because
the move was concentrated at the front of the curve.

Measured rejected alternatives include mixing
non-USD exposures into a USD scenario, substituting one parallel shift for the five
observed tenor moves, omitting the basis-point factor in duration, using another
scenario endpoint, and retaining filings without positive total exposure. The
query pins each of these calculation choices, so the alternatives are baseline
instruction-following probes rather than a live hidden-method trap. This is a hard
baseline case.

Interpretation used by the case: Form N-PORT DV01 is the reported portfolio-value
change associated with a one-basis-point rate move at each tenor; duration
normalizes the summed dollar sensitivity by net assets and one basis point. A
negative bucket value can represent a short or hedged exposure rather than an
error.

Boundaries: bucket exposures are a first-order sensitivity reported at one date
and carry no convexity, so applying a 74 basis point move to them approximates a
mark rather than reproducing it, and nothing here accounts for spread, credit or
trading over the window. The population is filings rather than funds: 5 series
identifiers appear twice at the date and 192 filings carry none at all, so counts
here are of reports.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and fund names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2025-06-30"
SCENARIO_DATE = "2025-12-31"
CURRENCY = "USD"
BUCKETS = ("dv01_3month", "dv01_1year", "dv01_5year", "dv01_10year", "dv01_30year")
CURVE_COLUMNS = {
    "dv01_3month": "3 Mo", "dv01_1year": "1 Yr", "dv01_5year": "5 Yr",
    "dv01_10year": "10 Yr", "dv01_30year": "30 Yr",
}
LONG_BUCKETS = ("dv01_10year", "dv01_30year")


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "filing_count", "exposed_filing_count", "median_duration_years", "ninetieth_duration_years",
    "longest_duration_fund", "longest_duration_years", "longest_duration_long_end_share_pct",
]
TURN_2_NAMES = [
    "three_month_change_bp", "one_year_change_bp", "five_year_change_bp",
    "ten_year_change_bp", "thirty_year_change_bp",
]
TURN_3_NAMES = [
    "total_estimated_mark_usd_millions", "median_estimated_mark_usd",
    "median_mark_share_of_assets_pct", "minimum_mark_share_pct", "maximum_mark_share_pct",
]
TURN_4_NAMES = [
    "largest_fund_name", "largest_fund_duration_years", "largest_fund_five_year_share_pct",
    "largest_fund_mark_share_pct", "second_fund_name", "second_fund_duration_years",
    "second_fund_long_end_share_pct", "second_fund_mark_share_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many qualifying filings the report date carries as an integer."),
    _v(TURN_1_NAMES[1], "Store how many of them report positive total exposure as an integer."),
    _v(TURN_1_NAMES[2], "Store the median implied duration in years, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store the ninetieth percentile implied duration in years, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the name of the fund with the longest implied duration as text."),
    _v(TURN_1_NAMES[5], "Store that fund's implied duration in years, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store the share of its exposure at the ten- and thirty-year points in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the change in the three-month yield in basis points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the change in the one-year yield in basis points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the change in the five-year yield in basis points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the change in the ten-year yield in basis points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the change in the thirty-year yield in basis points, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the estimated mark summed across the population in USD millions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the median estimated mark per filing in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[2], "Store the median estimated mark as a percent of net assets, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the lowest such percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the highest such percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the name of the largest fund by net assets as text."),
    _v(TURN_4_NAMES[1], "Store its implied duration in years, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the share of its exposure at the five-year point in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store its estimated mark as a percent of net assets, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the name of the second largest fund by net assets as text."),
    _v(TURN_4_NAMES[5], "Store its implied duration in years, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store the combined share of its exposure at the ten- and thirty-year points in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[7], "Store its estimated mark as a percent of net assets, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 4, 4, None, 4, 4,
    4, 4, 4, 4, 4,
    4, 2, 4, 4, 4,
    None, 4, 4, 4, None, 4, 4, 4,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def _curve_changes():
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["day"] = curve.Date.astype(str)
    start = curve.loc[curve.day.eq(REPORT_DATE)]
    end = curve.loc[curve.day.eq(SCENARIO_DATE)]
    if start.empty or end.empty:
        raise ValueError("the curve is missing one of the scenario dates")
    changes = {}
    for bucket, column in CURVE_COLUMNS.items():
        first = float(pd.to_numeric(start[column], errors="coerce").iloc[0])
        last = float(pd.to_numeric(end[column], errors="coerce").iloc[0])
        changes[bucket] = (last - first) * 100
    return changes


@lru_cache(maxsize=1)
def _population():
    frame = load_expansion_table("sec_nport_interest_rate_risk").copy()
    for column in (*BUCKETS, "net_assets_usd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    filings = frame.loc[
        frame.report_date.astype(str).eq(REPORT_DATE)
        & frame.currency_code.astype(str).eq(CURRENCY)
        & frame.net_assets_usd.gt(0)
    ].copy()
    filings["total_dv01"] = filings[list(BUCKETS)].sum(axis=1)
    exposed = filings.loc[filings.total_dv01.gt(0)].copy()
    exposed["duration"] = exposed.total_dv01 / (exposed.net_assets_usd * 1e-4)
    changes = _curve_changes()
    exposed["mark"] = -sum(exposed[bucket] * changes[bucket] for bucket in BUCKETS)
    exposed["mark_share"] = exposed.mark / exposed.net_assets_usd * 100
    exposed["five_year_share"] = exposed.dv01_5year / exposed.total_dv01 * 100
    exposed["long_share"] = exposed[list(LONG_BUCKETS)].sum(axis=1) / exposed.total_dv01 * 100
    return len(filings), exposed


@lru_cache(maxsize=1)
def ground_truth():
    filing_count, exposed = _population()
    changes = _curve_changes()
    longest = exposed.sort_values(["duration", "accession"], ascending=[False, True]).iloc[0]
    by_assets = exposed.sort_values(["net_assets_usd", "accession"], ascending=[False, True]).reset_index(drop=True)
    leader, second = by_assets.iloc[0], by_assets.iloc[1]

    return (
        int(filing_count), int(len(exposed)),
        float(exposed.duration.median()), float(exposed.duration.quantile(0.9)),
        str(longest.series_name), float(longest.duration), float(longest.long_share),
        changes["dv01_3month"], changes["dv01_1year"], changes["dv01_5year"],
        changes["dv01_10year"], changes["dv01_30year"],
        float(exposed.mark.sum()) / 1e6, float(exposed.mark.median()),
        float(exposed.mark_share.median()), float(exposed.mark_share.min()), float(exposed.mark_share.max()),
        str(leader.series_name), float(leader.duration), float(leader.five_year_share), float(leader.mark_share),
        str(second.series_name), float(second.duration), float(second.long_share), float(second.mark_share),
    )


NAME_OUTPUTS = ("longest_duration_fund", "largest_fund_name", "second_fund_name")


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
    "validate_exposure_cross_section": turn_validator(validate_turn_1),
    "validate_curve_move": turn_validator(validate_turn_2),
    "validate_scenario_mark": turn_validator(validate_turn_3),
    "validate_duration_versus_placement": turn_validator(validate_turn_4),
}
