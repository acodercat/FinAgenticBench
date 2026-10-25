"""Read a money market fund against the rule that governs it.

Rule 2a-7 caps two different maturity measures. Weighted average maturity may not
exceed 60 days and weighted average life may not exceed 120, and the two differ
because only the first may be computed to a floating security's next interest rate
reset; the second must run to final maturity, though a demand feature may be used
for either. At the 2025-06-30 report date, counting each fund once by keeping the
latest filing for its series identifier, 309 funds report. Their median weighted
average maturity is 31 days against a maximum of 59, and their median life 58 days
against a maximum of 116. No fund exceeds either cap, and none in this snapshot
reports a life shorter than its maturity. Keeping every filing
instead would count 322, because 13 amendments restate a filing already present and
sometimes change the figures.

The distance between the two measures describes a portfolio rather than measuring
slack. Its median is 30 days and its maximum 102. Government funds run the widest
gap, a median 49 days across 227 funds; prime funds run 19; and the 25 funds in the
broad tax exempt category run 0. Those category-level summaries establish a
difference in reported maturity profiles, but these two aggregate sources do not
identify the securities or features that cause it. Funds with the same weighted
average maturity can report different weighted average lives, so the maturity
measure alone does not determine the life measure.

Liquidity is reported daily and has to be screened before it is read. Across the
whole liquidity file 449 rows, 0.5950 percent, report zero weekly liquid assets in
both dollars and percent. The case treats them as an anomaly-screening population:
81 of the 317 series carry at least one, but for those series a zero is a median
0.8734 percent of their own observations and their median reading when positive is
84.7400 percent, and only 1 series reports zero throughout. A fund that genuinely
held nothing weekly liquid would not usually report eighty-five percent on the rest
of its days, but that pattern does not prove why a filer reported zero. All 309
funds carry a liquidity row dated the report date, 2 of them are
these zeros, and the 307 that remain hold a median 84.0000 percent of assets in
weekly liquid form and 71.8200 percent in daily liquid form.

Counting apparent breaches of the 25 percent daily and 50 percent weekly minimums
then takes two corrections the data does not supply. Taken raw, 18 funds sit below
the daily minimum and 2 below the weekly one. Dropping the zero rows leaves 16 and
0, so no fund is below the weekly minimum at all. Of the 16, 14 are tax exempt
funds, which the rule does not subject to the daily requirement, leaving 2 that it
does. Both of those report 100.0000 percent weekly liquid assets. The daily and
weekly measures cover different qualifying asset sets, so the readings are not
internally contradictory; what these two funds actually hold is not shown here.
Even for them the thresholds
condition what a fund may acquire rather than setting a level it must hold, so an
observation below one is not by itself a violation.

Measured rejected alternatives include
keeping every accession, retaining zero-liquidity rows, narrowing the tax-exempt
category set, using the pre-2024 daily threshold and substituting means for medians.
The query pins each calculation choice, so these are baseline instruction-following
probes rather than a live hidden-method trap. This is a hard baseline case.

Rule 2a-7 context used by the case: weighted average maturity and weighted average
life are distinct measures; tax-exempt money market funds are not subject to the
daily liquid asset acquisition test; and the daily and weekly minimums condition
acquisitions rather than requiring a fund to remain above the percentage at every
instant.

Boundaries: this is one report date and a fund's position moves daily. The zero rows
are treated as missing on the evidence of the same funds' other observations, which
is an inference rather than a statement from the filer. Whether a fund complied
depends on what it acquired and when, which these fields cannot show. The
single-state funds are read as tax exempt on the strength of their category name.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2025-06-30"
MATURITY_CAP = 60
LIFE_CAP = 120
DAILY_MINIMUM = 0.25
WEEKLY_MINIMUM = 0.50
TAX_EXEMPT_CATEGORIES = ("Other Tax Exempt", "Single State")
GOVERNMENT_CATEGORY = "Government"
PRIME_CATEGORY = "Prime"
BROAD_TAX_EXEMPT_CATEGORY = "Other Tax Exempt"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["filing_count", "median_maturity_days", "maximum_maturity_days", "over_maturity_cap",
                "median_life_days", "maximum_life_days", "over_life_cap", "minimum_gap_days"]
TURN_2_NAMES = ["median_gap_days", "maximum_gap_days", "government_filings", "government_gap_days",
                "prime_gap_days", "tax_exempt_filings", "tax_exempt_gap_days"]
TURN_3_NAMES = ["zero_liquidity_rows", "zero_liquidity_share_pct", "matched_filings",
                "zero_rows_at_date", "screened_filings", "median_weekly_liquid_pct"]
TURN_4_NAMES = ["median_daily_liquid_pct", "below_daily_raw", "below_daily_screened",
                "below_daily_tax_exempt", "below_daily_subject", "subject_minimum_weekly_pct",
                "below_weekly_screened"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many filings the report date carries as an integer."),
    _v(TURN_1_NAMES[1], "Store the median weighted average maturity in days, rounded to 4 decimals."),
    _v(TURN_1_NAMES[2], "Store the largest weighted average maturity in days, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store how many filings exceed the 60-day cap as an integer."),
    _v(TURN_1_NAMES[4], "Store the median weighted average life in days, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the largest weighted average life in days, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store how many filings exceed the 120-day cap as an integer."),
    _v(TURN_1_NAMES[7], "Store the smallest life-minus-maturity difference in days, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the median life-minus-maturity difference in days, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store its maximum in days, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store how many filings the government category holds as an integer."),
    _v(TURN_2_NAMES[3], "Store their median difference in days, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the prime funds' median difference in days, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store how many filings the broad tax exempt category holds as an integer."),
    _v(TURN_2_NAMES[6], "Store their median difference in days, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store how many rows in the whole liquidity file report zero weekly liquid assets as an integer."),
    _v(TURN_3_NAMES[1], "Store that as a percent of the file, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store how many report-date filings carry a same-date liquidity row as an integer."),
    _v(TURN_3_NAMES[3], "Store how many of those rows are the zero rows as an integer."),
    _v(TURN_3_NAMES[4], "Store how many filings survive the screen as an integer."),
    _v(TURN_3_NAMES[5], "Store their median weekly liquid assets in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the screened median daily liquid assets in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store how many unscreened filings sit below the 25 percent daily minimum as an integer."),
    _v(TURN_4_NAMES[2], "Store how many screened filings do as an integer."),
    _v(TURN_4_NAMES[3], "Store how many of those are tax exempt funds as an integer."),
    _v(TURN_4_NAMES[4], "Store how many are subject to the daily minimum as an integer."),
    _v(TURN_4_NAMES[5], "Store the lowest weekly liquid assets among those, in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store how many screened filings sit below the 50 percent weekly minimum as an integer."),
]

DECIMALS = [0, 4, 4, 0, 4, 4, 0, 4,
            4, 4, 0, 4, 4, 0, 4,
            0, 4, 0, 0, 0, 4,
            4, 0, 0, 0, 0, 4, 0]


@lru_cache(maxsize=1)
def _liquidity():
    frame = load_expansion_table("sec_nmfp_liquidity").copy()
    for column in ("daily_liquid_assets_usd", "weekly_liquid_assets_usd",
                   "daily_liquid_assets_fraction", "weekly_liquid_assets_fraction"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["day"] = frame.liquidity_date.astype(str)
    return frame


@lru_cache(maxsize=1)
def _filings():
    frame = load_expansion_table("sec_nmfp").copy()
    for column in ("weighted_average_maturity_days", "weighted_average_life_days", "net_assets_usd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    kept = frame.loc[frame.report_date.astype(str).eq(REPORT_DATE) & frame.series_id.notna()].copy()
    kept = kept.sort_values(["series_id", "filing_date", "accession"])
    kept = kept.drop_duplicates("series_id", keep="last")
    kept["gap"] = kept.weighted_average_life_days - kept.weighted_average_maturity_days
    kept["category"] = kept.fund_category.astype(str)
    return kept


@lru_cache(maxsize=1)
def _paired():
    liquidity = _liquidity()
    same_day = liquidity.loc[liquidity.day.eq(REPORT_DATE)]
    return _filings().merge(same_day, on="accession", how="inner")


@lru_cache(maxsize=1)
def ground_truth():
    filings = _filings()
    liquidity = _liquidity()
    paired = _paired()
    screened = paired.loc[paired.weekly_liquid_assets_usd.gt(0)].copy()
    screened["tax_exempt"] = screened.category.isin(TAX_EXEMPT_CATEGORIES)

    def gap_of(category):
        rows = filings.loc[filings.category.eq(category)]
        return int(len(rows)), float(rows.gap.median())

    government_count, government_gap = gap_of(GOVERNMENT_CATEGORY)
    _, prime_gap = gap_of(PRIME_CATEGORY)
    tax_exempt_count, tax_exempt_gap = gap_of(BROAD_TAX_EXEMPT_CATEGORY)

    below_daily = screened.daily_liquid_assets_fraction.lt(DAILY_MINIMUM)
    subject = screened.loc[below_daily & ~screened.tax_exempt]

    return (
        int(len(filings)), float(filings.weighted_average_maturity_days.median()),
        float(filings.weighted_average_maturity_days.max()),
        int(filings.weighted_average_maturity_days.gt(MATURITY_CAP).sum()),
        float(filings.weighted_average_life_days.median()),
        float(filings.weighted_average_life_days.max()),
        int(filings.weighted_average_life_days.gt(LIFE_CAP).sum()),
        float(filings.gap.min()),
        float(filings.gap.median()), float(filings.gap.max()),
        government_count, government_gap, prime_gap, tax_exempt_count, tax_exempt_gap,
        int(liquidity.weekly_liquid_assets_usd.eq(0).sum()),
        float(liquidity.weekly_liquid_assets_usd.eq(0).mean()) * 100,
        int(len(paired)), int(paired.weekly_liquid_assets_usd.eq(0).sum()), int(len(screened)),
        float(screened.weekly_liquid_assets_fraction.median()) * 100,
        float(screened.daily_liquid_assets_fraction.median()) * 100,
        int(paired.daily_liquid_assets_fraction.lt(DAILY_MINIMUM).sum()),
        int(below_daily.sum()), int((below_daily & screened.tax_exempt).sum()), int(len(subject)),
        float(subject.weekly_liquid_assets_fraction.min()) * 100,
        int(screened.weekly_liquid_assets_fraction.lt(WEEKLY_MINIMUM).sum()),
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
    "validate_maturity_caps": turn_validator(validate_turn_1),
    "validate_gap_composition": turn_validator(validate_turn_2),
    "validate_liquidity_screen": turn_validator(validate_turn_3),
    "validate_minimum_tests": turn_validator(validate_turn_4),
}
