"""Two ways to misread a fund's redemption record, and a market that settles both.

Form N-PORT gives each filing three months of share sales, share redemptions and
investment gains. Neither the months nor the redemptions read the way a frame
suggests, and this case is built on both.

The first trap is the sign, and it fires in the first turn. A redemption is an
amount paid out, but the filer chooses how to sign the field: 183,280 of the
874,476 redemption observations on this file are negative, 66,327 of the 290,709
retained filings carry at least one negative month, they come from 655 registrants,
and 4,416 series report both signs at different times. Adding the column up
therefore returns 35,417.3573 billion dollars against the 53,220.9308 actually
redeemed, understating the total by 33.4522 percent. At four requested decimals
the rounding-boundary tolerance is 0.00006 billion dollars, so the naive figure
cannot enter the accepted band.

The second trap is the calendar, and it fires in the second turn. The three
reported months are the first, second and third months of the reporting period,
so the third is the month the period ends in and the first is two months earlier.
Reading the first as the most recent inverts the order. On the lowest-return month
in the covered window the inverted reading returns 888.2351 billion dollars
redeemed against 1,552.7608, low by 42.7964 percent, and taking both wrong paths
at once returns 396.2804. The inversion is not a coverage artifact: both readings
draw on the same 11,597 filings and differ only in which of the three columns they
take.

The market settles the calendar without the query having to. Each reported month's
investment gain as a fraction of net assets carries a Spearman rank correlation of
0.6696 with the market return two months before the period end for the first
month, 0.7105 one month before for the second, and 0.6716 with the period-end
month for the third. Every off-diagonal entry of that table lies between -0.1227
and 0.0462. Tied values take their average rank, which the query states because the tie
handling moves the answer: minimum, maximum, first-seen and dense ranking return
0.6691, 0.6701, 0.6694 and 0.6653 against 0.6696, all outside a tolerance of
0.00006. Ties dominate one side of the pairing, since the market column carries
only 75 distinct values across 290,709 observations.

The rank statistic is used because Pearson is not robust here: 538 of
872,130 gain observations exceed half of net assets in absolute value, and leaving
them in collapses the diagonal from roughly 0.65 to 0.02, which would make the
diagnostic an artifact of a trimming threshold rather than a fact about the file.

The financial content is what the corrected record then shows. The lowest-return
month in the window is March 2020 at -13.1824 percent, ahead of the runner-up by
3.7683 percentage points, and it is the month of the dash for cash. Corrected, the
filings covering it record 1,552.7608 billion dollars redeemed against sales that
leave a net of -392.7647 billion; the sign-naive path reports a positive net
instead, which contradicts the episode. Within the period, one month carries a
median 0.4728 of the three-month redemption total, so quarterly aggregation halves
the visible stress; 2,431 filings redeemed more than a tenth of net assets in that
single month, and 503 of them -- 20.69 percent -- still show sales above
redemptions across the whole period, so a reader working at period grain sees an
inflow where a month of run-scale redemption happened.

A third rule earns its words too. Nineteen series moved their reporting cycle and
report the selected month in two periods at once; both filings state the same
month figure but rest on different period ends, so entering both would count
1,552.7608 billion dollars as 1,553.6928 and shift the net by 0.3656 billion,
against a tolerance of 0.00006. The series is counted once, keeping the filing
whose period ends earliest and so whose net assets are measured closest to the
month. Across the file 633 of 871,491 series-months are covered twice.

Two population rules earn their words in the query. 16,620 filings report no
series identifier, from 1,196 registrants of which only 10 also file with one, so
a grouping that keys on the identifier discards them by default; the query states
the exclusion instead of leaving it to that default. 15,673 of those filings do
carry a series name, so the identifier is what is absent rather than the fund. The
registrants visible in the excluded set are unit investment trusts and standalone
closed-end funds -- SPDR S&P 500 ETF Trust, General American Investors, Adams
Diversified Equity, AllianceBernstein Global High Income -- whose shares are not
redeemed for cash by the shareholder the way an open-end fund's are; the case
records that composition without asserting it of all 1,196. The exclusion is not
immaterial: those filings report a further 198.6938 billion dollars redeemed in
the selected month, 12.80 percent on top of the figure this case reports, so the
month total is the series-identified population's and not the whole file's.
And the surviving filing is taken whole: a column-wise retention carries a value
forward from the superseded filing wherever the successor leaves that column
empty, which spliced one filing here and moved the sold total by 0.0015 billion
dollars against a tolerance of 0.00006.

Sales are summed as reported. The sign of the sales field is not the filer's
convention the way the redemption field's is: 486 of the retained sales
observations are negative, in 465 filings, and 290 of the 293 series behind them
report positive sales in other filings; a negative month typically cancels the
one before it (5,233.44 sold, then -5,233.44). It reverses a sale, so taking its
magnitude would count the reversal as a further sale and move the sold total from
54,502.1314 to 54,512.7889 billion dollars. Of the other outputs only the
month's net flow reads it, and it moves by 0.0000025 billion, inside its tolerance.

Fragility of the anchors. The lowest-return month leads by 3.7683 percentage
points and the largest single filing, 69.0542 billion dollars, leads the runner-up
by 88.9458 percent, so neither selection is close. The retained-filing rule is not
idle either: 4,614 series-period pairs carry more than one filing and 88 of those
pairs disagree on a reported redemption.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; counts, the offset and the date match exactly.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


MONTHS = (1, 2, 3)
STRESS_SHARE = 0.10
OFFSETS = (0, 1, 2, 3)
BILLION = 1e9


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "retained_filing_count",
    "distinct_series_count",
    "total_gross_redeemed_usd_billions",
    "total_gross_sold_usd_billions",
]
TURN_2_NAMES = [
    "lowest_return_month_end_date",
    "lowest_return_month_return_pct",
    "filings_covering_that_month",
    "month_gross_redeemed_usd_billions",
]
TURN_3_NAMES = [
    "median_month_share_of_period",
    "filings_redeeming_over_one_tenth_count",
    "period_net_inflow_despite_stress_count",
    "largest_filing_month_redemption_usd_billions",
]
TURN_4_NAMES = [
    "first_reported_month_offset",
    "first_reported_month_rank_correlation",
    "third_reported_month_rank_correlation",
    "month_net_share_flow_usd_billions",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store how many filings the retention and completeness rules leave as an integer."),
    _v(TURN_1_NAMES[1],
       "Store how many distinct fund series those filings cover as an integer."),
    _v(TURN_1_NAMES[2],
       "Store the total amount redeemed across every month they report in billions of dollars, "
       "rounded to 4 decimals."),
    _v(TURN_1_NAMES[3],
       "Store the total amount sold across the same months in billions of dollars, rounded to 4 "
       "decimals."),
    _v(TURN_2_NAMES[0],
       "Store the date the selected month ends on as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1],
       "Store that month's compounded total market return in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2],
       "Store how many retained filings report a month falling in it as an integer."),
    _v(TURN_2_NAMES[3],
       "Store the total amount those filings redeemed in that month in billions of dollars, "
       "rounded to 4 decimals."),
    _v(TURN_3_NAMES[0],
       "Store the median share that month takes of a filing's three-month redemption total, "
       "rounded to 4 decimals."),
    _v(TURN_3_NAMES[1],
       "Store how many of those filings redeemed more than a tenth of period-end net assets in "
       "that month as an integer."),
    _v(TURN_3_NAMES[2],
       "Store how many of that group still sold more than they redeemed across the whole period "
       "as an integer."),
    _v(TURN_3_NAMES[3],
       "Store the largest amount any single filing redeemed in that month in billions of "
       "dollars, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0],
       "Store the whole number of months before the period end that maximises the first reported "
       "month's rank correlation as an integer."),
    _v(TURN_4_NAMES[1],
       "Store that maximised rank correlation, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2],
       "Store the third reported month's maximised rank correlation, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3],
       "Store sales less redemptions across the filings covering the selected month in billions "
       "of dollars, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 4, 4,
    None, 4, 0, 4,
    4, 0, 0, 4,
    0, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _filings():
    """One filing per series and period end, with every reported month present."""
    report = load_expansion_table("sec_nport")
    frame = report.loc[:, [
        "accession", "filing_date", "report_date", "series_id", "net_assets_usd",
        *[f"{field}_month_{month}_usd"
          for field in ("sales", "redemptions", "realized_gain", "unrealized_change")
          for month in MONTHS],
    ]].copy()
    del report
    frame["filed_on"] = pd.to_datetime(frame.filing_date)
    frame["period_end"] = pd.to_datetime(frame.report_date)
    # A filing without a series identifier cannot be placed on a series timeline at
    # all, and the registrants that file that way are not organised as series. The
    # query states the exclusion rather than leaving it to a grouping default.
    frame = frame.loc[frame.series_id.notna()]
    # An amendment supersedes the filing it corrects, so only the latest filing for
    # a series and period end is kept. The whole row is taken: a column-wise last
    # would carry a value forward from the superseded filing wherever the surviving
    # one leaves that column empty, splicing two filings into one.
    frame = (
        frame.sort_values(["filed_on", "accession"])
        .drop_duplicates(subset=["series_id", "period_end"], keep="last")
        .reset_index(drop=True)
    )
    columns = {}
    for month in MONTHS:
        columns[f"sold_{month}"] = pd.to_numeric(
            frame[f"sales_month_{month}_usd"], errors="coerce"
        )
        columns[f"redeemed_{month}"] = pd.to_numeric(
            frame[f"redemptions_month_{month}_usd"], errors="coerce"
        )
        columns[f"gain_{month}"] = pd.to_numeric(
            frame[f"realized_gain_month_{month}_usd"], errors="coerce"
        ) + pd.to_numeric(frame[f"unrealized_change_month_{month}_usd"], errors="coerce")
    columns["net_assets"] = pd.to_numeric(frame.net_assets_usd, errors="coerce")
    frame = frame.assign(**columns).dropna(subset=list(columns))
    frame = frame.loc[frame.net_assets.gt(0)].reset_index(drop=True)
    # The filer chooses the sign of the redemption field; an amount paid out is a
    # magnitude either way, so the reported figure is read as one. A sale is not
    # signed by convention: a negative one reverses an earlier month's, so it
    # stays as reported.
    for month in MONTHS:
        frame[f"redeemed_{month}"] = frame[f"redeemed_{month}"].abs()
    frame["period_month"] = frame.period_end.dt.to_period("M")
    return frame


@lru_cache(maxsize=1)
def _market_month_returns():
    """Daily total market return compounded inside each calendar month, in percent."""
    factors = load_ken_french_table("daily_factors").copy()
    factors["observed_on"] = pd.to_datetime(factors.date)
    factors["total"] = (
        pd.to_numeric(factors.mkt_rf_pct, errors="coerce")
        + pd.to_numeric(factors.rf_pct, errors="coerce")
    )
    daily = factors.dropna(subset=["observed_on", "total"]).set_index("observed_on").total
    return ((1.0 + daily / 100.0).resample("ME").prod() - 1.0) * 100.0


def _covered_months(filings):
    """The calendar months the reporting periods span, one per reported month slot."""
    months = set()
    for month in MONTHS:
        months |= set(filings.period_month - (3 - month))
    return sorted(months)


def _slot_for(filings, target):
    """Which reported month of each filing falls in the target calendar month."""
    slot = pd.Series(np.nan, index=filings.index)
    for month in MONTHS:
        slot[filings.period_month - (3 - month) == target] = month
    return slot


def _pick(frame, prefix, slot):
    return pd.Series(
        np.select(
            [slot.eq(month) for month in MONTHS],
            [frame[f"{prefix}_{month}"] for month in MONTHS],
            default=np.nan,
        ),
        index=frame.index,
    )


@lru_cache(maxsize=1)
def ground_truth():
    filings = _filings()
    returns = _market_month_returns()

    covered = [
        period for period in _covered_months(filings)
        if period.to_timestamp("M") in returns.index
    ]
    monthly = returns.reindex([period.to_timestamp("M") for period in covered])
    # The query breaks a tie by the earlier month; idxmin takes the first of a
    # sorted index, which is that month.
    selected_end = monthly.idxmin()
    selected = pd.Period(selected_end, "M")

    slot = _slot_for(filings, selected)
    covering = filings.loc[slot.notna()].copy()
    covering["reported_slot"] = slot[slot.notna()]
    # A series that moved its reporting cycle can report the same calendar month in
    # two periods, which would enter it twice. The filing whose period ends earliest
    # is kept, because its net assets are the ones measured closest to the month.
    covering = (
        covering.sort_values(["series_id", "period_end"])
        .drop_duplicates(subset=["series_id"], keep="first")
        .reset_index(drop=True)
    )
    covering_slot = covering.reported_slot
    covering["month_redeemed"] = _pick(covering, "redeemed", covering_slot)
    covering["month_sold"] = _pick(covering, "sold", covering_slot)
    covering["period_redeemed"] = sum(covering[f"redeemed_{m}"] for m in MONTHS)
    covering["period_sold"] = sum(covering[f"sold_{m}"] for m in MONTHS)

    with_redemptions = covering.loc[covering.period_redeemed.gt(0)]
    share = with_redemptions.month_redeemed / with_redemptions.period_redeemed
    stressed = covering.month_redeemed.div(covering.net_assets).gt(STRESS_SHARE)
    net_inflow = covering.period_sold.gt(covering.period_redeemed)

    # Only the amount is reported, never which filing paid it, so a tie at the top
    # would leave the answer unchanged and needs no rule. The margin is asserted in
    # the case-local test instead, where a collapse would be worth knowing about.
    ranked = covering.month_redeemed.sort_values(ascending=False)

    correlations = {}
    for month in MONTHS:
        rate = filings[f"gain_{month}"] / filings.net_assets
        for offset in OFFSETS:
            aligned = (filings.period_month - offset).dt.to_timestamp("M").map(returns)
            correlations[(month, offset)] = float(rate.corr(aligned, method="spearman"))
    best = {
        month: max(OFFSETS, key=lambda offset: correlations[(month, offset)])
        for month in MONTHS
    }

    return (
        int(len(filings)),
        int(filings.series_id.nunique()),
        float(sum(filings[f"redeemed_{m}"] for m in MONTHS).sum() / BILLION),
        float(sum(filings[f"sold_{m}"] for m in MONTHS).sum() / BILLION),
        str(selected_end.date()),
        float(monthly.min()),
        int(len(covering)),
        float(covering.month_redeemed.sum() / BILLION),
        float(share.median()),
        int(stressed.sum()),
        int((stressed & net_inflow).sum()),
        float(ranked.iloc[0] / BILLION),
        int(best[1]),
        correlations[(1, best[1])],
        correlations[(3, best[3])],
        float((covering.month_sold.sum() - covering.month_redeemed.sum()) / BILLION),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_retained_population(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_worst_month_redemptions(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_within_period_concentration(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_month_alignment(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_retained_population": turn_validator(validate_retained_population),
    "validate_worst_month_redemptions": turn_validator(validate_worst_month_redemptions),
    "validate_within_period_concentration": turn_validator(validate_within_period_concentration),
    "validate_month_alignment": turn_validator(validate_month_alignment),
}
