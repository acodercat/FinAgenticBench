"""Separate the Form 13F cover-page scale change from the filer-level holding unit.

Two unit conventions sit in the same 13F data and are routinely conflated. The
cover-page value total moved to whole dollars for filings made from 2023-01-03:
across original holdings reports the median raw cover total per holding row rises from
4,045.1296 at 2022-09-30 to 2,566,876.9847 at 2022-12-31, a factor of 634.5599, so
summing the field across that boundary produces a 155.5334-fold step that is a units
artifact. The holding rows carry a separate, per-filer unit that the governed
tables infer: at 2022-12-31, 5,758 filings report holdings in dollars and 1,308
in thousands, while the cover page reads as dollar scale for 5,799 and thousands
scale for 1,267. The two classifications agree on 96.5610 percent of filings but
disagree on 243, so one cannot be used as per-filing metadata for the other.

Reading the already-normalized holdings aggregation instead gives 75.484185
trillion USD over 36,120 issuer rows at 2022-12-31, led by Apple Inc. at
1.464846 trillion USD across 4,569 filing accessions. Dividing that USD total by
the raw, mixed-scale cover-page sum produces the number 2.1700, but not a
unit-consistent ratio. Carrying that issuer into the
registrant universe, its 10,535,889,677 reported shares stand against
15,821,946,000 shares outstanding as of 2023-01-20 from the latest count filed on
or before 2023-02-15, a 66.5904 percent reported-shares quotient. It is not total
institutional or beneficial ownership: the reporting population, security scope,
measurement dates, and shared-discretion reporting all differ. 974 of the 7,066
filings include other managers' holdings, but that count does not quantify duplicate
shares.

The `thirteen_f_unit_break` pinned-reading comparison records sensitivity to including
amendments, rescaling the cover page with the holding-row convention, changing the
cover heuristic, ranking by shares, and selecting a different outstanding-share
fact. The query fixes the reported convention, so these are robustness comparisons
rather than hidden answer paths.

The counts are of filings, not of filers. Three managers submit a duplicate qualifying
report with identical totals -- CIK 1767435 at 2022-09-30, CIK 93751 and CIK 1016021 at
2022-12-31 -- so collapsing to distinct filers reports 6,744 and 7,064 instead of 6,745
and 7,066. The medians and sums are taken over the same per-filing population.

Two source fields have to agree before a filing counts as an original holdings
report. One restatement per period carries submission type 13F-HR while its own
amendment flag reads Y, so filtering on the submission type alone admits it and
reports 6,746 and 7,067. Excluding it is what the query's "every amendment" asks
for, and it moves the two medians, both ratios, four of the six classification
counts and the other-managers count.

Two boundaries are load bearing. The cover-page scale is a per-filing property,
not a rule that holds without exception: 8.54 percent of filings for periods
from 2022Q4 still read as thousands scale and 9.29 percent of earlier filings
read as dollar scale, so the query classifies each filing rather than assuming
the period. The 100,000 reported-units-per-row cut is stated in the query because
nothing in the data labels each cover-page value's scale. It is an analytic
heuristic, not a consequence of the statutory $100 million manager threshold.
Sweeping it from 20,000 to 500,000 puts the disagreement count at
374, 263, 243, 250 and 416, so 100,000 sits at the minimum, the point where the
cover-page reading agrees most closely with the independently inferred
holding-row unit. The answer is fixed by the query either way.

Fragility: the leading issuer is 2.4829 percent ahead of the runner-up in
reported value, 1.464846 against 1.428476 trillion USD, so the top of the
ranking is a two-name race rather than a wide margin; the frozen artifact fixes
it.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, identifiers and names match exactly.
"""

from functools import lru_cache
import re

import pandas as pd
from cave_agent import Variable

from core.data import load_companies, load_facts, load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


PRE_BREAK_PERIOD = "2022-09-30"
POST_BREAK_PERIOD = "2022-12-31"
COVER_SCALE_CUT = 100_000.0
SHARES_CUTOFF_DATE = "2023-02-15"
SHARES_CONCEPT = "EntityCommonStockSharesOutstanding"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "pre_break_filing_count", "post_break_filing_count",
    "pre_break_median_cover_value_per_holding_reported",
    "post_break_median_cover_value_per_holding_reported",
    "median_value_per_holding_ratio", "cover_total_sum_ratio",
]
TURN_2_NAMES = [
    "post_break_dollar_holding_convention_count", "post_break_thousand_holding_convention_count",
    "post_break_dollar_cover_scale_count", "post_break_thousand_cover_scale_count",
    "convention_disagreement_filing_count", "dollar_holdings_thousand_cover_count",
]
TURN_3_NAMES = [
    "holdings_issuer_row_count", "holdings_total_usd_trillions", "leading_issuer_cusip",
    "leading_issuer_name", "leading_issuer_value_usd_trillions", "leading_issuer_filer_count",
    "holdings_usd_to_raw_cover_quotient",
]
TURN_4_NAMES = [
    "leading_issuer_ticker", "leading_issuer_reported_shares", "leading_issuer_shares_outstanding",
    "shares_outstanding_end_date", "reported_13f_shares_to_outstanding_pct",
    "filings_including_other_managers", "largest_included_manager_count",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the number of qualifying filings for the earlier period as an integer. Count "
       "submitted filings, not distinct filers: a manager that submits two qualifying 13F-HR "
       "reports for one period contributes two."),
    _v(TURN_1_NAMES[1],
       "Store the number of qualifying filings for the later period as an integer, on the same "
       "per-filing basis."),
    _v(TURN_1_NAMES[2], "Store the earlier period's median raw cover-page value per holding row in reported units, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store the later period's median raw cover-page value per holding row in reported units, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the later median divided by the earlier median, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store the later period's summed cover-page totals divided by the earlier period's, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store how many later-period filings report holding rows in dollars as an integer."),
    _v(TURN_2_NAMES[1], "Store how many report holding rows in thousands as an integer."),
    _v(TURN_2_NAMES[2], "Store how many later-period filings read as dollar scale on the cover page as an integer."),
    _v(TURN_2_NAMES[3], "Store how many read as thousands scale on the cover page as an integer."),
    _v(TURN_2_NAMES[4], "Store how many later-period filings the two classifications disagree on as an integer."),
    _v(TURN_2_NAMES[5], "Store how many report holding rows in dollars while reading as thousands scale on the cover page as an integer."),
    _v(TURN_3_NAMES[0], "Store the number of issuer rows in the later-period holdings aggregation as an integer."),
    _v(TURN_3_NAMES[1], "Store the aggregation's total reported value in USD trillions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[2], "Store the leading issuer's CUSIP as a string."),
    _v(TURN_3_NAMES[3], "Store the leading issuer's name verbatim from the aggregation as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_3_NAMES[4], "Store the leading issuer's reported value in USD trillions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[5], "Store how many filers report the leading issuer as an integer."),
    _v(TURN_3_NAMES[6], "Store the numerical quotient of the USD aggregation total divided by the later period's raw mixed-scale cover-page sum, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store the leading issuer's ticker verbatim from the registrant universe as a string."),
    _v(TURN_4_NAMES[1], "Store the shares reported for the leading issuer in the holdings aggregation as an integer."),
    _v(TURN_4_NAMES[2], "Store the registrant's shares outstanding from the selected count as an integer."),
    _v(TURN_4_NAMES[3], "Store the as-of date of that share count as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[4], "Store aggregated reported 13F shares as a percent of the selected shares-outstanding count, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store how many later-period filings include other managers' holdings as an integer."),
    _v(TURN_4_NAMES[6], "Store the largest number of other managers included on a single later-period filing as an integer."),
]

DECIMALS = [
    0, 0, 4, 4, 4, 4,
    0, 0, 0, 0, 0, 0,
    0, 6, None, None, 6, 0, 4,
    None, 0, 0, None, 4, 0, 0,
]


def _registry_key(value):
    return re.sub(r"[^a-z0-9 ]", "", str(value).casefold()).strip()


def _holdings_reports():
    filings = load_expansion_table("sec_13f_filings").copy()
    filings["period"] = filings.periodofreport.astype(str)
    for column in ("tablevaluetotal", "tableentrytotal", "otherincludedmanagerscount"):
        filings[column] = pd.to_numeric(filings[column], errors="coerce")
    # The submission type and the self-reported amendment flag disagree on a
    # handful of filings: a restatement can be submitted as 13F-HR rather than
    # 13F-HR/A, so the type alone does not exclude every amendment. Both
    # conditions are required. The flag is blank on about half the rows, so the
    # test is "not marked as an amendment", not "marked as an original".
    reports = filings.loc[
        filings.submissiontype.astype(str).eq("13F-HR")
        & filings.isamendment.astype(str).str.upper().ne("Y")
    ].copy()
    reports = reports.loc[reports.tablevaluetotal.gt(0) & reports.tableentrytotal.gt(0)].copy()
    reports["value_per_holding"] = reports.tablevaluetotal / reports.tableentrytotal
    return reports


@lru_cache(maxsize=1)
def ground_truth():
    reports = _holdings_reports()
    pre = reports.loc[reports.period.eq(PRE_BREAK_PERIOD)]
    post = reports.loc[reports.period.eq(POST_BREAK_PERIOD)].copy()
    if pre.empty or post.empty:
        raise ValueError("a bracketing period is missing from the filing table")
    pre_median = float(pre.value_per_holding.median())
    post_median = float(post.value_per_holding.median())
    pre_total = float(pre.tablevaluetotal.sum())
    post_total = float(post.tablevaluetotal.sum())

    post["cover_scale"] = post.value_per_holding.lt(COVER_SCALE_CUT).map(
        {True: "thousands", False: "dollars"}
    )
    holding_convention = post.value_reporting_convention.astype(str)
    dollar_holdings = holding_convention.eq("dollars")
    thousand_holdings = holding_convention.eq("thousands_of_dollars")
    dollar_cover = post.cover_scale.eq("dollars")
    disagreement = (dollar_holdings & ~dollar_cover) | (thousand_holdings & dollar_cover)

    holdings = load_expansion_table("sec_13f_holdings").copy()
    holdings = holdings.loc[holdings.period_of_report.astype(str).eq(POST_BREAK_PERIOD)].copy()
    ranked = holdings.sort_values(["value_usd_total", "cusip"], ascending=[False, True]).reset_index(drop=True)
    leader = ranked.iloc[0]

    companies = load_companies().copy()
    companies["registry_key"] = companies.name.map(_registry_key)
    match = companies.loc[companies.registry_key.eq(_registry_key(leader.issuer_name))]
    if len(match) != 1:
        raise ValueError(f"the leading issuer does not match exactly one registrant: {len(match)}")
    registrant = match.iloc[0]

    facts = load_facts().copy()
    counts = facts.loc[
        facts.cik.astype(str).eq(str(registrant.cik))
        & facts.concept.astype(str).eq(SHARES_CONCEPT)
        & facts.filed_date.astype(str).le(SHARES_CUTOFF_DATE)
    ].sort_values(["filed_date", "end_date"])
    if counts.empty:
        raise ValueError("no share count is available for the registrant by the cutoff")
    count = counts.iloc[-1]
    shares_outstanding = float(count.value)
    reported_shares = float(leader.share_amount_total)

    included = post.otherincludedmanagerscount.fillna(0)
    return (
        int(len(pre)), int(len(post)), pre_median, post_median,
        post_median / pre_median, post_total / pre_total,
        int(dollar_holdings.sum()), int(thousand_holdings.sum()),
        int(dollar_cover.sum()), int((~dollar_cover).sum()),
        int(disagreement.sum()), int((dollar_holdings & ~dollar_cover).sum()),
        int(len(holdings)), float(holdings.value_usd_total.sum()) / 1e12,
        str(leader.cusip), str(leader.issuer_name), float(leader.value_usd_total) / 1e12,
        int(leader.filer_count), float(holdings.value_usd_total.sum()) / post_total,
        str(registrant.ticker), int(reported_shares), int(shares_outstanding),
        str(count.end_date), reported_shares / shares_outstanding * 100,
        int(included.gt(0).sum()), int(included.max()),
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
    "validate_cover_page_break": turn_validator(validate_turn_1),
    "validate_convention_crosstab": turn_validator(validate_turn_2),
    "validate_normalized_aggregation": turn_validator(validate_turn_3),
    "validate_ownership_bridge": turn_validator(validate_turn_4),
}
