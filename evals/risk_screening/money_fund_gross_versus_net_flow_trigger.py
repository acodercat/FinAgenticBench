"""Compare a net-redemption proxy with gross money-fund flow activity.

The 2023 amendments to the money market fund rules require an institutional prime
or institutional tax exempt fund to impose a liquidity fee on a day when net
redemptions exceed 5 percent of net assets, unless the liquidity costs the fund
incurs are de minimis. The threshold is written on net flow, which is the
difference between what came in and what went out. Gross subscriptions plus
redemptions instead measure two-way transaction activity; they are not the
liquidity a portfolio had to raise.

The daily flow file reports 247,400 rows at the share class level across 1,053
classes, while net assets are reported once per series filing. A series files once
per period, and an amendment supersedes the filing it corrects, so the population
keeps the latest filing for each series period before the classes are summed:
2,419 series-days are covered by both an original N-MFP3 and its N-MFP3/A, and 226
of those pairs report different flows, so retaining both would enter the same day
twice with two different answers. Summing the classes of the retained filing on
each day gives 71,931 series-day observations over 316 series, a median of 2
classes each.

Dividing a daily flow by net assets needs care, because the asset figure is dated
the series report date at the end of the period while the flow is dated within it.
A fund that shrank sharply is measured against what was left. 492 series-days
report net assets below one million dollars, and 17 observations put a single day's
net flow beyond 100 percent of the fund's assets. Requiring at least a million
dollars leaves 71,439 observations, and even there the largest single-day net
redemption reaches -176.9093 percent of the reported base, so the ratio remains an
approximation rather than a measurement.

On that screened population the median absolute net balance is 0.3033 percent of
assets and median gross activity is 1.1841 percent. Their ratio has a median of
3.2406, computed on the 66,246 observations where net flow is not zero: 5,193 days,
7.2691 percent, show none, and 5,168 of those show no flow at all. Government funds
show a median 1.5743 percent gross-activity day against 0.8551 for prime.

Counted against 5 percent of the same period-end denominator, the two proxies do not
describe the same construct. Net redemptions exceed 5 percent on 1,788 series-days;
gross activity exceeds 5 percent on 15,783, a factor of 8.8272. Government funds
cross the gross line on 26.1362 percent of their days while prime funds cross the
net line on 0.8527 percent of theirs. Prime and the two tax-exempt categories,
which can contain covered funds, hold 18,895 of the screened days and 182 of the
1,788 proxy net crossings, 10.1790 percent. The file does not identify which series
are institutional.

The `money_fund_flow_trigger` pinned-reading comparison records sensitivity to retaining
share-class grain, dropping the analytic asset screen, using redemptions rather than
net flow, changing the percentage threshold, narrowing the category screen, and
retaining superseded filings instead of the latest one per series period. The query
fixes the reported convention, so these are robustness comparisons rather than
hidden answer paths.

Measurement boundary: flows are reported per share class while assets are reported
per series, and the available asset figure is a period-end snapshot rather than the
daily net asset base used by the rule.

Boundaries: the file carries no institutional or retail designation, so the
category screen is not a count or proven upper bound of funds subject to the rule,
and the fee is owed only when the estimated fee is not de minimis, which nothing
here observes. The threshold in the rule is evaluated by the fund on its own daily
asset base, which this file does not carry, so the ratios here are built on a period
end figure instead. The mandatory-fee compliance date was 2024-10-02 while the flow
file opens on 2024-06-03; the 1,136 series-days that precede it are 1.5902 percent
of the screened population and carry 3 of the 1,788 net crossings and none of the
182 in the fee-eligible categories, so the overlap does not move any figure
reported here.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and fund names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


ASSET_FLOOR = 1e6
TRIGGER_PERCENT = 5.0
EXTREME_PERCENT = 100.0
GOVERNMENT = "Government"
PRIME = "Prime"
FEE_CATEGORIES = ("Prime", "Other Tax Exempt", "Single State")


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["flow_rows", "share_classes", "series_day_observations", "series_count",
                "median_classes_per_series_day"]
TURN_2_NAMES = ["small_asset_days", "beyond_full_assets", "screened_observations",
                "largest_net_redemption_pct"]
TURN_3_NAMES = ["median_absolute_net_pct", "median_gross_pct", "median_gross_to_net_ratio",
                "government_median_gross_pct", "prime_median_gross_pct"]
TURN_4_NAMES = ["net_redemption_proxy_crossings", "gross_activity_over_5pct",
                "gross_to_net_proxy_crossing_ratio", "government_gross_share_pct",
                "prime_net_redemption_proxy_share_pct", "potentially_covered_category_days",
                "potentially_covered_category_proxy_crossings",
                "potentially_covered_share_of_proxy_crossings_pct"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many rows the daily flow file holds as an integer."),
    _v(TURN_1_NAMES[1], "Store how many distinct share classes appear as an integer."),
    _v(TURN_1_NAMES[2], "Store how many series-day observations the aggregation produces as an integer."),
    _v(TURN_1_NAMES[3], "Store how many distinct series they cover as an integer."),
    _v(TURN_1_NAMES[4], "Store the median number of share classes per series-day, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store how many series-days report net assets below one million dollars as an integer."),
    _v(TURN_2_NAMES[1], "Store how many observations put a day's net flow beyond 100 percent of assets as an integer."),
    _v(TURN_2_NAMES[2], "Store how many observations survive the minimum asset screen as an integer."),
    _v(TURN_2_NAMES[3], "Store the most negative net flow among them, in percent of assets, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the median absolute net flow as a percent of assets, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the median gross activity as a percent of assets, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the median ratio of gross to absolute net flow, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the government funds' median gross flow in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store the prime funds' median gross flow in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many series-days show proxy net redemptions beyond 5 percent of period-end assets as an integer."),
    _v(TURN_4_NAMES[1], "Store how many show gross activity beyond 5 percent of period-end assets as an integer."),
    _v(TURN_4_NAMES[2], "Store the second divided by the first, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the percent of government series-days with gross flow beyond 5 percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the percent of prime series-days with proxy net redemptions beyond 5 percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store how many screened series-days belong to categories that could contain covered funds as an integer."),
    _v(TURN_4_NAMES[6], "Store how many proxy net crossings those categories account for as an integer."),
    _v(TURN_4_NAMES[7], "Store that as a percent of all proxy net crossings, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 0, 0, 4,
            0, 0, 0, 4,
            4, 4, 4, 4, 4,
            0, 0, 4, 4, 4, 0, 0, 4]


@lru_cache(maxsize=1)
def _series_days():
    flows = load_expansion_table("sec_nmfp_flows").copy()
    for column in ("gross_subscriptions_usd", "gross_redemptions_usd"):
        flows[column] = pd.to_numeric(flows[column], errors="coerce")
    filings = load_expansion_table("sec_nmfp").copy()
    filings["assets"] = pd.to_numeric(filings.net_assets_usd, errors="coerce")
    # An amendment supersedes the filing it corrects, so a series reports once per
    # period: keep the latest filing date, and where an original and its amendment
    # carry the same date prefer the amendment, since an accession prefix is the
    # filing agent's CIK and does not order submissions in time. Without this a day
    # covered by both an original N-MFP3 and its N-MFP3/A enters the population
    # twice, and 226 of those pairs disagree about the flows themselves, so the
    # duplication is not even self-consistent.
    filings["is_amendment"] = filings.form.astype(str).str.endswith("/A")
    filings = filings.sort_values(
        ["series_id", "report_date", "filing_date", "is_amendment", "accession"]
    ).drop_duplicates(["series_id", "report_date"], keep="last")
    merged = flows.merge(
        filings[["accession", "series_id", "series_name", "fund_category", "assets"]],
        on="accession", how="inner",
    )
    merged["day"] = merged.flow_date.astype(str)
    grouped = merged.groupby(
        ["series_id", "series_name", "fund_category", "day"], as_index=False,
    ).agg(
        subscriptions=("gross_subscriptions_usd", "sum"),
        redemptions=("gross_redemptions_usd", "sum"),
        classes=("class_id", "nunique"),
        assets=("assets", "first"),
    )
    grouped = grouped.loc[grouped.assets.gt(0)].copy()
    grouped["net_pct"] = (grouped.subscriptions - grouped.redemptions) / grouped.assets * 100
    grouped["gross_pct"] = (grouped.subscriptions + grouped.redemptions) / grouped.assets * 100
    return flows, grouped


@lru_cache(maxsize=1)
def ground_truth():
    flows, raw = _series_days()
    screened = raw.loc[raw.assets.ge(ASSET_FLOOR)].copy()
    government = screened.loc[screened.fund_category.eq(GOVERNMENT)]
    prime = screened.loc[screened.fund_category.eq(PRIME)]
    eligible = screened.loc[screened.fund_category.isin(FEE_CATEGORIES)]

    net_crossings = int(screened.net_pct.lt(-TRIGGER_PERCENT).sum())
    gross_crossings = int(screened.gross_pct.gt(TRIGGER_PERCENT).sum())
    eligible_crossings = int(eligible.net_pct.lt(-TRIGGER_PERCENT).sum())

    return (
        int(len(flows)), int(flows.class_id.nunique()), int(len(raw)),
        int(raw.series_id.nunique()), float(raw.classes.median()),
        int(raw.assets.lt(ASSET_FLOOR).sum()),
        int(raw.net_pct.abs().gt(EXTREME_PERCENT).sum()),
        int(len(screened)), float(screened.net_pct.min()),
        float(screened.net_pct.abs().median()), float(screened.gross_pct.median()),
        float((screened.gross_pct / screened.net_pct.abs().replace(0, pd.NA)).median()),
        float(government.gross_pct.median()), float(prime.gross_pct.median()),
        net_crossings, gross_crossings, gross_crossings / net_crossings,
        float(government.gross_pct.gt(TRIGGER_PERCENT).mean()) * 100,
        float(prime.net_pct.lt(-TRIGGER_PERCENT).mean()) * 100,
        int(len(eligible)), eligible_crossings, eligible_crossings / net_crossings * 100,
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
    "validate_flow_grain": turn_validator(validate_turn_1),
    "validate_denominator_screen": turn_validator(validate_turn_2),
    "validate_flow_distribution": turn_validator(validate_turn_3),
    "validate_trigger_reach": turn_validator(validate_turn_4),
}
