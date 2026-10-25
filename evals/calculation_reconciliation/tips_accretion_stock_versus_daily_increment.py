"""Reconcile the accrued inflation compensation on TIPS against the daily flow.

An inflation-protected Treasury owes more principal than the Treasury borrowed.
The principal accretes with the price index, and the accrued amount is debt that
was never issued.  At 2025-12-31 it is USD385.656277 billion against
USD2,133.881769 billion of TIPS outstanding -- 18.072992 percent of the principal
the Treasury owes on them.

Two publications record it differently.  The monthly statement carries the
accrued amount as a **stock** beside each security, and its aggregate closes
exactly: issued plus accrued plus redeemed equals outstanding on every one of the
84 month ends, the largest residual across all of them being 5e-10 of a million
dollars.  The daily statement carries the accretion as a **flow**, one
transaction line per business day, 1,757 of them, 323 of which are negative
because the index fell.

The two do not meet, and the reason is the distinction the case turns on.  When a
security matures, the accrued principal it carried leaves the stock; the flow
never gives it back.  Over the 83 months both files cover, 22 month ends lose a
security and USD204.679361 billion of accrued principal leaves with them.
Comparing the month-to-month change in the stock with the sum of that month's
daily flow leaves a mean absolute gap of USD2,512.193653 million and reverses the
sign in 7 months.  Taking out the accrued principal that left brings the mean
absolute gap to USD48.235140 million, a fifty-two-fold reduction, with the worst
month at USD271.452971 million.  July 2025 is the clearest case: the stock fell
by USD10,197.547 million while the flow added USD4,387 million, because one
security matured carrying USD14,438.975 million of accrued principal.

Rejected conventions.  The stock is read from the security-level detail rows, not
the published class totals, which appear in the same file and would double count.
A security-level outstanding balance is reported once per security while issued
and accrued amounts are reported on each issue row, so the aggregate closes even
though only 53 of the final month end's 134 rows carry an outstanding balance.
Redemptions are stored as negative numbers and enter the identity as they stand.
The residual gap that survives the maturity adjustment is a settlement-timing
difference between a daily transaction file and a month-end snapshot, and the
case reports it rather than forcing it to zero.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TIPS_CLASS = "Inflation-Protected Securities"
INCREMENT_TYPE = "Inflation-Protected Securities Increment"
MILLIONS_PER_BILLION = 1_000.0


variables = [
    Variable("tips_month_end_count", None,
             "Store how many month ends the monthly statement covers as an integer."),
    Variable("tips_issue_row_count", None,
             "Store how many security-level inflation-protected rows the final month end carries as an integer."),
    Variable("tips_cusip_count", None,
             "Store how many distinct inflation-protected securities the final month end describes as an integer."),
    Variable("tips_outstanding_usd_billions", None,
             "Store the final month end's total inflation-protected outstanding balance in USD billions, rounded to 6 decimals."),

    Variable("accretion_share_of_outstanding_pct", None,
             "Store the accrued inflation compensation as a percent of that outstanding balance, in percent, rounded to 6 decimals."),
    Variable("cumulative_issued_usd_billions", None,
             "Store the total amount issued on the final month end's inflation-protected rows in USD billions, rounded to 6 decimals."),
    Variable("month_ends_with_a_departing_security_count", None,
             "Store how many month ends lose a security present the month before as an integer."),
    Variable("departing_accretion_usd_billions", None,
             "Store the accrued compensation carried by the inflation-protected securities that leave the monthly statement, in USD billions, rounded to 6 decimals."),

    Variable("increment_day_count", None,
             "Store how many daily statement lines record an inflation increment as an integer."),
    Variable("negative_increment_day_count", None,
             "Store how many daily inflation-increment lines are negative as an integer."),
    Variable("dts_increment_total_usd_billions", None,
             "Store the total of the daily inflation-increment lines in USD billions, rounded to 6 decimals."),
    Variable("months_compared_count", None,
             "Store how many months carry both a stock change and a daily total as an integer."),

    Variable("opposite_direction_month_count", None,
             "Store how many compared months move in opposite directions as an integer."),
    Variable("unadjusted_mean_absolute_gap_usd_millions", None,
             "Store the mean absolute monthly gap between the accrued-balance change and the daily increment total, in USD millions, rounded to 6 decimals."),
    Variable("adjusted_mean_absolute_gap_usd_millions", None,
             "Store the same mean once the departing securities' accrued compensation is removed from the daily total, in USD millions, rounded to 6 decimals."),
    Variable("largest_adjusted_gap_usd_millions", None,
             "Store the largest single month's absolute gap after that removal, in USD millions, rounded to 6 decimals."),
]

DECIMALS = [0, 0, 0, 6, 6, 6, 0, 6, 0, 0, 6, 0, 0, 6, 6, 6]


def _tips_detail():
    """Security-level TIPS rows from the monthly statement, totals excluded."""
    statement = load_expansion_table("treasury_marketable_securities")
    label = statement.security_identifier_or_total_label.astype(str)
    frame = pd.DataFrame({
        "security": label,
        "month": pd.to_datetime(statement.record_date, errors="coerce").dt.to_period("M"),
        "issued": pd.to_numeric(statement.issued_million_usd, errors="coerce"),
        "accrued": pd.to_numeric(statement.inflation_adjustment_million_usd, errors="coerce"),
        "redeemed": pd.to_numeric(statement.redeemed_million_usd, errors="coerce"),
        "outstanding": pd.to_numeric(statement.outstanding_million_usd, errors="coerce"),
        "cls": statement.security_class.astype(str),
    })
    del statement
    return frame.loc[
        frame.cls.eq(TIPS_CLASS) & ~frame.security.str.contains("otal", case=False, na=False)
    ]


def _daily_increments():
    """One transaction line per business day carrying the inflation increment."""
    daily = load_expansion_table("treasury_dts_public_debt")
    frame = pd.DataFrame({
        "day": pd.to_datetime(daily.record_date, errors="coerce"),
        "amount": pd.to_numeric(daily.transaction_today_amount_million_usd, errors="coerce"),
        "kind": daily.security_type.astype(str),
    })
    del daily
    return frame.loc[frame.kind.eq(INCREMENT_TYPE)]


@lru_cache(maxsize=1)
def ground_truth():
    detail = _tips_detail()
    final = detail.month.max()
    last = detail.loc[detail.month.eq(final)]
    outstanding = float(last.outstanding.sum())
    accrued = float(last.accrued.sum())

    by_security = detail.groupby(["month", "security"], observed=True).accrued.sum()
    stock = by_security.groupby(level="month").sum()

    months = sorted(stock.index)
    changes, departed, losing = {}, {}, 0
    for earlier, later in zip(months, months[1:]):
        before = by_security.loc[earlier]
        gone = before.loc[~before.index.isin(by_security.loc[later].index)]
        changes[later] = float(stock[later] - stock[earlier])
        departed[later] = float(gone.sum())
        losing += len(gone) > 0

    increments = _daily_increments()
    monthly_flow = increments.groupby(increments.day.dt.to_period("M")).amount.sum()

    compared = pd.DataFrame({
        "change": pd.Series(changes),
        "departed": pd.Series(departed),
        "flow": monthly_flow,
    }).dropna()
    unadjusted = (compared.change - compared.flow).abs()
    adjusted = (compared.change - (compared.flow - compared.departed)).abs()
    opposite = int(((compared.change > 0) & (compared.flow < 0)).sum()
                   + ((compared.change < 0) & (compared.flow > 0)).sum())

    return (
        int(detail.month.nunique()),
        len(last),
        int(last.security.nunique()),
        round(outstanding / MILLIONS_PER_BILLION, 6),

        round(100.0 * accrued / outstanding, 6),
        round(float(last.issued.sum()) / MILLIONS_PER_BILLION, 6),
        losing,
        round(sum(departed.values()) / MILLIONS_PER_BILLION, 6),

        len(increments),
        int(increments.amount.lt(0).sum()),
        round(float(increments.amount.sum()) / MILLIONS_PER_BILLION, 6),
        len(compared),

        opposite,
        round(float(unadjusted.mean()), 6),
        round(float(adjusted.mean()), 6),
        round(float(adjusted.max()), 6),
    )


TURN_1_NAMES = ["tips_month_end_count", "tips_issue_row_count", "tips_cusip_count",
                "tips_outstanding_usd_billions"]
TURN_2_NAMES = ["accretion_share_of_outstanding_pct", "cumulative_issued_usd_billions",
                "month_ends_with_a_departing_security_count", "departing_accretion_usd_billions"]
TURN_3_NAMES = ["increment_day_count", "negative_increment_day_count",
                "dts_increment_total_usd_billions", "months_compared_count"]
TURN_4_NAMES = ["opposite_direction_month_count", "unadjusted_mean_absolute_gap_usd_millions",
                "adjusted_mean_absolute_gap_usd_millions", "largest_adjusted_gap_usd_millions"]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        dict(outputs),
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_tips_stock": turn_validator(validate_turn_1),
    "validate_accrued_principal": turn_validator(validate_turn_2),
    "validate_daily_flow": turn_validator(validate_turn_3),
    "validate_stock_flow_bridge": turn_validator(validate_turn_4),
}
