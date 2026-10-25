"""Reconcile Regulation D offerings when a third of them state no size.

Form D lets an issuer decline to state how much it is offering.  Those filings
report an amount sold and nothing to measure it against, so they belong in a
count of the market and in a total of money raised, but they cannot enter a
subscription rate on either side of it.

The trap is the second turn.  Of 34,717 original 2025 notices, 10,152 state no
offering amount, and the 24,556 that state a positive one offer USD832.467540
billion against USD316.407218 billion sold -- a subscription share of
38.008355 percent.  Carrying every filing's sales into the numerator while only
the stated offerings can supply the denominator reports 81.812713 percent, more
than double, because the indefinite filings alone sold USD364.657063 billion,
more than the stated ones did.  The share is pinned to six decimals, so the
rounding-boundary tolerance excludes the naive reading by 43.804358 points.

Rejected conventions.  Only original notices count: a Form D amendment restates
the cumulative amount sold rather than an increment, so summing every 2025 filing
reports USD11,557.813107 billion against USD681.064281 billion, 16.9702 times
too much.  A stated offering must be positive -- nine notices state a zero
offering and would otherwise count as both fully subscribed and unsold at once,
moving the fully-subscribed count from 12,704 to 12,713.  On the positive stated
offerings the source identity holds exactly, with offered equal to sold plus
remaining on every one of them, and the three outcomes partition them:
12,704 fully subscribed, 8,130 partly, 3,722 not yet begun.

An issuer counts as a reporting company when it appears in the periodic-report
file, which holds only annual and quarterly reports.  784 of the originals are
theirs.  They pay far more to place the same dollar -- 4.1985 percent against
1.7619 -- and sell far more of what they offer, 79.907441 percent against
36.162780, because a placement agent sells a public company's private round
while a fund's notice records commitments its manager gathered itself.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, load_filings
from core.validation import turn_validator, validate_ordered_outputs


FILING_YEAR = 2025
ORIGINAL_TOKEN = "false"


variables = [
    Variable("original_filing_count", None,
             "Store the number of original notices filed in the year as an integer."),
    Variable("indefinite_offering_filing_count", None,
             "Store how many of them state no total offering amount as an integer."),
    Variable("stated_offering_usd_billions", None,
             "Store the total amount offered by the notices that state a positive one, in USD billions, rounded to 6 decimals."),
    Variable("indefinite_offering_amount_sold_usd_billions", None,
             "Store the amount sold by the notices stating no offering amount, in USD billions, rounded to 6 decimals."),

    Variable("subscription_share_pct", None,
             "Store the share of the amount offered that has been sold, in percent, rounded to 6 decimals."),
    Variable("fully_subscribed_filing_count", None,
             "Store how many stated offerings report nothing left to sell as an integer."),
    Variable("partially_subscribed_filing_count", None,
             "Store how many stated offerings have sold something and still have more to sell as an integer."),
    Variable("no_sales_yet_filing_count", None,
             "Store how many stated offerings report no sales at all as an integer."),

    Variable("intermediary_paying_filing_count", None,
             "Store how many original notices report a sales commission or a finder's fee as an integer."),
    Variable("sales_commission_usd_billions", None,
             "Store the total sales commission reported across the originals in USD billions, rounded to 6 decimals."),
    Variable("finders_fee_usd_billions", None,
             "Store the total finder's fee reported across them in USD billions, rounded to 6 decimals."),
    Variable("commission_rate_pct", None,
             "Store commission over amount sold across the notices reporting both a commission and a sale, in percent, rounded to 4 decimals."),

    Variable("public_issuer_filing_count", None,
             "Store how many originals were filed by a periodic-report issuer as an integer."),
    Variable("public_subscription_share_pct", None,
             "Store their share of the amount offered that has been sold, in percent, rounded to 6 decimals."),
    Variable("public_commission_rate_pct", None,
             "Store the periodic-report issuers' commission over amount sold on the same basis, in percent, rounded to 4 decimals."),
    Variable("private_commission_rate_pct", None,
             "Store that same rate for every other issuer, in percent, rounded to 4 decimals."),
]

# The three commission rates are pinned a decimal place coarser than the shares:
# at six decimals the private rate lands 0.0085 of a unit from a rounding
# boundary, which is not a precision the method reproduces safely.
DECIMALS = [0, 0, 6, 6, 6, 0, 0, 0, 0, 6, 6, 4, 0, 6, 4, 4]


def _original_notices():
    """Original Form D notices filed in the year, with their reported amounts."""
    notices = load_expansion_table("sec_form_d")
    filed_on = pd.to_datetime(notices.FILING_DATE, errors="coerce")
    frame = pd.DataFrame({
        "cik": pd.to_numeric(notices.CIK, errors="coerce"),
        "offered": pd.to_numeric(notices.TOTALOFFERINGAMOUNT, errors="coerce"),
        "sold": pd.to_numeric(notices.TOTALAMOUNTSOLD, errors="coerce"),
        "remaining": pd.to_numeric(notices.TOTALREMAINING, errors="coerce"),
        "commission": pd.to_numeric(notices.SALESCOMM_DOLLARAMOUNT, errors="coerce"),
        "finders": pd.to_numeric(notices.FINDERSFEE_DOLLARAMOUNT, errors="coerce"),
        "amendment": notices.ISAMENDMENT.astype(str).str.strip().str.lower(),
    }).loc[filed_on.dt.year.eq(FILING_YEAR)]
    del notices
    return frame.loc[frame.amendment.eq(ORIGINAL_TOKEN)]


def _periodic_report_issuers():
    """The issuers that file annual or quarterly reports."""
    filings = load_filings()
    identifiers = pd.to_numeric(filings.cik, errors="coerce").dropna()
    del filings
    return frozenset(identifiers.astype(int).tolist())


def _rate(commission, sold):
    return round(100.0 * float(commission.sum()) / float(sold.sum()), 4)


@lru_cache(maxsize=1)
def ground_truth():
    notices = _original_notices()
    indefinite = notices.offered.isna()
    stated = notices.offered.gt(0)

    fully = stated & notices.remaining.eq(0)
    partly = stated & notices.sold.gt(0) & notices.remaining.gt(0)
    unsold = stated & notices.sold.eq(0)

    paying = notices.commission.gt(0) | notices.finders.gt(0)
    commissioned = notices.commission.gt(0) & notices.sold.gt(0)

    reporting = notices.cik.isin(_periodic_report_issuers())

    return (
        len(notices),
        int(indefinite.sum()),
        round(float(notices.offered[stated].sum()) / 1e9, 6),
        round(float(notices.sold[indefinite].sum()) / 1e9, 6),

        round(100.0 * float(notices.sold[stated].sum()) / float(notices.offered[stated].sum()), 6),
        int(fully.sum()),
        int(partly.sum()),
        int(unsold.sum()),

        int(paying.sum()),
        round(float(notices.commission.sum()) / 1e9, 6),
        round(float(notices.finders.sum()) / 1e9, 6),
        _rate(notices.commission[commissioned], notices.sold[commissioned]),

        int(reporting.sum()),
        round(100.0 * float(notices.sold[reporting & stated].sum())
              / float(notices.offered[reporting & stated].sum()), 6),
        _rate(notices.commission[reporting & commissioned],
              notices.sold[reporting & commissioned]),
        _rate(notices.commission[~reporting & commissioned],
              notices.sold[~reporting & commissioned]),
    )


TURN_1_NAMES = ["original_filing_count", "indefinite_offering_filing_count",
                "stated_offering_usd_billions", "indefinite_offering_amount_sold_usd_billions"]
TURN_2_NAMES = ["subscription_share_pct", "fully_subscribed_filing_count",
                "partially_subscribed_filing_count", "no_sales_yet_filing_count"]
TURN_3_NAMES = ["intermediary_paying_filing_count", "sales_commission_usd_billions",
                "finders_fee_usd_billions", "commission_rate_pct"]
TURN_4_NAMES = ["public_issuer_filing_count", "public_subscription_share_pct",
                "public_commission_rate_pct", "private_commission_rate_pct"]


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
    "validate_offering_population": turn_validator(validate_turn_1),
    "validate_subscription": turn_validator(validate_turn_2),
    "validate_placement_cost": turn_validator(validate_turn_3),
    "validate_issuer_split": turn_validator(validate_turn_4),
}
