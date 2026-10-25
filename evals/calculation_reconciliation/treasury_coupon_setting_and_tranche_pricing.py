"""A Treasury coupon is set at the auction and never reset by a reopening.

For a new nominal note or bond, the coupon is the largest multiple of one
eighth of a percent that does not exceed the auction's high yield, and never
less than one eighth. Every one of the 623 original nominal note and bond
auctions from 2015 through 2025 obeys that rule: the high yield sits between
0 and 12.4 basis points above the coupon, except in one auction. The two-year
note auctioned on 2021-02-23 cleared at 0.119 percent, below the smallest
coupon the rule allows, took the one-eighth coupon, and so is the one original
issue priced above par. Everywhere else a new issue prices at par or below.

A reopening sells more of an existing security at its existing coupon, so its
price is whatever the market yield makes it. Of the 239 nominal reopenings in
the window, 83 priced above par, and each of them is exactly a reopening whose
high yield fell below the coupon.

The same holds across the stock. Of the 350 nominal notes and bonds
outstanding on 2025-12-31, 141 were sold in more than one tranche, and 62 of
those include both a tranche sold at a yield below the coupon and one sold
above it: the same security, the same coupon, issued at a premium and at a
discount. Two outstanding original tranches have a coupon their auction's high
yield does not reproduce. Both are 30-year bonds auctioned in 1996, when notes
and bonds of that term were still sold in multiple-price auctions and the
coupon followed the average accepted yield: 6.119 against a coupon of 6.0 and
6.619 against 6.5, where the high yields of 6.13 and 6.625 would have given
6.125 and 6.625.

The debt statement also records what the auction record cannot: 219.3 billion
dollars fewer outstanding than issued across those 350 securities, retired
before maturity. Weighted by the amount each security has lost this way,
their coupon is 2.7307 percent, against a median coupon of 3.625 percent
across the 350, so early retirement has fallen mostly on low-coupon issues
from years of low rates. Seven securities show outstanding amounts a
fraction of a million above issued, rounding in the statement, and enter the
weighting with those small negative weights as reported. The tranche
structure of the third turn can also be rebuilt from the auction record
alone; this turn is where the debt statement is indispensable. An
outstanding-weighted coupon of the stock is left out because an active case
already measures it.

Readings that were measured and do not move an output, stated so they are not
mistaken for oversights:

1. Bounding the window by issue date instead of auction date leaves 623
   originals, 239 reopenings and 83 premium reopenings.
2. Judging a tranche against the auction high yield instead of the yield the
   debt statement reports leaves 62 securities with both kinds of tranche. The
   two yields differ only for eight tranches from 1996 to 1998, the average
   against the stop of a multiple-price auction, and none changes side.
3. Seven tranches were sold at a yield equal to the coupon. The query counts a
   tranche above or below par strictly, so those seven count as neither.
4. No nominal note or bond matures on the record date itself.

One reading that does move an output: applying the coupon rule to the yield
the debt statement reports for each original tranche, rather than to the
auction's high yield, leaves no failures at all, because for the two 1996
bonds that reported yield is already the average the coupon was set from. The
query names the high yield for that reason. Separately, six original issues
cleared at a yield exactly equal to the coupon yet priced a few millionths
below par, 99.999837 to 99.999999; none is above par, so no count changes.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2015-01-01", "2025-12-31")
RECORD_DATE = "2025-12-31"
EIGHTH = 0.125
BASIS_POINTS = 100

OUTPUTS = (
    ("original_issue_count", 0, "the count of original nominal note and bond auctions as an integer"),
    ("max_yield_above_coupon_bp", 1, "the largest high yield less coupon across them in basis points"),
    ("premium_original_auction_date", None, "the auction date of the one original issue priced above par, as an ISO YYYY-MM-DD string"),
    ("reopening_count", 0, "the count of reopening auctions as an integer"),
    ("reopening_premium_count", 0, "how many reopenings are priced above par, as an integer"),
    ("outstanding_security_count", 0, "the count of outstanding nominal notes and bonds as an integer"),
    ("reopened_security_count", 0, "how many of them carry more than one tranche, as an integer"),
    ("mixed_tranche_security_count", 0, "how many carry both a tranche sold above and a tranche sold below par, as an integer"),
    ("high_yield_rule_failures", 0, "how many outstanding original tranches have a coupon the auction high yield does not reproduce, as an integer"),
    ("issued_less_outstanding_billion_usd", 1, "the amount issued less the amount outstanding across those securities, in billions of dollars"),
    ("issued_less_outstanding_weighted_coupon_pct", 4, "their coupon weighted by each security's amount issued less amount outstanding, in percent"),
    ("median_coupon_pct", 3, "the median coupon across those securities, in percent"),
)
variables = [Variable(name, None, f"Store {text}" + ("." if p in (None, 0) else f", rounded to {p} decimal{'s' if p > 1 else ''} from unrounded source rows."))
             for name, p, text in OUTPUTS]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:5]),
    tuple(n for n, _, _ in OUTPUTS[5:9]),
    tuple(n for n, _, _ in OUTPUTS[9:]),
)


def coupon_from_yield(yield_pct):
    """The largest multiple of an eighth not above the yield, never below an eighth."""
    return np.maximum(np.floor(yield_pct / EIGHTH + 1e-9) * EIGHTH, EIGHTH)


@lru_cache(maxsize=1)
def _auctions():
    auctions = load_expansion_table("treasury_auctions")
    frame = auctions.loc[
        auctions.security_type.astype(str).isin(["Note", "Bond"])
        & auctions.floating_rate.astype(str).ne("Yes")
        & auctions.inflation_index_security.astype(str).ne("Yes")
    ].copy()
    for column in ("int_rate", "high_yield", "price_per100"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("auction_date", "issue_date", "cusip", "reopening"):
        frame[column] = frame[column].astype(str)
    frame["reopened"] = frame.reopening.eq("Yes")
    return frame


@lru_cache(maxsize=1)
def _window_auctions():
    frame = _auctions()
    frame = frame.loc[frame.auction_date.between(*WINDOW)]
    if frame[["int_rate", "high_yield", "price_per100"]].isna().any().any():
        raise ValueError("a nominal note or bond auction in the window lacks a coupon, yield or price")
    originals = frame.loc[~frame.reopened]
    if not np.isclose(coupon_from_yield(originals.high_yield), originals.int_rate).all():
        raise ValueError("the coupon rule does not reproduce every original issue in the window")
    return frame


@lru_cache(maxsize=1)
def _tranches():
    """Every tranche of each nominal note and bond outstanding at the record date."""
    table = load_expansion_table("treasury_marketable_securities")
    tranches = table.loc[
        table.record_date.astype(str).eq(RECORD_DATE)
        & table.security_class.astype(str).isin(["Notes", "Bonds"])
        & table.maturity_date.notna()
    ].copy()
    # Class totals carry no maturity; a security is a row with one.
    tranches = tranches.loc[tranches.maturity_date.astype(str).gt(RECORD_DATE)]
    tranches["cusip"] = tranches.security_identifier_or_total_label.astype(str)
    tranches["issue_date"] = tranches.issue_date.astype(str)
    for column in ("interest_rate_pct", "yield_pct", "issued_million_usd", "outstanding_million_usd"):
        tranches[column] = pd.to_numeric(tranches[column], errors="coerce")
    if tranches[["interest_rate_pct", "yield_pct"]].isna().any().any():
        raise ValueError("an outstanding tranche lacks a coupon or a yield")
    if tranches.groupby("cusip").interest_rate_pct.nunique().gt(1).any():
        raise ValueError("a security carries more than one coupon across its tranches")
    return tranches


@lru_cache(maxsize=1)
def analysis():
    frame = _window_auctions()
    originals = frame.loc[~frame.reopened]
    reopenings = frame.loc[frame.reopened]
    premium_originals = originals.loc[originals.price_per100 > 100]
    if len(premium_originals) != 1:
        raise ValueError("the window no longer holds exactly one original issue above par")

    tranches = _tranches()
    per_security = tranches.assign(
        above=tranches.yield_pct < tranches.interest_rate_pct,
        below=tranches.yield_pct > tranches.interest_rate_pct,
    ).groupby("cusip").agg(tranches=("issue_date", "size"), above=("above", "any"), below=("below", "any"))

    first = tranches.sort_values("issue_date").drop_duplicates("cusip")
    stops = _auctions()[["cusip", "issue_date", "high_yield"]]
    matched = first.merge(stops, on=["cusip", "issue_date"], how="left", validate="one_to_one")
    if matched.high_yield.isna().any():
        raise ValueError("an outstanding original tranche has no matching auction")

    stock = tranches.groupby("cusip").agg(
        issued=("issued_million_usd", "sum"),
        outstanding=("outstanding_million_usd", "sum"),
        coupon=("interest_rate_pct", "first"),
    )
    if tranches.groupby("cusip").outstanding_million_usd.count().ne(1).any():
        raise ValueError("a security does not report its amount outstanding on exactly one row")
    retired = stock.issued - stock.outstanding

    return {
        "original_issue_count": len(originals),
        "max_yield_above_coupon_bp": float(((originals.high_yield - originals.int_rate) * BASIS_POINTS).max()),
        "premium_original_auction_date": str(premium_originals.auction_date.iloc[0]),
        "reopening_count": len(reopenings),
        "reopening_premium_count": int((reopenings.price_per100 > 100).sum()),
        "outstanding_security_count": len(per_security),
        "reopened_security_count": int(per_security.tranches.gt(1).sum()),
        "mixed_tranche_security_count": int((per_security.above & per_security.below).sum()),
        "high_yield_rule_failures": int(
            (~np.isclose(coupon_from_yield(matched.high_yield), matched.interest_rate_pct)).sum()
        ),
        "issued_less_outstanding_billion_usd": float(retired.sum() / 1e3),
        "issued_less_outstanding_weighted_coupon_pct": float((retired * stock.coupon).sum() / retired.sum()),
        "median_coupon_pct": float(stock.coupon.median()),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
