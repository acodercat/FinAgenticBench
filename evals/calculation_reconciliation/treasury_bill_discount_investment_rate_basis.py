"""A Treasury bill auction publishes two rates that measure different things.

The high discount rate states the discount from face value as an annual rate
on a 360-day year; the price per 100 follows from it exactly, and for every
one of the 867 regular bill auctions issued from 2023 through 2025 the
published price is 100 times one less the rate times the days to maturity
over 360, to the six decimals published. The high investment rate states the
return on the price paid, annualised over the bill's own year. Because the
price is below face and the year is longer than 360 days, the investment rate
is the higher of the two, by a median of 8.40 basis points on 4-week bills,
13.45 on 13-week, 18.65 on 26-week and 22.20 on 52-week bills.

The bill's own year is not decided by whether a 29 February falls inside the
bill's life. For the 821 auctions of bills running 182 days or fewer, the
published investment rate equals (100 - price) / price times y over the days
to maturity to three decimals for every auction only when y is the number of
days in the year that follows the issue date. A bill issued on 2023-03-07 and
maturing on 2023-04-04 takes y = 366 although its life contains no leap day,
and a bill issued on 2024-02-29 takes y = 365. Under that rule 273 of the 867
auctions carry a 366-day year. The competing rules reproduce fewer: a leap day
inside the bill's life 630 of 821 (and 81 at 366 days), the issue year being a
leap year 387, the maturity year being a leap year 522, a flat 365 days 563
and a flat 360 days none. The published price has to be used as published:
the price recomputed from the discount rate without its six-decimal rounding
reproduces 820 of the 821.

The difference splits into a day-count part, the discount rate times the
excess of y over 360 days, and a price part, the remainder from stating the
return on price rather than on face. The first depends on the rate level
alone, a median of 6.7167 basis points on 4-week bills and 6.6424 on 26-week;
the second grows with term, 1.6806 against 11.6583.

The par yield curve quotes bills on a coupon-equivalent basis. Set against the
curve on the auction date, the 26-week discount rate sits a median 18.50 basis
points below the 6-month point while the investment rate sits 0.00 from it;
the 52-week discount rate sits 23.00 below the 1-year point and the investment
rate 0.30 below. Every 26- and 52-week auction in the window has its
investment rate closer to the curve than its discount rate.

Three further conventions the query states because each moves an output:

1. Auctions are grouped by the term offered at that auction, as stated in
   weeks. 554 of the 867 are reopenings whose original term differs; grouped
   by original term the 26-week median difference reads 14.50 rather than
   18.65. The same offering also runs to a few different day counts, 143 of
   the 156 26-week auctions at 182 days and the rest at 181 or 183, so
   grouping by the day count instead gives 13.70, 18.90 and 22.45 for the
   13-, 26- and 52-week medians rather than 13.45, 18.65 and 22.20; only the
   4-week median is unchanged.
2. Cash management bills are left out. They run to irregular terms and would
   enter no offered-term group cleanly.
3. Medians are taken with linear interpolation. The 13-week and 26-week groups
   hold 156 auctions each, an even count whose two middle values differ.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2023-01-01", "2025-12-31")
SHORT_BILL_DAYS = 182
BASIS_POINTS = 100
RATES = ("high_discnt_rate", "high_investment_rate", "high_price")
CURVE_POINT = {"26-Week": "6 Mo", "52-Week": "1 Yr"}

OUTPUTS = (
    ("bill_auction_count", 0, "the count of regular bill auctions as an integer"),
    ("short_bill_count", 0, "the count of those running 182 days or fewer as an integer"),
    ("leap_basis_count", 0, "how many auctions take a 366-day year under the reproducing rule, as an integer"),
    ("median_gap_4_week_bp", 2, "the 4-week median of investment rate less discount rate in basis points"),
    ("median_gap_13_week_bp", 2, "the 13-week median of investment rate less discount rate in basis points"),
    ("median_gap_26_week_bp", 2, "the 26-week median of investment rate less discount rate in basis points"),
    ("median_gap_52_week_bp", 2, "the 52-week median of investment rate less discount rate in basis points"),
    ("day_part_4_week_bp", 4, "the 4-week median day-count part in basis points"),
    ("price_part_4_week_bp", 4, "the 4-week median price part in basis points"),
    ("day_part_26_week_bp", 4, "the 26-week median day-count part in basis points"),
    ("price_part_26_week_bp", 4, "the 26-week median price part in basis points"),
    ("discount_less_curve_26_week_bp", 2, "the 26-week median of discount rate less the 6-month curve point in basis points"),
    ("investment_less_curve_26_week_bp", 2, "the 26-week median of investment rate less the 6-month curve point in basis points"),
    ("discount_less_curve_52_week_bp", 2, "the 52-week median of discount rate less the 1-year curve point in basis points"),
    ("investment_less_curve_52_week_bp", 2, "the 52-week median of investment rate less the 1-year curve point in basis points"),
)
variables = [
    Variable(name, None, f"Store {text}" + ("." if p == 0 else f", rounded to {p} decimals from unrounded source rows."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:3]),
    tuple(n for n, _, _ in OUTPUTS[3:7]),
    tuple(n for n, _, _ in OUTPUTS[7:11]),
    tuple(n for n, _, _ in OUTPUTS[11:]),
)


def year_following(issue):
    """Days in the year that follows each issue date, the Treasury's y."""
    return ((issue + pd.DateOffset(years=1)) - issue).dt.days


@lru_cache(maxsize=1)
def _bills():
    auctions = load_expansion_table("treasury_auctions")
    bills = auctions.loc[
        auctions.security_type.astype(str).eq("Bill")
        & auctions.cash_management_bill_cmb.astype(str).ne("Yes")
    ].copy()
    for column in ("issue_date", "maturity_date", "auction_date", "security_term_week_year"):
        bills[column] = bills[column].astype(str)
    bills = bills.loc[bills.issue_date.between(*WINDOW)]
    for column in RATES:
        bills[column] = pd.to_numeric(bills[column], errors="coerce")
    if bills[list(RATES)].isna().any().any():
        raise ValueError("a retained bill auction is missing a published rate or price")

    issue = pd.to_datetime(bills.issue_date)
    bills["days"] = (pd.to_datetime(bills.maturity_date) - issue).dt.days
    bills["year_days"] = year_following(issue)
    bills["gap_bp"] = (bills.high_investment_rate - bills.high_discnt_rate) * BASIS_POINTS
    bills["day_part_bp"] = bills.high_discnt_rate * (bills.year_days / 360 - 1) * BASIS_POINTS
    bills["price_part_bp"] = bills.gap_bp - bills.day_part_bp

    short = bills.loc[bills.days.le(SHORT_BILL_DAYS)]
    reproduced = (
        (100 - short.high_price) / short.high_price * short.year_days / short.days * 100
    ).round(3)
    if not reproduced.eq(short.high_investment_rate.round(3)).all():
        raise ValueError("the year-following-issue basis does not reproduce every short bill")
    return bills


@lru_cache(maxsize=1)
def _against_curve():
    bills = _bills()
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = curve.Date.astype(str)
    frames = []
    for term, point in CURVE_POINT.items():
        group = bills.loc[bills.security_term_week_year.eq(term)]
        points = curve[["Date", point]].rename(columns={"Date": "auction_date", point: "curve"})
        matched = group.merge(points, on="auction_date", how="left", validate="many_to_one")
        matched["curve"] = pd.to_numeric(matched.curve, errors="coerce")
        if matched.curve.isna().any():
            raise ValueError(f"no curve point on an auction date for {term} bills")
        matched["term"] = term
        frames.append(matched)
    frame = pd.concat(frames, ignore_index=True)
    frame["discount_less_curve_bp"] = (frame.high_discnt_rate - frame.curve) * BASIS_POINTS
    frame["investment_less_curve_bp"] = (frame.high_investment_rate - frame.curve) * BASIS_POINTS
    return frame


def _median(frame, term, column):
    values = frame.loc[frame.security_term_week_year.eq(term), column]
    if values.empty:
        raise ValueError(f"no {term} auction in the window")
    return float(values.median())


@lru_cache(maxsize=1)
def analysis():
    bills = _bills()
    curve = _against_curve()
    answers = {
        "bill_auction_count": len(bills),
        "short_bill_count": int(bills.days.le(SHORT_BILL_DAYS).sum()),
        "leap_basis_count": int(bills.year_days.eq(366).sum()),
    }
    for weeks in (4, 13, 26, 52):
        answers[f"median_gap_{weeks}_week_bp"] = _median(bills, f"{weeks}-Week", "gap_bp")
    for weeks in (4, 26):
        answers[f"day_part_{weeks}_week_bp"] = _median(bills, f"{weeks}-Week", "day_part_bp")
        answers[f"price_part_{weeks}_week_bp"] = _median(bills, f"{weeks}-Week", "price_part_bp")
    for weeks in (26, 52):
        answers[f"discount_less_curve_{weeks}_week_bp"] = _median(
            curve, f"{weeks}-Week", "discount_less_curve_bp"
        )
        answers[f"investment_less_curve_{weeks}_week_bp"] = _median(
            curve, f"{weeks}-Week", "investment_less_curve_bp"
        )
    return answers


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


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
