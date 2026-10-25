"""A financed Treasury floating-rate note, and the credit line it breaches.

A March 2025 reopening of the two-year floating-rate note is bought at auction
and funded entirely at realized SOFR. The note pays a quarterly coupon indexed to
the thirteen-week bill; the loan accrues daily. Each coupon is swept into
repayment on arrival, so the loan drifts with the difference between the two
rates rather than with either one, and the position needs no view on direction to
be worth measuring.

The identification is a data contract before it is a calculation. The auction
table does not carry a floating-rate security type: these notes are recorded as
Notes, and the `floating_rate` flag is what separates them. Exactly one such
auction settles in March 2025.

The coupon convention decides the answer. The index is the high discount rate of
the most recent thirteen-week bill auction, converted to an ACT/360 money-market
yield over that bill's own term, and it is frozen for the final two business days
before each payment date. Without the freeze the first coupon comes to 1.075643
per 100 rather than 1.075714, which is 0.71 dollars on a million of face and more
than ten times the tolerance this case scores at, so the query pins it.

Cash amounts are reported in USD thousands to five decimals rather than four.
A coupon settled to the cent is 10,757.14 dollars, which is 10.75714 thousand: a
four-decimal contract would leave 10.7571 and the incorrectly rounded 10.7572
both inside the shared tolerance, and a band wider than the value's own precision
is exactly what the validator contract forbids.

The financing leg pins its own conventions: a SOFR observation date defines a
business day, one simple factor covers each interval including the non-business
days inside it, and factors compound between intervals.

Rejected alternative conventions, each measured against the pinned reading:

- Freezing nothing before the payment date gives a first coupon of 10,756.43
  against 10,757.14, and a second post-payment loan balance of 999,756.51.
- Taking the index from the bill's issue date rather than its auction date, so a
  rate applies three days later than it does, gives a first coupon of 10,754.58.
- Converting the bill's discount rate over a fixed 91 days rather than its own
  term gives 10,757.04.
- Financing only the clean price and leaving the accrued interest unfunded starts
  the loan 6,770.46 lower and never reaches the credit limit at all, so the final
  turn reports no contribution.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


FACE = Decimal("1000000")
CREDIT_LIMIT = Decimal("1005000")
RESTORE_TO = Decimal("1000000")
LOCKOUT_BUSINESS_DAYS = 2

TURN_1_NAMES = [
    "selected_cusip", "settlement_date", "first_payment_date",
    "settlement_cash_usd_thousands",
]
TURN_2_NAMES = ["first_coupon_usd_thousands", "first_post_payment_loan_usd_thousands"]
TURN_3_NAMES = [
    "second_payment_date", "second_coupon_usd_thousands",
    "second_post_payment_loan_usd_thousands",
]
TURN_4_NAMES = [
    "credit_contribution_count", "first_contribution_date",
    "total_contribution_usd_thousands",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store the selected note's CUSIP as its exact nine-character source string."),
    _v(TURN_1_NAMES[1], "Store the settlement date as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[2], "Store the first interest payment date after settlement as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[3], "Store the cash required at settlement in USD thousands as a float, rounded to 5 decimals."),
    _v(TURN_2_NAMES[0], "Store the first coupon received in USD thousands as a float, rounded to 5 decimals."),
    _v(TURN_2_NAMES[1], "Store the loan balance immediately after that coupon is applied, in USD thousands as a float, rounded to 5 decimals."),
    _v(TURN_3_NAMES[0], "Store the second interest payment date as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[1], "Store the second coupon received in USD thousands as a float, rounded to 5 decimals."),
    _v(TURN_3_NAMES[2], "Store the loan balance immediately after that coupon is applied, in USD thousands as a float, rounded to 5 decimals."),
    _v(TURN_4_NAMES[0], "Store the number of cash contributions the credit limit forces as a nonnegative integer."),
    _v(TURN_4_NAMES[1], "Store the first contribution date as an ISO YYYY-MM-DD string, or the exact token NONE when there is none."),
    _v(TURN_4_NAMES[2], "Store the total cash contributed in USD thousands as a float, rounded to 5 decimals."),
]

DECIMALS = [None, None, None, 5, 5, 5, None, 5, 5, 0, None, 5]


def _cents(value: Decimal) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _thousands(value: Decimal) -> float:
    return float(Decimal(value) / 1000)


@lru_cache(maxsize=1)
def _bill_index() -> pd.DataFrame:
    """Thirteen-week bill auctions with their own term, newest last."""
    rows = load_expansion_table("treasury_auctions")
    bills = rows.loc[
        rows.security_type.astype(str).eq("Bill")
        & rows.security_term.astype(str).str.contains("13-Week", na=False)
    ].copy()
    bills["auction"] = pd.to_datetime(bills.auction_date.astype(str))
    bills["issued"] = pd.to_datetime(bills.issue_date.astype(str))
    bills["matures"] = pd.to_datetime(bills.maturity_date.astype(str))
    bills["term_days"] = (bills.matures - bills.issued).dt.days
    return bills.sort_values("auction")


@lru_cache(maxsize=1)
def _note() -> pd.Series:
    """The one floating-rate auction settling in March 2025."""
    rows = load_expansion_table("treasury_auctions")
    frn = rows.loc[
        rows.floating_rate.astype(str).eq("Yes")
        & rows.auction_date.astype(str).str.startswith("2025-03")
    ]
    if len(frn) != 1:
        raise ValueError(f"expected one March 2025 floating-rate auction, found {len(frn)}")
    return frn.iloc[0]


def _money_market_yield(discount_rate: float, term_days: float) -> float:
    price = 100.0 * (1.0 - discount_rate / 100.0 * term_days / 360.0)
    return ((100.0 - price) / price) * (360.0 / term_days) * 100.0


def _coupon_per_hundred(start: pd.Timestamp, end: pd.Timestamp, spread: float) -> float:
    """Daily accrual over [start, end), with the index frozen before payment."""
    bills = _bill_index()
    days = list(pd.date_range(start, end - pd.Timedelta(days=1), freq="D"))
    business = [day for day in days if day.weekday() < 5]
    if len(business) < LOCKOUT_BUSINESS_DAYS:
        raise ValueError("period too short to apply the payment-date lockout")
    frozen_from = business[-LOCKOUT_BUSINESS_DAYS]
    total = 0.0
    for day in days:
        reference = frozen_from if day >= frozen_from else day
        prior = bills.loc[bills.auction < reference]
        if prior.empty:
            raise ValueError(f"no thirteen-week auction precedes {reference.date()}")
        row = prior.iloc[-1]
        index = round(_money_market_yield(float(row.high_discnt_rate), float(row.term_days)), 9)
        total += round((index + spread) / 360.0, 9)
    return total


@lru_cache(maxsize=1)
def _sofr() -> tuple:
    """Published SOFR observations in effective-date order; these are the business days."""
    rows = load_expansion_table("nyfed_reference_rates")
    rows = rows.loc[rows["Rate Type"].astype(str).eq("SOFR")].copy()
    rows["effective"] = pd.to_datetime(rows["Effective Date"].astype(str))
    rows = rows.sort_values("effective")
    return tuple(zip(rows.effective, rows["Rate (%)"].astype(float)))


def _financed(balance: Decimal, start: pd.Timestamp, end: pd.Timestamp) -> Decimal:
    """Compound the loan across SOFR intervals from start up to but excluding end."""
    points = [(day, rate) for day, rate in _sofr() if start <= day < end]
    for position, (day, rate) in enumerate(points):
        following = points[position + 1][0] if position + 1 < len(points) else end
        balance *= Decimal(1) + Decimal(str(rate)) / 100 * Decimal((following - day).days) / 360
    return balance


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    note = _note()
    spread = float(note.spread)
    settlement = pd.Timestamp(str(note.issue_date))
    first_payment = pd.Timestamp(str(note.first_int_payment_date))
    # Payment dates follow the note's month-end dated date, so the quarter after
    # 30 April ends on 31 July rather than three calendar months later.
    second_payment = first_payment + pd.offsets.MonthEnd(3)
    dated = pd.Timestamp(str(note.dated_date))

    principal = _cents(Decimal(str(note.price_per100)) / 100 * FACE)
    accrued = _cents(Decimal(str(note.accrued_int_per100)) / 100 * FACE)
    settlement_cash = principal + accrued

    first_coupon = _cents(Decimal(str(_coupon_per_hundred(dated, first_payment, spread))) / 100 * FACE)
    second_coupon = _cents(Decimal(str(_coupon_per_hundred(first_payment, second_payment, spread))) / 100 * FACE)
    first_loan = _financed(Decimal(settlement_cash), settlement, first_payment) - first_coupon
    second_loan = _financed(first_loan, first_payment, second_payment) - second_coupon

    balance, count, contributed, first_date = first_loan, 0, Decimal(0), None
    boundaries = [day for day, _ in _sofr() if first_payment <= day <= second_payment]
    rates = dict(_sofr())
    for position, day in enumerate(boundaries):
        if position:
            previous = boundaries[position - 1]
            balance *= Decimal(1) + Decimal(str(rates[previous])) / 100 * Decimal((day - previous).days) / 360
        if day == second_payment:
            balance -= second_coupon
        if balance > CREDIT_LIMIT:
            contribution = (balance - RESTORE_TO).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
            balance -= contribution
            contributed += contribution
            count += 1
            if first_date is None:
                first_date = day

    return (
        str(note.cusip), settlement.strftime("%Y-%m-%d"), first_payment.strftime("%Y-%m-%d"),
        _thousands(settlement_cash),
        _thousands(first_coupon), _thousands(first_loan),
        second_payment.strftime("%Y-%m-%d"), _thousands(second_coupon), _thousands(second_loan),
        count, first_date.strftime("%Y-%m-%d") if first_date is not None else "NONE",
        _thousands(contributed),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    places = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_purchase": turn_validator(validate_turn_1),
    "validate_first_coupon": turn_validator(validate_turn_2),
    "validate_second_coupon": turn_validator(validate_turn_3),
    "validate_credit_limit": turn_validator(validate_turn_4),
}
