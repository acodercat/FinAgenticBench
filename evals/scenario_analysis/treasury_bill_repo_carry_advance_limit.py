"""Treasury bill discount earnings versus floating repo carry.

The fixed redemption discount on an auctioned bill and a daily SOFR-funded
purchase have different day-count bases and economic cash-flow timing. The
retrospective financing path is hypothetical, not a 2025 executable repo quote.
The published 4.15% high bank-discount rate is not an investor's cash-on-cash
annualized yield; the latter is 4.2978% under the stipulated ACT/365 convention.
Using a same-effective-day SOFR before publication gives USD200,148.286765
of baseline interest rather than USD200,184.999994; ACT/365 in place of repo
ACT/360 gives USD197,442.7397201. The model uses the
9 a.m. published-fixing reset specified in the query and a bill maturity date
excluded from financing accrual. The base 90%-advance loan has positive cash
carry of USD9,620.6000 before other risks and costs; a Sep 1 rate shock of
100 basis points makes it negative USD13,386.3568. These are baseline
convention probes, not a live diagnostic trap.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


def D(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite numeric input")
    return result


TURNS = [
    ["bill_cusip", "issue_date", "maturity_date", "auction_price_per100", "bank_discount_rate_pct"],
    ["purchase_cash_usd", "investment_yield_pct"],
    ["repo_interest_usd", "break_even_spread_bp"],
    ["stressed_repo_interest_usd", "maximum_advance_pct"],
]
variables = [
    Variable("bill_cusip", None, "Store the nine-character Treasury CUSIP as an exact uppercase string."),
    Variable("issue_date", None, "Store the bill issue date as an ISO YYYY-MM-DD string."),
    Variable("maturity_date", None, "Store the bill maturity date as an ISO YYYY-MM-DD string."),
    Variable("auction_price_per100", None, "Store the published high price in USD per USD100 par as a finite number, rounded to 6 decimals from its source value."),
    Variable("bank_discount_rate_pct", None, "Store the published high bank-discount rate in percent as a finite number, rounded to 2 decimals from its source value."),
    Variable("purchase_cash_usd", None, "Store purchase cash in USD as a finite number, rounded to 2 decimals from the unrounded published price."),
    Variable("investment_yield_pct", None, "Store the annualized bill investment yield in percent as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("repo_interest_usd", None, "Store total baseline repo interest in USD as a finite number, rounded to 2 decimals from unrounded daily accrual."),
    Variable("break_even_spread_bp", None, "Store the constant break-even repo spread in basis points as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("stressed_repo_interest_usd", None, "Store total shocked repo interest in USD as a finite number, rounded to 2 decimals from unrounded daily accrual."),
    Variable("maximum_advance_pct", None, "Store the maximum purchase-cash advance percentage as a finite number, rounded to 4 decimals from unrounded inputs."),
]
DECIMALS = [None, None, None, 6, 2, 2, 4, 2, 4, 2, 4]


def _bill():
    auctions = load_expansion_table("treasury_auctions")
    rows = auctions.loc[
        auctions["security_type"].eq("Bill")
        & auctions["security_term"].eq("26-Week")
        & auctions["auction_date"].astype(str).eq("2025-06-02")
        & auctions["issue_date"].astype(str).eq("2025-06-05")
        & auctions["reopening"].eq("No")
    ]
    if len(rows) != 1:
        raise ValueError("ambiguous original bill auction")
    row = rows.iloc[0]
    if str(row["maturity_date"]) != "2025-12-04":
        raise ValueError("bill maturity changed")
    return row


def _rate_factors(issue, maturity):
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]].dropna()
    dates = pd.to_datetime(rates["Effective Date"])
    if dates.duplicated().any():
        raise ValueError("duplicate SOFR effective date")
    observed = pd.Series(rates["Rate (%)"].to_numpy(), index=dates).sort_index()
    if len(observed) < 2:
        raise ValueError("insufficient SOFR history")
    # The effective-date observation is available at the next publication day.
    published = pd.Series(observed.to_numpy()[:-1], index=observed.index[1:])
    accrual_days = pd.date_range(issue, maturity - pd.Timedelta(days=1), freq="D")
    available = published.reindex(accrual_days, method="ffill")
    if available.isna().any():
        raise ValueError("SOFR path unavailable")
    base_sofr = sum((D(rate) / 100) / 360 for rate in available)
    spread_days = D(len(accrual_days)) / 360
    shock_days = sum(day >= pd.Timestamp("2025-09-01") for day in accrual_days)
    return len(accrual_days), base_sofr, spread_days, D(shock_days) / 360


def analysis():
    row = _bill()
    issue = pd.Timestamp(str(row["issue_date"]))
    maturity = pd.Timestamp(str(row["maturity_date"]))
    days, base_sofr, spread_days, shock_years = _rate_factors(issue, maturity)
    if days != 182:
        raise ValueError("bill tenor changed")
    par = D(10_000_000)
    price = D(row["high_price"])
    purchase_cash = par * price / 100
    discount = par - purchase_cash
    if discount <= 0:
        raise ValueError("non-discount bill")
    investment_yield = 100 * discount / purchase_cash * D(365) / D(days)
    advance = D("0.90") * purchase_cash
    base_factor = base_sofr + D("0.0025") * spread_days
    stressed_factor = base_factor + D("0.01") * shock_years
    repo_interest = advance * base_factor
    stressed_interest = advance * stressed_factor
    break_even_spread = D(10_000) * (discount / advance - base_sofr) / spread_days
    maximum_advance = 100 * discount / (purchase_cash * stressed_factor)
    if not (0 < maximum_advance < 100):
        raise ValueError("financing boundary outside allowed range")
    return {
        "bill_cusip": str(row["cusip"]),
        "issue_date": issue.strftime("%Y-%m-%d"),
        "maturity_date": maturity.strftime("%Y-%m-%d"),
        "auction_price_per100": float(price),
        "bank_discount_rate_pct": float(D(row["high_discnt_rate"])),
        "purchase_cash_usd": float(purchase_cash),
        "investment_yield_pct": float(investment_yield),
        "repo_interest_usd": float(repo_interest),
        "break_even_spread_bp": float(break_even_spread),
        "stressed_repo_interest_usd": float(stressed_interest),
        "maximum_advance_pct": float(maximum_advance),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimal_by_name = {v.name: decimals for v, decimals in zip(variables, DECIMALS)}
    return validate_ordered_outputs(outputs, [by_name[name] for name in names], [truth[name] for name in names], [decimal_by_name[name] for name in names])


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


validators = {f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate(outputs, names)) for i, names in enumerate(TURNS, 1)}
