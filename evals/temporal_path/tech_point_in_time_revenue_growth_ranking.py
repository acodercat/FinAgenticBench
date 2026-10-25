"""Select point-in-time annual filings, then rank same-filing revenue growth.

The first turn uses EDGAR acceptance time to select each issuer's latest original
Form 10-K available at the cutoff. The second turn carries those accessions into
Company Facts and compares the current annual duration with the immediately
preceding annual comparative in the same filing. This preserves issuer fiscal
calendars and the comparative presentation available in that information set.

Four issuers report revenue from customer contracts excluding assessed tax;
Nvidia reports total revenues under its broader revenue concept. The natural-
language query pins that alias. Fiscal-year metadata describes the presenting
filing, so duration dates distinguish current and comparative rows.

The result is a historical reported-revenue ranking, not evidence about margins,
sustainability, valuation, causality, forecasts or investment merit. Because the
availability and concept rules are explicit, this is a hard baseline rather than
a hidden-method trap.

Numeric validation uses the shared rounding tolerance; mappings and identifiers
use exact key sets.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_filings
from core.validation import ValidatorResult, numeric_equal, select_extreme_key, turn_validator


TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
REVENUE_CONCEPT = {
    "AAPL": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "MSFT": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NVDA": "Revenues",
    "AMZN": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GOOGL": "RevenueFromContractWithCustomerExcludingAssessedTax",
}
CUTOFF = "2025-08-15T23:59:59"

TURN_1_NAMES = ["selected_accession_by_ticker", "acceptance_datetime_by_ticker"]
TURN_2_NAMES = [
    "revenue_growth_by_ticker_pct", "latest_annual_period_end_by_ticker",
    "highest_growth_ticker", "highest_growth_pct",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store a dict mapping each requested ticker to its selected original 10-K accession."),
    _v(TURN_1_NAMES[1], "Store a dict mapping each requested ticker to its EDGAR acceptance timestamp in ISO format."),
    _v(TURN_2_NAMES[0], "Store a dict mapping each requested ticker to annual revenue growth percentage rounded to 2 decimals."),
    _v(TURN_2_NAMES[1], "Store a dict mapping each requested ticker to its latest annual period end as YYYY-MM-DD."),
    _v(TURN_2_NAMES[2], "Store the ticker with the highest computed revenue growth."),
    _v(TURN_2_NAMES[3], "Store the highest computed revenue growth percentage rounded to 2 decimals."),
]


@lru_cache(maxsize=1)
def _selected_filings():
    filings = load_filings().copy()
    filings["acceptance"] = pd.to_datetime(filings.acceptance_datetime, errors="coerce")
    eligible = filings.loc[
        filings.ticker.astype(str).isin(TICKERS)
        & filings.form.astype(str).eq("10-K")
        & filings.is_amendment.eq(False)
        & filings.acceptance.notna()
        & filings.acceptance.le(pd.Timestamp(CUTOFF))
    ].copy()
    rows = []
    for ticker in TICKERS:
        candidates = eligible.loc[eligible.ticker.astype(str).eq(ticker)].sort_values(
            ["acceptance", "accession"], ascending=[False, True]
        )
        if candidates.empty:
            raise ValueError(f"no eligible original 10-K for {ticker}")
        rows.append(candidates.iloc[0])
    return pd.DataFrame(rows).set_index("ticker")


@lru_cache(maxsize=1)
def ground_truth():
    filings = _selected_filings()
    facts = load_facts()
    growth, period_ends = {}, {}
    for ticker in TICKERS:
        filing = filings.loc[ticker]
        rows = facts.loc[
            facts.ticker.astype(str).eq(ticker)
            & facts.accession.astype(str).eq(str(filing.accession))
            & facts.concept.astype(str).eq(REVENUE_CONCEPT[ticker])
            & facts.fiscal_period.astype(str).eq("FY")
            & facts.unit.astype(str).eq("USD")
            & facts.start_date.notna(),
            ["start_date", "end_date", "value"],
        ].drop_duplicates().copy()
        rows["amount"] = pd.to_numeric(rows.value, errors="coerce")
        current = rows.loc[rows.end_date.astype(str).eq(str(filing.report_date))]
        prior = rows.loc[rows.end_date.astype(str).lt(str(filing.report_date))].sort_values(
            ["end_date", "start_date"], ascending=[False, False]
        ).head(1)
        if len(current) != 1 or len(prior) != 1:
            raise ValueError(f"expected one current and prior annual revenue for {ticker}")
        current_value = float(current.amount.iloc[0])
        prior_value = float(prior.amount.iloc[0])
        growth[ticker] = (current_value / prior_value - 1) * 100
        period_ends[ticker] = str(current.end_date.iloc[0])
    winner = select_extreme_key(growth)
    return (
        filings.accession.astype(str).to_dict(),
        filings.acceptance_datetime.astype(str).to_dict(),
        growth, period_ends, winner, growth[winner],
    )


def _missing(outputs, names):
    return any(name not in outputs or outputs[name] is None for name in names)


def validate_turn_1(outputs):
    if _missing(outputs, TURN_1_NAMES):
        return ValidatorResult(False, "required output not set", True)
    expected_accessions, expected_times, *_ = ground_truth()
    errors = []
    if outputs[TURN_1_NAMES[0]] != expected_accessions:
        errors.append(f"accessions={outputs[TURN_1_NAMES[0]]!r}, expected {expected_accessions!r}")
    if outputs[TURN_1_NAMES[1]] != expected_times:
        errors.append(f"acceptance times={outputs[TURN_1_NAMES[1]]!r}, expected {expected_times!r}")
    return ValidatorResult(not errors, "; ".join(errors) or "correct")


def validate_turn_2(outputs):
    if _missing(outputs, TURN_2_NAMES):
        return ValidatorResult(False, "required output not set", True)
    _, _, expected_growth, expected_ends, expected_winner, expected_highest = ground_truth()
    actual_growth = outputs[TURN_2_NAMES[0]]
    actual_ends = outputs[TURN_2_NAMES[1]]
    errors = []
    if not isinstance(actual_growth, dict) or set(actual_growth) != set(TICKERS):
        errors.append(f"growth keys={getattr(actual_growth, 'keys', lambda: [])()}, expected {TICKERS}")
    else:
        for ticker, target in expected_growth.items():
            if not numeric_equal(actual_growth[ticker], target, decimals=2):
                errors.append(f"{ticker} growth={actual_growth[ticker]!r}, expected {target:.4f}")
    if not isinstance(actual_ends, dict) or actual_ends != expected_ends:
        errors.append(f"period ends={actual_ends!r}, expected {expected_ends!r}")
    if str(outputs[TURN_2_NAMES[2]]) != expected_winner:
        errors.append(f"winner={outputs[TURN_2_NAMES[2]]!r}, expected {expected_winner}")
    if not numeric_equal(outputs[TURN_2_NAMES[3]], expected_highest, decimals=2):
        errors.append(f"highest growth={outputs[TURN_2_NAMES[3]]!r}, expected {expected_highest:.4f}")
    return ValidatorResult(not errors, "; ".join(errors) or "correct")


def validate(outputs):
    first = validate_turn_1(outputs)
    return first if not first.success else validate_turn_2(outputs)


validators = {
    "validate_filing_cutoff": turn_validator(validate_turn_1),
    "validate_growth_ranking": turn_validator(validate_turn_2),
}
