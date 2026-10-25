"""Select Walmart's original FY2025 10-K, then retrieve its annual revenue fact.

Turn 1 uses submissions to establish the unique original filing, accession and
report date. Turn 2 carries those identifiers into CompanyFacts and selects the
single USD annual-revenue duration fact for fiscal period FY. Walmart's fiscal
2025 ended on 2025-01-31, so the fiscal-year label is not a calendar-year end.

Revenue is a flow over 2024-02-01 through 2025-01-31. It is not a balance-sheet
stock, profit, cash receipts, or a measure of revenue earned only during calendar
2025.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_facts, load_filings, one_fact
from core.validation import turn_validator, validate_ordered_outputs


TICKER = "WMT"
FORM = "10-K"
FILING_DATE = "2025-03-14"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["filing_accession", "fiscal_period_end", "acceptance_datetime"]
TURN_2_NAMES = ["revenue_period_start", "revenue_usd_billions"]

variables = [
    _v(TURN_1_NAMES[0], "Store the selected original filing accession as text."),
    _v(TURN_1_NAMES[1], "Store its report-date fiscal period end as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[2], "Store its acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(TURN_2_NAMES[0], "Store the annual revenue fact's start date as an ISO YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1], "Store annual total revenues in USD billions rounded to 2 decimals."),
]

DECIMALS = [None, None, None, None, 2]


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    filings = load_filings().copy()
    selected = filings.loc[
        filings.ticker.astype(str).eq(TICKER)
        & filings.form.astype(str).eq(FORM)
        & filings.filing_date.astype(str).eq(FILING_DATE)
        & ~filings.is_amendment.fillna(False)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one original Walmart 10-K on {FILING_DATE}, found {len(selected)}")
    filing = selected.iloc[0]
    accession = str(filing.accession)
    period_end = str(filing.report_date)
    acceptance = str(filing.acceptance_datetime)

    facts = load_facts()
    annual = facts.loc[
        facts.ticker.astype(str).eq(TICKER)
        & facts.accession.astype(str).eq(accession)
        & facts.concept.astype(str).eq("Revenues")
        & facts.end_date.astype(str).eq(period_end)
        & facts.unit.astype(str).eq("USD")
        & facts.fiscal_period.astype(str).eq("FY")
    ]
    if len(annual) != 1:
        raise ValueError(f"expected one annual revenue fact for {accession}, found {len(annual)}")
    start = str(annual.iloc[0].start_date)
    value = one_fact(
        facts,
        ticker=TICKER,
        accession=accession,
        concept="Revenues",
        unit="USD",
        start_date=start,
        end_date=period_end,
    )
    return accession, period_end, acceptance, start, value / 1e9


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


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_filing_selection": turn_validator(validate_turn_1),
    "validate_annual_revenue": turn_validator(validate_turn_2),
}
