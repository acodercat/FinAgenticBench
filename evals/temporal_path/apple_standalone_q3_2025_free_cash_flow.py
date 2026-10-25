"""Identify Apple's fiscal-2025 10-Qs and derive standalone Q3 free cash flow.

The first turn uses SEC submissions metadata to identify the non-amended 10-Qs
for the six- and nine-month report dates and records their EDGAR acceptance
timestamps. The second turn carries those accessions into Company Facts, requires
the cumulative cash-flow facts to share a fiscal-year start, and derives the
standalone quarter by differencing like-for-like cumulative periods.

Free cash flow here is the query-defined operating-cash-flow-minus-capital-
expenditures measure. It is derived rather than a directly reported GAAP subtotal,
and one quarter alone does not establish sustainability, quality, valuation or
investment merit. The method is stated and pinned, so this is a medium baseline
case rather than a hidden-method trap.

Numeric validation uses the shared rounding tolerance; accessions and timestamps
match exactly.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_facts, load_filings, one_fact
from core.validation import turn_validator, validate_ordered_outputs


TICKER = "AAPL"
FORM = "10-Q"
SIX_MONTH_REPORT_DATE = "2025-03-29"
NINE_MONTH_REPORT_DATE = "2025-06-28"
FISCAL_YEAR_START = "2024-09-29"

TURN_1_NAMES = [
    "six_month_accession", "six_month_acceptance_datetime",
    "nine_month_accession", "nine_month_acceptance_datetime",
]
TURN_2_NAMES = [
    "six_month_operating_cash_flow_usd_billions",
    "nine_month_operating_cash_flow_usd_billions",
    "six_month_capex_usd_billions",
    "nine_month_capex_usd_billions",
    "standalone_q3_free_cash_flow_usd_billions",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store the accession of the six-month 10-Q as text."),
    _v(TURN_1_NAMES[1], "Store its EDGAR acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(TURN_1_NAMES[2], "Store the accession of the nine-month 10-Q as text."),
    _v(TURN_1_NAMES[3], "Store its EDGAR acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(TURN_2_NAMES[0], "Store first-six-month operating cash flow in USD billions, rounded to 3 decimals."),
    _v(TURN_2_NAMES[1], "Store first-nine-month operating cash flow in USD billions, rounded to 3 decimals."),
    _v(TURN_2_NAMES[2], "Store first-six-month capex as a positive USD-billion outflow, rounded to 3 decimals."),
    _v(TURN_2_NAMES[3], "Store first-nine-month capex as a positive USD-billion outflow, rounded to 3 decimals."),
    _v(TURN_2_NAMES[4], "Store standalone Q3 operating cash flow minus standalone Q3 capex in USD billions, rounded to 3 decimals."),
]

DECIMALS = [None, None, None, None, 3, 3, 3, 3, 3]


@lru_cache(maxsize=1)
def _selected_filings():
    filings = load_filings()
    selected = filings.loc[
        filings.ticker.astype(str).eq(TICKER)
        & filings.form.astype(str).eq(FORM)
        & filings.report_date.astype(str).isin([SIX_MONTH_REPORT_DATE, NINE_MONTH_REPORT_DATE])
        & filings.is_amendment.eq(False)
    ].copy()
    counts = selected.groupby(selected.report_date.astype(str)).size()
    if counts.to_dict() != {SIX_MONTH_REPORT_DATE: 1, NINE_MONTH_REPORT_DATE: 1}:
        raise ValueError(f"expected one non-amended filing per report date, found {counts.to_dict()}")
    return selected.set_index(selected.report_date.astype(str))


@lru_cache(maxsize=1)
def ground_truth():
    filings = _selected_filings()
    six_filing = filings.loc[SIX_MONTH_REPORT_DATE]
    nine_filing = filings.loc[NINE_MONTH_REPORT_DATE]

    facts = load_facts()
    common = {"facts": facts, "ticker": TICKER, "start_date": FISCAL_YEAR_START}
    six_accession = str(six_filing.accession)
    nine_accession = str(nine_filing.accession)
    six_operating = one_fact(
        accession=six_accession, concept="NetCashProvidedByUsedInOperatingActivities",
        end_date=SIX_MONTH_REPORT_DATE, **common,
    )
    nine_operating = one_fact(
        accession=nine_accession, concept="NetCashProvidedByUsedInOperatingActivities",
        end_date=NINE_MONTH_REPORT_DATE, **common,
    )
    six_capex = one_fact(
        accession=six_accession, concept="PaymentsToAcquirePropertyPlantAndEquipment",
        end_date=SIX_MONTH_REPORT_DATE, **common,
    )
    nine_capex = one_fact(
        accession=nine_accession, concept="PaymentsToAcquirePropertyPlantAndEquipment",
        end_date=NINE_MONTH_REPORT_DATE, **common,
    )
    standalone = (nine_operating - six_operating) - (nine_capex - six_capex)
    return (
        six_accession, str(six_filing.acceptance_datetime),
        nine_accession, str(nine_filing.acceptance_datetime),
        *(value / 1e9 for value in (six_operating, nine_operating, six_capex, nine_capex)),
        standalone / 1e9,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = {
        name: str(value).strip() if decimals[name] is None else value
        for name, value in outputs.items()
    }
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
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
    "validate_standalone_q3_fcf": turn_validator(validate_turn_2),
}
