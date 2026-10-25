"""Select Microsoft's original FY2025 10-K, then compute its current ratio.

Turn 1 makes the submissions source indispensable: it selects the single
non-amended Microsoft 10-K filed on 2025-07-30 and returns its accession,
period end and acceptance timestamp. Turn 2 carries that filing identity into
CompanyFacts and retrieves the USD current-assets and current-liabilities facts
for the same accession and period end before dividing the former by the latter.

The result is a point-in-time balance-sheet liquidity indicator. It shows the
reported current-asset coverage of reported current liabilities at 2025-06-30;
it does not establish asset convertibility, liability timing, operating cash
flows, future payment capacity or solvency.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_facts, load_filings, one_fact
from core.validation import turn_validator, validate_ordered_outputs


TICKER = "MSFT"
FORM = "10-K"
FILING_DATE = "2025-07-30"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["filing_accession", "fiscal_period_end", "acceptance_datetime"]
TURN_2_NAMES = ["current_assets_usd_billions", "current_liabilities_usd_billions", "current_ratio"]

variables = [
    _v(TURN_1_NAMES[0], "Store the selected original filing accession as text."),
    _v(TURN_1_NAMES[1], "Store its report-date period end as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[2], "Store its acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(TURN_2_NAMES[0], "Store current assets in USD billions rounded to 2 decimals."),
    _v(TURN_2_NAMES[1], "Store current liabilities in USD billions rounded to 2 decimals."),
    _v(TURN_2_NAMES[2], "Store current assets divided by current liabilities rounded to 2 decimals."),
]

DECIMALS = [None, None, None, 2, 2, 2]


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
        raise ValueError(f"expected one original Microsoft 10-K on {FILING_DATE}, found {len(selected)}")
    filing = selected.iloc[0]
    accession = str(filing.accession)
    period_end = str(filing.report_date)
    acceptance = str(filing.acceptance_datetime)

    facts = load_facts()
    base = dict(ticker=TICKER, accession=accession, end_date=period_end, unit="USD")
    assets = one_fact(facts, **base, concept="AssetsCurrent") / 1e9
    liabilities = one_fact(facts, **base, concept="LiabilitiesCurrent") / 1e9
    return accession, period_end, acceptance, assets, liabilities, assets / liabilities


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
    "validate_current_ratio": turn_validator(validate_turn_2),
}
