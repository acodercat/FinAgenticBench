"""Build a JNJ historical-revenue candidate set and apply two knowledge cutoffs.

The first turn identifies annual Company Facts for one revenue concept and one
fiscal-2022 duration across later presentations. A fact's fiscal-year metadata
describes the presenting filing, so the historical duration is anchored by its
start and end dates rather than by requiring every later comparative row to carry
the same fiscal-year label.

The second turn joins those candidate accessions to SEC filing metadata and, for
each cutoff, selects the latest non-amended filing by EDGAR acceptance time. The
structured tables establish that the same concept and duration have different
presented values at the two information sets; they do not establish the accounting
or corporate cause of the change. The selection rule is explicit, so this is a
medium baseline rather than a hidden point-in-time trap.

Numeric validation uses the shared rounding tolerance; accessions and timestamps
match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_filings
from core.validation import turn_validator, validate_ordered_outputs


TICKER = "JNJ"
FORM = "10-K"
CONCEPT = "RevenueFromContractWithCustomerExcludingAssessedTax"
START = "2022-01-03"
END = "2023-01-01"
UNIT = "USD"
EARLY_CUTOFF = "2023-06-30T23:59:59"
LATE_CUTOFF = "2024-06-30T23:59:59"

TURN_1_NAMES = [
    "candidate_fact_rows", "candidate_accession_count", "candidate_distinct_value_count",
]
TURN_2_NAMES = [
    "earlier_source_accession", "earlier_acceptance_datetime",
    "fy2022_revenue_as_of_2023_06_30_usd_billions", "later_source_accession",
    "later_acceptance_datetime", "fy2022_revenue_as_of_2024_06_30_usd_billions",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store how many candidate fact rows qualify as an integer."),
    _v(TURN_1_NAMES[1], "Store how many distinct candidate accessions qualify as an integer."),
    _v(TURN_1_NAMES[2], "Store how many distinct revenue values the candidates contain as an integer."),
    _v(TURN_2_NAMES[0], "Store the accession supporting the earlier cutoff as text."),
    _v(TURN_2_NAMES[1], "Store its EDGAR acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(TURN_2_NAMES[2], "Store FY2022 revenue at the earlier cutoff in USD billions, rounded to 3 decimals."),
    _v(TURN_2_NAMES[3], "Store the accession supporting the later cutoff as text."),
    _v(TURN_2_NAMES[4], "Store its EDGAR acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(TURN_2_NAMES[5], "Store FY2022 revenue at the later cutoff in USD billions, rounded to 3 decimals."),
]

DECIMALS = [
    0, 0, 0,
    None, None, 3, None, None, 3,
]


@lru_cache(maxsize=1)
def _candidates():
    facts = load_facts()
    selected = facts.loc[
        facts.ticker.astype(str).eq(TICKER)
        & facts.form.astype(str).eq(FORM)
        & facts.concept.astype(str).eq(CONCEPT)
        & facts.start_date.astype(str).eq(START)
        & facts.end_date.astype(str).eq(END)
        & facts.unit.astype(str).eq(UNIT),
        ["accession", "cik", "value"],
    ].copy()
    selected["amount"] = pd.to_numeric(selected.value, errors="coerce")
    if selected.amount.isna().any():
        raise ValueError("candidate revenue contains a nonnumeric value")
    if selected.duplicated(["accession", "amount"]).any():
        raise ValueError("candidate accession and value are not unique")
    return selected


def _as_of(joined, cutoff):
    eligible = joined.loc[
        joined.acceptance.notna()
        & joined.acceptance.le(pd.Timestamp(cutoff))
        & joined.is_amendment.eq(False)
    ].copy()
    if eligible.empty:
        raise ValueError(f"no eligible filing by {cutoff}")
    return eligible.sort_values(
        ["acceptance", "accession"], ascending=[False, True]
    ).iloc[0]


@lru_cache(maxsize=1)
def ground_truth():
    candidates = _candidates()
    filings = load_filings().loc[
        :, ["accession", "cik", "acceptance_datetime", "is_amendment"]
    ].copy()
    filings["acceptance"] = pd.to_datetime(filings.acceptance_datetime, errors="coerce")
    joined = candidates.merge(
        filings, on=["accession", "cik"], how="left", validate="one_to_one"
    )
    early = _as_of(joined, EARLY_CUTOFF)
    late = _as_of(joined, LATE_CUTOFF)
    return (
        int(len(candidates)),
        int(candidates.accession.nunique()),
        int(candidates.amount.nunique()),
        str(early.accession),
        str(early.acceptance_datetime),
        float(early.amount) / 1e9,
        str(late.accession),
        str(late.acceptance_datetime),
        float(late.amount) / 1e9,
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
    "validate_candidate_set": turn_validator(validate_turn_1),
    "validate_cutoff_presentations": turn_validator(validate_turn_2),
}
