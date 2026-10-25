"""Point-in-time amendment comparison where normalized facts are unchanged.

The trap becomes live in the sole turn when the shared normalized fact keys are
counted. Missing ``start_date`` is a legitimate identity value for instant facts.
Naive pandas ``groupby`` with its default NaN-dropping behavior removes the 135
instant keys and reports 195 equal shared keys; a null-safe comparison reports the
correct 330 equal shared keys and 0 changed keys. Both accessions report Q1 Revenue
of 54.373 billion USD.

Numeric validation uses 0.6 × 10^-N rounding-boundary tolerance for N requested
decimals. No numeric tolerance applies to discrete outputs; labels, dates, Booleans
and timestamps use exact or explicitly documented semantic matching. The fact counts
match exactly, so the wrong value 195 cannot enter the accepted band around 330.
"""

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_filings, one_fact
from core.validation import ValidatorResult, datetime_equal, numeric_equal, turn_validator


ORIGINAL = "0000093410-22-000026"
AMENDMENT = "0000093410-22-000028"
KEYS = ["taxonomy", "concept", "unit", "start_date", "end_date"]

variables = [
    Variable("original_acceptance_datetime", None, "Store the original 10-Q SEC acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    Variable("amendment_acceptance_datetime", None, "Store the 10-Q/A SEC acceptance timestamp as an ISO YYYY-MM-DDTHH:MM:SS string."),
    Variable("original_q1_revenue_usd_billions", None, "Store original-filing Q1 Revenues in USD billions, rounded to 3 decimals."),
    Variable("amended_q1_revenue_usd_billions", None, "Store amendment-filing Q1 Revenues in USD billions, rounded to 3 decimals."),
    Variable("matched_normalized_fact_count", None, "Store the number of fact keys present in both accessions with equal values."),
    Variable("changed_normalized_fact_count", None, "Store the number of shared fact keys whose values differ."),
]


def ground_truth() -> tuple[str, str, float, float, int, int]:
    filings = load_filings().set_index("accession")
    facts = load_facts()
    kwargs = {
        "facts": facts, "ticker": "CVX", "concept": "Revenues",
        "start_date": "2022-01-01", "end_date": "2022-03-31",
    }
    original_revenue = one_fact(accession=ORIGINAL, **kwargs)
    amended_revenue = one_fact(accession=AMENDMENT, **kwargs)
    original = facts.loc[
        facts["ticker"].eq("CVX") & facts["accession"].eq(ORIGINAL),
        KEYS + ["value"],
    ].drop_duplicates()
    amended = facts.loc[
        facts["ticker"].eq("CVX") & facts["accession"].eq(AMENDMENT),
        KEYS + ["value"],
    ].drop_duplicates()
    merged = original.merge(amended, on=KEYS, suffixes=("_original", "_amended"))
    changed = ~merged.apply(
        lambda row: pd.isna(row["value_original"]) and pd.isna(row["value_amended"])
        or row["value_original"] == row["value_amended"], axis=1,
    )
    return (
        str(filings.loc[ORIGINAL, "acceptance_datetime"]),
        str(filings.loc[AMENDMENT, "acceptance_datetime"]),
        original_revenue / 1e9,
        amended_revenue / 1e9,
        int((~changed).sum()),
        int(changed.sum()),
    )


def validate(outputs: dict) -> ValidatorResult:
    if any(value is None for value in outputs.values()):
        return ValidatorResult(False, "required output not set", True)
    expected = ground_truth()
    names = [variable.name for variable in variables]
    errors = []
    for name, target in zip(names[:2], expected[:2]):
        if not datetime_equal(outputs[name], target):
            errors.append(f"{name}={outputs[name]!r}, expected instant {target}")
    for name, target, places in zip(names[2:], expected[2:], [3, 3, 0, 0]):
        if not numeric_equal(outputs[name], target, decimals=places):
            errors.append(f"{name}={outputs[name]!r}, expected {target:.{places}f}")
    return ValidatorResult(not errors, "; ".join(errors) or "correct")



TURN_1_NAMES = [
    "original_acceptance_datetime",
    "amendment_acceptance_datetime",
    "original_q1_revenue_usd_billions",
    "amended_q1_revenue_usd_billions",
]
TURN_2_NAMES = [
    "matched_normalized_fact_count",
    "changed_normalized_fact_count",
]


def _turn_subset(names):
    """Score only `names`, filling the other outputs with their own answers."""
    def run(outputs):
        merged = dict(zip((variable.name for variable in variables), ground_truth()))
        merged.update({name: outputs.get(name) for name in names})
        return validate(merged)

    return run


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_filing_order": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_fact_comparison": turn_validator(_turn_subset(TURN_2_NAMES)),
}
