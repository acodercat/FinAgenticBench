"""Retrospective disaster-county mortgage origination collateral exercise.

FEMA disaster 4827 has 39 declared NC county rows and one separate
tribal-area row in the frozen snapshot.
HMDA 2024 originated application records are not a loan-servicing book, and
their reported origination amounts are not current unpaid principal. The
scenario keeps those amounts unchanged solely to measure a counterfactual
collateral-value and per-record principal-paydown requirement. No actual storm
damage, delinquency, loss or public relief allocation is inferred.

Pinned alternatives: admitting manufactured housing changes the eligible
record count from 33,526 to 34,939; treating all NC counties as declared
changes the geography. A portfolio-wide paydown after netting collateral
surpluses would understate the sum of per-record paydowns because surplus
collateral on one loan cannot cure another loan under the stipulated test.
Counts use strict greater-than thresholds and exact integer-scaled arithmetic;
the validator uses the shared display-precision tolerance only.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TURNS = [
    ["declared_counties", "eligible_origination_records", "origination_amount_m"],
    ["above_eighty_records", "above_eighty_amount_m"],
    ["stressed_underwater_records", "newly_underwater_records", "stressed_collateral_shortfall_m"],
    ["minimum_paydown_m", "largest_paydown_county", "largest_county_paydown_m", "highest_intensity_county", "highest_intensity_pct"],
]

variables = [
    Variable("declared_counties", None, "Store the number of declared counties as a finite integer number."),
    Variable("eligible_origination_records", None, "Store the number of eligible originated application records as a finite integer number."),
    Variable("origination_amount_m", None, "Store the total reported origination amount in USD millions, rounded to 4 decimals from unrounded source rows."),
    Variable("above_eighty_records", None, "Store the number of records above the stated origination leverage threshold as a finite integer number."),
    Variable("above_eighty_amount_m", None, "Store their total reported origination amount in USD millions, rounded to 4 decimals from unrounded source rows."),
    Variable("stressed_underwater_records", None, "Store the number of stressed underwater records as a finite integer number."),
    Variable("newly_underwater_records", None, "Store the number newly underwater under the stress as a finite integer number."),
    Variable("stressed_collateral_shortfall_m", None, "Store total positive per-record collateral shortfall in USD millions, rounded to 4 decimals from unrounded inputs."),
    Variable("minimum_paydown_m", None, "Store the minimum sum of per-record principal paydowns in USD millions, rounded to 4 decimals from unrounded inputs."),
    Variable("largest_paydown_county", None, "Store the five-digit county FIPS code as an exact string, preserving leading zeros."),
    Variable("largest_county_paydown_m", None, "Store that county's per-record paydown sum in USD millions, rounded to 4 decimals from unrounded inputs."),
    Variable("highest_intensity_county", None, "Store the five-digit county FIPS code as an exact string, preserving leading zeros."),
    Variable("highest_intensity_pct", None, "Store the county paydown as a percentage of its reported origination amount, rounded to 4 decimals from unrounded inputs."),
]
DECIMALS = [0, 0, 4, 0, 4, 0, 0, 4, 4, None, 4, None, 4]


@lru_cache(maxsize=1)
def _eligible():
    declarations = load_expansion_table("fema_disasters")
    event = declarations.loc[declarations["disaster_number"].astype(str).eq("4827")]
    if len(event) != 40 or not event["state_usps"].eq("NC").all():
        raise ValueError("disaster geography changed")
    county_rows = event.loc[event["fips_county_code"].astype(str).ne("000")]
    counties = county_rows["county_fips"].astype(str)
    if len(counties) != 39 or counties.duplicated().any() or not counties.str.fullmatch(r"\d{5}").all():
        raise ValueError("ambiguous county list")
    loans = load_expansion_table("hmda")
    selected = loans.loc[
        loans["activity_year"].astype(str).eq("2024")
        & loans["state_code"].astype(str).eq("NC")
        & loans["county_code"].astype(str).isin(set(counties))
        & loans["action_taken"].astype(str).eq("1")
        & loans["loan_type"].astype(str).eq("1")
        & loans["loan_purpose"].astype(str).eq("1")
        & loans["lien_status"].astype(str).eq("1")
        & loans["occupancy_type"].astype(str).eq("1")
        & loans["derived_dwelling_category"].eq("Single Family (1-4 Units):Site-Built")
    , ["county_code", "loan_amount", "property_value"]].copy()
    for column in ("loan_amount", "property_value"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected["loan_amount"])
        & np.isfinite(selected["property_value"])
        & selected["loan_amount"].gt(0)
        & selected["property_value"].gt(0)
    ].copy()
    for column in ("loan_amount", "property_value"):
        if not selected[column].eq(selected[column].round()).all():
            raise ValueError("nonintegral HMDA source amount")
        selected[column] = selected[column].astype("int64")
    if selected.empty:
        raise ValueError("empty origination cohort")
    return len(counties), selected


def analysis():
    county_count, selected = _eligible()
    amount = selected["loan_amount"]
    value = selected["property_value"]
    # Integer-scaled comparisons prevent binary-float ties at 80% and 100%.
    above_eighty = (5 * amount > 4 * value)
    underwater_before = amount > value
    underwater_after = 100 * amount > 88 * value
    collateral_shortfall_cents = (100 * amount - 88 * value).clip(lower=0)
    paydown_mills = (1000 * amount - 704 * value).clip(lower=0)
    county = pd.DataFrame({
        "county": selected["county_code"].astype(str),
        "amount": amount,
        "paydown_mills": paydown_mills,
    }).groupby("county", sort=True).agg(
        record_count=("amount", "size"),
        total_amount=("amount", "sum"),
        total_paydown_mills=("paydown_mills", "sum"),
    )
    dollar_order = county.sort_values(["total_paydown_mills"], ascending=False, kind="stable")
    largest = str(dollar_order.index[0])
    eligible_county = county.loc[county["record_count"].ge(100)].copy()
    eligible_county["intensity"] = eligible_county["total_paydown_mills"] / (1000 * eligible_county["total_amount"])
    intensity_order = eligible_county.sort_values("intensity", ascending=False, kind="stable")
    intensive = str(intensity_order.index[0])
    if len(largest) != 5 or len(intensive) != 5:
        raise ValueError("county identifier lost fixed width")
    return {
        "declared_counties": county_count,
        "eligible_origination_records": len(selected),
        "origination_amount_m": amount.sum() / 1_000_000,
        "above_eighty_records": int(above_eighty.sum()),
        "above_eighty_amount_m": amount.loc[above_eighty].sum() / 1_000_000,
        "stressed_underwater_records": int(underwater_after.sum()),
        "newly_underwater_records": int((underwater_after & ~underwater_before).sum()),
        "stressed_collateral_shortfall_m": collateral_shortfall_cents.sum() / 100_000_000,
        "minimum_paydown_m": paydown_mills.sum() / 1_000_000_000,
        "largest_paydown_county": largest,
        "largest_county_paydown_m": county.loc[largest, "total_paydown_mills"] / 1_000_000_000,
        "highest_intensity_county": intensive,
        "highest_intensity_pct": 100 * eligible_county.loc[intensive, "intensity"],
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[name] for name in names], [truth[name] for name in names], [DECIMALS[[v.name for v in variables].index(name)] for name in names])


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


validators = {f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate(outputs, names)) for i, names in enumerate(TURNS, 1)}
