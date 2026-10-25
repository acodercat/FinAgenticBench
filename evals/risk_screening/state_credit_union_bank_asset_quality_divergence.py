"""Screen state credit-union deterioration and benchmark bank asset quality.

The baseline builds a two-date balanced NCUA panel and selects the state or
territory with the largest increase in the ratio of delinquent loans two months
or more to total loans.  A current NIC certificate-to-state bridge then locates
the matching balanced FDIC bank panel and measures the change in noncurrent loans
to net loans.  The case identifies the credit union driving the selected state's
dollar increase and applies the bank-panel basis-point change as a transparent
counterfactual benchmark to that institution's ending loan balance.

The NCUA and FDIC delinquency concepts are not identical, and the NIC geography
is a current identifier attribute rather than a historical organization record.
The final calculation is a sensitivity, not an observed bank-equivalent outcome.
The query pins balanced populations, ratio-of-sums aggregation, direction and
tie rules; unbalanced panels, means of institution ratios, gross-loan denominators
and treating NIC as historical are regression probes.  This is a hard baseline
with one canonical result per output.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


START_DATE = "2024-09-30"
END_DATE = "2025-09-30"

TURN_1_NAMES = [
    "selected_state_code", "balanced_credit_union_count",
    "credit_union_delinquency_change_bp",
]
TURN_2_NAMES = ["balanced_bank_count", "bank_noncurrent_change_bp"]
TURN_3_NAMES = [
    "credit_union_number", "credit_union_delinquent_increase_usd_millions",
    "leader_share_of_state_increase_pct",
]
TURN_4_NAMES = [
    "bank_change_counterfactual_delinquency_pct",
    "bank_change_counterfactual_delinquent_usd_millions",
    "actual_minus_counterfactual_delinquent_usd_millions",
]

variables = [
    Variable(
        TURN_1_NAMES[0], None,
        "Store the selected two-letter NCUA state or territory code verbatim as an uppercase string.",
    ),
    Variable(
        TURN_1_NAMES[1], None,
        "Store the number of credit unions in the selected balanced panel as an integer.",
    ),
    Variable(
        TURN_1_NAMES[2], None,
        "Store the signed ending-minus-starting delinquency-rate change in basis points, rounded to 4 decimals.",
    ),
    Variable(
        TURN_2_NAMES[0], None,
        "Store the number of banks in the selected balanced panel as an integer.",
    ),
    Variable(
        TURN_2_NAMES[1], None,
        "Store the signed ending-minus-starting bank noncurrent-loan-rate change in basis points, rounded to 4 decimals.",
    ),
    Variable(
        TURN_3_NAMES[0], None,
        "Store the selected NCUA credit union number verbatim as a string.",
    ),
    Variable(
        TURN_3_NAMES[1], None,
        "Store the signed delinquent-loan increase in USD millions, rounded to 4 decimals.",
    ),
    Variable(
        TURN_3_NAMES[2], None,
        "Store the selected credit union's share of the state's delinquent-dollar increase in percent, rounded to 4 decimals.",
    ),
    Variable(
        TURN_4_NAMES[0], None,
        "Store the counterfactual delinquency ratio in percent, rounded to 4 decimals.",
    ),
    Variable(
        TURN_4_NAMES[1], None,
        "Store the counterfactual delinquent-loan amount in USD millions, rounded to 4 decimals.",
    ),
    Variable(
        TURN_4_NAMES[2], None,
        "Store actual ending delinquent loans minus the counterfactual amount in USD millions, rounded to 4 decimals.",
    ),
]

DECIMALS = [None, 0, 4, 0, 4, None, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def _credit_union_pairs() -> pd.DataFrame:
    frame = load_expansion_table("ncua_call_reports").copy()
    eligible = frame.loc[
        frame.report_date.isin([START_DATE, END_DATE])
        & frame.total_loans_and_leases_usd.gt(0)
        & frame.delinquent_loans_two_plus_months_usd.ge(0)
    ].copy()
    start = eligible.loc[eligible.report_date.eq(START_DATE)]
    end = eligible.loc[eligible.report_date.eq(END_DATE)]
    pairs = start.merge(
        end, on="credit_union_number", suffixes=("_start", "_end"),
        validate="one_to_one",
    )
    return pairs.loc[pairs.state_start.eq(pairs.state_end)].copy()


def _state_screen(pairs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for state, rows in pairs.groupby("state_end", observed=True):
        start_rate = (
            rows.delinquent_loans_two_plus_months_usd_start.sum()
            / rows.total_loans_and_leases_usd_start.sum()
        )
        end_rate = (
            rows.delinquent_loans_two_plus_months_usd_end.sum()
            / rows.total_loans_and_leases_usd_end.sum()
        )
        records.append({
            "state": str(state), "count": len(rows),
            "change_bp": float((end_rate - start_rate) * 10_000),
        })
    return pd.DataFrame(records).sort_values(
        ["change_bp", "state"], ascending=[False, True]
    )


def _bank_pairs(state: str) -> pd.DataFrame:
    banks = load_expansion_table("fdic_bankfind").copy()
    eligible = banks.loc[
        banks.REPDTE.astype(str).isin([START_DATE, END_DATE])
        & banks.LNLSNET.gt(0)
        & banks.NCLNLS.ge(0)
    ].copy()
    eligible["certificate"] = eligible.CERT.astype("Int64").astype("string")
    start = eligible.loc[eligible.REPDTE.astype(str).eq(START_DATE)]
    end = eligible.loc[eligible.REPDTE.astype(str).eq(END_DATE)]
    pairs = start.merge(
        end, on="certificate", suffixes=("_start", "_end"), validate="one_to_one"
    )
    institutions = load_expansion_table("ffiec_nic_institutions")
    mapping = institutions.loc[
        institutions.fdic_certificate.notna() & institutions.state.notna(),
        ["fdic_certificate", "state"],
    ].copy()
    if mapping.fdic_certificate.duplicated().any():
        raise ValueError("NIC FDIC-certificate mapping is not unique")
    pairs = pairs.merge(
        mapping, left_on="certificate", right_on="fdic_certificate",
        how="left", validate="many_to_one",
    )
    return pairs.loc[pairs.state.eq(state)].copy()


@lru_cache(maxsize=1)
def _trajectory() -> dict[str, object]:
    credit_unions = _credit_union_pairs()
    state_screen = _state_screen(credit_unions)
    selected = state_screen.iloc[0]
    state = str(selected.state)
    state_credit_unions = credit_unions.loc[
        credit_unions.state_end.eq(state)
    ].copy()
    state_credit_unions["delinquent_increase"] = (
        state_credit_unions.delinquent_loans_two_plus_months_usd_end
        - state_credit_unions.delinquent_loans_two_plus_months_usd_start
    )
    state_credit_unions = state_credit_unions.sort_values(
        ["delinquent_increase", "credit_union_number"],
        ascending=[False, True],
    )
    leader = state_credit_unions.iloc[0]
    state_increase = float(state_credit_unions.delinquent_increase.sum())
    if state_increase <= 0:
        raise ValueError("selected state has no positive delinquent-dollar increase")

    banks = _bank_pairs(state)
    bank_start_rate = banks.NCLNLS_start.sum() / banks.LNLSNET_start.sum()
    bank_end_rate = banks.NCLNLS_end.sum() / banks.LNLSNET_end.sum()
    bank_change_bp = float((bank_end_rate - bank_start_rate) * 10_000)

    credit_union_start_rate = (
        leader.delinquent_loans_two_plus_months_usd_start
        / leader.total_loans_and_leases_usd_start
    )
    counterfactual_rate = credit_union_start_rate + bank_change_bp / 10_000
    counterfactual_amount = (
        counterfactual_rate * leader.total_loans_and_leases_usd_end
    )
    actual_amount = float(leader.delinquent_loans_two_plus_months_usd_end)
    return {
        "state": state,
        "credit_union_count": int(selected["count"]),
        "credit_union_change_bp": float(selected.change_bp),
        "bank_count": len(banks),
        "bank_change_bp": bank_change_bp,
        "credit_union_number": str(leader.credit_union_number),
        "credit_union_increase_millions": float(leader.delinquent_increase) / 1e6,
        "leader_share_pct": float(leader.delinquent_increase) / state_increase * 100,
        "counterfactual_rate_pct": float(counterfactual_rate * 100),
        "counterfactual_amount_millions": float(counterfactual_amount) / 1e6,
        "actual_gap_millions": (actual_amount - float(counterfactual_amount)) / 1e6,
    }


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    result = _trajectory()
    return (
        result["state"], result["credit_union_count"], result["credit_union_change_bp"],
        result["bank_count"], result["bank_change_bp"],
        result["credit_union_number"], result["credit_union_increase_millions"],
        result["leader_share_pct"], result["counterfactual_rate_pct"],
        result["counterfactual_amount_millions"], result["actual_gap_millions"],
    )


def _validate_subset(outputs: dict, names: list[str]):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip((variable.name for variable in variables), ground_truth()))
    places = dict(zip((variable.name for variable in variables), DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [places[name] for name in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_state_screen": turn_validator(lambda outputs: _validate_subset(outputs, TURN_1_NAMES)),
    "validate_bank_benchmark": turn_validator(lambda outputs: _validate_subset(outputs, TURN_2_NAMES)),
    "validate_credit_union_driver": turn_validator(lambda outputs: _validate_subset(outputs, TURN_3_NAMES)),
    "validate_counterfactual": turn_validator(lambda outputs: _validate_subset(outputs, TURN_4_NAMES)),
}
