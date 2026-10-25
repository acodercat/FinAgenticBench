"""Separate a bank CET1 denominator comparison from its parent's stress path.

The retained 2025-Q3 Call Report identifies JPMorgan Chase Bank, N.A. by RSSD
852218.  The current NIC snapshot observed 2026-08-13 maps that bank to immediate
parent JPMorgan Chase & Co., RSSD 1039502 and entity type FHD, on a relationship-
level-1 row.  That parent, not the bank, is the legal entity in the 2025
supervisory severely adverse stress results.  Joining the bank RSSD directly to
the stress table returns zero rows.  Omitting the relationship-level filter gives
the same parent in this frozen snapshot because the bank has only one current
controlled/regulatory relationship; that numerical equivalence does not make the
unfiltered rule a general definition of immediate parent.

At the bank, $291.288 billion of CET1 divided by $1,856.46605 billion of
standardized RWA and $1,737.022 billion of advanced-approaches RWA gives
15.6904566% and 16.7693904%, a 1.0789338 percentage-point gap.  At the parent,
the stress disclosure reports 15.7% actual CET1 as of 2024-12-31 and a 14.2%
projected minimum, a -1.5 percentage-point change.  Substituting the nearby Tier
1 and supplementary leverage ratios gives 7.8478% and 6.4731%, while selecting
the stress end ratio instead of the projected minimum gives 15.8%; these are
measured regression probes, not designated traps because the query pins the
requested measures.

Numeric validation uses 0.6 × 10^-N rounding-boundary tolerance for N requested
decimals: three for USD billions, four for bank ratios and their gap, and one for
stress ratios and their change. Identifiers, the entity-type code and the date
match exactly; parent-name capitalization and repeated whitespace are normalized.
"""

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


BANK_DATE = "2025-09-30"
BANK_NAME = "JPMORGAN CHASE BANK, NATIONAL ASSOCIATION"
STRESS_SCENARIO = "Supervisory Severely Adverse"

variables = [
    Variable("bank_rssd_id", None, "Store the bank RSSD ID as a string."),
    Variable("parent_rssd_id", None, "Store the immediate-parent RSSD ID as a string."),
    Variable("parent_legal_name", None, "Store the immediate parent's legal name; capitalization and repeated whitespace are not significant."),
    Variable("parent_nic_entity_type", None, "Store the immediate parent's NIC entity-type code."),
    Variable("cet1_capital_billion_usd", None, "Store bank common equity tier 1 capital in USD billions, rounded to 3 decimals."),
    Variable("standardized_rwa_billion_usd", None, "Store bank standardized-approach risk-weighted assets in USD billions, rounded to 3 decimals."),
    Variable("advanced_rwa_billion_usd", None, "Store bank advanced-approaches risk-weighted assets in USD billions, rounded to 3 decimals."),
    Variable("standardized_cet1_ratio_pct", None, "Store the bank standardized-approach CET1 ratio in percent, rounded to 4 decimals."),
    Variable("advanced_cet1_ratio_pct", None, "Store the bank advanced-approaches CET1 ratio in percent, rounded to 4 decimals."),
    Variable("stress_actual_as_of_date", None, "Store the stress disclosure's actual-ratio date as a YYYY-MM-DD string."),
    Variable("parent_stress_actual_cet1_ratio_pct", None, "Store the parent's actual CET1 ratio in percent, rounded to 1 decimal."),
    Variable("parent_stress_projected_minimum_cet1_ratio_pct", None, "Store the parent's projected minimum CET1 ratio in percent, rounded to 1 decimal."),
]


def _one(frame, description):
    if len(frame) != 1:
        raise ValueError(f"expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def ground_truth():
    banks = load_expansion_table("ffiec_call_reports_capital")
    bank = _one(
        banks.loc[banks.report_date.eq(BANK_DATE) & banks.bank_name.eq(BANK_NAME)],
        "JPMorgan bank Call Report",
    )

    relationships = load_expansion_table("ffiec_nic_ownership")
    relationship = _one(
        relationships.loc[
            relationships.offspring_rssd_id.eq(bank.bank_id)
            & relationships.relationship_level.eq(1)
            & relationships.control_indicator.eq(1)
            & relationships.regulatory_relationship_indicator.eq(1)
        ],
        "current NIC immediate-parent relationship",
    )
    institutions = load_expansion_table("ffiec_nic_institutions")
    parent = _one(
        institutions.loc[institutions.rssd_id.eq(relationship.parent_rssd_id)],
        "current NIC parent institution",
    )
    if parent.legal_name != relationship.parent_legal_name:
        raise ValueError("NIC parent legal names disagree across retained tables")

    stress = load_expansion_table("fed_stress_tests")
    stress_row = _one(
        stress.loc[
            stress.id_rssd.eq(parent.rssd_id)
            & stress.exercise_name.eq("2025 Stress Test")
            & stress.scenario_name.eq(STRESS_SCENARIO)
        ],
        "matched parent stress-test",
    )

    capital = bank.common_equity_tier1_capital_thousand_usd
    standardized_rwa = bank.standardized_risk_weighted_assets_thousand_usd
    advanced_rwa = bank.advanced_risk_weighted_assets_thousand_usd
    standardized_ratio = capital / standardized_rwa * 100
    advanced_ratio = capital / advanced_rwa * 100
    actual = stress_row.common_equity_tier1_actual_rat
    projected_minimum = stress_row.common_equity_tier1_min_rat
    actual_date = pd.to_datetime(stress_row.dt_exercise_quarter).strftime("%Y-%m-%d")
    return (
        str(bank.bank_id),
        str(parent.rssd_id),
        str(parent.legal_name),
        str(parent.entity_type),
        capital / 1e6,
        standardized_rwa / 1e6,
        advanced_rwa / 1e6,
        standardized_ratio,
        advanced_ratio,
        actual_date,
        actual,
        projected_minimum,
    )


def validate(outputs):
    expected = ground_truth()
    candidate = dict(outputs)
    actual_name = candidate.get("parent_legal_name")
    expected_name = expected[2]
    if isinstance(actual_name, str):
        normalized_actual = " ".join(actual_name.split()).casefold()
        normalized_expected = " ".join(expected_name.split()).casefold()
        if normalized_actual == normalized_expected:
            candidate["parent_legal_name"] = expected_name
    return validate_ordered_outputs(
        candidate,
        variables,
        expected,
        [
    None, None, None, None, 3, 3, 3, 4, 4, None, 1, 1,
],
    )



TURN_1_NAMES = [
    "bank_rssd_id",
    "cet1_capital_billion_usd",
    "standardized_rwa_billion_usd",
    "advanced_rwa_billion_usd",
    "standardized_cet1_ratio_pct",
    "advanced_cet1_ratio_pct",
]
TURN_2_NAMES = [
    "parent_rssd_id",
    "parent_legal_name",
    "parent_nic_entity_type",
]
TURN_3_NAMES = [
    "stress_actual_as_of_date",
    "parent_stress_actual_cet1_ratio_pct",
    "parent_stress_projected_minimum_cet1_ratio_pct",
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
    "validate_bank_capital": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_parent_identity": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_stress_disclosure": turn_validator(_turn_subset(TURN_3_NAMES)),
}
