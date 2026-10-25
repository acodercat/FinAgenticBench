"""Compare bank assets with the nearest Y-9C-reporting controlling ancestor.

For the four fixed banks at 2025-09-30, the current NIC direct-control graph
requires 1, 3, 2 and 2 relationship edges to reach the nearest ancestor with a
Y-9C balance-sheet row.  Their Call Report assets sum to $10,075.815 billion and
the matched holding-company assets sum to $12,669.373 billion, a 79.5289%
ratio of sums.  Wells Fargo Bank has the highest bank-to-parent share at
85.6580%; Citibank has the lowest at 69.7902%, a 15.8678-point spread.
Two of the four Call Report rows have source-recorded last-submission dates
after 2025-12-31; Citibank is latest, on 2026-06-30.  This revision metadata is
not available from the otherwise numerically equivalent FDIC BankFind asset
rows and makes the Call Report source necessary to the complete task.

Stopping at each immediate parent leaves three cohort members without a Y-9C
row.  The domestic-office RCON asset field is unreported for all four banks;
averaging the four bank-level shares or substituting parent liabilities changes
the requested outputs.  The query pins the Y-9C-reporting ancestor, consolidated
asset scope and ratio-of-sums aggregation, so these are regression probes for a
hard baseline rather than live traps.  Ignoring the direct/control/regulatory
filters happens to be numerically equivalent along these four frozen lineages.
"""

from cave_agent import Variable
import pandas as pd

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


REPORT_DATE = "2025-09-30"
BANK_RSSD_IDS = {"852218", "480228", "476810", "451965"}

variables = [
    Variable("matched_bank_parent_pair_count", None, "Store the matched bank-parent pair count as an integer."),
    Variable("jpmorgan_matched_parent_legal_name", None, "Store JPMorgan Chase Bank's matched Y-9C parent legal name; casing and repeated whitespace are not significant."),
    Variable("jpmorgan_lineage_edge_count", None, "Store JPMorgan Chase Bank's lineage-edge count as an integer."),
    Variable("bank_of_america_matched_parent_legal_name", None, "Store Bank of America, N.A.'s matched Y-9C parent legal name; casing and repeated whitespace are not significant."),
    Variable("bank_of_america_lineage_edge_count", None, "Store Bank of America, N.A.'s lineage-edge count as an integer."),
    Variable("citibank_matched_parent_legal_name", None, "Store Citibank's matched Y-9C parent legal name; casing and repeated whitespace are not significant."),
    Variable("citibank_lineage_edge_count", None, "Store Citibank's lineage-edge count as an integer."),
    Variable("wells_fargo_matched_parent_legal_name", None, "Store Wells Fargo Bank's matched Y-9C parent legal name; casing and repeated whitespace are not significant."),
    Variable("wells_fargo_lineage_edge_count", None, "Store Wells Fargo Bank's lineage-edge count as an integer."),
    Variable("total_nic_relationship_edge_count", None, "Store the total relationship-edge count across the four matched lineages as an integer."),
    Variable("cohort_bank_total_assets_usd_billions", None, "Store aggregate bank consolidated total assets in USD billions, rounded to 3 decimals."),
    Variable("cohort_parent_total_assets_usd_billions", None, "Store aggregate matched-parent consolidated total assets in USD billions, rounded to 3 decimals."),
    Variable("cohort_bank_to_parent_asset_ratio_pct", None, "Store the cohort bank-to-parent asset ratio in percent, rounded to 4 decimals."),
    Variable("highest_asset_share_bank_legal_name", None, "Store the highest-share bank legal name; casing and repeated whitespace are not significant."),
    Variable("highest_asset_share_parent_legal_name", None, "Store its matched parent legal name; casing and repeated whitespace are not significant."),
    Variable("highest_asset_share_lineage_edge_count", None, "Store its lineage-edge count as an integer."),
    Variable("highest_bank_to_parent_asset_share_pct", None, "Store the highest bank-to-parent asset share in percent, rounded to 4 decimals."),
    Variable("lowest_asset_share_bank_legal_name", None, "Store the lowest-share bank legal name; casing and repeated whitespace are not significant."),
    Variable("lowest_asset_share_parent_legal_name", None, "Store its matched parent legal name; casing and repeated whitespace are not significant."),
    Variable("lowest_asset_share_lineage_edge_count", None, "Store its lineage-edge count as an integer."),
    Variable("lowest_bank_to_parent_asset_share_pct", None, "Store the lowest bank-to-parent asset share in percent, rounded to 4 decimals."),
    Variable("post_2025_cutoff_call_report_update_count", None, "Store the count of selected Call Report rows whose source-recorded last-submission date is after 2025-12-31 as an integer."),
    Variable("latest_updated_call_report_bank_legal_name", None,
       "Store the legal name of the selected bank with the latest source-recorded Call Report "
       "submission-update date; casing and repeated whitespace are not significant."),
    Variable("latest_call_report_submission_update_date", None, "Store that latest source-recorded submission-update date as a YYYY-MM-DD string."),
]


def _normalized_label(value):
    if not isinstance(value, str):
        return value
    return " ".join(value.split()).casefold()


def _nearest_y9c_ancestor(bank_rssd_id, relationships, y9c_ids):
    current = bank_rssd_id
    visited = {current}
    edge_count = 0
    while current not in y9c_ids:
        parents = relationships.loc[
            relationships.offspring_rssd_id.eq(current), "parent_rssd_id"
        ].drop_duplicates()
        if len(parents) != 1:
            raise ValueError(
                f"expected one direct controlling parent for {current}, found {len(parents)}"
            )
        current = str(parents.iloc[0])
        edge_count += 1
        if current in visited:
            raise ValueError("cycle in NIC controlling-relationship graph")
        visited.add(current)
        if edge_count > 10:
            raise ValueError("unexpectedly deep NIC controlling lineage")
    return current, edge_count


def _comparison():
    banks = load_expansion_table("ffiec_call_reports_balance").loc[
        lambda frame: frame.report_date.eq(REPORT_DATE)
        & frame.bank_rssd_id.isin(BANK_RSSD_IDS)
    ].copy()
    if len(banks) != 4 or banks.bank_rssd_id.nunique() != 4:
        raise ValueError("unexpected fixed-bank Call Report cohort")
    if banks.consolidated_total_assets_thousand_usd.isna().any():
        raise ValueError("missing consolidated bank assets")

    y9c = load_expansion_table("ffiec_y9c").loc[
        lambda frame: frame.report_date.eq(REPORT_DATE)
    ].copy()
    if y9c.reporter_rssd_id.duplicated().any():
        raise ValueError("duplicate Y-9C reporter")
    y9c_ids = set(y9c.reporter_rssd_id)

    relationships = load_expansion_table("ffiec_nic_ownership").loc[
        lambda frame: frame.relationship_level.eq(1)
        & frame.control_indicator.eq(1)
        & frame.regulatory_relationship_indicator.eq(1)
    ].copy()

    matched = []
    for bank in banks.itertuples(index=False):
        parent_rssd_id, edges = _nearest_y9c_ancestor(
            str(bank.bank_rssd_id), relationships, y9c_ids
        )
        parent = y9c.loc[y9c.reporter_rssd_id.eq(parent_rssd_id)]
        if len(parent) != 1:
            raise ValueError("expected one matched Y-9C parent")
        parent = parent.iloc[0]
        matched.append(
            {
                "bank_rssd_id": str(bank.bank_rssd_id),
                "bank_legal_name": bank.bank_legal_name,
                "parent_legal_name": parent.reporter_legal_name,
                "lineage_edge_count": edges,
                "bank_assets": float(bank.consolidated_total_assets_thousand_usd),
                "parent_assets": float(parent.consolidated_total_assets_thousand_usd),
                "submission_update_date": str(bank.last_submission_update)[:10],
            }
        )
    result = pd.DataFrame(matched)
    result["asset_share_pct"] = result.bank_assets / result.parent_assets * 100
    return result


def ground_truth():
    result = _comparison()
    by_rssd = result.set_index("bank_rssd_id")
    highest = result.sort_values(
        ["asset_share_pct", "bank_legal_name"], ascending=[False, True]
    ).iloc[0]
    lowest = result.sort_values(
        ["asset_share_pct", "bank_legal_name"], ascending=[True, True]
    ).iloc[0]
    bank_total = result.bank_assets.sum()
    parent_total = result.parent_assets.sum()
    post_cutoff = result.loc[result.submission_update_date.gt("2025-12-31")]
    latest_updated = result.sort_values(
        ["submission_update_date", "bank_legal_name"], ascending=[False, True]
    ).iloc[0]
    return (
        len(result),
        by_rssd.loc["852218", "parent_legal_name"],
        int(by_rssd.loc["852218", "lineage_edge_count"]),
        by_rssd.loc["480228", "parent_legal_name"],
        int(by_rssd.loc["480228", "lineage_edge_count"]),
        by_rssd.loc["476810", "parent_legal_name"],
        int(by_rssd.loc["476810", "lineage_edge_count"]),
        by_rssd.loc["451965", "parent_legal_name"],
        int(by_rssd.loc["451965", "lineage_edge_count"]),
        int(result.lineage_edge_count.sum()),
        bank_total / 1e6,
        parent_total / 1e6,
        bank_total / parent_total * 100,
        highest.bank_legal_name,
        highest.parent_legal_name,
        int(highest.lineage_edge_count),
        highest.asset_share_pct,
        lowest.bank_legal_name,
        lowest.parent_legal_name,
        int(lowest.lineage_edge_count),
        lowest.asset_share_pct,
        len(post_cutoff),
        latest_updated.bank_legal_name,
        latest_updated.submission_update_date,
    )


DECIMALS = [
    0, None, 0, None, 0, None, 0, None, 0, 0, 3, 3, 4, None, None, 0, 4, None, None, 0, 4, 0,
    None, None,
]
LEGAL_NAME_OUTPUTS = (
    "jpmorgan_matched_parent_legal_name", "bank_of_america_matched_parent_legal_name",
    "citibank_matched_parent_legal_name", "wells_fargo_matched_parent_legal_name",
    "highest_asset_share_bank_legal_name", "highest_asset_share_parent_legal_name",
    "lowest_asset_share_bank_legal_name", "lowest_asset_share_parent_legal_name",
    "latest_updated_call_report_bank_legal_name",
)


TURN_1_NAMES = [
    "matched_bank_parent_pair_count",
    "jpmorgan_matched_parent_legal_name", "jpmorgan_lineage_edge_count",
    "bank_of_america_matched_parent_legal_name", "bank_of_america_lineage_edge_count",
    "citibank_matched_parent_legal_name", "citibank_lineage_edge_count",
    "wells_fargo_matched_parent_legal_name", "wells_fargo_lineage_edge_count",
    "total_nic_relationship_edge_count",
]
TURN_2_NAMES = [
    "cohort_bank_total_assets_usd_billions", "cohort_parent_total_assets_usd_billions",
    "cohort_bank_to_parent_asset_ratio_pct",
]
TURN_3_NAMES = [
    "highest_asset_share_bank_legal_name", "highest_asset_share_parent_legal_name",
    "highest_asset_share_lineage_edge_count", "highest_bank_to_parent_asset_share_pct",
    "lowest_asset_share_bank_legal_name", "lowest_asset_share_parent_legal_name",
    "lowest_asset_share_lineage_edge_count", "lowest_bank_to_parent_asset_share_pct",
]
TURN_4_NAMES = [
    "post_2025_cutoff_call_report_update_count",
    "latest_updated_call_report_bank_legal_name", "latest_call_report_submission_update_date",
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    candidate = dict(outputs)
    for name in LEGAL_NAME_OUTPUTS:
        if _normalized_label(candidate.get(name)) == _normalized_label(truth[name]):
            candidate[name] = truth[name]
    return validate_ordered_outputs(
        candidate,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_lineage_match": turn_validator(validate_turn_1),
    "validate_scope_aggregate": turn_validator(validate_turn_2),
    "validate_share_extremes": turn_validator(validate_turn_3),
    "validate_submission_vintage": turn_validator(validate_turn_4),
}
