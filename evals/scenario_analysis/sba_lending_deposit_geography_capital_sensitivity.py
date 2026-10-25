"""Bridge SBA lending geography to deposits, employment and bank equity.

For FY2024 SBA 7(a) lenders with at least 500 reported rows and a 2025 SOD
match, Northeast Bank has the greatest total-variation distance between its
state approval-dollar distribution and its branch-deposit distribution:
99.743974%, versus 99.146516% for the runner-up.  Its deposits are entirely
reported in Maine while approvals span all 50 states and DC.

This four-turn hard baseline then overlays states with negative 2024-to-2025
private QCEW employment growth and runs a mechanical 10% loss sensitivity on
unguaranteed approval dollars against domestic-office Call Report equity.  The
query pins all-row preservation, the 51-jurisdiction population, ratio-of-sums,
TVD, preliminary retrospective QCEW levels and the hypothetical loss rule.
Deduplicating public SBA rows, averaging row shares, using rounded QCEW change
fields, or treating approval flow as retained exposure are regression paths, not
unannounced traps.

Fragility: the total-variation-distance leader is 0.60 percentage points ahead
of the runner-up at 99.74 percent, and it is a bank with Maine deposits and a
national SBA book.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table, qcew_state_totals
from core.validation import turn_validator, validate_ordered_outputs


STATE_USPS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}


TURN_1_NAMES = [
    "eligible_sba_sod_lender_count", "geography_mismatch_leader_bank_name",
    "geography_mismatch_leader_fdic_certificate", "leader_fy2024_sba_row_count",
    "leader_fy2024_gross_approval_usd_billions", "leader_sod_deposits_usd_billions",
    "leader_sba_approval_jurisdiction_count", "leader_sod_deposit_jurisdiction_count",
    "leader_geography_tvd_pct", "leader_geography_overlap_pct", "tvd_leader_margin_pp",
]
TURN_2_NAMES = [
    "leader_largest_approval_state", "largest_approval_state_approval_usd_billions",
    "largest_approval_state_approval_share_pct", "largest_approval_state_deposits_usd_billions",
    "largest_approval_state_deposit_share_pct", "leader_largest_deposit_state",
    "largest_deposit_state_deposits_usd_billions", "largest_deposit_state_deposit_share_pct",
    "largest_deposit_state_approval_usd_billions", "largest_deposit_state_approval_share_pct",
]
TURN_3_NAMES = [
    "matched_qcew_jurisdiction_count", "negative_private_employment_jurisdiction_count",
    "leader_negative_employment_state_sba_row_count",
    "leader_negative_employment_state_gross_approval_usd_billions",
    "leader_negative_employment_state_approval_share_pct",
    "leader_negative_employment_state_unguaranteed_approval_usd_billions",
    "leader_negative_employment_state_deposits_usd_billions",
    "leader_negative_employment_state_deposit_share_pct",
    "largest_negative_state_approval_state", "largest_negative_state_private_employment_growth_pct",
]
TURN_4_NAMES = [
    "selected_bank_call_report_match_count", "selected_bank_rssd_id",
    "selected_bank_call_report_legal_name", "selected_bank_domestic_equity_usd_billions",
    "hypothetical_sba_loss_usd_billions", "hypothetical_pro_forma_equity_usd_billions",
    "hypothetical_equity_reduction_pct",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("eligible_sba_sod_lender_count", "Store the eligible matched-lender count as an integer."),
    _v("geography_mismatch_leader_bank_name", "Store the selected SBA lender name exactly as reported."),
    _v("geography_mismatch_leader_fdic_certificate", "Store the selected FDIC certificate number as a string."),
    _v("leader_fy2024_sba_row_count", "Store the selected lender's reported FY2024 SBA row count as an integer."),
    _v("leader_fy2024_gross_approval_usd_billions", "Store gross approvals in USD billions, rounded to 6 decimals."),
    _v("leader_sod_deposits_usd_billions", "Store branch deposits in USD billions, rounded to 6 decimals."),
    _v("leader_sba_approval_jurisdiction_count", "Store the positive-approval jurisdiction count as an integer."),
    _v("leader_sod_deposit_jurisdiction_count", "Store the positive-deposit jurisdiction count as an integer."),
    _v("leader_geography_tvd_pct", "Store total-variation distance in percent, rounded to 4 decimals."),
    _v("leader_geography_overlap_pct", "Store the jurisdiction-wise minimum-share overlap in percent, rounded to 4 decimals."),
    _v("tvd_leader_margin_pp", "Store the leader's TVD margin over the runner-up in percentage points, rounded to 4 decimals."),
    _v("leader_largest_approval_state", "Store the largest-approval state USPS abbreviation."),
    _v("largest_approval_state_approval_usd_billions", "Store that state's approvals in USD billions, rounded to 6 decimals."),
    _v("largest_approval_state_approval_share_pct", "Store that state's approval share in percent, rounded to 4 decimals."),
    _v("largest_approval_state_deposits_usd_billions", "Store that state's deposits in USD billions, rounded to 6 decimals."),
    _v("largest_approval_state_deposit_share_pct", "Store that state's deposit share in percent, rounded to 4 decimals."),
    _v("leader_largest_deposit_state", "Store the largest-deposit state USPS abbreviation."),
    _v("largest_deposit_state_deposits_usd_billions", "Store that state's deposits in USD billions, rounded to 6 decimals."),
    _v("largest_deposit_state_deposit_share_pct", "Store that state's deposit share in percent, rounded to 4 decimals."),
    _v("largest_deposit_state_approval_usd_billions", "Store that state's approvals in USD billions, rounded to 6 decimals."),
    _v("largest_deposit_state_approval_share_pct", "Store that state's approval share in percent, rounded to 4 decimals."),
    _v("matched_qcew_jurisdiction_count", "Store the matched state-level QCEW jurisdiction count as an integer."),
    _v("negative_private_employment_jurisdiction_count", "Store the negative-growth jurisdiction count as an integer."),
    _v("leader_negative_employment_state_sba_row_count", "Store the selected lender's SBA row count in negative-growth jurisdictions as an integer."),
    _v("leader_negative_employment_state_gross_approval_usd_billions", "Store gross approvals in negative-growth jurisdictions in USD billions, rounded to 6 decimals."),
    _v("leader_negative_employment_state_approval_share_pct", "Store the negative-growth-jurisdiction approval share in percent, rounded to 4 decimals."),
    _v("leader_negative_employment_state_unguaranteed_approval_usd_billions", "Store gross minus SBA-guaranteed approvals in negative-growth jurisdictions in USD billions, rounded to 6 decimals."),
    _v("leader_negative_employment_state_deposits_usd_billions", "Store deposits in negative-growth jurisdictions in USD billions, rounded to 6 decimals."),
    _v("leader_negative_employment_state_deposit_share_pct", "Store the negative-growth-jurisdiction deposit share in percent, rounded to 4 decimals."),
    _v("largest_negative_state_approval_state", "Store the negative-growth jurisdiction with the largest selected-lender approvals as a USPS abbreviation."),
    _v("largest_negative_state_private_employment_growth_pct", "Store that jurisdiction's private-employment growth in percent, rounded to 4 decimals."),
    _v("selected_bank_call_report_match_count", "Store the matching Call Report row count as an integer."),
    _v("selected_bank_rssd_id", "Store the selected bank's RSSD ID as a string."),
    _v("selected_bank_call_report_legal_name", "Store the selected bank legal name exactly as reported in the Call Report."),
    _v("selected_bank_domestic_equity_usd_billions", "Store domestic-office total equity in USD billions, rounded to 6 decimals."),
    _v("hypothetical_sba_loss_usd_billions", "Store the hypothetical loss in USD billions, rounded to 6 decimals."),
    _v("hypothetical_pro_forma_equity_usd_billions", "Store hypothetical pro forma equity in USD billions, rounded to 6 decimals."),
    _v("hypothetical_equity_reduction_pct", "Store hypothetical loss divided by reported equity in percent, rounded to 4 decimals."),
]


def _state_bridge(sod):
    bridge = sod.assign(state_fips=sod.branch_county_fips.str[:2])[
        ["branch_state", "state_fips"]
    ].drop_duplicates().rename(columns={"branch_state": "USPS"})
    bridge = bridge.loc[bridge.USPS.isin(STATE_USPS)].copy()
    if len(bridge) != 51 or bridge.USPS.nunique() != 51 or bridge.state_fips.nunique() != 51:
        raise ValueError("expected 50 states plus District of Columbia")
    return bridge.sort_values("USPS").reset_index(drop=True)


@lru_cache(maxsize=1)
def _analysis():
    sod = load_expansion_table("fdic_sod")
    sod = sod.loc[sod.report_date.eq("2025-06-30")].copy()
    bridge = _state_bridge(sod)
    jurisdictions = set(bridge.USPS)
    sba = load_expansion_table("sba_7a")
    sba = sba.loc[
        sba.ApprovalFY.eq(2024) & sba.ProjectState.isin(jurisdictions)
        & sba.fdic_certificate.notna()
    ].copy()
    sba["certificate"] = sba.fdic_certificate.astype("string")
    sod = sod.loc[sod.branch_state.isin(jurisdictions)].copy()
    sod["certificate"] = sod.fdic_certificate.astype("string")
    counts = sba.groupby("certificate").size()
    eligible = sorted(set(counts.loc[counts.ge(500)].index) & set(sod.certificate))
    approval = sba.loc[sba.certificate.isin(eligible)].groupby(["certificate", "ProjectState"], as_index=False).GrossApproval.sum()
    deposits = sod.loc[sod.certificate.isin(eligible)].groupby(["certificate", "branch_state"], as_index=False).branch_deposits_thousand_usd.sum()
    deposits["deposits_usd"] = deposits.branch_deposits_thousand_usd * 1_000
    deposits = deposits.rename(columns={"branch_state": "ProjectState"})
    ranked = []
    grids = {}
    for certificate in eligible:
        grid = bridge[["USPS"]].rename(columns={"USPS": "ProjectState"}).merge(
            approval.loc[approval.certificate.eq(certificate), ["ProjectState", "GrossApproval"]], on="ProjectState", how="left"
        ).merge(
            deposits.loc[deposits.certificate.eq(certificate), ["ProjectState", "deposits_usd"]], on="ProjectState", how="left"
        ).fillna({"GrossApproval": 0, "deposits_usd": 0})
        grid["approval_share"] = grid.GrossApproval / grid.GrossApproval.sum()
        grid["deposit_share"] = grid.deposits_usd / grid.deposits_usd.sum()
        tvd = float(0.5 * (grid.approval_share - grid.deposit_share).abs().sum())
        overlap = float(np.minimum(grid.approval_share, grid.deposit_share).sum())
        rows = sba.loc[sba.certificate.eq(certificate)]
        name_values = rows.BankName.dropna().drop_duplicates()
        if len(name_values) != 1:
            raise ValueError(f"ambiguous SBA lender name for {certificate}")
        ranked.append({"certificate": certificate, "bank_name": str(name_values.iloc[0]), "tvd": tvd, "overlap": overlap})
        grids[certificate] = grid
    ranking = pd.DataFrame(ranked).sort_values(["tvd", "bank_name", "certificate"], ascending=[False, True, True]).reset_index(drop=True)
    leader = ranking.iloc[0]
    certificate = str(leader.certificate)
    leader_sba = sba.loc[sba.certificate.eq(certificate)].copy()
    leader_sod = sod.loc[sod.certificate.eq(certificate)].copy()
    grid = grids[certificate]

    qcew = qcew_state_totals(load_expansion_table("bls_qcew"))
    employment = qcew.loc[qcew.ownership_scope.eq("private") & qcew.year.isin([2024, 2025])].pivot(index="state_fips", columns="year", values="annual_avg_employment").dropna()
    employment["growth"] = (employment[2025] / employment[2024] - 1) * 100
    employment = bridge.merge(employment[["growth"]], on="state_fips", how="inner")
    if len(employment) != 51:
        raise ValueError("QCEW state population is incomplete")
    negative = employment.loc[employment.growth.lt(0)].copy()
    negative_states = set(negative.USPS)
    negative_sba = leader_sba.loc[leader_sba.ProjectState.isin(negative_states)].copy()
    negative_sod = leader_sod.loc[leader_sod.branch_state.isin(negative_states)].copy()
    negative_by_state = negative_sba.groupby("ProjectState", as_index=False).GrossApproval.sum().merge(
        negative[["USPS", "growth"]], left_on="ProjectState", right_on="USPS", how="left"
    ).sort_values(["GrossApproval", "ProjectState"], ascending=[False, True])

    call = load_expansion_table("ffiec_call_reports_balance")
    call_match = call.loc[call.fdic_certificate.eq(certificate) & call.report_date.eq("2024-12-31")]
    if len(call_match) != 1:
        raise ValueError("expected one selected-bank Call Report")
    return ranking, leader, leader_sba, leader_sod, grid, employment, negative, negative_sba, negative_sod, negative_by_state, call_match


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    ranking, leader, sba, sod, grid, employment, negative, negative_sba, negative_sod, negative_by_state, call_match = _analysis()
    approval_rank = grid.sort_values(["GrossApproval", "ProjectState"], ascending=[False, True]).iloc[0]
    deposit_rank = grid.sort_values(["deposits_usd", "ProjectState"], ascending=[False, True]).iloc[0]
    total_approval = float(grid.GrossApproval.sum())
    total_deposits = float(grid.deposits_usd.sum())
    unguaranteed = float((negative_sba.GrossApproval - negative_sba.SBAGuaranteedApproval).sum())
    call = call_match.iloc[0]
    equity = float(call.domestic_office_total_equity_thousand_usd) * 1_000
    loss = unguaranteed * 0.10
    return (
        len(ranking), str(leader.bank_name), str(leader.certificate), len(sba), total_approval / 1e9,
        total_deposits / 1e9, int(grid.GrossApproval.gt(0).sum()), int(grid.deposits_usd.gt(0).sum()),
        float(leader.tvd) * 100, float(leader.overlap) * 100,
        float(leader.tvd - ranking.iloc[1].tvd) * 100,
        str(approval_rank.ProjectState), float(approval_rank.GrossApproval) / 1e9,
        float(approval_rank.approval_share) * 100, float(approval_rank.deposits_usd) / 1e9,
        float(approval_rank.deposit_share) * 100, str(deposit_rank.ProjectState),
        float(deposit_rank.deposits_usd) / 1e9, float(deposit_rank.deposit_share) * 100,
        float(deposit_rank.GrossApproval) / 1e9, float(deposit_rank.approval_share) * 100,
        len(employment), len(negative), len(negative_sba), float(negative_sba.GrossApproval.sum()) / 1e9,
        float(negative_sba.GrossApproval.sum()) / total_approval * 100, unguaranteed / 1e9,
        float(negative_sod.branch_deposits_thousand_usd.sum()) / 1e6,
        float(negative_sod.branch_deposits_thousand_usd.sum() * 1_000) / total_deposits * 100,
        str(negative_by_state.iloc[0].ProjectState), float(negative_by_state.iloc[0].growth),
        len(call_match), str(call.bank_rssd_id), str(call.bank_legal_name), equity / 1e9,
        loss / 1e9, (equity - loss) / 1e9, loss / equity * 100,
    )


DECIMALS = [0, None, None, 0, 6, 6, 0, 0, 4, 4, 4,
            None, 6, 4, 6, 4, None, 6, 4, 6, 4,
            0, 0, 0, 6, 4, 6, 6, 4, None, 4,
            0, None, None, 6, 6, 6, 4]


def _validate_subset(outputs, names):
    by_name = {v.name:v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    places = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_geography_screen":turn_validator(validate_turn_1),
    "validate_geography_decomposition":turn_validator(validate_turn_2),
    "validate_employment_overlay":turn_validator(validate_turn_3),
    "validate_equity_sensitivity":turn_validator(validate_turn_4),
}
