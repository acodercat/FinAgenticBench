"""Carry an NFIP county loss-share screen into bank deposit concentration.

The canonical screen aggregates every physical 2024 NFIP claim row by five-digit
county code, retains counties with at least 500 rows, at least $100m of building
plus contents coverage, and at least three represented banks from the cohort of
the twelve largest FDIC-insured institutions by total assets at 2025-06-30.  Buncombe County, NC (37021) leads net payment divided
by reported coverage at 48.3584%.  Wells Fargo leads selected-county branch deposits,
but removing its largest branch switches the leader to Truist.

The pinned-reading comparison measures physical-row versus represented-policy threshold,
net versus gross payments, ratio-of-sums versus mean row ratios, all SOD banks versus
the BankFind cohort, zero-deposit-row handling, total versus domestic bank denominator,
and largest-branch versus largest-bank removal.  The query pins these choices, so the
case is a hard baseline.  The cohort size is the size of that twelve-bank cohort;
reading it as the number of institutions the 2025-06-30 snapshot holds reports
4,494 instead.  NFIP claims, SOD branch allocation and BankFind institution
stocks have different populations and do not form an accounting or causal identity.

Fragility: the selected county clears the 500-record threshold with 506
records, and the turn-4 leadership flip rests on a single 419 million USD
branch.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import largest_fdic_institutions, load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


REPORT_DATE = "2025-06-30"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "bankfind_snapshot_bank_count", "eligible_nfip_bank_county_count",
    "selected_county_fips", "selected_county_state", "selected_county_claim_row_count",
    "selected_county_represented_policy_count", "selected_county_net_payment_usd_millions",
    "selected_county_coverage_usd_millions", "selected_county_net_payment_to_coverage_pct",
    "selected_county_bankfind_cohort_bank_count", "selected_county_loss_share_margin_pp",
]
TURN_2_NAMES = [
    "selected_county_deposit_bank_count", "selected_county_deposit_leader_certificate",
    "selected_county_deposit_leader_name", "selected_county_deposit_leader_branch_count",
    "selected_county_deposit_leader_usd_billions", "selected_county_cohort_deposits_usd_billions",
    "selected_county_deposit_leader_share_pct", "selected_county_deposit_leader_margin_usd_millions",
]
TURN_3_NAMES = [
    "deposit_leader_total_deposits_usd_billions", "deposit_leader_domestic_deposits_usd_billions",
    "deposit_leader_foreign_office_residual_usd_billions", "selected_county_share_of_total_deposits_pct",
    "selected_county_share_of_domestic_deposits_pct", "deposit_leader_total_assets_usd_billions",
    "deposit_leader_net_loans_usd_billions", "deposit_leader_noncurrent_loans_usd_billions",
    "deposit_leader_noncurrent_loan_ratio_pct",
]
TURN_4_NAMES = [
    "removed_largest_branch_id", "removed_largest_branch_name",
    "removed_largest_branch_deposits_usd_millions", "adjusted_selected_county_leader_certificate",
    "adjusted_selected_county_leader_name", "adjusted_selected_county_leader_deposits_usd_billions",
    "original_leader_adjusted_deposits_usd_billions", "original_leader_adjusted_rank",
]

variables = [
    _v("bankfind_snapshot_bank_count",
       "Store the size of the BankFind cohort, the largest FDIC-insured institutions by total assets at the "
       "2025-06-30 snapshot date, as an integer. This is the cohort count, not the number of institutions "
       "the snapshot holds."),
    _v("eligible_nfip_bank_county_count", "Store the count of counties passing every screen as an integer."),
    _v("selected_county_fips", "Store the selected county's five-digit FIPS code as a string."),
    _v("selected_county_state", "Store the selected county's USPS state abbreviation as a string."),
    _v("selected_county_claim_row_count", "Store the selected county's physical NFIP claim-row count as an integer."),
    _v("selected_county_represented_policy_count", "Store the selected county's represented-policy count as an integer."),
    _v("selected_county_net_payment_usd_millions", "Store aggregate signed net building, contents and ICC payments in USD millions, rounded to 6 decimals."),
    _v("selected_county_coverage_usd_millions", "Store aggregate building plus contents coverage in USD millions, rounded to 6 decimals."),
    _v("selected_county_net_payment_to_coverage_pct", "Store aggregate net payments divided by aggregate coverage in percent, rounded to 4 decimals."),
    _v("selected_county_bankfind_cohort_bank_count", "Store the distinct BankFind-cohort bank count represented by selected-county SOD rows as an integer."),
    _v("selected_county_loss_share_margin_pp", "Store the selected county's payment-to-coverage rate minus the runner-up rate in percentage points, rounded to 4 decimals."),
    _v("selected_county_deposit_bank_count", "Store the distinct represented BankFind-cohort bank count in the selected county as an integer."),
    _v("selected_county_deposit_leader_certificate", "Store the selected-county deposit leader's unpadded FDIC certificate as a string."),
    _v("selected_county_deposit_leader_name", "Store the selected-county deposit leader's SOD legal name as a string."),
    _v("selected_county_deposit_leader_branch_count", "Store the selected-county deposit leader's branch-row count as an integer."),
    _v("selected_county_deposit_leader_usd_billions", "Store the selected-county deposit leader's branch deposits in USD billions, rounded to 6 decimals."),
    _v("selected_county_cohort_deposits_usd_billions", "Store all BankFind-cohort branch deposits in the selected county in USD billions, rounded to 6 decimals."),
    _v("selected_county_deposit_leader_share_pct", "Store the leader's share of selected-county BankFind-cohort deposits in percent, rounded to 4 decimals."),
    _v("selected_county_deposit_leader_margin_usd_millions", "Store leader minus runner-up selected-county deposits in USD millions, rounded to 6 decimals."),
    _v("deposit_leader_total_deposits_usd_billions", "Store the deposit leader's BankFind total deposits in USD billions, rounded to 6 decimals."),
    _v("deposit_leader_domestic_deposits_usd_billions", "Store the deposit leader's BankFind domestic deposits in USD billions, rounded to 6 decimals."),
    _v("deposit_leader_foreign_office_residual_usd_billions", "Store total minus domestic deposits in USD billions, rounded to 6 decimals."),
    _v("selected_county_share_of_total_deposits_pct", "Store selected-county deposits divided by BankFind total deposits in percent, rounded to 4 decimals."),
    _v("selected_county_share_of_domestic_deposits_pct", "Store selected-county deposits divided by BankFind domestic deposits in percent, rounded to 4 decimals."),
    _v("deposit_leader_total_assets_usd_billions", "Store the deposit leader's total assets in USD billions, rounded to 6 decimals."),
    _v("deposit_leader_net_loans_usd_billions", "Store the deposit leader's net loans and leases in USD billions, rounded to 6 decimals."),
    _v("deposit_leader_noncurrent_loans_usd_billions", "Store the deposit leader's noncurrent loans in USD billions, rounded to 6 decimals."),
    _v("deposit_leader_noncurrent_loan_ratio_pct", "Store noncurrent loans divided by net loans and leases in percent, rounded to 4 decimals."),
    _v("removed_largest_branch_id", "Store the removed branch's source branch identifier as a string."),
    _v("removed_largest_branch_name", "Store the removed branch name as a string."),
    _v("removed_largest_branch_deposits_usd_millions", "Store the removed branch's deposits in USD millions, rounded to 6 decimals."),
    _v("adjusted_selected_county_leader_certificate", "Store the adjusted selected-county leader's unpadded FDIC certificate as a string."),
    _v("adjusted_selected_county_leader_name", "Store the adjusted selected-county leader's SOD legal name as a string."),
    _v("adjusted_selected_county_leader_deposits_usd_billions", "Store the adjusted selected-county leader's deposits in USD billions, rounded to 6 decimals."),
    _v("original_leader_adjusted_deposits_usd_billions", "Store the original leader's deposits after branch removal in USD billions, rounded to 6 decimals."),
    _v("original_leader_adjusted_rank", "Store the original leader's adjusted selected-county deposit rank as an integer."),
]


@lru_cache(maxsize=1)
def ground_truth():
    banks = largest_fdic_institutions(load_expansion_table("fdic_bankfind"), REPORT_DATE)
    banks["CERT"] = banks.CERT.astype("string")
    if len(banks) != 12 or banks.CERT.duplicated().any():
        raise ValueError("unexpected BankFind snapshot cohort")
    certificates = set(banks.CERT)

    sod = load_expansion_table("fdic_sod").copy()
    sod["fdic_certificate"] = sod.fdic_certificate.astype("string")
    sod["branch_county_fips"] = sod.branch_county_fips.astype("string").str.zfill(5)
    cohort_sod = sod.loc[
        sod.report_date.eq(REPORT_DATE) & sod.fdic_certificate.isin(certificates)
    ].copy()
    bank_counts = cohort_sod.groupby("branch_county_fips").fdic_certificate.nunique()

    claims = load_expansion_table("fema_nfip").copy()
    claims = claims.loc[claims.date_of_loss.astype(str).str.startswith("2024")]
    claims["county_code"] = claims.county_code.astype("string").str.zfill(5)
    claims = claims.loc[claims.county_code.str.fullmatch(r"\d{5}", na=False)].copy()
    claims["net_payment"] = claims[
        ["net_building_payment_usd", "net_contents_payment_usd", "net_icc_payment_usd"]
    ].sum(axis=1)
    claims["coverage"] = claims[["building_coverage_usd", "contents_coverage_usd"]].sum(axis=1)
    counties = claims.groupby(["county_code", "state_usps"], as_index=False).agg(
        claim_rows=("claim_record_id", "size"), represented_policies=("policy_count", "sum"),
        net_payment=("net_payment", "sum"), coverage=("coverage", "sum"),
    )
    counties["cohort_banks"] = counties.county_code.map(bank_counts).fillna(0).astype(int)
    counties["loss_share"] = counties.net_payment / counties.coverage * 100
    eligible = counties.loc[
        counties.claim_rows.ge(500) & counties.coverage.ge(100e6) & counties.cohort_banks.ge(3)
    ].sort_values(["loss_share", "county_code"], ascending=[False, True]).reset_index(drop=True)
    if len(eligible) != 17 or len(eligible) < 2:
        raise ValueError("unexpected NFIP/SOD county population")
    county, county_runner = eligible.iloc[0], eligible.iloc[1]

    selected_sod = cohort_sod.loc[cohort_sod.branch_county_fips.eq(county.county_code)].copy()
    bank_deposits = selected_sod.groupby(["fdic_certificate", "bank_name"], as_index=False).agg(
        branch_count=("branch_id", "size"), deposits_thousand=("branch_deposits_thousand_usd", "sum")
    ).sort_values(["deposits_thousand", "fdic_certificate"], ascending=[False, True]).reset_index(drop=True)
    leader, bank_runner = bank_deposits.iloc[0], bank_deposits.iloc[1]
    county_deposits = float(bank_deposits.deposits_thousand.sum()) * 1000
    leader_county_deposits = float(leader.deposits_thousand) * 1000
    bank = banks.loc[banks.CERT.eq(leader.fdic_certificate)]
    if len(bank) != 1:
        raise ValueError("selected BankFind row is not unique")
    bank = bank.iloc[0]

    largest_branch = selected_sod.sort_values(
        ["branch_deposits_thousand_usd", "branch_id"], ascending=[False, True]
    ).iloc[0]
    adjusted_rows = selected_sod.loc[~selected_sod.branch_id.eq(largest_branch.branch_id)]
    adjusted = adjusted_rows.groupby(["fdic_certificate", "bank_name"], as_index=False).agg(
        deposits_thousand=("branch_deposits_thousand_usd", "sum")
    ).sort_values(["deposits_thousand", "fdic_certificate"], ascending=[False, True]).reset_index(drop=True)
    adjusted_leader = adjusted.iloc[0]
    original_adjusted = adjusted.loc[adjusted.fdic_certificate.eq(leader.fdic_certificate)].iloc[0]
    original_rank = int(adjusted.index[adjusted.fdic_certificate.eq(leader.fdic_certificate)][0]) + 1

    return (
        len(banks), len(eligible), str(county.county_code), str(county.state_usps),
        int(county.claim_rows), int(county.represented_policies), float(county.net_payment) / 1e6,
        float(county.coverage) / 1e6, float(county.loss_share), int(county.cohort_banks),
        float(county.loss_share - county_runner.loss_share),
        len(bank_deposits), str(leader.fdic_certificate), str(leader.bank_name), int(leader.branch_count),
        leader_county_deposits / 1e9, county_deposits / 1e9,
        leader_county_deposits / county_deposits * 100,
        float(leader.deposits_thousand - bank_runner.deposits_thousand) / 1000,
        float(bank.DEP) / 1e6, float(bank.DEPDOM) / 1e6, float(bank.DEP - bank.DEPDOM) / 1e6,
        leader_county_deposits / (float(bank.DEP) * 1000) * 100,
        leader_county_deposits / (float(bank.DEPDOM) * 1000) * 100,
        float(bank.ASSET) / 1e6, float(bank.LNLSNET) / 1e6, float(bank.NCLNLS) / 1e6,
        float(bank.NCLNLS) / float(bank.LNLSNET) * 100,
        str(largest_branch.branch_id), str(largest_branch.branch_name),
        float(largest_branch.branch_deposits_thousand_usd) / 1000,
        str(adjusted_leader.fdic_certificate), str(adjusted_leader.bank_name),
        float(adjusted_leader.deposits_thousand) / 1e6,
        float(original_adjusted.deposits_thousand) / 1e6, original_rank,
    )


DECIMALS = [0, 0, None, None, 0, 0, 6, 6, 4, 0, 4, 0, None, None, 0, 6, 6, 4, 6, 6, 6, 6, 4, 4, 6, 6, 6, 4, None, None, 6, None, None, 6, 6, 0]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_nfip_county_screen": turn_validator(validate_turn_1),
    "validate_county_bank_concentration": turn_validator(validate_turn_2),
    "validate_bank_balance_context": turn_validator(validate_turn_3),
    "validate_branch_removal_sensitivity": turn_validator(validate_turn_4),
}
