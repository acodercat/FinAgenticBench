"""Trace the largest county flood-loss year into branch deposits and declarations.

Across loss years 2019-2024 the county-year with the largest signed net NFIP
payment is Lee County, FL in 2022 (Hurricane Ian): 3,391.645 million USD over
28,889 claim records representing 54,906 policies, only 14.684 million USD
ahead of Pinellas County, FL in 2024 (Helene/Milton). The case then reads the
Summary of Deposits snapshots that bracket the loss year (2022-06-30 and
2023-06-30): Bank of America led county branch deposits before the event and
Wells Fargo after it, with the original leader's county deposits falling by
250.844 million USD while the county total fell by 169.693 million USD. Four
FEMA declarations cover the county for incidents beginning in 2022; the
earliest major-disaster (DR) declaration is 4673, made six days after the
2022-09-23 incident began.

Rejected alternatives (each pinned by the query): gross building payments
instead of signed net payments (same leader, 113.621 million USD margin);
claim-record counts instead of dollars (same leader); including statewide
"000" county rows or non-five-digit codes; using the loss year's snapshot as
the post-event snapshot; counting emergency (EM) and major-disaster (DR)
declarations together when the DR number is requested; grouping county-years
on the reported state as well as the county code, which splits FIPS 46109 in
2023 into a one-record MN row and a three-record SD row and reports 7,171
county-years instead of 7,170. Because the leader margin is 0.43 percent of
the leading value, the net-versus-gross and signed-versus-positive-only
choices are the live risks, and the query pins them.

The loss window stops at 2024 because the 2025 year of loss is only partially
reported in the claims snapshot and because the post-event Summary of Deposits
snapshot (June 30 of the following year) does not yet exist for 2025. The
leader identity is snapshot-sensitive: the claims table is a current OpenFEMA
extract in which payments are revised retroactively, and the 0.43 percent
margin could reverse in a later extract; the frozen artifact fixes it.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, codes, dates and labels match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


LOSS_YEARS = ("2019", "2024")
NET_COLUMNS = ["net_building_payment_usd", "net_contents_payment_usd", "net_icc_payment_usd"]


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "county_year_count", "leader_county_fips", "leader_state_usps", "leader_loss_year",
    "leader_net_payment_usd_millions", "leader_claim_record_count", "leader_represented_policy_count",
    "runner_up_county_fips", "runner_up_loss_year", "leader_margin_usd_millions",
]
TURN_2_NAMES = [
    "pre_event_snapshot_date", "post_event_snapshot_date", "pre_event_bank_count", "post_event_bank_count",
    "pre_event_county_deposits_usd_billions", "post_event_county_deposits_usd_billions",
    "pre_event_leader_certificate", "pre_event_leader_name", "pre_event_leader_share_pct",
    "post_event_leader_certificate", "post_event_leader_name", "post_event_leader_share_pct",
    "pre_event_leader_deposit_change_usd_millions", "county_deposit_change_usd_millions",
]
TURN_3_NAMES = [
    "declaration_count", "major_disaster_declaration_count", "emergency_declaration_count",
    "earliest_major_disaster_number", "earliest_major_disaster_declaration_date",
    "earliest_major_disaster_incident_begin_date", "earliest_major_disaster_incident_type",
    "days_from_incident_begin_to_declaration",
]
TURN_4_NAMES = [
    "gross_building_leader_county_fips", "gross_building_leader_loss_year", "gross_building_leader_margin_usd_millions",
    "claim_record_rank_of_leader", "net_payment_to_pre_event_deposits_pct",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the count of county-years with at least one claim record as an integer. A county-year is one "
       "county code in one loss year: the reported state is an attribute of the county, not part of the "
       "grain, so claim rows of one county that disagree on the reported state still form a single "
       "county-year."),
    _v(TURN_1_NAMES[1], "Store the leading county's five-digit FIPS code as a string."),
    _v(TURN_1_NAMES[2], "Store the leading county's USPS state code as text, taking the state that most of its claim rows report."),
    _v(TURN_1_NAMES[3], "Store the leading loss year as a four-character string."),
    _v(TURN_1_NAMES[4], "Store the leading county-year's signed net NFIP payment in USD millions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[5], "Store the leading county-year's claim record count as an integer."),
    _v(TURN_1_NAMES[6], "Store the leading county-year's represented-policy count as an integer."),
    _v(TURN_1_NAMES[7], "Store the runner-up county-year's five-digit FIPS code as a string."),
    _v(TURN_1_NAMES[8], "Store the runner-up loss year as a four-character string."),
    _v(TURN_1_NAMES[9], "Store the leader minus runner-up net payment in USD millions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[0], "Store the pre-event SOD snapshot date as a YYYY-MM-DD string."),
    _v(TURN_2_NAMES[1], "Store the post-event SOD snapshot date as a YYYY-MM-DD string."),
    _v(TURN_2_NAMES[2], "Store the count of banks with branch rows in the county at the pre-event snapshot as an integer."),
    _v(TURN_2_NAMES[3], "Store the count of banks with branch rows in the county at the post-event snapshot as an integer."),
    _v(TURN_2_NAMES[4], "Store total county branch deposits at the pre-event snapshot in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[5], "Store total county branch deposits at the post-event snapshot in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[6], "Store the pre-event deposit leader's FDIC certificate as an unpadded string."),
    _v(TURN_2_NAMES[7], "Store the pre-event deposit leader's SOD bank name as text; casing and repeated whitespace are not significant."),
    _v(TURN_2_NAMES[8], "Store the pre-event leader's share of county branch deposits in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[9], "Store the post-event deposit leader's FDIC certificate as an unpadded string."),
    _v(TURN_2_NAMES[10], "Store the post-event deposit leader's SOD bank name as text; casing and repeated whitespace are not significant."),
    _v(TURN_2_NAMES[11], "Store the post-event leader's share of county branch deposits in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[12], "Store the pre-event leader's county deposits at the post-event snapshot minus its pre-event county deposits, in USD millions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[13], "Store post-event minus pre-event total county branch deposits in USD millions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[0], "Store the count of declared-area rows for the county whose incident began in the loss year as an integer."),
    _v(TURN_3_NAMES[1], "Store the count of those rows with declaration type DR as an integer."),
    _v(TURN_3_NAMES[2], "Store the count of those rows with declaration type EM as an integer."),
    _v(TURN_3_NAMES[3], "Store the earliest-declared DR disaster number as an integer."),
    _v(TURN_3_NAMES[4], "Store that declaration's date as a YYYY-MM-DD string."),
    _v(TURN_3_NAMES[5], "Store that declaration's incident begin date as a YYYY-MM-DD string."),
    _v(TURN_3_NAMES[6], "Store that declaration's incident type as text."),
    _v(TURN_3_NAMES[7], "Store the calendar days from incident begin date to declaration date as an integer."),
    _v(TURN_4_NAMES[0], "Store the leading county FIPS when county-years are ranked by gross building payments instead, as a string."),
    _v(TURN_4_NAMES[1], "Store that leader's loss year as a four-character string."),
    _v(TURN_4_NAMES[2], "Store the gross-building leader margin over its runner-up in USD millions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[3], "Store the rank of the turn-1 leader county-year when county-years are ranked by claim record count, as an integer."),
    _v(TURN_4_NAMES[4], "Store the leader's net payment divided by pre-event county branch deposits in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, None, None, None, 6, 0, 0, None, None, 6,
    None, None, 0, 0, 6, 6, None, None, 4, None, None, 4, 6, 6,
    0, 0, 0, 0, None, None, None, 0,
    None, None, 6, 0, 4,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


def _county_years():
    claims = load_expansion_table("fema_nfip").copy()
    claims["loss_year"] = claims.date_of_loss.astype(str).str.slice(0, 4)
    claims = claims.loc[claims.loss_year.between(*LOSS_YEARS)]
    claims["county_code"] = claims.county_code.astype("string")
    claims = claims.loc[
        claims.county_code.str.fullmatch(r"\d{5}", na=False) & ~claims.county_code.str.endswith("000")
    ].copy()
    claims["net_payment"] = claims[NET_COLUMNS].fillna(0).sum(axis=1)
    claims["gross_building"] = claims.gross_building_payment_usd.fillna(0)
    # A county-year is one county in one year. Grouping on the reported state as well
    # would split FIPS 46109 in 2023 into an MN row of 1 claim and an SD row of 3,
    # because one claim carries the wrong state for a South Dakota county, and the
    # count would be of reported (county, year, state) triples rather than of
    # county-years. The state is carried as the one the county's claims mostly report.
    grouped = claims.groupby(["loss_year", "county_code"], as_index=False).agg(
        net_payment=("net_payment", "sum"), gross_building=("gross_building", "sum"),
        claim_rows=("claim_record_id", "size"), policies=("policy_count", "sum"),
    )
    states = (
        claims.groupby(["loss_year", "county_code", "state_usps"], as_index=False)
        .agg(rows=("claim_record_id", "size"))
        .sort_values(["loss_year", "county_code", "rows", "state_usps"],
                     ascending=[True, True, False, True])
        .drop_duplicates(["loss_year", "county_code"], keep="first")
        .loc[:, ["loss_year", "county_code", "state_usps"]]
    )
    return grouped.merge(states, on=["loss_year", "county_code"], how="left")


def _county_banks(sod, county, snapshot):
    rows = sod.loc[sod.county_code.eq(county) & sod.report_date.astype(str).eq(snapshot)]
    banks = rows.groupby(["fdic_certificate", "bank_name"], as_index=False).agg(
        deposits_thousand=("branch_deposits_thousand_usd", "sum")
    ).sort_values(["deposits_thousand", "fdic_certificate"], ascending=[False, True]).reset_index(drop=True)
    return banks


@lru_cache(maxsize=1)
def ground_truth():
    county_years = _county_years()
    by_net = county_years.sort_values(["net_payment", "county_code", "loss_year"], ascending=[False, True, True]).reset_index(drop=True)
    leader, runner = by_net.iloc[0], by_net.iloc[1]
    by_gross = county_years.sort_values(["gross_building", "county_code", "loss_year"], ascending=[False, True, True]).reset_index(drop=True)
    by_rows = county_years.sort_values(["claim_rows", "county_code", "loss_year"], ascending=[False, True, True]).reset_index(drop=True)
    rows_rank = int(by_rows.index[by_rows.county_code.eq(leader.county_code) & by_rows.loss_year.eq(leader.loss_year)][0]) + 1

    sod = load_expansion_table("fdic_sod").copy()
    sod["county_code"] = sod.branch_county_fips.astype("string").str.zfill(5)
    sod["fdic_certificate"] = sod.fdic_certificate.astype(str)
    pre_date = f"{leader.loss_year}-06-30"
    post_date = f"{int(leader.loss_year) + 1}-06-30"
    pre, post = _county_banks(sod, leader.county_code, pre_date), _county_banks(sod, leader.county_code, post_date)
    if pre.empty or post.empty:
        raise ValueError("SOD snapshots bracketing the loss year are missing")
    pre_total, post_total = float(pre.deposits_thousand.sum()), float(post.deposits_thousand.sum())
    pre_leader, post_leader = pre.iloc[0], post.iloc[0]
    pre_leader_after = post.loc[post.fdic_certificate.eq(pre_leader.fdic_certificate), "deposits_thousand"]
    pre_leader_after = float(pre_leader_after.iloc[0]) if len(pre_leader_after) else 0.0

    declarations = load_expansion_table("fema_disasters").copy()
    declarations["county_code"] = declarations.county_fips.astype("string").str.zfill(5)
    covered = declarations.loc[
        declarations.county_code.eq(leader.county_code)
        & declarations.incident_begin_date.astype(str).str.startswith(str(leader.loss_year))
    ].copy()
    majors = covered.loc[covered.declaration_type.astype(str).eq("DR")].sort_values(
        ["declaration_date", "disaster_number"]).reset_index(drop=True)
    if majors.empty:
        raise ValueError("no major-disaster declaration covers the leader county-year")
    first = majors.iloc[0]
    days = (pd.Timestamp(str(first.declaration_date)) - pd.Timestamp(str(first.incident_begin_date))).days

    return (
        len(county_years), str(leader.county_code), str(leader.state_usps), str(leader.loss_year),
        float(leader.net_payment) / 1e6, int(leader.claim_rows), int(leader.policies),
        str(runner.county_code), str(runner.loss_year), float(leader.net_payment - runner.net_payment) / 1e6,
        pre_date, post_date, len(pre), len(post), pre_total * 1000 / 1e9, post_total * 1000 / 1e9,
        str(pre_leader.fdic_certificate), str(pre_leader.bank_name), float(pre_leader.deposits_thousand) / pre_total * 100,
        str(post_leader.fdic_certificate), str(post_leader.bank_name), float(post_leader.deposits_thousand) / post_total * 100,
        (pre_leader_after - float(pre_leader.deposits_thousand)) * 1000 / 1e6, (post_total - pre_total) * 1000 / 1e6,
        len(covered), int(covered.declaration_type.astype(str).eq("DR").sum()), int(covered.declaration_type.astype(str).eq("EM").sum()),
        int(first.disaster_number), str(first.declaration_date), str(first.incident_begin_date), str(first.incident_type), int(days),
        str(by_gross.iloc[0].county_code), str(by_gross.iloc[0].loss_year),
        float(by_gross.iloc[0].gross_building - by_gross.iloc[1].gross_building) / 1e6,
        rows_rank, float(leader.net_payment) / (pre_total * 1000) * 100,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in ("pre_event_leader_name", "post_event_leader_name", "earliest_major_disaster_incident_type", "leader_state_usps"):
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
    return validate_ordered_outputs(normalized, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_county_year_screen": turn_validator(validate_turn_1),
    "validate_deposit_leadership_shift": turn_validator(validate_turn_2),
    "validate_declaration_lineage": turn_validator(validate_turn_3),
    "validate_ranking_sensitivity": turn_validator(validate_turn_4),
}
