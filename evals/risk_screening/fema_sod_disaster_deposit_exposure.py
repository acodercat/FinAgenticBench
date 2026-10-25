"""Measure large-bank branch-deposit exposure to declared disaster counties.

Nineteen FEMA major-disaster declarations beginning and declared in the first
half of 2025 cover 359 distinct valid county FIPS.  Among the 50 banks with the
largest 2025-06-30 SOD branch-deposit totals, Bank of America has the largest
exposed dollar amount ($124.117307B, 6.2843%), while City National Bank has the
largest exposed share ($50.241248B of $77.761501B, or 64.6094%).  City National's
46.9697% exposed branch-row share demonstrates that branch counts and deposits
weight geography differently.

Joining FEMA declaration-area rows without first forming the county set,
including emergency declarations, allowing later declaration dates, changing
the incident-year rule, using a top-20 cohort or weighting by branch counts
changes at least one requested result.  The query pins these conventions, so
they are regression probes for a hard baseline rather than live traps.  Numeric
validation uses 0.6 x 10^-N rounding-boundary tolerance; counts and identifiers
are exact, and bank-name casing and repeated whitespace are normalized.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


variables = [
    Variable("qualifying_major_disaster_count", None, "Store the distinct qualifying major-disaster number count as an integer."),
    Variable("distinct_qualifying_county_count", None, "Store the distinct qualifying five-digit county-FIPS count as an integer."),
    Variable("largest_exposed_deposit_amount_bank_name", None, "Store the legal bank name leading exposed branch-deposit dollars; casing and repeated whitespace are not significant."),
    Variable("largest_exposed_deposit_amount_bank_certificate", None, "Store that bank's FDIC certificate number as an integer."),
    Variable("amount_leader_exposed_deposits_usd_billions", None, "Store that bank's exposed branch deposits in USD billions, rounded to 6 decimals."),
    Variable("amount_leader_total_deposits_usd_billions", None, "Store that bank's total SOD branch deposits in USD billions, rounded to 6 decimals."),
    Variable("amount_leader_exposed_deposit_share_pct", None, "Store that bank's exposed-deposit share in percent, rounded to 4 decimals."),
    Variable("largest_exposed_deposit_share_bank_name", None, "Store the legal bank name leading exposed branch-deposit share; casing and repeated whitespace are not significant."),
    Variable("largest_exposed_deposit_share_bank_certificate", None, "Store that bank's FDIC certificate number as an integer."),
    Variable("share_leader_exposed_deposits_usd_billions", None, "Store that bank's exposed branch deposits in USD billions, rounded to 6 decimals."),
    Variable("share_leader_total_deposits_usd_billions", None, "Store that bank's total SOD branch deposits in USD billions, rounded to 6 decimals."),
    Variable("share_leader_exposed_deposit_share_pct", None, "Store that bank's exposed-deposit share in percent, rounded to 4 decimals."),
    Variable("share_leader_exposed_branch_row_share_pct", None, "Store that bank's exposed branch-row share in percent, rounded to 4 decimals."),
]


def _normalized_label(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else value


def _bank_exposure_population():
    declarations = load_expansion_table("fema_disasters")
    qualifying = declarations.loc[
        declarations.declaration_type.eq("DR")
        & declarations.declaration_date.le("2025-06-30")
        & declarations.incident_begin_date.between("2025-01-01", "2025-06-30")
        & declarations.fips_county_code.ne("000")
    ].copy()
    county_fips = set(qualifying.county_fips.unique())
    branches = load_expansion_table("fdic_sod").loc[
        lambda frame: frame.report_date.eq("2025-06-30")
    ].copy()
    if branches.branch_id.duplicated().any():
        raise ValueError("duplicate SOD branch IDs")
    branches["is_exposed"] = branches.branch_county_fips.isin(county_fips)
    banks = branches.groupby(
        ["fdic_certificate", "bank_name"], as_index=False
    ).agg(
        total_branch_rows=("branch_id", "size"),
        total_deposits=("branch_deposits_thousand_usd", "sum"),
        exposed_branch_rows=("is_exposed", "sum"),
        exposed_deposits=(
            "branch_deposits_thousand_usd",
            lambda values: values[branches.loc[values.index, "is_exposed"]].sum(),
        ),
    )
    banks = banks.sort_values(
        ["total_deposits", "bank_name"], ascending=[False, True]
    ).head(50).copy()
    if len(banks) != 50:
        raise ValueError("unexpected top-bank cohort")
    banks["exposed_deposit_share_pct"] = (
        banks.exposed_deposits / banks.total_deposits * 100
    )
    banks["exposed_branch_row_share_pct"] = (
        banks.exposed_branch_rows / banks.total_branch_rows * 100
    )
    return qualifying, banks


def ground_truth():
    qualifying, banks = _bank_exposure_population()
    amount_leader = banks.sort_values(
        ["exposed_deposits", "bank_name"], ascending=[False, True]
    ).iloc[0]
    share_leader = banks.sort_values(
        ["exposed_deposit_share_pct", "bank_name"], ascending=[False, True]
    ).iloc[0]
    return (
        qualifying.disaster_number.nunique(),
        qualifying.county_fips.nunique(),
        amount_leader.bank_name,
        int(amount_leader.fdic_certificate),
        amount_leader.exposed_deposits / 1e6,
        amount_leader.total_deposits / 1e6,
        amount_leader.exposed_deposit_share_pct,
        share_leader.bank_name,
        int(share_leader.fdic_certificate),
        share_leader.exposed_deposits / 1e6,
        share_leader.total_deposits / 1e6,
        share_leader.exposed_deposit_share_pct,
        share_leader.exposed_branch_row_share_pct,
    )


def validate(outputs):
    expected = ground_truth()
    candidate = dict(outputs)
    for index, name in (
        (2, "largest_exposed_deposit_amount_bank_name"),
        (7, "largest_exposed_deposit_share_bank_name"),
    ):
        if _normalized_label(candidate.get(name)) == _normalized_label(expected[index]):
            candidate[name] = expected[index]
    return validate_ordered_outputs(
        candidate,
        variables,
        expected,
        [0, 0, None, 0, 6, 6, 4, None, 0, 6, 6, 4, 4],
    )



TURN_1_NAMES = [
    "qualifying_major_disaster_count",
    "distinct_qualifying_county_count",
]
TURN_2_NAMES = [
    "largest_exposed_deposit_amount_bank_name",
    "largest_exposed_deposit_amount_bank_certificate",
    "amount_leader_exposed_deposits_usd_billions",
    "amount_leader_total_deposits_usd_billions",
    "amount_leader_exposed_deposit_share_pct",
]
TURN_3_NAMES = [
    "largest_exposed_deposit_share_bank_name",
    "largest_exposed_deposit_share_bank_certificate",
    "share_leader_exposed_deposits_usd_billions",
    "share_leader_total_deposits_usd_billions",
    "share_leader_exposed_deposit_share_pct",
    "share_leader_exposed_branch_row_share_pct",
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
    "validate_qualifying_counties": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_amount_leader": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_share_leader": turn_validator(_turn_subset(TURN_3_NAMES)),
}
