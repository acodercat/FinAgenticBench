"""Reconcile JPMorgan's institution deposits with its SOD branch geography.

The frozen 2025-06-30 BankFind row reports 2,669.161 billion USD of total
deposits and 2,132.981 billion of domestic deposits, leaving a 536.180 billion
foreign-office residual.  Independently summing 5,000 SOD branch rows produces
exactly 2,132.981 billion, while New York contains 43.7718% of branch deposits
but only 11.5400% of branch rows.  Computing concentration from branch counts or
using total deposits as the state-share denominator therefore answers a
different question.

The query pins the date, certificate, branch-deposit measure, domestic
reconciliation, state geography, HHI scale and tie break, so these alternatives
are regression probes rather than an unprompted trap.  Numeric validation uses
0.6 x 10^-N rounding-boundary tolerance for N requested decimals; counts are
exact, and USPS labels normalize casing and repeated whitespace.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


REPORT_DATE = "2025-06-30"
CERTIFICATE = "628"

variables = [
    Variable("sod_branch_row_count", None, "Store the SOD branch-row count as an integer."),
    Variable("represented_branch_state_count", None, "Store the represented branch-state count as an integer."),
    Variable("bankfind_total_deposits_usd_billions", None, "Store BankFind total deposits in USD billions, rounded to 6 decimals."),
    Variable("bankfind_domestic_deposits_usd_billions", None, "Store BankFind domestic deposits in USD billions, rounded to 6 decimals."),
    Variable("derived_foreign_office_deposits_usd_billions", None, "Store the derived foreign-office deposit residual in USD billions, rounded to 6 decimals."),
    Variable("sod_branch_deposits_usd_billions", None, "Store aggregate SOD branch deposits in USD billions, rounded to 6 decimals."),
    Variable("sod_minus_bankfind_domestic_usd_millions", None, "Store SOD branch deposits minus BankFind domestic deposits in USD millions, rounded to 3 decimals."),
    Variable("largest_branch_deposit_state_usps", None, "Store the leading branch-deposit state's USPS abbreviation; casing and repeated whitespace are not significant."),
    Variable("largest_state_branch_row_count", None, "Store that state's branch-row count as an integer."),
    Variable("largest_state_branch_row_share_pct", None, "Store that state's share of branch rows in percent, rounded to 4 decimals."),
    Variable("largest_state_branch_deposits_usd_billions", None, "Store that state's branch deposits in USD billions, rounded to 6 decimals."),
    Variable("largest_state_share_of_domestic_deposits_pct", None, "Store that state's share of domestic deposits in percent, rounded to 4 decimals."),
    Variable("state_deposit_hhi", None, "Store the state-level branch-deposit HHI on the 0-to-10,000 scale, rounded to 4 decimals."),
]


def _normalized_label(value):
    if not isinstance(value, str):
        return value
    return " ".join(value.split()).casefold()


def ground_truth():
    financials = load_expansion_table("fdic_bankfind")
    bank = financials.loc[
        financials.CERT.eq(CERTIFICATE)
        & financials.REPDTE.eq(REPORT_DATE)
    ]
    if len(bank) != 1:
        raise ValueError(f"expected one BankFind row, found {len(bank)}")
    bank = bank.iloc[0]

    branches = load_expansion_table("fdic_sod").loc[
        lambda frame: frame.report_date.eq(REPORT_DATE)
        & frame.fdic_certificate.eq(CERTIFICATE)
    ].copy()
    if branches.branch_id.duplicated().any() or branches.empty:
        raise ValueError("invalid JPMorgan SOD branch population")
    states = branches.groupby("branch_state", as_index=False).agg(
        branch_rows=("branch_id", "size"),
        branch_deposits=("branch_deposits_thousand_usd", "sum"),
    )
    branch_total = states.branch_deposits.sum()
    states["branch_row_share_pct"] = states.branch_rows / len(branches) * 100
    states["deposit_share_pct"] = states.branch_deposits / branch_total * 100
    leader = states.sort_values(
        ["branch_deposits", "branch_state"], ascending=[False, True]
    ).iloc[0]
    hhi = (states.deposit_share_pct ** 2).sum()
    foreign = bank.DEP - bank.DEPDOM
    return (
        len(branches),
        len(states),
        bank.DEP / 1e6,
        bank.DEPDOM / 1e6,
        foreign / 1e6,
        branch_total / 1e6,
        (branch_total - bank.DEPDOM) / 1e3,
        leader.branch_state,
        int(leader.branch_rows),
        leader.branch_row_share_pct,
        leader.branch_deposits / 1e6,
        leader.deposit_share_pct,
        hhi,
    )


def validate(outputs):
    expected = ground_truth()
    candidate = dict(outputs)
    actual_state = candidate.get("largest_branch_deposit_state_usps")
    if _normalized_label(actual_state) == _normalized_label(expected[7]):
        candidate["largest_branch_deposit_state_usps"] = expected[7]
    return validate_ordered_outputs(
        candidate,
        variables,
        expected,
        [0, 0, 6, 6, 6, 6, 3, None, 0, 4, 6, 4, 4],
    )



TURN_1_NAMES = [
    "bankfind_total_deposits_usd_billions",
    "bankfind_domestic_deposits_usd_billions",
    "derived_foreign_office_deposits_usd_billions",
]
TURN_2_NAMES = [
    "sod_branch_row_count",
    "represented_branch_state_count",
    "sod_branch_deposits_usd_billions",
    "sod_minus_bankfind_domestic_usd_millions",
]
TURN_3_NAMES = [
    "largest_branch_deposit_state_usps",
    "largest_state_branch_row_count",
    "largest_state_branch_row_share_pct",
    "largest_state_branch_deposits_usd_billions",
    "largest_state_share_of_domestic_deposits_pct",
    "state_deposit_hhi",
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
    "validate_reported_balances": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_branch_reconciliation": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_geographic_concentration": turn_validator(_turn_subset(TURN_3_NAMES)),
}
