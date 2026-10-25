"""Compare Treasury coupon stock with 2025 nominal auction-yield flow.

At 2025-12-31, conventional notes have 15,636.8393201 billion USD outstanding
at a 3.08855463% outstanding-weighted stated coupon; bonds have 5,248.9742071
billion at 3.30224808%, a 0.21369344-point bond-minus-note gap. Calendar-2025
nominal fixed-rate issuance contains 60 note auctions, including 9 reopenings,
with 3,372 billion USD offered at a 3.94448606% offering-weighted high yield;
24 bond auctions, including 16 reopenings, offer 444 billion at 4.78658108%.
The respective auction-yield-minus-stock-coupon gaps are 0.85593143 and
1.48433300 percentage points.

Equal-weighting security rows or auctions, excluding reopenings, and admitting
TIPS are all measured alternatives. The query pins outstanding principal for
the stock, offering amount for the flow, and nominal fixed-rate original plus
reopening auctions, so this is a hard baseline. Numeric validation uses
0.6 x 10^-N rounding-boundary tolerance for N requested decimals; counts are
exact.
"""

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


variables = [
    Variable("notes_outstanding_billion_usd", None, "Store conventional Treasury notes outstanding in USD billions, rounded to 3 decimals."),
    Variable("bonds_outstanding_billion_usd", None, "Store conventional Treasury bonds outstanding in USD billions, rounded to 3 decimals."),
    Variable("notes_weighted_coupon_pct", None, "Store the outstanding-principal-weighted stated coupon for notes in percent, rounded to 4 decimals."),
    Variable("bonds_weighted_coupon_pct", None, "Store the outstanding-principal-weighted stated coupon for bonds in percent, rounded to 4 decimals."),
    Variable("note_auction_count", None, "Store the 2025 nominal fixed-rate note auction count as an integer."),
    Variable("note_reopening_count", None, "Store the note reopening-auction count as an integer."),
    Variable("note_auction_offering_billion_usd", None, "Store aggregate note auction offering amount in USD billions, rounded to 3 decimals."),
    Variable("note_offering_weighted_high_yield_pct", None, "Store the note offering-amount-weighted auction high yield in percent, rounded to 4 decimals."),
    Variable("bond_auction_count", None, "Store the 2025 nominal fixed-rate bond auction count as an integer."),
    Variable("bond_reopening_count", None, "Store the bond reopening-auction count as an integer."),
    Variable("bond_auction_offering_billion_usd", None, "Store aggregate bond auction offering amount in USD billions, rounded to 3 decimals."),
    Variable("bond_offering_weighted_high_yield_pct", None, "Store the bond offering-amount-weighted auction high yield in percent, rounded to 4 decimals."),
]


def _stock_result(security_class):
    rows = load_expansion_table("treasury_marketable_securities").loc[
        lambda frame: frame.record_date.eq("2025-12-31")
        & frame.security_class.eq(security_class)
        & frame.interest_rate_pct.notna()
        & frame.outstanding_million_usd.notna()
    ]
    if rows.empty:
        raise ValueError(f"no eligible {security_class} stock rows")
    outstanding = rows.outstanding_million_usd.sum()
    coupon = (
        rows.interest_rate_pct * rows.outstanding_million_usd
    ).sum() / outstanding
    return outstanding, coupon


def _auction_result(security_type):
    auctions = load_expansion_table("treasury_auctions").copy()
    auctions["issue_date_dt"] = pd.to_datetime(auctions.issue_date)
    auctions["high_yield_numeric"] = pd.to_numeric(
        auctions.high_yield, errors="coerce"
    )
    auctions["offering_numeric"] = pd.to_numeric(
        auctions.offering_amt, errors="coerce"
    )
    rows = auctions.loc[
        auctions.issue_date_dt.dt.year.eq(2025)
        & auctions.security_type.eq(security_type)
        & auctions.inflation_index_security.eq("No")
        & auctions.floating_rate.eq("No")
        & auctions.high_yield_numeric.notna()
        & auctions.offering_numeric.gt(0)
    ]
    if rows.empty or rows.duplicated(["cusip", "auction_date", "issue_date"]).any():
        raise ValueError(f"invalid {security_type} auction population")
    offering = rows.offering_numeric.sum()
    weighted_yield = (
        rows.high_yield_numeric * rows.offering_numeric
    ).sum() / offering
    return len(rows), int(rows.reopening.eq("Yes").sum()), offering, weighted_yield


def ground_truth():
    notes_outstanding, notes_coupon = _stock_result("Notes")
    bonds_outstanding, bonds_coupon = _stock_result("Bonds")
    note_count, note_reopenings, note_offering, note_yield = _auction_result("Note")
    bond_count, bond_reopenings, bond_offering, bond_yield = _auction_result("Bond")
    return (
        notes_outstanding / 1e3,
        bonds_outstanding / 1e3,
        notes_coupon,
        bonds_coupon,
        note_count,
        note_reopenings,
        note_offering / 1e9,
        note_yield,
        bond_count,
        bond_reopenings,
        bond_offering / 1e9,
        bond_yield,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs,
        variables,
        ground_truth(),
        [
    3, 3, 4, 4, 0, 0, 3, 4, 0, 0, 3, 4,
],
    )



TURN_1_NAMES = [
    "notes_outstanding_billion_usd",
    "bonds_outstanding_billion_usd",
    "notes_weighted_coupon_pct",
    "bonds_weighted_coupon_pct",
]
TURN_2_NAMES = [
    "note_auction_count",
    "note_reopening_count",
    "note_auction_offering_billion_usd",
    "note_offering_weighted_high_yield_pct",
    "bond_auction_count",
    "bond_reopening_count",
    "bond_auction_offering_billion_usd",
    "bond_offering_weighted_high_yield_pct",
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
    "validate_outstanding_stock": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_auction_flow": turn_validator(_turn_subset(TURN_2_NAMES)),
}
