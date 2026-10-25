"""Reconcile Treasury auction awards and compare bidder-share denominators.

The selected nominal ten-year auctions reconcile total accepted to the three
reported competitive bidder categories, SOMA accepted and noncompetitive
accepted. That arithmetic identity establishes the relationship among these
fields in the frozen rows; it does not prove that the fields exhaust every
economic way of describing an auction.

Bidder-category shares use the sum of primary-dealer, direct and indirect
accepted amounts. Dividing by total accepted instead incorporates SOMA and
noncompetitive awards in the denominator even though they are outside those
three competitive bidder categories. The case reports both bases rather than
calling one a foreign-demand measure.

Primary dealers are expected by the Federal Reserve Bank of New York to bid for
their pro-rata share in all Treasury auctions at reasonably competitive prices.
That participation expectation does not require them to receive unsold residual
securities. Dealer share is the arithmetic complement of the other two bidder
shares on the three-category base, while its correlation with bid-to-cover in
this small sample remains descriptive rather than causal.

The clearing-yield comparison uses the same-day ten-year par yield, not the
when-issued quotation conventionally used to discuss an auction tail. It is a
curve context measure and cannot by itself determine whether an auction was
well received. The query pins the population, denominator and curve choices, so
this is a hard baseline rather than a live hidden-method trap.

Numeric validation uses the shared rounding tolerance; counts and dates match
exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


YEARS = ("2024", "2025")
SECURITY_TYPE = "Note"
TEN_YEAR_TERMS = ("10-Year", "9-Year 10-Month")
NOMINAL_FLAG = "No"
CURVE_TENOR = "10 Yr"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "auction_count", "new_issue_count", "reopening_count", "total_accepted_usd_billions",
    "competitive_award_usd_billions", "soma_accepted_usd_billions",
    "noncompetitive_usd_billions", "identity_residual_usd",
]
TURN_2_NAMES = [
    "median_indirect_share_competitive_pct", "median_dealer_share_competitive_pct",
    "median_direct_share_competitive_pct", "median_indirect_share_total_pct",
    "median_dealer_share_total_pct",
]
TURN_3_NAMES = [
    "highest_dealer_share_date", "highest_dealer_share_pct", "highest_dealer_share_bid_to_cover",
    "lowest_dealer_share_date", "lowest_dealer_share_pct", "lowest_dealer_share_bid_to_cover",
    "dealer_share_bid_to_cover_correlation", "median_bid_to_cover",
]
TURN_4_NAMES = [
    "auctions_clearing_above_curve", "median_yield_gap_bp", "widest_yield_gap_date",
    "widest_yield_gap_bp", "widest_gap_dealer_share_pct", "positive_soma_auction_count",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many auctions qualify as an integer."),
    _v(TURN_1_NAMES[1], "Store how many of them are new issues as an integer."),
    _v(TURN_1_NAMES[2], "Store how many are reopenings as an integer."),
    _v(TURN_1_NAMES[3], "Store total accepted across them in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[4], "Store the competitively awarded total in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[5], "Store SOMA accepted across the auctions in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[6], "Store the non-competitive total in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[7], "Store the largest absolute per-auction difference between total accepted and the five components, in USD rounded to 2 decimals."),
    _v(TURN_2_NAMES[0], "Store the median indirect share of the competitive award in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the median dealer share of the competitive award in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the median direct share of the competitive award in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the median indirect share of total accepted in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the median dealer share of total accepted in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the auction date with the highest dealer share as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[1], "Store that share in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store that auction's bid-to-cover ratio, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the auction date with the lowest dealer share as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[4], "Store that share in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store that auction's bid-to-cover ratio, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the correlation between dealer share and bid-to-cover across the auctions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[7], "Store the median bid-to-cover ratio, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many auctions clear above the same-day curve reading as an integer."),
    _v(TURN_4_NAMES[1], "Store the median clearing yield minus curve reading in basis points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the date of the widest positive difference as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[3], "Store that difference in basis points, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store that auction's dealer share of the competitive award in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store how many auctions carry a positive SOMA accepted amount as an integer."),
]

DECIMALS = [
    0, 0, 0, 6, 6, 6, 6, 2,
    4, 4, 4, 4, 4,
    None, 4, 4, None, 4, 4, 4, 4,
    0, 4, None, 4, 4, 0,
]

NUMERIC_COLUMNS = (
    "high_yield", "total_accepted", "primary_dealer_accepted", "direct_bidder_accepted",
    "indirect_bidder_accepted", "soma_accepted", "noncomp_accepted", "bid_to_cover_ratio",
)


@lru_cache(maxsize=1)
def _auctions():
    frame = load_expansion_table("treasury_auctions").copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    selected = frame.loc[
        frame.security_type.astype(str).eq(SECURITY_TYPE)
        & frame.auction_date.astype(str).str.startswith(YEARS)
        & frame.security_term.astype(str).isin(TEN_YEAR_TERMS)
        & frame.inflation_index_security.astype(str).eq(NOMINAL_FLAG)
    ].copy()
    selected["competitive"] = (
        selected.primary_dealer_accepted + selected.direct_bidder_accepted + selected.indirect_bidder_accepted
    )
    selected["dealer_share"] = selected.primary_dealer_accepted / selected.competitive * 100
    selected["indirect_share"] = selected.indirect_bidder_accepted / selected.competitive * 100
    selected["direct_share"] = selected.direct_bidder_accepted / selected.competitive * 100
    selected["dealer_share_total"] = selected.primary_dealer_accepted / selected.total_accepted * 100
    selected["indirect_share_total"] = selected.indirect_bidder_accepted / selected.total_accepted * 100
    selected["residual"] = (
        selected.total_accepted - selected.competitive - selected.soma_accepted - selected.noncomp_accepted
    )
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["day"] = curve.Date.astype(str)
    curve["tenor"] = pd.to_numeric(curve[CURVE_TENOR], errors="coerce")
    joined = selected.merge(
        curve[["day", "tenor"]], left_on=selected.auction_date.astype(str), right_on="day", how="left"
    )
    if joined.tenor.isna().any():
        raise ValueError("an auction date has no same-day curve reading")
    joined["gap_bp"] = (joined.high_yield - joined.tenor) * 100
    return joined


@lru_cache(maxsize=1)
def ground_truth():
    auctions = _auctions()
    by_dealer = auctions.sort_values(["dealer_share", "auction_date"], ascending=[False, True]).reset_index(drop=True)
    by_gap = auctions.sort_values(["gap_bp", "auction_date"], ascending=[False, True]).reset_index(drop=True)
    reopenings = auctions.reopening.astype(str).eq("Yes")

    return (
        int(len(auctions)),
        int((~reopenings).sum()),
        int(reopenings.sum()),
        float(auctions.total_accepted.sum()) / 1e9,
        float(auctions.competitive.sum()) / 1e9,
        float(auctions.soma_accepted.sum()) / 1e9,
        float(auctions.noncomp_accepted.sum()) / 1e9,
        float(auctions.residual.abs().max()),
        float(auctions.indirect_share.median()),
        float(auctions.dealer_share.median()),
        float(auctions.direct_share.median()),
        float(auctions.indirect_share_total.median()),
        float(auctions.dealer_share_total.median()),
        str(by_dealer.auction_date.iloc[0]),
        float(by_dealer.dealer_share.iloc[0]),
        float(by_dealer.bid_to_cover_ratio.iloc[0]),
        str(by_dealer.auction_date.iloc[-1]),
        float(by_dealer.dealer_share.iloc[-1]),
        float(by_dealer.bid_to_cover_ratio.iloc[-1]),
        float(auctions.dealer_share.corr(auctions.bid_to_cover_ratio)),
        float(auctions.bid_to_cover_ratio.median()),
        int(auctions.gap_bp.gt(0).sum()),
        float(auctions.gap_bp.median()),
        str(by_gap.auction_date.iloc[0]),
        float(by_gap.gap_bp.iloc[0]),
        float(by_gap.dealer_share.iloc[0]),
        int(auctions.soma_accepted.gt(0).sum()),
    )


NAME_OUTPUTS = ("highest_dealer_share_date", "lowest_dealer_share_date", "widest_yield_gap_date")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = str(normalized[name]).strip()
            truth[name] = str(truth[name]).strip()
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_award_identity": turn_validator(validate_turn_1),
    "validate_demand_base": turn_validator(validate_turn_2),
    "validate_dealer_residual": turn_validator(validate_turn_3),
    "validate_clearing_yield_context": turn_validator(validate_turn_4),
}
