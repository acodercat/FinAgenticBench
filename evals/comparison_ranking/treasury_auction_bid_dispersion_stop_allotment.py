"""Auction demand read from the accepted-bid distribution, not from cover alone.

A Treasury auction publishes three points of the accepted-bid distribution, the
low, the median and the high, together with the percentage allotted at the high.
Bid-to-cover counts how much was bid; it says nothing about where those bids
sat. Across the 168 nominal note and bond auctions in this window the allotment
at the stop and the bid-to-cover ratio correlate at 0.1091, so the two headline
numbers are close to independent readings of the same event.

Three filters the data forces, each of which changes the answer:

1. ``security_type`` of Note or Bond still admits floating-rate notes and TIPS.
   The 24 floating-rate auctions publish a discount margin rather than a yield,
   and the 24 TIPS auctions publish a real yield: the 2024-12-19 five-year TIPS
   stopped at 2.121% against a nominal five-year par yield near 4.4%, which
   would read as a 231 basis point concession. Excluding both leaves 168
   auctions of 216.
2. ``high_yield``, ``avg_med_yield`` and ``low_yield`` are stored as text, and
   negative yields carry a leading apostrophe that defeats a plain numeric cast.
   No such row falls in this window, but the cast has to be explicit.
3. ``original_security_term`` takes exactly seven values here, 2, 3, 5, 7, 10,
   20 and 30 years, each mapping to one published par-yield tenor.
   ``security_term`` does not: a reopened ten-year note carries a term such as
   "9-Year 10-Month" and would not map.

A CUSIP does not identify an auction here. A reopened security is sold again
under the same CUSIP, so 24 of the window's CUSIPs carry more than one auction
and only 80.5% appear once; 91282CJZ5 alone is auctioned three times, stopping
at 4.093%, 4.166% and 4.560%. The auction date is reported alongside the CUSIP
for that reason.

Every median here is taken with linear interpolation, which the query states
because three of the reported ones have an even count whose two middle values
differ. Taking the lower or the upper middle value instead gives 52.8500 or
53.4100 for the median allotment against the reported 53.1300, 6.3000 or 6.4000
for the thirty-year gap against 6.3500, and 27.1900 or 44.5400 for the
thirty-year allotment against 35.8650. The remaining medians are unaffected:
the two middle values coincide for the overall gap, the overall concession and
the two-year concession. None of the three term rankings moves under any of the
three definitions.

Two stated rules do not currently bind and are kept anyway, with the case test
asserting that they do not: every one of the 168 retained auctions publishes all
five demand fields, so the completeness requirement excludes nothing, and no two
original terms share a median gap or a median allotment, so the shorter-term tie
rule never has to decide.

Bills are outside the case. They publish a discount rate rather than a yield,
and ``avg_med_investment_rate`` is empty throughout this window, so there is no
bond-equivalent median with which to place them on the same scale.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2024-01-01", "2025-12-31")
TENOR_COLUMN = {
    "2-Year": "2 Yr", "3-Year": "3 Yr", "5-Year": "5 Yr", "7-Year": "7 Yr",
    "10-Year": "10 Yr", "20-Year": "20 Yr", "30-Year": "30 Yr",
}
NUMERIC = ("high_yield", "avg_med_yield", "low_yield", "allocation_pctage", "bid_to_cover_ratio")

OUTPUTS = (
    ("auction_count", 0, "the eligible auction count as an integer"),
    ("distinct_original_terms", 0, "the count of distinct original security terms as an integer"),
    ("median_dispersion_bp", 4, "the median gap from the median accepted yield to the stop, in basis points"),
    ("median_stop_allotment_pct", 4, "the median percentage allotted at the stop"),
    ("allotment_cover_correlation", 4, "the Pearson correlation between that allotment and the bid-to-cover ratio"),
    ("widest_dispersion_term", None, "the original security term with the widest median gap, as a source-verbatim string"),
    ("widest_dispersion_median_bp", 4, "that term's median gap in basis points"),
    ("lowest_allotment_term", None, "the original security term with the lowest median allotment, as a source-verbatim string"),
    ("lowest_allotment_median_pct", 4, "that term's median allotment in percent"),
    ("median_concession_bp", 4, "the median stop yield less the matching par yield, in basis points"),
    ("widest_concession_term", None, "the original security term with the largest median such difference, as a source-verbatim string"),
    ("widest_concession_median_bp", 4, "that term's median difference in basis points"),
    ("widest_auction_cusip", None, "the single auction with the widest gap, by its exact CUSIP"),
    ("widest_auction_date", None, "that auction's date as an ISO YYYY-MM-DD string"),
    ("widest_auction_allotment_pct", 4, "that auction's percentage allotted at the stop"),
    ("widest_auction_concession_bp", 4, "that auction's stop yield less its matching par yield, in basis points"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:5]),
    tuple(n for n, _, _ in OUTPUTS[5:9]),
    tuple(n for n, _, _ in OUTPUTS[9:12]),
    tuple(n for n, _, _ in OUTPUTS[12:]),
)


@lru_cache(maxsize=1)
def _auctions():
    raw = load_expansion_table("treasury_auctions")
    frame = raw.loc[
        raw.auction_date.astype(str).between(*WINDOW)
        & raw.security_type.isin(["Note", "Bond"])
        & raw.floating_rate.eq("No")
        & raw.inflation_index_security.eq("No")
    ].copy()
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(NUMERIC)].isna().any().any():
        raise ValueError("a retained auction is missing a published demand field")
    if not frame.original_security_term.isin(TENOR_COLUMN).all():
        raise ValueError("an original security term has no matching par-yield tenor")
    ordered = frame.low_yield.le(frame.avg_med_yield) & frame.avg_med_yield.le(frame.high_yield)
    if not ordered.all():
        raise ValueError("accepted-bid yields are not ordered low, median, high")
    if frame.cusip.isna().any():
        raise ValueError("a retained auction reports no CUSIP")
    frame["dispersion_bp"] = (frame.high_yield - frame.avg_med_yield) * 100
    return frame


def _par_yield(curve, as_of, tenor):
    window = curve.loc[curve.Date.le(as_of)].sort_values("Date")
    if window.empty:
        raise ValueError(f"no published par curve on or before {as_of}")
    value = pd.to_numeric(window.iloc[-1][tenor], errors="coerce")
    if pd.isna(value):
        raise ValueError(f"par yield for {tenor} is absent at {as_of}")
    return float(value)


@lru_cache(maxsize=1)
def analysis():
    frame = _auctions().copy()
    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = curve.Date.astype(str)

    frame["par_yield"] = [
        _par_yield(curve, str(row.auction_date), TENOR_COLUMN[row.original_security_term])
        for row in frame.itertuples()
    ]
    frame["concession_bp"] = (frame.high_yield - frame.par_yield) * 100

    by_term = frame.groupby("original_security_term")
    dispersion = by_term.dispersion_bp.median().sort_index()
    allotment = by_term.allocation_pctage.median().sort_index()
    concession = by_term.concession_bp.median().sort_index()

    widest = dispersion.sort_values(ascending=False, kind="stable")
    lowest = allotment.sort_values(kind="stable")
    richest = concession.sort_values(ascending=False, kind="stable")
    for series, label in ((widest, "dispersion"), (richest, "concession")):
        if series.iloc[0] == series.iloc[1]:
            raise ValueError(f"term ranking on {label} is not a strict leader")
    if lowest.iloc[0] == lowest.iloc[1]:
        raise ValueError("term ranking on allotment is not a strict leader")

    # Two auctions share the widest gap, so the stated tie rule decides.
    single = frame.sort_values(
        ["dispersion_bp", "auction_date", "cusip"], ascending=[False, True, True]
    ).iloc[0]

    return {
        "auction_count": len(frame),
        "distinct_original_terms": int(frame.original_security_term.nunique()),
        "median_dispersion_bp": float(frame.dispersion_bp.median()),
        "median_stop_allotment_pct": float(frame.allocation_pctage.median()),
        "allotment_cover_correlation": float(frame.allocation_pctage.corr(frame.bid_to_cover_ratio)),
        "widest_dispersion_term": str(widest.index[0]),
        "widest_dispersion_median_bp": float(widest.iloc[0]),
        "lowest_allotment_term": str(lowest.index[0]),
        "lowest_allotment_median_pct": float(lowest.iloc[0]),
        "median_concession_bp": float(frame.concession_bp.median()),
        "widest_concession_term": str(richest.index[0]),
        "widest_concession_median_bp": float(richest.iloc[0]),
        "widest_auction_cusip": str(single.cusip),
        "widest_auction_date": str(single.auction_date),
        "widest_auction_allotment_pct": float(single.allocation_pctage),
        "widest_auction_concession_bp": float(single.concession_bp),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
