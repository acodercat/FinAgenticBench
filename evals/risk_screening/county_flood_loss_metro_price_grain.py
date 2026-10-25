"""Carry county flood losses to a metropolitan price index without a county index.

No governed table holds a county-to-metropolitan crosswalk. The house price
panel is published for metropolitan areas and states and never for counties, and
the county file carries no metropolitan column, so the 2024 mortgage register is
the only place a county and its metropolitan area appear on the same row. That
crosswalk is a year of lender behaviour, not an administrative boundary, and the
case says so rather than presenting it as a definition.

The grain gap is the point. 2024 flood losses reach 1,193 counties and
8,798.2578 million USD of signed net payments. 710 of those counties sit in a
metropolitan area and 472 sit outside one, yet the metropolitan counties hold
97.0167 percent of the money: a metropolitan reading drops roughly two fifths of
the affected places while keeping almost all the dollars, and the two counts
answer different questions. A further 11 counties never appear in the mortgage
register at all.

Loss reaches 326 metropolitan areas, 322 of them covered by the index, and the
three largest hold 74.5948 percent of metropolitan loss, so the panel is an
event sample shaped by one storm season rather than a national cross-section.
Among the five largest-loss covered areas the index growth from 2023 Q4 to
2025 Q4 runs from -7.3460 percent to 2.9369 percent. That cohort has no boundary
tie: the fifth area carries 103.97 percent more loss than the sixth, and no rank
used anywhere in the case is tied. Nor does the word covered have to be pinned
further: the 322 areas the panel holds at all and the 322 it holds at both
quarter endpoints are the same set, in the same order, so the query leaves the
distinction out and a case-local test asserts it. Both extremes sit on the
same Florida coast and absorbed the same storms, so insured loss and measured
price moved independently; a repeat-transaction index prices the homes that
changed hands, not the stock that was damaged.

Measured alternative readings, all of which the query pins. Summing the gross
rather than the signed net payment fields gives 8,801.1170 million USD, and
using the net building payment alone gives 7,725.4177 million. Screening on the
year a claim was opened rather than its date of loss gives 8,760.9865 million
across 100,955 claims and 1,197 counties. Excluding county numbers that end in
three zeros was tested and then removed from both the query and the module: the
2024 claims contain none, so the clause changed no output, and leaving it in the
code alone would have applied a rule the question never stated. A case-local
test asserts the absence so a refresh that introduces one is caught rather than
silently filtered. On the index side, the purchase-only series
covers only four of the five leading areas and puts the lowest growth at
-13.9289 percent on area 15980, while the expanded-data series keeps the same
weakest area at -7.3070 percent but moves the strongest to 0.4635 percent. The
seasonally adjusted column is empty for this metropolitan series.

1,982 of the qualifying claims were opened in a calendar year after the loss
year, so a loss year is still accumulating when the year ends.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and identifiers match exactly. Summed payments are
reported in USD millions because the summation-order spread of the underlying
USD figures, measured over sixteen orderings, is 0.0 for the national total and
4.8e-07 for the leading area, and the millions scale keeps the tolerance several
orders of magnitude wider than that spread.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


LOSS_YEAR = "2024"
NON_METRO_SENTINEL = "99999"
BASE_YEAR, BASE_QUARTER = 2023, 4
FINAL_YEAR, FINAL_QUARTER = 2025, 4
LARGEST_METRO_COUNT = 5
NET_PAYMENT_COLUMNS = (
    "net_building_payment_usd",
    "net_contents_payment_usd",
    "net_icc_payment_usd",
)


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "county_claim_count",
    "affected_county_count",
    "total_net_paid_usd_millions",
    "largest_loss_county_code",
    "loss_year_claims_opened_later_count",
]
TURN_2_NAMES = [
    "metropolitan_county_count",
    "non_metropolitan_county_count",
    "unmatched_county_count",
    "metropolitan_net_paid_share_pct",
]
TURN_3_NAMES = [
    "affected_metro_area_count",
    "indexed_metro_area_count",
    "largest_loss_metro_code",
    "largest_loss_metro_net_paid_usd_millions",
    "top_three_metro_net_paid_share_pct",
]
TURN_4_NAMES = [
    "weakest_price_metro_code",
    "weakest_price_index_growth_pct",
    "strongest_price_metro_code",
    "strongest_price_index_growth_pct",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store the number of qualifying claim records as an integer."),
    _v(TURN_1_NAMES[1],
       "Store the number of distinct counties they cover as an integer."),
    _v(TURN_1_NAMES[2],
       "Store their summed net paid amount in USD millions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3],
       "Store the leading county's five-digit code as its exact source string."),
    _v(TURN_1_NAMES[4],
       "Store how many qualifying claims were opened in a later calendar year than the loss "
       "year, as an integer."),
    _v(TURN_2_NAMES[0],
       "Store how many affected counties carry a metropolitan-area code as an integer."),
    _v(TURN_2_NAMES[1],
       "Store how many carry the non-metropolitan code as an integer."),
    _v(TURN_2_NAMES[2],
       "Store how many affected counties the mortgage-application geography leaves without any "
       "code, as an integer."),
    _v(TURN_2_NAMES[3],
       "Store the metropolitan group's share of the summed net paid amount in percent, computed "
       "from the unrounded sums, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0],
       "Store the number of distinct metropolitan areas carrying loss as an integer."),
    _v(TURN_3_NAMES[1],
       "Store how many of them the quarterly price panel covers as an integer."),
    _v(TURN_3_NAMES[2],
       "Store the leading metropolitan area's five-digit code as its exact source string."),
    _v(TURN_3_NAMES[3],
       "Store its summed net paid amount in USD millions, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4],
       "Store the three leading metropolitan areas' combined share of metropolitan net paid in "
       "percent, computed from the unrounded sums, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0],
       "Store the five-digit code of the metropolitan area with the lowest index growth as its "
       "exact source string."),
    _v(TURN_4_NAMES[1],
       "Store that growth in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2],
       "Store the five-digit code of the metropolitan area with the highest index growth as its "
       "exact source string."),
    _v(TURN_4_NAMES[3],
       "Store that growth in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 4, None, 0,
    0, 0, 0, 4,
    0, 0, None, 4, 4,
    None, 4, None, 4,
]


@lru_cache(maxsize=1)
def _claims():
    """2024 claim records that identify a county, with their signed net payment."""
    claims = load_expansion_table("fema_nfip")
    columns = ["county_code", "date_of_loss", "open_date", *NET_PAYMENT_COLUMNS]
    frame = claims.loc[
        claims.date_of_loss.astype(str).str.slice(0, 4).eq(LOSS_YEAR), columns
    ].copy()
    del claims
    frame["county_code"] = frame.county_code.astype(str)
    frame = frame.loc[frame.county_code.str.len().eq(5)].copy()
    for column in NET_PAYMENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["net_paid"] = sum(frame[column] for column in NET_PAYMENT_COLUMNS)
    if frame.net_paid.isna().any():
        raise ValueError("a qualifying claim reports an unparseable payment component")
    return frame


@lru_cache(maxsize=1)
def _county_to_metro():
    """One metropolitan code per county, from 2024 mortgage-application geography.

    No governed table carries a county-to-metropolitan crosswalk of its own; the
    mortgage register is the only place a county and its metropolitan area appear
    on the same row.
    """
    register = load_expansion_table("hmda")
    frame = register.loc[:, ["county_code", "derived_msa_md"]].copy()
    del register
    frame["county_code"] = frame.county_code.astype(str)
    frame["metro"] = frame.derived_msa_md.astype(str)
    frame = frame.loc[frame.county_code.str.len().eq(5)]
    counts = frame.groupby(["county_code", "metro"]).size().rename("rows").reset_index()
    counts = counts.sort_values(["county_code", "rows", "metro"], ascending=[True, False, True])
    return counts.drop_duplicates("county_code").set_index("county_code").metro


@lru_cache(maxsize=1)
def _metro_index():
    """Quarterly, not seasonally adjusted, all-transactions metropolitan index levels."""
    panel = load_expansion_table("fhfa_hpi")
    frame = panel.loc[
        panel.level.astype(str).eq("MSA")
        & panel.frequency.astype(str).eq("quarterly")
        & panel.hpi_flavor.astype(str).eq("all-transactions")
        & panel.hpi_type.astype(str).eq("traditional"),
        ["place_id", "yr", "period", "index_nsa"],
    ].copy()
    del panel
    frame["place_id"] = frame.place_id.astype(str)
    frame["index_nsa"] = pd.to_numeric(frame.index_nsa, errors="coerce")
    frame = frame.loc[frame.index_nsa.notna()]
    return frame.drop_duplicates(["place_id", "yr", "period"]).set_index(
        ["place_id", "yr", "period"]
    ).index_nsa


def _leading_code(totals):
    """Largest total, breaking an exact tie by the lower code."""
    target = totals.max()
    return str(sorted(totals.loc[totals.eq(target)].index)[0])


@lru_cache(maxsize=1)
def ground_truth():
    claims = _claims()
    by_county = claims.groupby("county_code").net_paid.sum()
    opened_later = (
        claims.open_date.astype(str).str.slice(0, 4) > claims.date_of_loss.astype(str).str.slice(0, 4)
    )

    bridge = _county_to_metro()
    metro_of = pd.Series(by_county.index.map(bridge), index=by_county.index, dtype="object")
    unmatched = metro_of.isna()
    non_metro = metro_of.eq(NON_METRO_SENTINEL)
    metropolitan = ~unmatched & ~non_metro

    total_paid = float(claims.net_paid.sum())
    metro_paid = float(by_county.loc[metropolitan].sum())
    by_metro = (
        by_county.loc[metropolitan]
        .groupby(metro_of.loc[metropolitan])
        .sum()
        .sort_values(ascending=False)
    )

    levels = _metro_index()
    covered = [
        code for code in by_metro.index
        if (code, BASE_YEAR, BASE_QUARTER) in levels.index
        and (code, FINAL_YEAR, FINAL_QUARTER) in levels.index
    ]
    growth = {
        code: 100.0
        * (
            float(levels[(code, FINAL_YEAR, FINAL_QUARTER)])
            / float(levels[(code, BASE_YEAR, BASE_QUARTER)])
            - 1.0
        )
        for code in covered[:LARGEST_METRO_COUNT]
    }
    ordered = sorted(growth.items(), key=lambda item: (item[1], item[0]))
    leading_metro = _leading_code(by_metro)

    return (
        int(len(claims)),
        int(by_county.size),
        total_paid / 1e6,
        _leading_code(by_county),
        int(opened_later.sum()),
        int(metropolitan.sum()),
        int(non_metro.sum()),
        int(unmatched.sum()),
        100.0 * metro_paid / total_paid,
        int(by_metro.size),
        int(len(covered)),
        leading_metro,
        float(by_metro[leading_metro]) / 1e6,
        100.0 * float(by_metro.head(3).sum()) / metro_paid,
        ordered[0][0],
        float(ordered[0][1]),
        ordered[-1][0],
        float(ordered[-1][1]),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs,
        [by_name[name] for name in names],
        [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_county_loss_population(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_metropolitan_bridge(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_metro_concentration(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_price_path_divergence(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_county_loss_population": turn_validator(validate_county_loss_population),
    "validate_metropolitan_bridge": turn_validator(validate_metropolitan_bridge),
    "validate_metro_concentration": turn_validator(validate_metro_concentration),
    "validate_price_path_divergence": turn_validator(validate_price_path_divergence),
}
