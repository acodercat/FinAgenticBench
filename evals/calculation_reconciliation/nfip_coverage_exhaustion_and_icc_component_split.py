"""Reconcile NFIP payment components against the coverage each one actually consumes.

A flood policy is not one pot of money.  The building coverage, the contents
coverage and increased cost of compliance are three statutory coverages with
three separate limits, and a payment made under one of them does not draw down
another.  Increased cost of compliance in particular is additional coverage: it
carries its own USD30,000 cap and sits outside the building limit entirely.

The trap is the second turn, which asks how many claims exhausted their building
coverage.  Only the gross building payment is drawn against the building
coverage.  Summing the payment components before the comparison reports 12,097
claims when increased cost of compliance is folded in, 18,137 when the contents
payment is folded in and 18,308 when both are, against 11,934 on the pinned
reading; the same fold-in carries the exhausted group's gross building payment
to USD2,262.423430 million against USD2,233.450134 million.  Counts are compared
for exact equality and the payment is pinned to six decimals, so the
rounding-boundary tolerance of 0.6e-6 excludes every one of those readings --
the smallest of them misses by 163 claims and by USD28.973296 million.  The
error propagates into the third turn, where the count of exhausted claims that
also drew increased cost of compliance (229, carrying USD4.936133 million) is
the evidence that the coverage pays past an exhausted building limit.

The first turn carries a second, independent axis.  The cause-of-damage column
stores the same code in two representations: 133,349 rows read "4" and a further
12,564 read "4.0".  Reading the column as recorded reports 19 distinct causes and
a leading cause of 133,349 claims; coercing it to a number instead silences the
702 rows carrying the letter codes A, B, C and D and reports 8 distinct causes.
The pinned reading treats a code as one cause regardless of numeric formatting
and keeps the letter codes, giving 12 causes and 145,913 claims.  Both counts are
compared for exact equality, so both naive readings are excluded.  The pinned
answer does not depend on how the formatting is reconciled: casting to an integer,
stripping a trailing ".0" as text, and casefolding before either all give 12 and
145,913.  Numeric coercion recovers the leading count but not the cause count, so
the two outputs fail the two naive readings between them.

Rejected conventions.  Building coverage counts as exhausted when the gross
building payment reaches it, not only when it passes it -- 11,140 of the 11,934
settle at exactly the coverage amount and passing alone would report 794.  Storm
rows are restricted to the ones carrying a recorded flood cause; on the frozen
extract those are exactly the 49,617 rows labelled Flash Flood, Flood or Debris
Flow, and every one of them already sits on a county, so the county-granularity
filter below is a guard that currently removes nothing.  It is what the restriction
costs that matters: 5,485 water-related rows -- Tropical Storm, Coastal Flood, Storm
Surge/Tide, Hurricane, Lakeshore Flood, Tsunami -- carry no cause at all and are filed
against a forecast zone, so coastal and tropical flooding never enters the pairing.
The storm side is restricted to the fifty states and the District of Columbia,
which settles a reading rather than changing an answer: the territories are
numbered outside the standard range and could never have paired as recorded, so
the restriction moves none of the sixteen figures, while a reader who repaired
the numbering instead would report 6,877 shared county-months, 138,693 matched
claims, USD13,816.571187 million unmatched and 6,284 on the leading cause.  The
month is the month of loss on the claim side and the month the event began on
the storm side.  683 claim rows carry no county code at all; they can never be
paired and the query counts them outside the pairing, which is worth
USD21.153562 million of the unmatched total -- dropping them instead would report
USD13,797.319565 million, so the rule is stated rather than left to the reader.

Sixteen combinations of payment basis, comparison boundary and coverage scope were
swept over the second turn and exactly one reproduces the pinned pair.  The three
dollar outputs lie 0.31, 0.18 and 0.11 of a unit from the value they are rounded
to -- a rounding boundary would be half a unit away -- and are invariant to
summation order to the cent.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


RESIDENTIAL_BUILDING_LIMIT_USD = 250_000.0
ICC_STATUTORY_CAP_USD = 30_000.0
COUNTY_GRANULARITY_TOKEN = "C"
STATE_USPS = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
})


variables = [
    Variable("claim_record_count", None,
             "Store the number of claim rows the flood claim file holds as an integer."),
    Variable("cause_coded_claim_count", None,
             "Store the number of those rows carrying a recorded cause of damage as an integer."),
    Variable("distinct_cause_code_count", None,
             "Store the number of distinct causes of damage those coded rows describe as an integer."),
    Variable("leading_cause_claim_count", None,
             "Store the number of claim rows carrying the single most frequent cause as an integer."),

    Variable("positive_building_coverage_claim_count", None,
             "Store the number of claims reporting a positive building coverage amount as an integer."),
    Variable("building_coverage_exhausted_claim_count", None,
             "Store how many of those claims exhausted their building coverage as an integer."),
    Variable("exhausted_gross_building_payment_usd_millions", None,
             "Store the exhausted claims' total gross building payment in USD millions, rounded to 6 decimals."),
    Variable("exhausted_at_maximum_residential_coverage_claim_count", None,
             "Store how many exhausted claims carried building coverage at the maximum residential amount as an integer."),

    Variable("icc_paid_claim_count", None,
             "Store the number of claims receiving a gross increased cost of compliance payment as an integer."),
    Variable("icc_at_statutory_cap_claim_count", None,
             "Store how many of those were paid exactly at the statutory cap as an integer."),
    Variable("icc_gross_payment_usd_millions", None,
             "Store the total gross increased cost of compliance paid in USD millions, rounded to 6 decimals."),
    Variable("exhausted_claim_icc_recipient_count", None,
             "Store how many exhausted claims also received such a payment as an integer."),

    Variable("noaa_matched_county_month_count", None,
             "Store the number of county-months the two files share as an integer."),
    Variable("noaa_matched_claim_count", None,
             "Store the number of claim rows falling in those shared county-months as an integer."),
    Variable("unmatched_claim_gross_payment_usd_millions", None,
             "Store the total gross payment on claim rows outside them in USD millions, rounded to 6 decimals."),
    Variable("leading_noaa_flood_cause_county_month_count", None,
             "Store how many shared county-months the most widely recorded flood cause covers as an integer."),
]

DECIMALS = [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 6, 0]


def canonical_cause_code(value):
    """One cause per code: "4" and "4.0" name the same cause, "A" names another."""
    if value is None or value is pd.NA or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text.upper()


def _flood_cause_label(value):
    if value is None or value is pd.NA or (isinstance(value, float) and value != value):
        return None
    return " ".join(str(value).split()) or None


def _digits(series, width):
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(width)


def _claims():
    claims = load_expansion_table("fema_nfip")
    frame = pd.DataFrame({
        "cause": claims.cause_of_damage.map(canonical_cause_code),
        "gross_building": pd.to_numeric(claims.gross_building_payment_usd, errors="coerce"),
        "gross_contents": pd.to_numeric(claims.gross_contents_payment_usd, errors="coerce"),
        "gross_icc": pd.to_numeric(claims.gross_icc_payment_usd, errors="coerce"),
        "building_coverage": pd.to_numeric(claims.building_coverage_usd, errors="coerce"),
        "county": _digits(claims.county_code, 5),
        "month": pd.to_datetime(claims.date_of_loss, errors="coerce").dt.to_period("M").astype(str),
    })
    del claims
    return frame


def _county_flood_events():
    """Storm rows in the states, placed on a county, carrying a recorded flood cause.

    The state restriction settles the reading rather than changing the answer.  The
    storm file numbers the outlying territories outside the standard range -- Puerto
    Rico 99, American Samoa 97, Guam 98, the Virgin Islands 96, against the 72, 60,
    66 and 78 the claim file uses -- so those 1,766 rows can never meet a claim
    county as they stand, and a reader who repaired the codes instead would pair 22
    more county-months.  Pinning the states makes both readings agree.
    """
    storms = load_expansion_table("noaa_storm_events")
    frame = pd.DataFrame({
        "cause": storms.flood_cause.map(_flood_cause_label),
        "county": _digits(storms.state_fips, 2) + _digits(storms.county_zone_fips, 3),
        "month": pd.to_datetime(storms.begin_datetime, errors="coerce").dt.to_period("M").astype(str),
        "granularity": storms.county_zone_type.astype(str).str.strip().str.upper(),
        "state": storms.state_usps.astype(str).str.strip().str.upper(),
    })
    del storms
    return frame.loc[
        frame.cause.notna()
        & frame.granularity.eq(COUNTY_GRANULARITY_TOKEN)
        & frame.state.isin(STATE_USPS)
    ]


@lru_cache(maxsize=1)
def ground_truth():
    claims = _claims()

    coded = claims.cause.notna()
    leading_cause_claims = int(claims.cause.value_counts().iloc[0])

    insured = claims.building_coverage.gt(0)
    exhausted = insured & claims.gross_building.ge(claims.building_coverage)
    at_residential_limit = exhausted & claims.building_coverage.eq(RESIDENTIAL_BUILDING_LIMIT_USD)

    icc_paid = claims.gross_icc.gt(0)

    events = _county_flood_events()
    claim_months = set(zip(claims.county, claims.month))
    event_months = set(zip(events.county, events.month))
    shared = claim_months & event_months
    matched = pd.Series([key in shared for key in zip(claims.county, claims.month)],
                        index=claims.index)
    inside = events.loc[[key in shared for key in zip(events.county, events.month)]]
    widest_cause_months = max(
        len(set(zip(group.county, group.month))) for _, group in inside.groupby("cause")
    )
    gross_total = claims.gross_building.fillna(0.0) + claims.gross_contents.fillna(0.0) + claims.gross_icc.fillna(0.0)

    return (
        len(claims),
        int(coded.sum()),
        int(claims.cause.nunique()),
        leading_cause_claims,

        int(insured.sum()),
        int(exhausted.sum()),
        round(float(claims.gross_building[exhausted].sum()) / 1e6, 6),
        int(at_residential_limit.sum()),

        int(icc_paid.sum()),
        int((icc_paid & claims.gross_icc.eq(ICC_STATUTORY_CAP_USD)).sum()),
        round(float(claims.gross_icc[icc_paid].sum()) / 1e6, 6),
        int((exhausted & icc_paid).sum()),

        len(shared),
        int(matched.sum()),
        round(float(gross_total[~matched].sum()) / 1e6, 6),
        widest_cause_months,
    )


TURN_1_NAMES = [
    "claim_record_count", "cause_coded_claim_count",
    "distinct_cause_code_count", "leading_cause_claim_count",
]
TURN_2_NAMES = [
    "positive_building_coverage_claim_count", "building_coverage_exhausted_claim_count",
    "exhausted_gross_building_payment_usd_millions",
    "exhausted_at_maximum_residential_coverage_claim_count",
]
TURN_3_NAMES = [
    "icc_paid_claim_count", "icc_at_statutory_cap_claim_count",
    "icc_gross_payment_usd_millions", "exhausted_claim_icc_recipient_count",
]
TURN_4_NAMES = [
    "noaa_matched_county_month_count", "noaa_matched_claim_count",
    "unmatched_claim_gross_payment_usd_millions",
    "leading_noaa_flood_cause_county_month_count",
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        dict(outputs),
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
    "validate_cause_population": turn_validator(validate_turn_1),
    "validate_building_exhaustion": turn_validator(validate_turn_2),
    "validate_compliance_coverage": turn_validator(validate_turn_3),
    "validate_storm_scope_bridge": turn_validator(validate_turn_4),
}
