"""Who keeps a mortgage, who sells it, and the two ceilings on what can be sold.

Two questions sit on one field. Form HMDA records, for each originated loan, the
type of institution that bought it within the reporting year, and that single
column carries both the originate-to-distribute margin between depositories and
everyone else, and the boundary of what the two housing agencies are allowed to
buy at all.

The originator split needs the second source. HMDA names a lender only by its
legal entity identifier, so whether that lender is an insured depository can only
come from the supervisory institution file, which carries both the identifier and
a deposit-insurance certificate. Of the 4,869 identifiers that originated in
2024, 1,710 are tied to a certificate and 3,159 are not; the first group made
25.3243 percent of the 6,191,825 originations. The second group is named for what
it is -- identifiers the snapshot does not tie to a certificate -- and not for
what it is assumed to be: 3,106 of the 3,159 do not appear in the snapshot at all
and only 53 appear without a certificate, and the group also contains credit
unions, which are insured elsewhere.

The module keeps the further rule that an identifier carrying several
certificates is discarded, because one carrying several cannot be assigned a
supervisor. On this release that rule removes nothing -- no identifier carries
more than one, and a case-local test asserts the rule stays idle -- which is what
would tell a later author that the clause has started doing work.

The margin is large. Depository-linked originators sold 34.6269 percent of what
they originated inside the year; the rest sold 68.4937. The gap is not spread
evenly across buyers: the widest is the government mortgage agency at 10.7173
percentage points, 2.8598 against 13.5771, ahead of the next widest by 1.6271.

The purchase limit is the second half. The two housing agencies may buy a loan
only below a statutory amount, and the schedule has two tiers: a national figure
and a higher one for designated high-cost areas. Both are visible as cliffs in
the amount distribution of the 1,320,636 loans they bought. Grouping into ten
thousand dollar bands, the steepest fall in the half-million to one-and-a-half
million range is from the band opening at 1,140,000, which holds 1,307 loans
against the 20 in the band above it, a fall of 98.4698 percent. The second
steepest opens at 760,000 and falls 97.8944 percent, from 16,337 to 344.

Reading only the national figure turns the high-cost tier into an anomaly. 23,117
agency-bought loans exceed 766,550, and 47.0995 percent of them are in one state,
which leads the next by 471 percent; six states hold three quarters. A further
4,645 exceed even the high-cost ceiling, and they are not single-family jumbos:
their unit counts run above 149, then 50 to 99, then 100 to 149, so they are
multifamily loans under a different schedule again.

Conventions the query pins because each moves an answer. The population is loans
the filer originated, since the purchaser field is also populated on records the
filer bought rather than made. The band width is ten thousand dollars and bands
open at a multiple of it; the reported amounts are not themselves rounded to that
grid. The identifier match keeps only identifiers carrying exactly one
certificate, because an identifier carrying several cannot be assigned one
supervisor.

Fragility. The steepest and second steepest falls are 98.4698 and 97.8944
percent, 0.5754 apart, so the ordering of the two bands is the closest call the
case makes; both bands are reported, and a case-local test pins the gap.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for N
requested decimals; counts, the band boundaries, the buyer code and the state
match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


ORIGINATED = "1"
NOT_SOLD = "0"
AGENCY_CODES = ("1", "3")
BAND = 10_000
SEARCH_LOW = 500_000
SEARCH_HIGH = 1_500_000
NATIONAL_LIMIT = 766_550


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "origination_count",
    "certificate_linked_lender_count",
    "unlinked_lender_count",
    "certificate_linked_origination_share_pct",
]
TURN_2_NAMES = [
    "certificate_linked_sold_share_pct",
    "unlinked_sold_share_pct",
    "widest_buyer_gap_code",
    "widest_buyer_gap_pp",
]
TURN_3_NAMES = [
    "agency_purchased_count",
    "steepest_fall_band_floor",
    "steepest_fall_band_count",
    "steepest_fall_pct",
]
TURN_4_NAMES = [
    "second_steepest_fall_band_floor",
    "above_national_limit_count",
    "largest_above_limit_state",
    "largest_above_limit_state_share_pct",
]

variables = [
    _v(TURN_1_NAMES[0],
       "Store how many loans the filers originated in the file as an integer."),
    _v(TURN_1_NAMES[1],
       "Store how many originating identifiers carry exactly one certificate as an integer."),
    _v(TURN_1_NAMES[2],
       "Store how many originating identifiers do not as an integer."),
    _v(TURN_1_NAMES[3],
       "Store the first group's share of originations in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0],
       "Store the share of that group's originations sold inside the year in percent, rounded "
       "to 4 decimals."),
    _v(TURN_2_NAMES[1],
       "Store the other group's share in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2],
       "Store the buyer's source code verbatim as it appears in the file."),
    _v(TURN_2_NAMES[3],
       "Store that buyer's gap in percentage points, computed from the unrounded shares, "
       "rounded to 4 decimals."),
    _v(TURN_3_NAMES[0],
       "Store how many originated loans the two agencies bought that carry a reported amount as an integer."),
    _v(TURN_3_NAMES[1],
       "Store the opening amount of the band with the steepest fall in dollars as an integer."),
    _v(TURN_3_NAMES[2],
       "Store how many loans that band holds as an integer."),
    _v(TURN_3_NAMES[3],
       "Store that fall in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0],
       "Store the opening amount of the band with the second steepest fall in dollars as an "
       "integer."),
    _v(TURN_4_NAMES[1],
       "Store how many agency-bought loans exceed the stated national amount as an integer."),
    _v(TURN_4_NAMES[2],
       "Store the leading state's two-letter source code verbatim."),
    _v(TURN_4_NAMES[3],
       "Store that state's share of them in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 0, 4,
    4, 4, None, 4,
    0, 0, 0, 4,
    0, 0, None, 4,
]


@lru_cache(maxsize=1)
def _certificate_linked_identifiers():
    """Identifiers the supervisory file ties to exactly one deposit-insurance certificate."""
    institutions = load_expansion_table("ffiec_nic_institutions")
    frame = institutions.loc[
        institutions.lei.notna() & institutions.fdic_certificate.notna(),
        ["lei", "fdic_certificate"],
    ].copy()
    del institutions
    counts = frame.groupby(frame.lei.astype(str)).fdic_certificate.nunique()
    # An identifier tied to several certificates cannot be assigned one supervisor,
    # so it is left out rather than resolved by picking one.
    return frozenset(counts.loc[counts.eq(1)].index)


@lru_cache(maxsize=1)
def _originations():
    """Loans the filer originated, with the lender split and the buyer code."""
    records = load_expansion_table("hmda")
    frame = records.loc[
        records.action_taken.astype(str).eq(ORIGINATED),
        ["lei", "purchaser_type", "loan_amount", "state_code"],
    ].copy()
    del records
    frame["identifier"] = frame.lei.astype(str)
    frame["buyer"] = frame.purchaser_type.astype(str)
    frame["amount"] = pd.to_numeric(frame.loan_amount, errors="coerce")
    frame["certificate_linked"] = frame.identifier.isin(_certificate_linked_identifiers())
    if frame.empty:
        raise ValueError("the file reports no originated loans")
    return frame


def _band_counts(amounts):
    """Every band on the grid, including the ones no loan falls in.

    ``value_counts`` returns only populated bands, so a plain ``shift(-1)``
    would compare a band with the next *populated* one rather than with the band
    one width above it.  Reindexing onto the full grid keeps the comparison the
    query describes.
    """
    floors = (amounts // BAND) * BAND
    counts = floors.value_counts().sort_index()
    grid = range(int(counts.index.min()), int(counts.index.max()) + BAND, BAND)
    return counts.reindex(grid, fill_value=0)


@lru_cache(maxsize=1)
def ground_truth():
    originations = _originations()
    linked = originations.certificate_linked

    sold = originations.buyer.ne(NOT_SOLD)
    gaps = {}
    for code in originations.buyer.unique():
        if code == NOT_SOLD:
            continue
        taken = originations.buyer.eq(code)
        gaps[code] = 100.0 * (
            float(taken[~linked].mean()) - float(taken[linked].mean())
        )
    # The query breaks a tie by the lower code; sorting the codes first makes the
    # maximum stable when two buyers share a gap.
    widest = max(sorted(gaps, key=int), key=lambda code: gaps[code])

    agency = originations.loc[originations.buyer.isin(AGENCY_CODES)]
    amounts = agency.amount.dropna()
    counts = _band_counts(amounts)
    ratios = (counts.shift(-1) / counts).dropna()
    window = ratios.loc[SEARCH_LOW:SEARCH_HIGH]
    ordered = window.nsmallest(2)

    above = agency.loc[agency.amount.gt(NATIONAL_LIMIT)]
    # The query breaks a tie by the alphabetically earlier code.
    states = above.state_code.astype(str).value_counts()
    states = states.sort_index().sort_values(ascending=False, kind="mergesort")

    return (
        int(len(originations)),
        int(originations.loc[linked, "identifier"].nunique()),
        int(originations.loc[~linked, "identifier"].nunique()),
        100.0 * float(linked.mean()),
        100.0 * float(sold[linked].mean()),
        100.0 * float(sold[~linked].mean()),
        str(widest),
        gaps[widest],
        int(len(amounts)),
        int(ordered.index[0]),
        int(counts.loc[ordered.index[0]]),
        100.0 * (1.0 - float(ordered.iloc[0])),
        int(ordered.index[1]),
        int(len(above)),
        str(states.index[0]),
        100.0 * float(states.iloc[0]) / float(states.sum()),
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


def validate_lender_split(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_retention_and_channel(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_purchase_limit_cliff(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_high_cost_tier(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_lender_split": turn_validator(validate_lender_split),
    "validate_retention_and_channel": turn_validator(validate_retention_and_channel),
    "validate_purchase_limit_cliff": turn_validator(validate_purchase_limit_cliff),
    "validate_high_cost_tier": turn_validator(validate_high_cost_tier),
}
