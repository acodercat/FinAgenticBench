"""Two federal responses to the same flood, over two different sets of counties.

A disaster declaration designates counties one at a time and names which
programmes each one gets. Flood insurance pays policyholders wherever they are.
The two footprints are not the same and do not line up the way a county count
suggests. Of the 1,193 counties with a 2024 flood claim, 811 were never inside
a major disaster declaration for a water-driven incident that year, so by count
most insured loss sits outside the declaration system; those 811 counties paid
247.0510 USD millions against 8798.2578 in total, so by money almost none of it
does.

Four things about the declaration file decide the population:

1. ``individual_assistance_declared`` is false on every one of the 19,167 rows
   in this extract, so a screen written against it returns nothing. The live
   household flag is ``individuals_households_program_declared``, true on 368
   of the rows here. Both names describe household aid and only one carries
   information.
2. Emergency declarations never authorise household assistance: all 368 rows
   carrying that flag are major disaster declarations, and none of the 813
   emergency rows do. Counting designations without separating the declaration
   type puts 877 counties in the population where 630 can receive the
   programme being measured.
3. A code ending 000 is a tribal or statewide designation rather than a county,
   and the 21 such rows here name areas as distinct as the Eastern Band of
   Cherokee Indians and the St. Regis Mohawk Indian Reservation while sharing
   six state-level codes between them. Keeping them would put 1,646 rows and
   883 counties in the population and would count several tribal areas as one
   place. The claim side drops the same codes, so the two sides are screened
   alike.
4. A county is designated once per disaster, and 149 of them are designated
   under more than one 2024 disaster. A county counts as having the household
   programme if any of its designations carries the flag, which is what makes
   the county rather than the designation the unit.

Two conventions on the claim side. Payments are the signed net building,
contents and increased-cost-of-compliance amounts: the gross columns are
absent on 5,029 of these rows while the net columns are complete, and gross
gives 8801.1170 against the reported 8798.2578, while dropping the
compliance component gives 8787.3735. And 27 claims carry a negative net
payment and 16,444 a zero; both are reported outcomes of a claim and neither
is filtered.

The incident window is the incident begin date rather than the declaration
date, which follows the event instead of the paperwork and moves the row count
from 1,642 to 1,625. The three incident types are the ones whose name states a
water-driven event. The 2024 window holds six others; the largest excluded is
Severe Storm at 491 rows, whose admission would take the designated county
count under a major declaration from 630 to 964, and the smallest is a single
Mud/Landslide row.

One reading the query leaves open because the data closes it: the household
programme row count is 368 whether it is struck over the whole window or over
the major declarations alone, since no emergency row carries the flag. The
county figures are not like that, and the query names the major declarations
for each of them: multiply designated counties read 149 inside them against
411 across every designation, and the paying counties outside them number 811
against 685 outside every designation.

One coincidence worth naming so it is not mistaken for a relation: the 368
household rows happen to equal the 49 disasters plus the 319 household
counties. The three are counted over different things and nothing ties them
together; the equality is arithmetic accident at these values.

Nothing here says the two programmes should agree. Flood insurance pays a
contract to a policyholder; household assistance is a grant of last resort for
needs insurance did not meet. A county with a large paid loss and no household
designation is what the two rules produce, not evidence that either failed.
"""

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


INCIDENT_YEAR = "2024"
INCIDENT_TYPES = ("Flood", "Hurricane", "Tropical Storm")
MAJOR_DECLARATION = "DR"
HOUSEHOLD_FLAG = "individuals_households_program_declared"
PUBLIC_FLAG = "public_assistance_declared"
NET_PAYMENTS = (
    "net_building_payment_usd",
    "net_contents_payment_usd",
    "net_icc_payment_usd",
)
MILLION = Decimal(10) ** 6

OUTPUTS = (
    ("designation_row_count", 0, "the count of designated-area rows in the window as an integer"),
    ("disaster_count", 0, "the count of distinct disasters behind them as an integer"),
    ("designated_county_count", 0, "the count of distinct counties designated as an integer"),
    ("major_declaration_row_count", 0, "how many of those rows are major disaster declarations, as an integer"),
    ("ihp_declared_row_count", 0, "the count of rows carrying the household programme as an integer"),
    ("ihp_county_count", 0, "the count of counties carrying it on any designation as an integer"),
    ("public_assistance_only_county_count", 0, "the count carrying public assistance without it as an integer"),
    ("multiply_designated_county_count", 0, "the count designated under more than one major disaster declaration, as an integer"),
    ("claim_record_count", 0, "the count of retained claim records as an integer"),
    ("claim_net_paid_mn", 4, "their net paid amount in USD millions"),
    ("matched_county_count", 0, "the count of counties both designated and paying claims, as an integer"),
    ("matched_ihp_county_count", 0, "how many of those carry the household programme, as an integer"),
    ("ihp_county_paid_mn", 4, "the net paid amount in those counties, in USD millions"),
    ("non_ihp_county_paid_mn", 4, "the net paid amount in the designated counties without it, in USD millions"),
    ("undeclared_county_count", 0, "the count of paying counties outside every major declaration, as an integer"),
    ("undeclared_county_paid_mn", 4, "their net paid amount in USD millions"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:8]),
    tuple(n for n, _, _ in OUTPUTS[8:12]),
    tuple(n for n, _, _ in OUTPUTS[12:]),
)


@lru_cache(maxsize=1)
def _designations():
    frame = load_expansion_table("fema_disasters")
    window = frame.loc[
        frame.incident_begin_date.astype(str).str[:4].eq(INCIDENT_YEAR)
        & frame.incident_type.isin(INCIDENT_TYPES)
    ].copy()
    if window.empty:
        raise ValueError("no designation falls in the incident window")
    if window.county_fips.isna().any():
        raise ValueError("a designated area reports no county code")
    window["county_fips"] = window.county_fips.astype(str)
    # A code ending 000 is a tribal or statewide designation rather than a
    # county, and several distinct tribal areas in one state share the code.
    window = window.loc[~window.county_fips.str.endswith("000")]
    if window.empty:
        raise ValueError("every designation in the window is tribal or statewide")
    return window


@lru_cache(maxsize=1)
def _counties():
    """One row per county, carrying the programmes any of its designations did."""
    major = _designations()
    major = major.loc[major.declaration_type.eq(MAJOR_DECLARATION)]
    if major.empty:
        raise ValueError("no major disaster declaration falls in the window")
    return major.groupby("county_fips").agg(
        household=(HOUSEHOLD_FLAG, "max"),
        public=(PUBLIC_FLAG, "max"),
        disasters=("disaster_number", "nunique"),
    )


@lru_cache(maxsize=1)
def _claims():
    frame = load_expansion_table("fema_nfip")
    claims = frame.loc[frame.date_of_loss.astype(str).str[:4].eq(INCIDENT_YEAR)].copy()
    claims["county_fips"] = claims.county_code.astype(str)
    # A statewide or unallocated code cannot be placed in a county.
    claims = claims.loc[
        claims.county_fips.str.len().eq(5) & ~claims.county_fips.str.endswith("000")
    ]
    if claims.empty:
        raise ValueError("no claim in the window carries a county code")
    if claims[list(NET_PAYMENTS)].isna().any().any():
        raise ValueError("a retained claim reports no net payment")
    claims["paid_usd"] = [
        sum((Decimal(str(row[column])) for column in NET_PAYMENTS), Decimal(0))
        for _, row in claims[list(NET_PAYMENTS)].iterrows()
    ]
    return claims


def _paid(claims, counties):
    return sum(
        (v for k, v in claims.items() if k in counties), Decimal(0)
    )


@lru_cache(maxsize=1)
def analysis():
    designations = _designations()
    counties = _counties()
    claims = _claims()

    by_county = claims.groupby("county_fips").paid_usd.apply(
        lambda values: sum(values, Decimal(0))
    ).to_dict()

    household = frozenset(counties.index[counties.household])
    public_only = counties.index[~counties.household & counties.public]
    paying = frozenset(by_county)
    matched = paying & frozenset(counties.index)
    if not matched:
        raise ValueError("no designated county reports a claim")
    undeclared = paying - frozenset(counties.index)

    return {
        "designation_row_count": len(designations),
        "disaster_count": int(designations.disaster_number.nunique()),
        "designated_county_count": int(designations.county_fips.nunique()),
        "major_declaration_row_count": int(
            designations.declaration_type.eq(MAJOR_DECLARATION).sum()
        ),
        "ihp_declared_row_count": int(designations[HOUSEHOLD_FLAG].sum()),
        "ihp_county_count": len(household),
        "public_assistance_only_county_count": len(public_only),
        "multiply_designated_county_count": int((counties.disasters > 1).sum()),
        "claim_record_count": len(claims),
        "claim_net_paid_mn": float(
            sum(by_county.values(), Decimal(0)) / MILLION
        ),
        "matched_county_count": len(matched),
        "matched_ihp_county_count": len(matched & household),
        "ihp_county_paid_mn": float(_paid(by_county, matched & household) / MILLION),
        "non_ihp_county_paid_mn": float(_paid(by_county, matched - household) / MILLION),
        "undeclared_county_count": len(undeclared),
        "undeclared_county_paid_mn": float(_paid(by_county, undeclared) / MILLION),
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
