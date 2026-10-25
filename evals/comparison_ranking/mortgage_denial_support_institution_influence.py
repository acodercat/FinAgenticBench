"""Standardize a linked mortgage denial-rate gap and test lender influence.

The retrospective linked population uses current NIC identifiers observed
2026-08-13 to map HMDA LEIs with exactly one nonmissing FDIC certificate.
No separate FDIC financial-report source is required. It contains 175 represented
LEIs before the race restriction.  Among 2024 California owner-occupied,
first-lien home-purchase applications whose outcome is originated, approved but
not accepted, or denied, Black or African American and White groups contain
1,022 and 18,503 applications, with 21.8200 and 13.5275 percent denied.

Seven county-by-loan-type cells have at least thirty applications in each group.
They retain 678 and 9,631 applications and have an unweighted pooled gap of
7.8239 percentage points.  Direct standardization to each retained cell's share
of the two groups' combined applications gives 6.4440 points.  This separates
common-support restriction from within-support reweighting; neither step is a
causal adjustment for unobserved credit risk or underwriting information.

Keeping those cells and original weights fixed, 153 lender omissions preserve
positive group denominators throughout.  Removing LEI 593C3GZG957YOJPS2Z63
(City National Bank in the frozen HMDA name field) produces the largest absolute
change and a 5.6482-point gap.  Re-estimating weights after each omission instead
gives 5.6053 for that omission and answers a different question.  Excluding
approved-not-accepted outcomes changes the two raw rates to 22.5709 and 13.9148
percent.  The public query pins both conventions.

This is a baseline: the support, weights and leave-one-out policy are explicit
to ensure one financial object.  A synthetic swap of the two most influential
LEI labels leaves every turn-1 through turn-3 aggregate unchanged but changes
the turn-4 winning LEI, demonstrating that the terminal result requires lender-
level observations rather than earlier outputs alone.
"""

from fractions import Fraction
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs

ACTIVITY_YEAR = "2024"
STATE = "CA"
DECIDED_ACTIONS = {"1", "2", "3"}
# The pinned outcome reading first, then the alternatives the pinned-reading comparison measures.
OUTCOME_ACTIONS = {
    "decided": DECIDED_ACTIONS,
    "accepted_or_denied": {"1", "3"},
    "all_actions": {"1", "2", "3", "4", "5"},
}
DENIED_ACTION = "3"
RACES = ("Black or African American", "White")
MINIMUM_CELL_COUNT = 30


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "linked_bank_count",
    "black_application_count",
    "white_application_count",
    "black_denial_pct",
    "white_denial_pct",
]
TURN_2_NAMES = [
    "support_cell_count",
    "black_support_count",
    "white_support_count",
    "support_gap_pp",
]
TURN_3_NAMES = ["standardized_gap_pp"]
TURN_4_NAMES = ["influential_lei", "leave_out_gap_pp", "valid_leave_out_count"]

variables = [
    _v(
        TURN_1_NAMES[0],
        "Store the represented linked-lender LEI count before the race restriction as an integer.",
    ),
    _v(TURN_1_NAMES[1], "Store the Black or African American application count as an integer."),
    _v(TURN_1_NAMES[2], "Store the White application count as an integer."),
    _v(
        TURN_1_NAMES[3],
        "Store the Black or African American denial rate in percent, rounded to 4 decimals.",
    ),
    _v(TURN_1_NAMES[4], "Store the White denial rate in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store the retained county-by-loan-type cell count as an integer."),
    _v(
        TURN_2_NAMES[1],
        "Store the retained Black or African American application count as an integer.",
    ),
    _v(TURN_2_NAMES[2], "Store the retained White application count as an integer."),
    _v(
        TURN_2_NAMES[3],
        "Store the retained population's unweighted denial-rate gap in percentage points, rounded to 4 decimals.",
    ),
    _v(
        TURN_3_NAMES[0],
        "Store the directly standardized denial-rate gap in percentage points, rounded to 4 decimals.",
    ),
    _v(TURN_4_NAMES[0], "Store the selected omitted lender's exact 20-character LEI string."),
    _v(
        TURN_4_NAMES[1],
        "Store the selected omission's standardized denial-rate gap in percentage points, rounded to 4 decimals.",
    ),
    _v(TURN_4_NAMES[2], "Store the valid institutional omission count as an integer."),
]

DECIMALS = [0, 0, 0, 4, 4, 0, 0, 0, 4, 4, None, 4, 0]


def _rate(numerator, denominator):
    if int(denominator) <= 0:
        raise ValueError("rate denominator must be positive")
    return Fraction(int(numerator), int(denominator))


def _standardized_gap(cells, weights):
    result = Fraction(0)
    for cell, weight in weights.items():
        black = _rate(
            cells.loc[cell, ("denials", RACES[0])],
            cells.loc[cell, ("applications", RACES[0])],
        )
        white = _rate(
            cells.loc[cell, ("denials", RACES[1])],
            cells.loc[cell, ("applications", RACES[1])],
        )
        result += weight * (black - white)
    return 100 * result


@lru_cache(maxsize=None)
def _population(outcomes="decided"):
    """The linked two-group application sample under one outcome reading."""
    nic = load_expansion_table("ffiec_nic_institutions")
    mappings = nic.loc[nic.lei.notna() & nic.fdic_certificate.notna()].copy()
    distinct_certificates = mappings.groupby("lei").fdic_certificate.nunique()
    mappings = mappings.loc[
        mappings.lei.isin(distinct_certificates.loc[distinct_certificates.eq(1)].index)
    ].drop_duplicates("lei")
    eligible_leis = set(mappings.lei.astype(str))
    hmda = load_expansion_table("hmda")
    applications = hmda.loc[
        hmda.activity_year.astype(str).eq(ACTIVITY_YEAR)
        & hmda.state_code.astype(str).eq(STATE)
        & hmda.loan_purpose.astype(str).eq("1")
        & hmda.lien_status.astype(str).eq("1")
        & hmda.occupancy_type.astype(str).eq("1")
        & hmda.action_taken.astype(str).isin(OUTCOME_ACTIONS[outcomes])
        & hmda.lei.astype(str).isin(eligible_leis)
    ].copy()
    if applications.empty:
        raise ValueError("linked application population is empty")
    linked_bank_count = int(applications.lei.astype(str).nunique())
    comparison = applications.loc[
        applications.derived_race.astype(str).isin(RACES),
        ["lei", "county_code", "loan_type", "derived_race", "action_taken"],
    ].copy()
    comparison["lei"] = comparison.lei.astype(str)
    comparison["derived_race"] = comparison.derived_race.astype(str)
    comparison["denials"] = comparison.action_taken.astype(str).eq(DENIED_ACTION).astype(int)
    comparison["applications"] = 1
    return linked_bank_count, comparison


def _cell_weights(cells, weight_basis):
    """Standardization weights over the retained cells."""
    if weight_basis == "combined":
        counts = {cell: int(row.applications.sum()) for cell, row in cells.iterrows()}
    else:
        counts = {cell: int(row.applications[RACES[1]]) for cell, row in cells.iterrows()}
    total = sum(counts.values())
    return {cell: Fraction(count, total) for cell, count in counts.items()}


@lru_cache(maxsize=None)
def _analysis(outcomes="decided", weight_basis="combined", leave_out_weights="fixed"):
    """Every output; the defaults are the conventions the query pins."""
    linked_bank_count, comparison = _population(outcomes)
    cell_keys = ["county_code", "loan_type", "derived_race"]
    full = comparison.groupby("derived_race")[["applications", "denials"]].sum()
    if any(race not in full.index for race in RACES):
        raise ValueError("both comparison groups must be present")

    # Default groupby dropna=True excludes rows without a complete cell key from
    # common support while retaining them in the turn-1 totals.
    grouped = (
        comparison.groupby(cell_keys)[["applications", "denials"]].sum().unstack("derived_race")
    )
    cells = grouped.dropna().loc[lambda x: (x.applications >= MINIMUM_CELL_COUNT).all(axis=1)]
    if cells.empty:
        raise ValueError("common support is empty")
    weights = _cell_weights(cells, weight_basis)
    if sum(weights.values()) != 1:
        raise ValueError("standardization weights do not sum to one")
    standardized = _standardized_gap(cells, weights)

    leave_one_out = []
    for lei in sorted(comparison.lei.unique()):
        remaining = (
            comparison.loc[comparison.lei.ne(lei)]
            .groupby(cell_keys)[["applications", "denials"]]
            .sum()
            .unstack("derived_race")
            .reindex(cells.index)
        )
        if remaining.applications.isna().any().any() or remaining.applications.le(0).any().any():
            continue
        omission_weights = (
            weights if leave_out_weights == "fixed" else _cell_weights(remaining, weight_basis)
        )
        result = _standardized_gap(remaining, omission_weights)
        leave_one_out.append((abs(result - standardized), lei, result))
    leave_one_out.sort(key=lambda item: (-item[0], item[1]))
    if not leave_one_out:
        raise ValueError("no valid lender omission")

    raw_rates = {
        race: _rate(full.loc[race, "denials"], full.loc[race, "applications"]) for race in RACES
    }
    support_rates = {
        race: _rate(cells.denials[race].sum(), cells.applications[race].sum()) for race in RACES
    }
    winner = leave_one_out[0]
    return (
        linked_bank_count,
        int(full.loc[RACES[0], "applications"]),
        int(full.loc[RACES[1], "applications"]),
        float(100 * raw_rates[RACES[0]]),
        float(100 * raw_rates[RACES[1]]),
        int(len(cells)),
        int(cells.applications[RACES[0]].sum()),
        int(cells.applications[RACES[1]].sum()),
        float(100 * (support_rates[RACES[0]] - support_rates[RACES[1]])),
        float(standardized),
        str(winner[1]),
        float(winner[2]),
        int(len(leave_one_out)),
    )


@lru_cache(maxsize=1)
def ground_truth():
    return _analysis()


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


def validate_turn_1(outputs):
    return _validate_subset(outputs, TURN_1_NAMES)


def validate_turn_2(outputs):
    return _validate_subset(outputs, TURN_2_NAMES)


def validate_turn_3(outputs):
    return _validate_subset(outputs, TURN_3_NAMES)


def validate_turn_4(outputs):
    return _validate_subset(outputs, TURN_4_NAMES)


def validate(outputs):
    return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_linked_decision_population": turn_validator(validate_turn_1),
    "validate_common_support": turn_validator(validate_turn_2),
    "validate_direct_standardization": turn_validator(validate_turn_3),
    "validate_lender_influence": turn_validator(validate_turn_4),
}
