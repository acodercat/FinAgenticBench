"""Map a bank to a county by its ZIP code and measure how often that is wrong.

Postal geography and county geography are different partitions of the same land, and
the Census relationship file says how they overlap: 46,960 rows pair 33,791 ZIP code
tabulation areas with 3,232 counties, each row carrying the land and water area the
two share, and no pair appearing twice. 10,186 of the areas, 30.1441 percent, reach
into more than one county, and one reaches into 6.

The Census also publishes a single primary county per area, chosen by the largest
shared land area with water area breaking ties. That rule is exactly reproducible:
applied to the relationship file it returns the published county for all 33,791
areas, with 0 disagreements. What the rule cannot do is make a split area
unambiguous. Across the 10,186 split areas the largest county holds a median of
82.8811 percent of the land, but the tenth percentile is 55.1992 percent and the
smallest is 30.1308; on 1,629 of them, 15.9925 percent, the largest county holds
under 60 percent. That threshold identifies weak dominance; it does not establish
that the published selection is random.

The FFIEC structure file records both a postal code and a county for the same
institution, which turns this into a measurement rather than an argument. Of its
61,914 institutions, 52,324 carry a five-digit postal code together with a county
code, and 98.3927 percent of the counties so recorded appear in the county file.
4,953 institution records have postal codes that match no tabulation area in the
published primary-county file, leaving 47,371 institutions to test. On those,
assigning a county from the postal code through the primary rule returns the
recorded county 95.6091 percent of the time.

The 4.4 percent that miss are not scattered at random. Splitting the tested
institutions by how many counties their area reaches, the rule is right 96.9805
percent of the time on the 39,377 in areas confined to one county, 90.1042 percent
on the 5,760 in areas reaching two, and 85.6312 percent on the 2,234 in areas
reaching three or more. The error is predictable from the crosswalk itself before
any answer is checked, which means an analysis can report which of its own
assignments are unreliable. The residual error on single-county areas is not
explained here and has other sources, among them a recorded county that is wrong
and a postal address that is not the site.

The `zip_county_crosswalk_error` pinned-reading comparison records sensitivity to combining
land and water, retaining full postal strings, including unmatched records in the
denominator, restricting institution types, and changing the reach bands. These are
alternative population or mapping conventions rather than hidden answer paths.

Interpretive boundary: a tabulation area is the geographic key available in these
tables, and assigning the whole area to one primary county is an approximation.
The observed cross-county reach can flag records exposed to that approximation;
it cannot identify the correct county for an individual institution.

Boundaries: the relationship file is the 2020 tabulation and the county file is
2024, so a county changed in between is matched by identifier rather than by
boundary. Institutions are counted one per record, not weighted by size, so this
measures how often a mapping is wrong and not how much of the banking system it
misplaces. The recorded county is treated as truth, which it is not always.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


WEAK_PRIMARY_THRESHOLD = 0.60
MANY_COUNTY_THRESHOLD = 3


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["relationship_rows", "area_count", "county_count", "duplicate_pairs",
                "split_area_count", "split_area_share_pct", "maximum_counties"]
TURN_2_NAMES = ["reproduced_areas", "rule_disagreements", "median_largest_share_pct",
                "tenth_percentile_share_pct", "minimum_share_pct", "weak_primary_count",
                "weak_primary_share_pct"]
TURN_3_NAMES = ["eligible_institutions", "recorded_county_present_pct", "unmatched_postal_codes",
                "tested_institutions", "assignment_accuracy_pct"]
TURN_4_NAMES = ["single_county_tested", "single_county_accuracy_pct", "two_county_tested",
                "two_county_accuracy_pct", "many_county_tested", "many_county_accuracy_pct"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many rows the relationship file carries as an integer."),
    _v(TURN_1_NAMES[1], "Store how many distinct tabulation areas appear as an integer."),
    _v(TURN_1_NAMES[2], "Store how many distinct counties appear as an integer."),
    _v(TURN_1_NAMES[3], "Store how many rows repeat an area-and-county pair already present as an integer."),
    _v(TURN_1_NAMES[4], "Store how many areas reach into more than one county as an integer."),
    _v(TURN_1_NAMES[5], "Store that as a percent of all areas, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store the largest number of counties any one area reaches as an integer."),
    _v(TURN_2_NAMES[0], "Store how many areas the derived rule assigns a county to as an integer."),
    _v(TURN_2_NAMES[1], "Store how many of them disagree with the published primary county as an integer."),
    _v(TURN_2_NAMES[2], "Store the median share of a split area's land held by its largest county, in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the tenth percentile of that share, in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store its minimum, in percent, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store how many split areas have a largest county below 60 percent as an integer."),
    _v(TURN_2_NAMES[6], "Store that as a percent of split areas, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store how many institutions carry both a five-digit postal code and a county code as an integer."),
    _v(TURN_3_NAMES[1], "Store the percent of their recorded counties present in the county file, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store how many eligible institution records have a postal code matching no tabulation area as an integer."),
    _v(TURN_3_NAMES[3], "Store how many institutions remain to test as an integer."),
    _v(TURN_3_NAMES[4], "Store the percent where the assigned county equals the recorded one, rounded to 4 decimals."),
    _v(TURN_4_NAMES[0], "Store how many tested institutions sit in an area confined to one county as an integer."),
    _v(TURN_4_NAMES[1], "Store the accuracy on them in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store how many sit in an area reaching two counties as an integer."),
    _v(TURN_4_NAMES[3], "Store the accuracy on them in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store how many sit in an area reaching three or more as an integer."),
    _v(TURN_4_NAMES[5], "Store the accuracy on them in percent, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 0, 0, 0, 4, 0,
            0, 0, 4, 4, 4, 0, 4,
            0, 4, 0, 0, 4,
            0, 4, 0, 4, 0, 4]


@lru_cache(maxsize=1)
def _relationships():
    frame = load_expansion_table("census_zcta_county_relationships").copy()
    frame["area"] = frame.GEOID_ZCTA5_20.astype(str).str.zfill(5)
    frame["county"] = frame.GEOID_COUNTY_20.astype(str).str.zfill(5)
    frame["land"] = pd.to_numeric(frame.AREALAND_PART, errors="coerce")
    frame["water"] = pd.to_numeric(frame.AREAWATER_PART, errors="coerce")
    return frame


@lru_cache(maxsize=1)
def _primary():
    frame = load_expansion_table("census_zcta_primary_counties").copy()
    frame["area"] = frame.zcta5.astype(str).str.zfill(5)
    frame["county"] = frame.county_geoid.astype(str).str.zfill(5)
    return frame[["area", "county"]]


@lru_cache(maxsize=1)
def _counties():
    frame = load_expansion_table("census_counties").copy()
    frame["geoid"] = frame.GEOID.astype(str).str.zfill(5)
    frame["state_fips"] = frame.geoid.str[:2]
    return frame


@lru_cache(maxsize=1)
def _tested():
    counties = _counties()
    state_fips = counties.drop_duplicates("USPS").set_index("USPS").state_fips.to_dict()
    institutions = load_expansion_table("ffiec_nic_institutions").copy()
    institutions["postal"] = institutions.zip_code.astype(str).str.extract(r"^(\d{5})")[0]
    institutions["county_number"] = pd.to_numeric(institutions.county_code, errors="coerce")
    institutions["recorded"] = (
        institutions.state.astype(str).map(state_fips).fillna("")
        + institutions.county_number.apply(lambda value: "" if pd.isna(value) else str(int(value)).zfill(3))
    )
    eligible = institutions.loc[
        institutions.postal.notna() & institutions.county_number.notna()
        & institutions.recorded.str.len().eq(5)
    ].copy()
    assigned = eligible.merge(
        _primary().rename(columns={"area": "postal", "county": "assigned"}), on="postal", how="left",
    )
    counts = _relationships().groupby("area").county.nunique().rename("reach")
    assigned = assigned.merge(counts, left_on="postal", right_index=True, how="left")
    return eligible, assigned


@lru_cache(maxsize=1)
def ground_truth():
    relationships = _relationships()
    published = _primary()
    counties = _counties()
    eligible, assigned = _tested()

    reach = relationships.groupby("area").county.nunique()
    ordered = relationships.sort_values(
        ["area", "land", "water", "county"], ascending=[True, False, False, True],
    )
    derived = ordered.drop_duplicates("area")[["area", "county"]].rename(columns={"county": "derived"})
    checked = published.merge(derived, on="area", how="left")

    totals = relationships.groupby("area").land.sum()
    largest = ordered.drop_duplicates("area").set_index("area").land
    share = (largest / totals.replace(0, pd.NA)).loc[reach[reach.gt(1)].index]

    tested = assigned.loc[assigned.assigned.notna()].copy()
    tested["correct"] = tested.assigned.eq(tested.recorded)

    def band(mask):
        rows = tested.loc[mask]
        return int(len(rows)), float(rows.correct.mean()) * 100

    return (
        int(len(relationships)), int(relationships.area.nunique()), int(relationships.county.nunique()),
        int(relationships.duplicated(["area", "county"]).sum()),
        int(reach.gt(1).sum()), float(reach.gt(1).mean()) * 100, int(reach.max()),
        int(checked.derived.notna().sum()), int(checked.county.ne(checked.derived).sum()),
        float(share.median()) * 100, float(share.quantile(0.10)) * 100, float(share.min()) * 100,
        int(share.lt(WEAK_PRIMARY_THRESHOLD).sum()), float(share.lt(WEAK_PRIMARY_THRESHOLD).mean()) * 100,
        int(len(eligible)), float(eligible.recorded.isin(set(counties.geoid)).mean()) * 100,
        int(assigned.assigned.isna().sum()), int(len(tested)), float(tested.correct.mean()) * 100,
        *band(tested.reach.eq(1)), *band(tested.reach.eq(2)),
        *band(tested.reach.ge(MANY_COUNTY_THRESHOLD)),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_crosswalk_shape": turn_validator(validate_turn_1),
    "validate_primary_rule": turn_validator(validate_turn_2),
    "validate_assignment_accuracy": turn_validator(validate_turn_3),
    "validate_error_structure": turn_validator(validate_turn_4),
}
