"""Compare state HPI leaders with aligned covered-employment growth.

The 51-state-or-district population is linked from FHFA USPS abbreviations to
two-digit state FIPS through distinct Census county rows, then to BLS QCEW state
annual totals. For 2024-Q4 to 2025-Q4, Montana has the highest ending HPI level
at 726.20 but its HPI and total-covered-employment growth are -1.4841% and
-0.0842%. Delaware has the fastest HPI growth at 7.0184% with 0.7913%
employment growth. Idaho has the fastest employment growth at 1.7943% while
its HPI falls 0.9318%.

The query pins traditional purchase-only, unadjusted HPI and total-covered QCEW
employment. All-transactions HPI, seasonally adjusted HPI and private-only QCEW
employment are measured regression alternatives, not live traps, because those
choices are disclosed for uniqueness. Numeric validation uses 0.6 x 10^-N
rounding-boundary tolerance for N requested decimals. State label comparison
normalizes casing and repeated whitespace.
"""

from cave_agent import Variable

from core.data import load_expansion_table, qcew_state_totals
from core.validation import validate_ordered_outputs, turn_validator


STATE_OUTPUTS = {
    "highest_2025q4_index_state",
    "fastest_hpi_growth_state",
    "fastest_employment_growth_state",
}

variables = [
    Variable("matched_state_count", None, "Store the matched population count as an integer."),
    Variable("highest_2025q4_index_state", None, "Store the state with the highest 2025-Q4 HPI index level; casing and repeated whitespace are not significant."),
    Variable("highest_state_2024q4_index", None, "Store that state's 2024-Q4 HPI index, rounded to 2 decimals."),
    Variable("highest_state_2025q4_index", None, "Store that state's 2025-Q4 HPI index, rounded to 2 decimals."),
    Variable("highest_state_hpi_growth_pct", None, "Store that state's one-year HPI growth in percent, rounded to 4 decimals."),
    Variable("highest_state_employment_growth_pct", None, "Store that state's 2024-to-2025 total-covered annual-average employment growth in percent, rounded to 4 decimals."),
    Variable("fastest_hpi_growth_state", None, "Store the state with the fastest one-year HPI growth; casing and repeated whitespace are not significant."),
    Variable("fastest_hpi_state_2024q4_index", None, "Store that state's 2024-Q4 HPI index, rounded to 2 decimals."),
    Variable("fastest_hpi_state_2025q4_index", None, "Store that state's 2025-Q4 HPI index, rounded to 2 decimals."),
    Variable("fastest_hpi_growth_pct", None, "Store that state's one-year HPI growth in percent, rounded to 4 decimals."),
    Variable("fastest_hpi_state_employment_growth_pct", None, "Store that state's 2024-to-2025 total-covered annual-average employment growth in percent, rounded to 4 decimals."),
    Variable("fastest_employment_growth_state", None,
       "Store the state with the fastest 2024-to-2025 total-covered annual-average employment "
       "growth; casing and repeated whitespace are not significant."),
    Variable("fastest_employment_growth_pct", None, "Store that state's employment growth in percent, rounded to 4 decimals."),
    Variable("fastest_employment_state_hpi_growth_pct", None, "Store that state's one-year HPI growth in percent, rounded to 4 decimals."),
]


def _normalized_state(value):
    if not isinstance(value, str):
        return value
    return " ".join(value.split()).casefold()


def _joined_population():
    counties = load_expansion_table("census_counties").copy()
    counties["state_fips"] = counties.GEOID.str[:2]
    state_map = counties[["USPS", "state_fips"]].drop_duplicates()
    if state_map.USPS.duplicated().any():
        raise ValueError("Census USPS abbreviation maps to multiple state FIPS codes")

    hpi = load_expansion_table("fhfa_hpi")
    selected_hpi = hpi.loc[
        hpi.hpi_type.eq("traditional")
        & hpi.hpi_flavor.eq("purchase-only")
        & hpi.frequency.eq("quarterly")
        & hpi.level.eq("State")
        & hpi.period.eq(4)
        & hpi.yr.isin([2024, 2025])
    ]
    hpi_pivot = selected_hpi.pivot(
        index=["place_id", "place_name"], columns="yr", values="index_nsa"
    ).dropna().reset_index().rename(columns={2024: "hpi_2024", 2025: "hpi_2025"})
    hpi_pivot["hpi_growth"] = (
        hpi_pivot.hpi_2025 / hpi_pivot.hpi_2024 - 1
    ) * 100
    hpi_pivot = hpi_pivot.merge(
        state_map, left_on="place_id", right_on="USPS", validate="one_to_one"
    )

    qcew = qcew_state_totals(load_expansion_table("bls_qcew")).loc[
        lambda frame: frame.ownership_scope.eq("total_covered")
        & frame.aggregation_level.eq("state_total")
        & frame.industry_code.eq("10")
        & frame.year.isin([2024, 2025]),
        ["state_fips", "year", "annual_avg_employment"],
    ]
    employment = qcew.pivot(
        index="state_fips", columns="year", values="annual_avg_employment"
    ).dropna().reset_index().rename(
        columns={2024: "employment_2024", 2025: "employment_2025"}
    )
    employment["employment_growth"] = (
        employment.employment_2025 / employment.employment_2024 - 1
    ) * 100
    joined = hpi_pivot.merge(employment, on="state_fips", validate="one_to_one")
    if len(joined) != 51:
        raise ValueError(f"expected 51 matched states or districts, found {len(joined)}")
    return joined


def ground_truth():
    joined = _joined_population()
    level = joined.sort_values(
        ["hpi_2025", "place_name"], ascending=[False, True]
    ).iloc[0]
    hpi_growth = joined.sort_values(
        ["hpi_growth", "place_name"], ascending=[False, True]
    ).iloc[0]
    employment_growth = joined.sort_values(
        ["employment_growth", "place_name"], ascending=[False, True]
    ).iloc[0]
    return (
        len(joined),
        level.place_name,
        level.hpi_2024,
        level.hpi_2025,
        level.hpi_growth,
        level.employment_growth,
        hpi_growth.place_name,
        hpi_growth.hpi_2024,
        hpi_growth.hpi_2025,
        hpi_growth.hpi_growth,
        hpi_growth.employment_growth,
        employment_growth.place_name,
        employment_growth.employment_growth,
        employment_growth.hpi_growth,
    )


def validate(outputs):
    normalized_outputs = dict(outputs)
    for name in STATE_OUTPUTS:
        normalized_outputs[name] = _normalized_state(outputs.get(name))
    expected = list(ground_truth())
    for index, variable in enumerate(variables):
        if variable.name in STATE_OUTPUTS:
            expected[index] = _normalized_state(expected[index])
    return validate_ordered_outputs(
        normalized_outputs,
        variables,
        expected,
        [0, None, 2, 2, 4, 4, None, 2, 2, 4, 4, None, 4, 4],
    )



TURN_1_NAMES = [
    "matched_state_count",
    "highest_2025q4_index_state",
    "highest_state_2024q4_index",
    "highest_state_2025q4_index",
    "highest_state_hpi_growth_pct",
    "highest_state_employment_growth_pct",
]
TURN_2_NAMES = [
    "fastest_hpi_growth_state",
    "fastest_hpi_state_2024q4_index",
    "fastest_hpi_state_2025q4_index",
    "fastest_hpi_growth_pct",
    "fastest_hpi_state_employment_growth_pct",
]
TURN_3_NAMES = [
    "fastest_employment_growth_state",
    "fastest_employment_growth_pct",
    "fastest_employment_state_hpi_growth_pct",
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
    "validate_index_level_leader": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_price_growth_leader": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_employment_leader": turn_validator(_turn_subset(TURN_3_NAMES)),
}
