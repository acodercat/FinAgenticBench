"""Rank credit stock, trend gap and debt-service burden on one country cohort.

For the 32 countries with all three requested 2025-Q3 measures, Hong Kong SAR
leads both the actual private-sector credit-to-GDP ratio (332.4073%) and the
private-sector DSR (34.1%), while its credit gap ranks 31st at -54.2594 points.
Japan instead leads the credit gap at +5.1514 points, but ranks 14th on the
actual ratio and 18th on DSR.

Using 2025-Q4, a household rather than private-sector DSR, the credit trend as
the level measure, or the actual ratio as the gap measure changes the population,
leaders, values or ranks.  The query pins all four choices, so these are
regression probes for a hard baseline rather than live traps.  Numeric validation
uses 0.6 x 10^-N rounding-boundary tolerance; ranks and the population count are
exact, and country labels accept the source's optional two-letter prefix.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


COUNTRY = "BORROWERS_CTY:Borrowers' country"
PERIOD = "TIME_PERIOD:Time period or range"
VALUE = "OBS_VALUE:Observation Value"
ACTUAL = "A: Credit-to-GDP ratios (actual data)"
GAP = "C: Credit-to-GDP gaps (actual-trend)"
PRIVATE = "P: Private non-financial sector"

variables = [
    Variable("joint_country_count", None, "Store the common-country count as an integer."),
    Variable("highest_actual_credit_ratio_country", None, "Store the country name leading the actual credit-to-GDP ratio ranking; an optional two-letter source prefix is accepted."),
    Variable("highest_actual_credit_ratio_pct_of_gdp", None, "Store the leading actual credit-to-GDP ratio in percent of GDP, rounded to 4 decimals."),
    Variable("highest_credit_gap_country", None, "Store the country name leading the credit-gap ranking; an optional two-letter source prefix is accepted."),
    Variable("highest_credit_gap_percentage_points", None, "Store the leading actual-minus-trend credit gap in percentage points, rounded to 4 decimals."),
    Variable("highest_dsr_country", None, "Store the country name leading the debt-service-ratio ranking; an optional two-letter source prefix is accepted."),
    Variable("highest_dsr_pct", None, "Store the leading debt-service ratio in percent, rounded to 1 decimal."),
    Variable("actual_ratio_leader_credit_gap_rank", None, "Store the actual-ratio leader's ordinal credit-gap rank as an integer."),
    Variable("credit_gap_leader_actual_ratio_rank", None, "Store the credit-gap leader's ordinal actual-ratio rank as an integer."),
    Variable("credit_gap_leader_dsr_rank", None, "Store the credit-gap leader's ordinal debt-service-ratio rank as an integer."),
]


def _display_country(value):
    text = " ".join(str(value).split())
    prefix, separator, remainder = text.partition(": ")
    return remainder if separator and len(prefix) == 2 else text


def _normalized_country(value):
    return _display_country(value).casefold() if isinstance(value, str) else value


def _population():
    credit = load_expansion_table("bis_credit_gap")
    selected = credit.loc[
        credit["TC_BORROWERS:Borrowing sector"].eq(PRIVATE)
        & credit["TC_LENDERS:Lending sector"].str.startswith("A:")
        & credit[PERIOD].eq("2025-Q3")
    ]
    credit_measures = selected.pivot_table(
        index=COUNTRY,
        columns="CG_DTYPE:Credit gap data type",
        values=VALUE,
        aggfunc="first",
    ).dropna(subset=[ACTUAL, GAP])[[ACTUAL, GAP]]

    dsr = load_expansion_table("bis_debt_service")
    private_dsr = dsr.loc[
        dsr["DSR_BORROWERS:Borrowers"].eq(PRIVATE)
        & dsr[PERIOD].eq("2025-Q3")
    ].set_index(COUNTRY)[VALUE].rename("dsr")
    population = credit_measures.join(private_dsr, how="inner").dropna().reset_index()
    if len(population) != 32 or population[COUNTRY].duplicated().any():
        raise ValueError("unexpected BIS joint-country population")
    return population


def _rank_map(population, measure):
    ordered = population.assign(
        _country_name=population[COUNTRY].map(_display_country)
    ).sort_values([measure, "_country_name"], ascending=[False, True])
    return {country: rank for rank, country in enumerate(ordered[COUNTRY], start=1)}


def ground_truth():
    population = _population()
    actual_ranks = _rank_map(population, ACTUAL)
    gap_ranks = _rank_map(population, GAP)
    dsr_ranks = _rank_map(population, "dsr")
    actual_leader = population.loc[population[COUNTRY].map(actual_ranks).eq(1)].iloc[0]
    gap_leader = population.loc[population[COUNTRY].map(gap_ranks).eq(1)].iloc[0]
    dsr_leader = population.loc[population[COUNTRY].map(dsr_ranks).eq(1)].iloc[0]
    return (
        len(population),
        _display_country(actual_leader[COUNTRY]),
        actual_leader[ACTUAL],
        _display_country(gap_leader[COUNTRY]),
        gap_leader[GAP],
        _display_country(dsr_leader[COUNTRY]),
        dsr_leader["dsr"],
        gap_ranks[actual_leader[COUNTRY]],
        actual_ranks[gap_leader[COUNTRY]],
        dsr_ranks[gap_leader[COUNTRY]],
    )


def validate(outputs):
    expected = ground_truth()
    candidate = dict(outputs)
    for index, name in [(1, "highest_actual_credit_ratio_country"), (3, "highest_credit_gap_country"), (5, "highest_dsr_country")]:
        if _normalized_country(candidate.get(name)) == _normalized_country(expected[index]):
            candidate[name] = expected[index]
    return validate_ordered_outputs(
        candidate,
        variables,
        expected,
        [0, None, 4, None, 4, None, 1, 0, 0, 0],
    )



TURN_1_NAMES = [
    "joint_country_count",
    "highest_actual_credit_ratio_country",
    "highest_actual_credit_ratio_pct_of_gdp",
]
TURN_2_NAMES = [
    "highest_credit_gap_country",
    "highest_credit_gap_percentage_points",
]
TURN_3_NAMES = [
    "highest_dsr_country",
    "highest_dsr_pct",
]
TURN_4_NAMES = [
    "actual_ratio_leader_credit_gap_rank",
    "credit_gap_leader_actual_ratio_rank",
    "credit_gap_leader_dsr_rank",
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
    "validate_common_population": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_credit_gap_leader": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_debt_service_leader": turn_validator(_turn_subset(TURN_3_NAMES)),
    "validate_cross_ranks": turn_validator(_turn_subset(TURN_4_NAMES)),
}
