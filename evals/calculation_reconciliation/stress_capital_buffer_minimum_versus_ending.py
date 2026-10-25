"""Compare minimum- and ending-ratio measures of the stress capital decline.

Under 12 CFR 225.8(f)(2), one input to a firm's stress capital buffer is the fall
in its common equity tier 1 ratio from the starting ratio to the lowest projected
ratio in the planning horizon. Planned common dividends are added to that decline,
and the resulting calculation is compared with a 2.5 percent floor. This case
isolates only the decline component; it does not reconstruct an assigned buffer.

The 2025 exercise publishes 23 rows for the severely adverse scenario, but one of
them is an aggregate labelled for the participating banks and carries no identifier,
leaving 22 firms. Their median starting ratio is 12.9500 percent, their median
trough 11.1000 and their median ending ratio 12.8000, so the level a firm finishes
with sits far closer to where it began than its projected minimum. On
17 of the 22 the trough and the ending ratio differ at all.

Measured to the trough the median decline is 1.9500 points; measured to the ending
ratio it is 1.1000, a median gap of 0.4000. Taken to the trough, 12 of the
22 have a decline at or below the 2.5-point benchmark; taken to the ending ratio,
16 do. The ending reference adds 4 firms to that descriptive group. These counts
do not identify which assigned buffers equal the floor because the planned-dividend
add-on is absent. Four firms show a minimum ratio above their own starting ratio,
and the largest decline is 11.0000 points; the ratios alone do not separate
numerator-capital from denominator-RWA effects.

The pattern holds across the four exercises run under this framework. They cover 109
firm-scenario results, of which 55 have a minimum-based decline at or below the
benchmark and 70 have an ending-based decline at or below it; the 2022 exercise
diverges most, by 6 firms.

None of these firms can be looked up in the Call Report capital file, and the reason
is structural rather than a matching failure: 0 of the 22 appear there, because the
identifiers name holding companies, 15 of them financial holding companies, 6
intermediate holding companies of foreign banking organizations and 1 a savings and
loan holding company. A Call Report is filed by an insured depository, which is a
subsidiary of these firms and not the entity the test covers.

The `stress_capital_buffer` pinned-reading comparison records sensitivity to using the
ending ratio, retaining aggregate rows, widening the exercise set, changing the
2.5-point boundary, and using means. The query fixes the reported convention, so
these are robustness comparisons rather than hidden answer paths.

Entity boundary: the stress-test identifiers in the focus population are holding
companies, while Call Report filer identifiers represent insured depository
institutions. A direct identifier match therefore does not bridge parent and
subsidiary entities.

Boundaries: the files do not carry the planned-dividend add-on or enough information
to apply any exercise-specific averaging or subsequent adjustment, so the computed
quantity is the stress capital decline component and not the buffer a firm is
assigned. The ratios are published to one decimal, so declines are differences of
rounded figures. The four exercises cover overlapping but non-identical panels.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and exercise names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


FOCUS_EXERCISE = "2025 Stress Test"
SEVERE_SCENARIO = "Supervisory Severely Adverse"
FRAMEWORK_EXERCISES = ("2022 Stress Test", "2023 Stress Test", "2024 Stress Test", "2025 Stress Test")
BUFFER_FLOOR = 2.5
RATIO_COLUMNS = (
    "common_equity_tier1_actual_rat", "common_equity_tier1_end_rat", "common_equity_tier1_min_rat",
)
FINANCIAL_HOLDING = "FHD"
INTERMEDIATE_HOLDING = "IHC"
SAVINGS_LOAN_HOLDING = "SLHC"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["exercise_rows", "aggregate_rows", "named_firms", "median_starting_ratio",
                "median_trough_ratio", "median_ending_ratio", "trough_differs_from_ending"]
TURN_2_NAMES = ["median_decline_to_trough", "median_decline_to_ending", "median_decline_gap",
                "maximum_decline", "minimum_above_start_firms",
                "decline_at_or_below_floor_by_minimum", "decline_at_or_below_floor_by_ending"]
TURN_3_NAMES = ["framework_results", "framework_decline_at_or_below_floor_by_minimum",
                "framework_decline_at_or_below_floor_by_ending",
                "widest_exercise", "widest_exercise_gap"]
TURN_4_NAMES = ["call_report_matches", "identified_firms", "financial_holding_firms",
                "intermediate_holding_firms", "savings_loan_holding_firms"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many severely adverse rows the focus exercise carries as an integer."),
    _v(TURN_1_NAMES[1], "Store how many of them carry no identifier as an integer."),
    _v(TURN_1_NAMES[2], "Store how many named firms remain as an integer."),
    _v(TURN_1_NAMES[3], "Store their median starting ratio in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store their median minimum projected ratio in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store their median ending ratio in percent, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store how many firms have a minimum ratio different from their ending ratio as an integer."),
    _v(TURN_2_NAMES[0], "Store the median decline from starting to minimum in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store the median decline from starting to ending in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the median of minimum-based decline minus ending-based decline in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the largest decline to the minimum in percentage points, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store how many firms have a minimum ratio above their own starting ratio as an integer."),
    _v(TURN_2_NAMES[5], "Store how many firms have a minimum-based decline at or below the 2.5-point benchmark as an integer."),
    _v(TURN_2_NAMES[6], "Store how many have an ending-based decline at or below that benchmark as an integer."),
    _v(TURN_3_NAMES[0], "Store how many named results the four framework exercises hold as an integer."),
    _v(TURN_3_NAMES[1], "Store how many have a minimum-based decline at or below the benchmark as an integer."),
    _v(TURN_3_NAMES[2], "Store how many have an ending-based decline at or below it as an integer."),
    _v(TURN_3_NAMES[3], "Store exactly one exercise label from: 2022 Stress Test, 2023 Stress Test, 2024 Stress Test, 2025 Stress Test."),
    _v(TURN_3_NAMES[4], "Store that divergence in firms as an integer."),
    _v(TURN_4_NAMES[0], "Store how many of the focus exercise's firms appear as a Call Report filer as an integer."),
    _v(TURN_4_NAMES[1], "Store how many appear in the institution identifier file as an integer."),
    _v(TURN_4_NAMES[2], "Store how many are financial holding companies as an integer."),
    _v(TURN_4_NAMES[3], "Store how many are intermediate holding companies as an integer."),
    _v(TURN_4_NAMES[4], "Store how many are savings and loan holding companies as an integer."),
]

DECIMALS = [0, 0, 0, 4, 4, 4, 0,
            4, 4, 4, 4, 0, 0, 0,
            0, 0, 0, None, 0,
            0, 0, 0, 0, 0]


@lru_cache(maxsize=1)
def _results():
    frame = load_expansion_table("fed_stress_tests").copy()
    for column in RATIO_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    severe = frame.loc[frame.scenario_name.astype(str).eq(SEVERE_SCENARIO)].copy()
    severe["decline_to_trough"] = severe.common_equity_tier1_actual_rat - severe.common_equity_tier1_min_rat
    severe["decline_to_ending"] = severe.common_equity_tier1_actual_rat - severe.common_equity_tier1_end_rat
    severe["exercise"] = severe.exercise_name.astype(str)
    return severe


def _named(frame):
    return frame.loc[frame.id_rssd.notna()].copy()


@lru_cache(maxsize=1)
def ground_truth():
    severe = _results()
    focus = severe.loc[severe.exercise.eq(FOCUS_EXERCISE)]
    named = _named(focus)
    framework = _named(severe.loc[severe.exercise.isin(FRAMEWORK_EXERCISES)])

    by_exercise = framework.groupby("exercise").apply(
        lambda rows: pd.Series({
            "trough": int(rows.decline_to_trough.le(BUFFER_FLOOR).sum()),
            "ending": int(rows.decline_to_ending.le(BUFFER_FLOOR).sum()),
        }),
        include_groups=False,
    )
    by_exercise["gap"] = by_exercise.ending - by_exercise.trough
    widest = by_exercise.sort_values(["gap"], ascending=False).index[0]

    identifiers = load_expansion_table("ffiec_nic_institutions").copy()
    identifiers["rssd"] = identifiers.rssd_id.astype(str)
    entity_types = dict(zip(identifiers.rssd, identifiers.entity_type.astype(str)))
    capital = load_expansion_table("ffiec_call_reports_capital")
    filers = set(capital.bank_id.astype(str).str.strip())
    keys = named.id_rssd.astype(str).str.strip()

    def entity_count(code):
        return int(sum(1 for key in keys if entity_types.get(key) == code))

    return (
        int(len(focus)), int(focus.id_rssd.isna().sum()), int(len(named)),
        float(named.common_equity_tier1_actual_rat.median()),
        float(named.common_equity_tier1_min_rat.median()),
        float(named.common_equity_tier1_end_rat.median()),
        int(named.common_equity_tier1_min_rat.ne(named.common_equity_tier1_end_rat).sum()),
        float(named.decline_to_trough.median()), float(named.decline_to_ending.median()),
        float((named.decline_to_trough - named.decline_to_ending).median()),
        float(named.decline_to_trough.max()), int(named.decline_to_trough.lt(0).sum()),
        int(named.decline_to_trough.le(BUFFER_FLOOR).sum()),
        int(named.decline_to_ending.le(BUFFER_FLOOR).sum()),
        int(len(framework)), int(framework.decline_to_trough.le(BUFFER_FLOOR).sum()),
        int(framework.decline_to_ending.le(BUFFER_FLOOR).sum()),
        str(widest), int(by_exercise.loc[widest, "gap"]),
        int(keys.isin(filers).sum()), int(sum(1 for key in keys if key in entity_types)),
        entity_count(FINANCIAL_HOLDING), entity_count(INTERMEDIATE_HOLDING),
        entity_count(SAVINGS_LOAN_HOLDING),
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
    "validate_population_and_ratios": turn_validator(validate_turn_1),
    "validate_decline_reference": turn_validator(validate_turn_2),
    "validate_framework_pattern": turn_validator(validate_turn_3),
    "validate_entity_scope": turn_validator(validate_turn_4),
}
