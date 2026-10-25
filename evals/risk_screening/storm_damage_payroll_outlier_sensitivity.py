"""Screen state storm damage, normalize by payroll scale, and test outlier sensitivity.

Rejected alternatives (each pinned by the query and measured in the
`storm_payroll_sensitivity` comparison): treating a missing damage
component as a reported zero, keeping forecast-zone (Z) rows alongside county
(C) rows, ranking by event counts instead of reported damage, normalizing by
private-sector instead of total-covered QCEW employment, and removing the
largest bank-style unit rather than the single largest damage row. The North
Carolina leader is dominated by one Flash Flood episode (Helene), so the type
concentration in turn 2 is close to degenerate by construction; that is the
point the outlier turn tests.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table, qcew_state_totals
from core.validation import turn_validator, validate_ordered_outputs


T1 = [
    "matched_state_jurisdiction_count", "county_event_row_count",
    "event_rows_with_any_reported_damage_count", "cohort_reported_damage_usd_billions",
    "absolute_damage_leader_state_usps", "absolute_leader_event_row_count",
    "absolute_leader_reported_damage_row_count", "absolute_leader_damage_usd_billions",
    "absolute_damage_runner_up_state_usps", "absolute_damage_leader_margin_usd_billions",
]
T2 = [
    "absolute_leader_event_type_count", "absolute_leader_top_damage_event_type",
    "top_event_type_row_count", "top_event_type_damage_usd_billions",
    "top_event_type_damage_share_pct", "absolute_leader_event_type_damage_hhi",
    "largest_damage_event_id", "largest_damage_event_county_name",
    "largest_damage_event_begin_datetime", "largest_damage_event_usd_billions",
    "largest_event_share_of_state_damage_pct",
]
T3 = [
    "damage_qcew_matched_state_count", "damage_per_employee_leader_state_usps",
    "normalized_leader_total_covered_employment", "normalized_leader_total_annual_wages_usd_billions",
    "normalized_leader_damage_per_employee_usd", "normalized_leader_damage_to_annual_wages_pct",
    "absolute_leader_damage_per_employee_usd", "absolute_leader_damage_to_annual_wages_pct",
    "absolute_leader_damage_per_employee_rank", "normalized_leader_margin_usd_per_employee",
]
T4 = [
    "top_event_type_rows_excluding_largest_event_count",
    "top_event_type_damage_excluding_largest_event_usd_billions",
    "top_event_type_damage_retained_pct", "state_damage_excluding_largest_event_usd_billions",
    "top_type_remainder_share_of_adjusted_state_damage_pct",
    "state_damage_outside_top_event_type_usd_billions",
    "top_type_remainder_exceeds_all_other_types",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(T1[0], "Store the matched state-level jurisdiction count as an integer."),
    _v(T1[1], "Store the retained NOAA event-row count as an integer."),
    _v(T1[2], "Store the event-row count with any nonmissing damage amount as an integer."),
    _v(T1[3], "Store summed reported damage in USD billions, rounded to 3 decimals."),
    _v(T1[4], "Store the leader USPS abbreviation."),
    _v(T1[5], "Store the leader's retained event-row count as an integer."),
    _v(T1[6], "Store the leader's rows with any reported damage as an integer."),
    _v(T1[7], "Store the leader's reported damage in USD billions, rounded to 3 decimals."),
    _v(T1[8], "Store the runner-up USPS abbreviation."),
    _v(T1[9], "Store the leader-over-runner-up damage margin in USD billions, rounded to 3 decimals."),
    _v(T2[0], "Store the distinct event-type count as an integer."),
    _v(T2[1], "Store the selected source event-type label; casing and repeated whitespace are not significant."),
    _v(T2[2], "Store the selected event-type row count as an integer."),
    _v(T2[3], "Store selected event-type damage in USD billions, rounded to 3 decimals."),
    _v(T2[4], "Store selected event-type share of state damage in percent, rounded to 4 decimals."),
    _v(T2[5], "Store the event-type damage-share HHI on a 0-to-10,000 scale, rounded to 4 decimals."),
    _v(T2[6], "Store the NOAA event identifier as text."),
    _v(T2[7], "Store the source county/zone name; casing and repeated whitespace are not significant."),
    _v(T2[8], "Store the source begin datetime as an ISO YYYY-MM-DDTHH:MM:SS string."),
    _v(T2[9], "Store the event's reported damage in USD billions, rounded to 3 decimals."),
    _v(T2[10], "Store the event's share of state damage in percent, rounded to 4 decimals."),
    _v(T3[0], "Store the matched NOAA-QCEW state count as an integer."),
    _v(T3[1], "Store the damage-per-employee leader USPS abbreviation."),
    _v(T3[2], "Store total covered annual-average employment as an integer."),
    _v(T3[3], "Store total annual wages in USD billions, rounded to 3 decimals."),
    _v(T3[4], "Store reported damage per annual-average covered employee in USD, rounded to 2 decimals."),
    _v(T3[5], "Store reported damage divided by total annual wages in percent, rounded to 4 decimals."),
    _v(T3[6], "Store the turn-1 leader's damage per employee in USD, rounded to 2 decimals."),
    _v(T3[7], "Store the turn-1 leader's damage-to-wages ratio in percent, rounded to 4 decimals."),
    _v(T3[8], "Store the turn-1 leader's damage-per-employee rank as an integer."),
    _v(T3[9], "Store leader minus runner-up damage per employee in USD, rounded to 2 decimals."),
    _v(T4[0], "Store the selected event type's row count after excluding the largest event row as an integer."),
    _v(T4[1], "Store selected event-type damage after excluding the largest event row in USD billions, rounded to 3 decimals."),
    _v(T4[2], "Store the retained share of selected event-type damage in percent, rounded to 4 decimals."),
    _v(T4[3], "Store state damage after excluding the largest event row in USD billions, rounded to 3 decimals."),
    _v(T4[4], "Store the remaining selected event type's share of adjusted state damage in percent, rounded to 4 decimals."),
    _v(T4[5], "Store state damage outside the selected event type in USD billions, rounded to 3 decimals."),
    _v(T4[6], "Store whether remaining selected event-type damage exceeds all damage outside that type as a Boolean."),
]


def _norm(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def ground_truth():
    storms = load_expansion_table("noaa_storm_events")
    storms = storms.loc[storms.begin_datetime.astype(str).str.startswith("2024")]
    events = storms.loc[
        storms.state_usps.notna() & storms.county_zone_type.eq("C")
    ].copy()
    damage_columns = ["property_damage_usd", "crop_damage_usd"]
    events["any_reported_damage"] = events[damage_columns].notna().any(axis=1)
    events["reported_damage_usd"] = events[damage_columns].fillna(0).sum(axis=1)
    states = events.groupby(["state_fips", "state_usps"], as_index=False).agg(
        event_rows=("event_id", "size"),
        reported_damage_rows=("any_reported_damage", "sum"),
        reported_damage_usd=("reported_damage_usd", "sum"),
    ).sort_values(["reported_damage_usd", "state_usps"], ascending=[False, True]).reset_index(drop=True)
    leader, runner = states.iloc[0], states.iloc[1]

    leader_events = events.loc[events.state_usps.eq(leader.state_usps)].copy()
    types = leader_events.groupby("event_type", as_index=False).agg(
        event_rows=("event_id", "size"), reported_damage_usd=("reported_damage_usd", "sum")
    ).sort_values(["reported_damage_usd", "event_type"], ascending=[False, True]).reset_index(drop=True)
    types["share_pct"] = types.reported_damage_usd / leader.reported_damage_usd * 100
    top_type = types.iloc[0]
    hhi = float(types.share_pct.pow(2).sum())
    top_event = leader_events.sort_values(
        ["reported_damage_usd", "event_id"], ascending=[False, True]
    ).iloc[0]

    qcew = qcew_state_totals(load_expansion_table("bls_qcew")).loc[
        lambda frame: frame.year.eq(2024) & frame.ownership_scope.eq("total_covered")
    ]
    panel = states.merge(qcew, on="state_fips", validate="one_to_one")
    panel["damage_per_employee"] = panel.reported_damage_usd / panel.annual_avg_employment
    panel["damage_to_wages_pct"] = panel.reported_damage_usd / panel.total_annual_wages_usd * 100
    panel = panel.sort_values(["damage_per_employee", "state_usps"], ascending=[False, True]).reset_index(drop=True)
    normalized_leader = panel.iloc[0]
    absolute_in_panel = panel.loc[panel.state_usps.eq(leader.state_usps)].iloc[0]
    absolute_rank = int(panel.index[panel.state_usps.eq(leader.state_usps)][0]) + 1

    top_type_remainder = float(top_type.reported_damage_usd - top_event.reported_damage_usd)
    adjusted_state_damage = float(leader.reported_damage_usd - top_event.reported_damage_usd)
    outside_top_type = float(leader.reported_damage_usd - top_type.reported_damage_usd)

    return (
        len(states),
        len(events),
        int(events.any_reported_damage.sum()),
        float(events.reported_damage_usd.sum()) / 1e9,
        leader.state_usps,
        int(leader.event_rows),
        int(leader.reported_damage_rows),
        float(leader.reported_damage_usd) / 1e9,
        runner.state_usps,
        (float(leader.reported_damage_usd) - float(runner.reported_damage_usd)) / 1e9,
        len(types),
        top_type.event_type,
        int(top_type.event_rows),
        float(top_type.reported_damage_usd) / 1e9,
        float(top_type.share_pct),
        hhi,
        top_event.event_id,
        top_event.county_zone_name,
        top_event.begin_datetime,
        float(top_event.reported_damage_usd) / 1e9,
        float(top_event.reported_damage_usd / leader.reported_damage_usd * 100),
        len(panel),
        normalized_leader.state_usps,
        int(normalized_leader.annual_avg_employment),
        float(normalized_leader.total_annual_wages_usd) / 1e9,
        float(normalized_leader.damage_per_employee),
        float(normalized_leader.damage_to_wages_pct),
        float(absolute_in_panel.damage_per_employee),
        float(absolute_in_panel.damage_to_wages_pct),
        absolute_rank,
        float(normalized_leader.damage_per_employee - panel.iloc[1].damage_per_employee),
        int(top_type.event_rows) - 1,
        top_type_remainder / 1e9,
        top_type_remainder / float(top_type.reported_damage_usd) * 100,
        adjusted_state_damage / 1e9,
        top_type_remainder / adjusted_state_damage * 100,
        outside_top_type / 1e9,
        bool(top_type_remainder > outside_top_type),
    )


DECIMALS = [
    0, 0, 0, 3, None, 0, 0, 3, None, 3,
    0, None, 0, 3, 4, 4, None, None, None, 3, 4,
    0, None, 0, 3, 2, 4, 2, 4, 0, 2,
    0, 3, 4, 3, 4, 3, None,
]


def _validate(outputs, names):
    expected = dict(zip((v.name for v in variables), ground_truth()))
    candidate = dict(outputs)
    for name in [T2[1], T2[7]]:
        if _norm(candidate.get(name)) == _norm(expected[name]):
            candidate[name] = expected[name]
    if str(candidate.get(T2[6], "")).strip() == str(expected[T2[6]]):
        candidate[T2[6]] = expected[T2[6]]
    by_name = {v.name: v for v in variables}
    places = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(candidate, [by_name[n] for n in names], [expected[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate(outputs, T1)
def validate_turn_2(outputs): return _validate(outputs, T2)
def validate_turn_3(outputs): return _validate(outputs, T3)
def validate_turn_4(outputs): return _validate(outputs, T4)
def validate(outputs): return _validate(outputs, [v.name for v in variables])


validators = {
    "validate_damage_screen": turn_validator(validate_turn_1),
    "validate_leader_concentration": turn_validator(validate_turn_2),
    "validate_payroll_normalization": turn_validator(validate_turn_3),
    "validate_outlier_sensitivity": turn_validator(validate_turn_4),
}
