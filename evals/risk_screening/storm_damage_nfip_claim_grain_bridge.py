"""Bridge NOAA storm-damage state-months to NFIP claim-record grain.

Florida in September 2024 is the unique 50-state-plus-DC state-month with the
largest sum of reported NOAA property damage.  The matching NFIP loss-month is
dominated by Hurricane Helene, but physical claim records and represented
policy counts differ materially.  The two dollar measures describe different
universes and are deliberately not reconciled.

The pinned-reading comparison measures the geography population, the damage scope, the
claim month field, the payment basis and the event weight.  Adding crop damage to
the property total reports 264 NOAA rows and $5.4075439 billion instead of 261
rows and $5.1075439 billion; keying NFIP claims on the open date instead of the
date of loss reports 38,187 claim records and 73,748 represented policies instead
of 55,067 and 117,036; using gross rather than signed net payments reports
$6.493308 billion instead of $6.491534 billion; and weighting the dominant event
by physical records instead of represented policies reports 54,673 instead of
115,932.  Retaining every raw geography rather than the 50 states plus DC is
numerically equivalent on this frozen extract, because no non-state geography
reaches the leading damage total.  The query pins all five choices.
"""

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


STATE_USPS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}


variables = [
    Variable("noaa_damage_leader_state_name", None, "Store the NOAA state name as text; casing and repeated whitespace are not significant."),
    Variable("noaa_damage_leader_state_usps", None, "Store the two-letter USPS abbreviation as text; casing and surrounding whitespace are not significant."),
    Variable("noaa_damage_leader_month", None, "Store the selected calendar month as a YYYY-MM string."),
    Variable("noaa_selected_event_row_count", None, "Store the selected NOAA event-row count as an integer."),
    Variable("noaa_reported_property_damage_row_count", None, "Store the count of selected NOAA rows with a reported property-damage amount as an integer."),
    Variable("noaa_property_damage_usd_billions", None, "Store reported NOAA property damage in USD billions, rounded to 6 decimals."),
    Variable("nfip_claim_record_count", None, "Store the matching NFIP physical claim-record count as an integer."),
    Variable("nfip_represented_policy_count", None, "Store the matching sum of reported policy counts as an integer."),
    Variable("nfip_dominant_named_flood_event", None, "Store the leading nonblank NFIP flood-event label as text; casing and repeated whitespace are not significant."),
    Variable("nfip_dominant_event_policy_count", None, "Store that event's represented policy count as an integer."),
    Variable("nfip_dominant_event_policy_share_pct", None, "Store that event's share of all matching represented policy counts in percent, rounded to 4 decimals."),
    Variable("nfip_total_net_claim_payment_usd_billions", None, "Store total signed net NFIP building, contents and ICC payments in USD billions, rounded to 6 decimals."),
    Variable("nfip_total_reported_coverage_usd_billions", None, "Store total reported NFIP building and contents coverage in USD billions, rounded to 6 decimals."),
    Variable("nfip_net_payment_to_coverage_pct", None, "Store aggregate signed net payments divided by aggregate reported coverage in percent, rounded to 4 decimals."),
]


def _normalized_label(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else value


def _selected_populations():
    storms = load_expansion_table("noaa_storm_events")
    storms = storms.loc[storms.begin_datetime.astype(str).str.startswith("2024")]
    storms = storms.loc[storms.state_usps.isin(STATE_USPS)].copy()
    storms["begin_month"] = storms.begin_datetime.str[:7]
    state_months = storms.groupby(
        ["state_name", "state_usps", "begin_month"], as_index=False
    ).agg(
        property_damage_usd=(
            "property_damage_usd", lambda values: values.sum(min_count=1)
        ),
        event_row_count=("event_id", "size"),
        reported_property_damage_row_count=("property_damage_usd", "count"),
    )
    state_months = state_months.loc[state_months.property_damage_usd.notna()]
    leader = state_months.sort_values(
        ["property_damage_usd", "state_name", "begin_month"],
        ascending=[False, True, True],
    ).iloc[0]
    selected_storms = storms.loc[
        storms.state_usps.eq(leader.state_usps)
        & storms.begin_month.eq(leader.begin_month)
    ].copy()

    claims = load_expansion_table("fema_nfip")
    selected_claims = claims.loc[
        claims.state_usps.eq(leader.state_usps)
        & claims.date_of_loss.str[:7].eq(leader.begin_month)
    ].copy()
    named = selected_claims.loc[selected_claims.flood_event.notna()].copy()
    event_counts = named.groupby("flood_event", as_index=False).policy_count.sum()
    dominant = event_counts.sort_values(
        ["policy_count", "flood_event"], ascending=[False, True]
    ).iloc[0]
    return leader, selected_storms, selected_claims, dominant


def ground_truth():
    leader, storms, claims, dominant = _selected_populations()
    net_payment = claims[[
        "net_building_payment_usd",
        "net_contents_payment_usd",
        "net_icc_payment_usd",
    ]].sum().sum()
    coverage = claims[["building_coverage_usd", "contents_coverage_usd"]].sum().sum()
    represented_policies = claims.policy_count.sum()
    return (
        leader.state_name,
        leader.state_usps,
        leader.begin_month,
        len(storms),
        int(storms.property_damage_usd.notna().sum()),
        storms.property_damage_usd.sum() / 1e9,
        len(claims),
        int(represented_policies),
        dominant.flood_event,
        int(dominant.policy_count),
        dominant.policy_count / represented_policies * 100,
        net_payment / 1e9,
        coverage / 1e9,
        net_payment / coverage * 100,
    )


def validate(outputs):
    expected = ground_truth()
    candidate = dict(outputs)
    for index, name in (
        (0, "noaa_damage_leader_state_name"),
        (1, "noaa_damage_leader_state_usps"),
        (8, "nfip_dominant_named_flood_event"),
    ):
        if _normalized_label(candidate.get(name)) == _normalized_label(expected[index]):
            candidate[name] = expected[index]
    return validate_ordered_outputs(
        candidate,
        variables,
        expected,
        [None, None, None, 0, 0, 6, 0, 0, None, 0, 4, 6, 6, 4],
    )



TURN_1_NAMES = [
    "noaa_damage_leader_state_name",
    "noaa_damage_leader_state_usps",
    "noaa_damage_leader_month",
    "noaa_selected_event_row_count",
    "noaa_reported_property_damage_row_count",
    "noaa_property_damage_usd_billions",
]
TURN_2_NAMES = [
    "nfip_claim_record_count",
    "nfip_represented_policy_count",
    "nfip_dominant_named_flood_event",
    "nfip_dominant_event_policy_count",
    "nfip_dominant_event_policy_share_pct",
]
TURN_3_NAMES = [
    "nfip_total_net_claim_payment_usd_billions",
    "nfip_total_reported_coverage_usd_billions",
    "nfip_net_payment_to_coverage_pct",
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
    "validate_damage_selection": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_claim_population": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_payment_coverage": turn_validator(_turn_subset(TURN_3_NAMES)),
}
