"""Historical paid-claim replay under a hypothetical occurrence/aggregate treaty.

The disaster declarations define this hypothetical treaty's county/date scope;
they are not an official NFIP claim-to-disaster link. Signed net building plus
contents payments are historical snapshot proxies, not ultimate losses.

Measured rejected paths (USD millions): gross-first annual cover pays 750 rather
than 349.54474344; no reinstatement leaves 706.36750470 of Milton claims after
both layers rather than zero; forgetting reinstatement premium understates full
economic cost by 90. A lower aggregate attachment was rejected during design
because gross-first and correct paths exhausted the same cover and gave the same
annual payment. All downstream values use unrounded source amounts.
"""

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


M = Decimal(1_000_000)
EVENTS = (("Debby", "4806"), ("Helene", "4828"), ("Milton", "4834"))
OUTPUTS = (
    ("debby_loss_m", 4, "the Debby contract subject loss in USD millions"),
    ("helene_loss_m", 4, "the Helene contract subject loss in USD millions"),
    ("milton_loss_m", 4, "the Milton contract subject loss in USD millions"),
    ("helene_occurrence_recovery_m", 4, "the Helene occurrence recovery in USD millions"),
    ("pre_milton_occurrence_retained_m", 3, "cumulative Debby and Helene claim loss retained after the occurrence layer, in USD millions"),
    ("reinstatement_premium_m", 4, "the full reinstatement premium in USD millions"),
    ("milton_occurrence_recovery_m", 4, "the Milton occurrence recovery in USD millions"),
    ("ending_occurrence_capacity_m", 4, "unused occurrence capacity after Milton, in USD millions"),
    ("annual_aggregate_recovery_m", 3, "total annual aggregate-cover recovery in USD millions"),
    ("full_treaty_economic_cost_m", 4, "retained claim loss plus base, aggregate and reinstatement premiums, in USD millions"),
    ("minimum_reinstatement_m", 3, "the minimum pre-agreed restored occurrence capacity satisfying the Milton retained-loss ceiling, in USD millions"),
)
variables = [Variable(name, None, f"Store {description} as a finite number, rounded to {decimals} decimals from unrounded source inputs.")
             for name, decimals, description in OUTPUTS]
PRECISION = {name: decimals for name, decimals, _ in OUTPUTS}
TURN_NAMES = (
    ("debby_loss_m", "helene_loss_m", "milton_loss_m"),
    ("helene_occurrence_recovery_m", "pre_milton_occurrence_retained_m"),
    ("reinstatement_premium_m", "milton_occurrence_recovery_m", "ending_occurrence_capacity_m"),
    ("annual_aggregate_recovery_m", "full_treaty_economic_cost_m"),
    ("minimum_reinstatement_m",),
)


def D(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite claim amount")
    return result


def _event_losses():
    claims = load_expansion_table("fema_nfip")
    declarations = load_expansion_table("fema_disasters")
    result = {}
    for event, number in EVENTS:
        area = declarations.loc[declarations.disaster_number.eq(number) & declarations.state_usps.eq("FL")]
        if area.empty or area.county_fips.dropna().empty:
            raise ValueError(f"Missing declaration geography for {number}")
        if area.incident_begin_date.isna().any() or area.incident_end_date.isna().any():
            raise ValueError(f"Incomplete incident period for {number}")
        pool = claims.loc[
            claims.state_usps.eq("FL") & claims.flood_event.eq(f"Hurricane {event}")
            & claims.date_of_loss.str.startswith("2024")
        ]
        eligible = pool.loc[
            pool.county_code.isin(set(area.county_fips.dropna()))
            & pool.date_of_loss.between(area.incident_begin_date.min(), area.incident_end_date.max())
        ]
        if eligible.empty or eligible.claim_record_id.duplicated().any():
            raise ValueError(f"Missing or duplicated eligible claims for {event}")
        cols = ("net_building_payment_usd", "net_contents_payment_usd")
        if eligible[list(cols)].isna().any().any():
            raise ValueError(f"Incomplete net payments for {event}")
        loss = sum((D(x) + D(y) for x, y in eligible[list(cols)].itertuples(index=False, name=None)), Decimal(0)) / M
        if loss <= 0:
            raise ValueError(f"Nonpositive event subject loss for {event}")
        result[event] = loss
    return result


def _replay(losses, restored):
    capacity = Decimal(1500)
    cumulative = Decimal(0)
    annual_paid = Decimal(0)
    events = {}
    for event, _ in EVENTS:
        loss = losses[event]
        occurrence = min(max(loss - 100, Decimal(0)), Decimal(1500), capacity)
        capacity -= occurrence
        cumulative += loss - occurrence
        new_annual_paid = min(max(cumulative - 4800, Decimal(0)), Decimal(750))
        annual = new_annual_paid - annual_paid
        annual_paid = new_annual_paid
        events[event] = {"occurrence": occurrence, "annual": annual,
                         "retained": loss - occurrence - annual, "capacity": capacity,
                         "cumulative_occurrence_retained": cumulative}
        if event == "Helene":
            if capacity != 0:
                raise ValueError("Expected exhausted occurrence capacity before reinstatement")
            capacity += restored
    retained = sum((v["retained"] for v in events.values()), Decimal(0))
    if sum(losses.values()) != retained + sum((v["occurrence"] for v in events.values()), Decimal(0)) + annual_paid:
        raise ValueError("Treaty loss waterfall does not reconcile")
    return events, annual_paid, retained + 90 + 40 + restored * Decimal("0.06")


def analysis():
    losses = _event_losses()
    full, aggregate, cost = _replay(losses, Decimal(1500))
    # For the observed Helene-exhausted state, annual capacity remaining after
    # Helene and the Milton subject loss set the linear partial-restoration
    # boundary. Verify that it is in the assumed piecewise regime.
    pre_milton_aggregate_paid = full["Helene"]["annual"]
    remaining_aggregate_capacity = Decimal(750) - pre_milton_aggregate_paid
    boundary = losses["Milton"] - remaining_aggregate_capacity - Decimal(100)
    if not (Decimal(0) < boundary < Decimal(1500)):
        raise ValueError("Reinstatement boundary outside available contract range")
    partial, partial_annual, _ = _replay(losses, boundary)
    if partial["Milton"]["retained"] != 100 or partial_annual != 750:
        raise ValueError("Reinstatement boundary is in a different treaty regime")
    if _replay(losses, boundary - Decimal("0.01"))[0]["Milton"]["retained"] <= 100:
        raise ValueError("Smaller restoration unexpectedly satisfies loss ceiling")
    return {
        "debby_loss_m": float(losses["Debby"]),
        "helene_loss_m": float(losses["Helene"]),
        "milton_loss_m": float(losses["Milton"]),
        "helene_occurrence_recovery_m": float(full["Helene"]["occurrence"]),
        "pre_milton_occurrence_retained_m": float(full["Helene"]["cumulative_occurrence_retained"]),
        "reinstatement_premium_m": 90.0,
        "milton_occurrence_recovery_m": float(full["Milton"]["occurrence"]),
        "ending_occurrence_capacity_m": float(full["Milton"]["capacity"]),
        "annual_aggregate_recovery_m": float(aggregate),
        "full_treaty_economic_cost_m": float(cost),
        "minimum_reinstatement_m": float(boundary),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names],
                                    [truth[n] for n in names], [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _validate(outputs, TURN_NAMES[0])
def validate_turn_2(outputs): return _validate(outputs, TURN_NAMES[1])
def validate_turn_3(outputs): return _validate(outputs, TURN_NAMES[2])
def validate_turn_4(outputs): return _validate(outputs, TURN_NAMES[3])
def validate_turn_5(outputs): return _validate(outputs, TURN_NAMES[4])
def validate(outputs): return _validate(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 6)}
