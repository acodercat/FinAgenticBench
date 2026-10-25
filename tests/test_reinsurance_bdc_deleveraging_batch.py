"""Financial-contract and acceptance regressions for two scenario cases."""

from decimal import Decimal, ROUND_HALF_UP

from evals.scenario_analysis import (
    bdc_collateral_sale_deleveraging as bdc,
    flood_occurrence_reinstatement_aggregate_cover as flood,
)


def _answers(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def _round(number, places):
    return float(Decimal(str(number)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def test_flood_loss_layers_and_wrong_conventions():
    answer = _answers(flood)
    assert flood.validate(answer).success
    assert 82 < answer["debby_loss_m"] < 83
    assert answer["helene_occurrence_recovery_m"] == 1500
    assert answer["reinstatement_premium_m"] == 90
    assert answer["full_treaty_economic_cost_m"] == 5020
    assert 606 < answer["minimum_reinstatement_m"] < 607
    # Gross-first aggregation, free restoration and no restoration each change
    # a real treaty result rather than an arbitrary model output.
    for wrong in (
        dict(answer, annual_aggregate_recovery_m=750),
        dict(answer, full_treaty_economic_cost_m=4930),
        dict(answer, milton_occurrence_recovery_m=0),
        dict(answer, minimum_reinstatement_m=0),
    ):
        assert not flood.validate(wrong).success
    assert flood.validate_turn_4({"annual_aggregate_recovery_m": answer["annual_aggregate_recovery_m"],
                                  "full_treaty_economic_cost_m": answer["full_treaty_economic_cost_m"]}).success
    assert not flood.validate_turn_4({"annual_aggregate_recovery_m": 750,
                                      "full_treaty_economic_cost_m": answer["full_treaty_economic_cost_m"]}).success


def test_flood_reported_input_rounding_is_accepted_without_a_second_financial_answer():
    answer = _answers(flood)
    rounded_event_losses = sum((_round(answer[name], 4) for name in
                                ("debby_loss_m", "helene_loss_m", "milton_loss_m")))
    reported_path_aggregate = rounded_event_losses - 1500 - (_round(answer["milton_loss_m"], 4) - 100) - 4800
    reported_path_restoration = rounded_event_losses - 7150
    assert _round(reported_path_aggregate, 3) == _round(answer["annual_aggregate_recovery_m"], 3)
    assert _round(reported_path_restoration, 3) == _round(answer["minimum_reinstatement_m"], 3)
    assert flood.validate_turn_4({"annual_aggregate_recovery_m": _round(reported_path_aggregate, 3),
                                  "full_treaty_economic_cost_m": 5020}).success
    assert flood.validate_turn_5({"minimum_reinstatement_m": _round(reported_path_restoration, 3)}).success


def test_bdc_sale_feedback_and_financing_path():
    answer = _answers(bdc)
    assert bdc.validate(answer).success
    assert answer["eligible_investment_count"] == 953
    assert answer["stressed_borrowing_deficiency_m"] > 800
    assert 24 < answer["immediate_minimum_sale_pct"] < 25
    assert 27 < answer["whole_window_minimum_sale_pct"] < 28
    assert answer["immediate_realized_pnl_m"] < answer["immediate_incremental_nav_change_m"] < 0
    assert answer["year_end_borrowing_deficiency_m"] > 0
    for wrong in (
        dict(answer, immediate_minimum_sale_pct=5.5338427587),
        dict(answer, immediate_incremental_nav_change_m=answer["immediate_realized_pnl_m"]),
        dict(answer, year_end_borrowing_deficiency_m=0),
        dict(answer, whole_window_minimum_sale_pct=answer["immediate_minimum_sale_pct"]),
        dict(answer, eligible_investment_count=954),
    ):
        assert not bdc.validate(wrong).success
    rounded_sale = _round(answer["immediate_minimum_sale_pct"], 7)
    assert bdc.validate_turn_3({
        "immediate_minimum_sale_pct": rounded_sale,
        "immediate_realized_pnl_m": _round(answer["immediate_realized_pnl_m"], 4),
        "immediate_incremental_nav_change_m": _round(answer["immediate_incremental_nav_change_m"], 4),
    }).success
