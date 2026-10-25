"""Local boundary checks for the disaster-county mortgage scenario."""

from evals.scenario_analysis import disaster_county_mortgage_collateral_paydown as case


def _answers():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def test_disaster_county_collateral_outputs_and_wrong_boundaries():
    answer = _answers()
    assert case.validate(answer).success
    rounded = dict(answer)
    for variable, decimals in zip(case.variables, case.DECIMALS):
        if decimals and decimals > 0:
            rounded[variable.name] = round(rounded[variable.name], decimals)
    assert case.validate(rounded).success
    assert answer["declared_counties"] == 39
    assert answer["eligible_origination_records"] == 33526
    assert answer["minimum_paydown_m"] > 1500
    assert answer["stressed_collateral_shortfall_m"] < answer["minimum_paydown_m"]
    assert answer["largest_paydown_county"] != answer["highest_intensity_county"]
    for name, wrong in (
        ("declared_counties", 40),  # includes the separate tribal-area record
        ("eligible_origination_records", 34939),  # includes manufactured housing
        ("minimum_paydown_m", answer["stressed_collateral_shortfall_m"]),
        ("largest_paydown_county", "37193"),
        ("highest_intensity_county", "37119"),
    ):
        assert not case.validate(dict(answer, **{name: wrong})).success
