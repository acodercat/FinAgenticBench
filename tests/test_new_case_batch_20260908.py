from importlib import import_module

import pytest


OTC_MODULE = "evals.temporal_path.otc_rate_valuation_futures_positioning_path"
ASSET_QUALITY_MODULE = (
    "evals.risk_screening.state_credit_union_bank_asset_quality_divergence"
)


@pytest.mark.runtime_data
def test_otc_rate_futures_path_ground_truth_and_boundaries():
    module = import_module(OTC_MODULE)
    truth = module.ground_truth()
    assert truth[:6] == (
        "2020-S1", pytest.approx(40.3024423353), "042601", 452523,
        "2020-01-07", pytest.approx(-28.5607321625),
    )
    assert truth[6] == pytest.approx(-30.9721691070)
    outputs = dict(zip((variable.name for variable in module.variables), truth))
    assert module.validate(outputs).success
    assert not module.validate({
        **outputs, "otc_interest_rate_gmv_change_pct": -truth[1]
    }).success
    assert not module.validate({
        **outputs, "treasury_net_position_change_contracts": -truth[3]
    }).success
    assert not module.validate({
        **outputs, "peak_signed_net_open_interest_pct": truth[5] * 100
    }).success


@pytest.mark.runtime_data
def test_state_asset_quality_ground_truth_and_boundaries():
    module = import_module(ASSET_QUALITY_MODULE)
    truth = module.ground_truth()
    assert truth[:8] == (
        "NH", 12, pytest.approx(53.5603357456), 16,
        pytest.approx(-13.6663802559), "24921",
        pytest.approx(58.89375), pytest.approx(94.4351706071),
    )
    assert truth[8:] == (
        pytest.approx(0.3702129850), pytest.approx(19.7742745866),
        pytest.approx(62.7899794134),
    )
    outputs = dict(zip((variable.name for variable in module.variables), truth))
    assert module.validate(outputs).success
    assert not module.validate({
        **outputs, "bank_noncurrent_change_bp": abs(truth[4])
    }).success
    assert not module.validate({
        **outputs, "leader_share_of_state_increase_pct": truth[7] / 100
    }).success
    assert not module.validate({
        **outputs, "credit_union_number": int(truth[5])
    }).success
