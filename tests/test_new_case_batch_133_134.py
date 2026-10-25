"""Numerical boundaries and financial-state checks for the reverse-stress/SCF pair."""

import numpy as np

from core.data import load_expansion_table
from evals.risk_screening import nport_bounded_curve_reverse_stress as fund
from evals.scenario_analysis import scf_imputation_robust_annuity_support as scf


def _reported(module):
    full = module.analysis()
    return {
        variable.name: round(full[variable.name], precision)
        if precision is not None else full[variable.name]
        for variable, precision in zip(module.variables, module.DECIMALS)
    }


def test_fund_box_binding_and_redemption_risk_state():
    values = fund.analysis()
    selected, dv, covariance, changes = fund._state()
    target = .02 * float(selected.net_assets_usd)
    unboxed, distance = fund._unboxed(dv, covariance, target)
    boxed, bounded_distance = fund._boxed(dv, covariance, target)
    assert changes == 1250
    assert np.linalg.eigvalsh(covariance)[0] > 0
    assert abs(dv @ unboxed - target) < 1e-6
    assert abs(dv @ boxed - target) < 1e-4
    assert np.max(np.abs(boxed)) <= 50 + 1e-7
    assert abs(boxed[2] + 50) < 1e-5
    assert bounded_distance > distance + 2
    assert values["boxed_first_binding"] == "5Y_LOW"
    assert values["redemption_dv01_retained_pct"] > 80
    assert values["redemption_unboxed_distance"] < distance
    assert values["required_new_equity_mn"] > 16
    assert fund.validate(_reported(fund)).success
    assert not fund.validate(dict(values, boxed_first_binding="NONE")).success
    assert not fund.validate(dict(values, redemption_dv01_retained_pct=80.0)).success


def test_scf_mean_policy_breach_and_shared_robust_feasibility():
    values = scf.analysis()
    _, worlds, cost = scf._worlds((20, 15, 10))
    source = load_expansion_table("fed_scf")
    first = source.loc[
        source.implicate.eq(1)
        & source.reference_person_age.between(45, 74)
        & source.net_worth_usd.lt(50_000)
    ]
    groups = worlds[0][0]
    assert np.all(groups[first.reference_person_age.to_numpy() == 55] == 1)
    assert np.all(groups[first.reference_person_age.to_numpy() == 65] == 2)
    mean_policy = np.array([values[f"mean_policy_{g}_kusd"] for g in ("45_54", "55_64", "65_74")])
    robust_policy = np.array([values[f"robust_{g}_kusd"] for g in ("45_54", "55_64", "65_74")])
    revised_policy = np.array([values[f"revised_{g}_kusd"] for g in ("45_54", "55_64", "65_74")])
    assert (cost @ mean_policy).mean() <= 650 + 1e-5
    assert (cost @ mean_policy).max() > 656
    assert (cost @ robust_policy).max() <= 650 + 1e-5
    assert np.argmax(cost @ robust_policy) == 3
    assert np.argmax(scf._scores(worlds, robust_policy)[0]) == 0
    assert revised_policy[0] > robust_policy[0] + .6
    assert values["revised_worst_squared_shortfall"] < values["robust_worst_squared_shortfall"]
    assert scf.validate(_reported(scf)).success
    assert not scf.validate(dict(values, mean_policy_worst_budget_world="I1")).success
    assert not scf.validate(dict(values, mean_policy_budget_overage_bn=values["mean_policy_budget_overage_bn"] / 5)).success


def test_strict_display_boundary_for_each_numeric_output():
    for module in (fund, scf):
        full = module.analysis()
        reported = _reported(module)
        for variable, precision in zip(module.variables, module.DECIMALS):
            if precision in (None, 0):
                continue
            outside = dict(reported)
            outside[variable.name] = full[variable.name] + .7 * 10 ** (-precision)
            assert not module.validate(outside).success, variable.name
