"""Consolidation identities and attainable default-law/collateral bounds."""
import numpy as np
from scipy.optimize import linprog

from evals.calculation_reconciliation import bank_acquisition_goodwill_nci_capital_bridge as a
from evals.risk_screening import otc_netting_collateral_wrong_way_cva_bounds as c
from scripts.verdict_baseline import mutate


def test_acquisition_journal_identity_and_internal_capital_lifecycle():
    buyer, target, _, _ = a.inputs()
    result = a.analysis()
    price = result['consideration_mn']
    nci = result['nci_fair_value_mn']
    goodwill = result['new_goodwill_mn']
    parent = result['postdeal_parent_equity_mn']
    markdown = .01 * target['assets']
    cash_and_costs = .5 * price + .01 * price + .02 * .5 * price
    consolidated_assets = buyer['assets'] + target['assets'] - markdown + goodwill - cash_and_costs
    liabilities = buyer['liabilities'] + target['liabilities']
    inherited_nci = buyer['inherited_nci'] + target['inherited_nci']
    assert abs(target['inherited_nci'] - 27.407) < 1e-9
    assert abs(consolidated_assets - liabilities - parent - nci - inherited_nci) < 1e-8
    # Impairment reduces full goodwill and both equity owners, keeping the identity.
    impairment = .3 * goodwill
    impaired_nci = nci - .2 * impairment
    impaired_parent = result['impairment_parent_equity_mn']
    assert abs(consolidated_assets - impairment - liabilities - impaired_parent - impaired_nci - inherited_nci) < 1e-8
    assert abs(result['impairment_internal_capital_mn'] - result['postdeal_internal_capital_mn']) < 1e-8
    # Control-retaining purchase moves half impaired NCI at a premium; no new goodwill.
    bought = .5 * impaired_nci
    final_assets = consolidated_assets - impairment - 1.2 * bought
    final_parent = impaired_parent - .2 * bought
    assert abs(final_assets - liabilities - final_parent - .5 * impaired_nci - inherited_nci) < 1e-8
    original_adjustment = buyer['cet1'] - buyer['equity']
    assert abs(final_parent + original_adjustment - .9 * .7 * goodwill - result['buyout_internal_capital_mn']) < 1e-8
    assert result['buyout_internal_capital_mn'] < result['impairment_internal_capital_mn']
    # Both tempting double additions/deductions are financially material.
    truth = {v.name: x for v, x in zip(a.variables, a.ground_truth())}
    for name, wrong in [('postdeal_internal_capital_mn', result['postdeal_internal_capital_mn'] + target['equity']),
                        ('impairment_internal_capital_mn', result['impairment_internal_capital_mn'] - .8 * impairment)]:
        bad = truth.copy(); bad[name] = wrong
        assert not a.validate(bad).success


def test_acquisition_loan_capacity_is_an_unexecuted_binding_frontier():
    buyer, target, _, _ = a.inputs()
    result = a.analysis()
    rwa = buyer['rwa'] + target['rwa'] - .01 * target['assets']
    capacity = result['new_loan_rwa_capacity_mn']
    cap = result['postdeal_internal_capital_mn']
    assert capacity > 0
    assert abs(cap / (rwa + capacity) - .12) < 1e-14
    assert cap / (rwa + capacity + .01) < .12
    assert abs(result['buyout_capital_ratio_pct'] - 100 * result['buyout_internal_capital_mn'] / rwa) < 1e-10
    assert abs(result['buyout_capital_ratio_pct'] - 100 * result['buyout_internal_capital_mn'] / (rwa + capacity)) > 1


def test_credit_sharp_bounds_joint_laws_and_collateral_boundary():
    scales, netting, discount = c.inputs()
    result = c.analysis()
    assert np.all(scales > 0)
    collateral = result['minimum_extra_collateral_each_mn']
    for perturbation in [-1e-6, 0, 1e-6]:
        perturbed = netting * (1 + perturbation)
        for margin in [0, collateral, 4.5]:
            exposure = c.exposures(perturbed, margin)
            for largest in [False, True]:
                direction = -1 if largest else 1
                lp = linprog(direction * exposure, A_eq=np.ones((1, 6)), b_eq=[.25], bounds=(0, 1 / 6), method='highs')
                assert lp.success
                assert np.min(lp.x) >= -1e-12 and np.max(lp.x) <= 1 / 6 + 1e-12
                assert abs(sum(lp.x) - .25) < 1e-12
                assert abs(c.extremal_loss(exposure, discount, largest) - direction * lp.fun * .6 * discount) < 1e-12
    assert result['sharp_cva_lower_mn'] < result['independent_cva_mn'] < result['sharp_cva_upper_mn']
    assert abs(c.extremal_loss(c.exposures(netting, collateral), discount, True) - .10) < 1e-10
    assert c.extremal_loss(c.exposures(netting, collateral - 1e-6), discount, True) > .10
    assert c.extremal_loss(c.exposures(netting, collateral + 1e-6), discount, True) < .10
    assert result['boundary_independent_cva_mn'] < .10
    wrong = .6 * discount * .25 * np.maximum(netting.sum(axis=1) - .3, 0).mean()
    assert abs(wrong - result['independent_cva_mn']) > .01
    # Default mass25% cannot all fit the highest state of mass1/6.
    lp = linprog(-c.exposures(netting, 0), A_eq=np.ones((1, 6)), b_eq=[.25], bounds=(0, 1 / 6), method='highs')
    assert np.count_nonzero(lp.x > 1e-12) == 2


def test_all_public_outputs_precision_and_strict_errors():
    for module in [a, c]:
        truth = dict(zip((v.name for v in module.variables), module.ground_truth()))
        assert module.validate(truth).success
        rounded = {k: round(value, 4) for k, value in truth.items()}
        assert module.validate(rounded).success
        for name in truth:
            bad = truth.copy(); bad[name] = mutate(bad[name])
            assert not module.validate(bad).success
            bad[name] = float('nan')
            assert not module.validate(bad).success
        for names in module.TURN_NAMES:
            assert module._subset({n: rounded[n] for n in names}, names).success
