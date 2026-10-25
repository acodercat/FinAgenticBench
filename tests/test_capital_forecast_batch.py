"""Case-local financial and scoring challenges; no framework sweep dependency."""
import importlib
import json
from decimal import Decimal as D
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import pytest

EXPECTED = {'bank_cet1_growth_attribution_capacity': {'eligible_bank_count': 29, 'selected_bank_rssd': '722777', 'ending_cet1_pct': 19.50399330746406, 'capital_contribution_bp': 179.87161999792013, 'rwa_contribution_bp': 170.31891141590606, 'funding_expense_mn': 255.15, 'proforma_cet1_pct': 14.960475277799477, 'maximum_new_loans_bn': 14.399294415134815, 'additional_cet1_for_plan_bn': 0.8854099728354778}, 'wti_budget_allowance_holdout_liquidity': {'calibration_month_count': 9, 'evaluation_month_count': 12, 'calibration_mean_error_usd_barrel': 6.212217675281749, 'selected_allowance_usd_barrel': -5, 'calibration_loss_mn': 74.51562323793674, 'calibration_runner_up_margin_mn': 2.728708133971292, 'held_out_selected_loss_mn': 235.12739077436675, 'held_out_zero_loss_mn': 125.9834382844463, 'held_out_underbudget_month_count': 10, 'first_half_reserve_mn': 13.514705513784461, 'first_binding_month': '2025-03'}}
MODULES = {name: importlib.import_module('evals.' + ('calculation_reconciliation' if name.startswith('bank_') else 'scenario_analysis') + '.' + name) for name in EXPECTED}

@pytest.mark.parametrize('name', EXPECTED)
def test_complete_maps_turns_and_correct_representations(name):
    m = MODULES[name]; expected = EXPECTED[name]
    assert dict(zip((v.name for v in m.variables), m.ground_truth())) == expected
    assert m.validate(expected).success
    spec = json.loads(Path(m.__file__).with_suffix('.json').read_text())
    turns = spec['conversations'][0]['turns']
    assert [n for t in turns for n in t['stores']] == list(expected)
    for i, t in enumerate(turns, 1):
        local = {n:expected[n] for n in t['stores']}; validate = getattr(m, f'validate_turn_{i}')
        assert validate(local).success
        for n in local:
            assert not validate({k:v for k,v in local.items() if k != n}).success
        # Other-turn outputs cannot affect a turn-local result.
        assert validate(local | {n:None for n in expected if n not in local}).success
    for var, places in zip(m.variables, m.DECIMALS):
        value = expected[var.name]
        invalid = [value+' ', 'UNKNOWN', None] if places is None else [True, None, str(value), float('nan'), float('inf'), value+1, value*1000]
        for candidate in invalid:
            assert not m.validate(expected | {var.name:candidate}).success
        if places is not None:
            assert m.validate(expected | {var.name:round(value, places)}).success
            if places:
                for sign in (-1,1):
                    assert m.validate(expected | {var.name:value+sign*0.4e-4}).success
                    assert not m.validate(expected | {var.name:value+sign*0.8e-4}).success


def test_bank_interacting_wrong_conventions():
    name='bank_cet1_growth_attribution_capacity';m=MODULES[name];e=EXPECTED[name]
    for cap,expense,capacity in product([e['capital_contribution_bp'],171.22229905334708,188.52094094249317], [255.15,183.01666666666668,265.77777777777777], [e['maximum_new_loans_bn'],14.225456870247918]):
        candidate=e | dict(capital_contribution_bp=cap, funding_expense_mn=expense, maximum_new_loans_bn=capacity)
        assert m.validate(candidate).success == (candidate == e)


def test_bank_exact_allocation_and_capacity_boundary():
    c0,c1,r0,r1=map(F,[11452872,12678329,71571110,65003760])
    # Explicit finite-game marginal averaging, independent of production formulas.
    capital=((c1/r0-c0/r0)+(c1/r1-c0/r1))/2
    risk=((c0/r1-c0/r0)+(c1/r1-c1/r0))/2
    assert capital+risk==c1/r1-c0/r0
    assert float(capital*10000)==EXPECTED['bank_cet1_growth_attribution_capacity']['capital_contribution_bp']
    margin=F('0.001931875');floor=c0/r0
    amount=F(str(EXPECTED['bank_cet1_growth_attribution_capacity']['maximum_new_loans_bn']))*1000000
    for change,above in [(F(-1),True),(F(1),False)]:
        ratio=(c1+(amount+change)*margin)/(r1+amount+change)
        assert (ratio>floor)==above


def test_forecast_policy_and_path_wrong_conventions():
    name='wti_budget_allowance_holdout_liquidity';m=MODULES[name];e=EXPECTED[name]
    for allowance,reserve,count in product([-5,5], [e['first_half_reserve_mn'],8.89], [9,12]):
        candidate=e | dict(selected_allowance_usd_barrel=allowance,first_half_reserve_mn=reserve,calibration_month_count=count)
        assert m.validate(candidate).success == (candidate==e)
    rows=m.forecast_panel();train=[r for r in rows if r[0].startswith('2024')]
    assert len(train)==9 and train[0][0]=='2024-04'
    # Symmetric loss happens to select the same policy, but is not the same objective.
    symmetric=sum(abs(a-f+D(5)) for _,f,a in train)
    assert not m.validate(e | {'calibration_loss_mn':float(symmetric)}).success
    half=[r for r in rows if '2025-01'<=r[0]<='2025-06']
    gaps=[sum((a-f+D(5) for _,f,a in half[:j]),D(0)) for j in range(1,7)]
    reserve=D(str(e['first_half_reserve_mn']))
    assert abs(min(reserve-g for g in gaps))<D("1e-12")
    assert min(reserve-D('0.00000001')-g for g in gaps)<0


def test_liquidity_surplus_reuse_and_earliest_binding():
    m=MODULES['wti_budget_allowance_holdout_liquidity']
    rows=[('2025-01',D(10),D(8)),('2025-02',D(10),D(13)),('2025-03',D(10),D(10))]
    assert m.liquidity_buffer(rows,D(0))==(D(1),'2025-02')
    assert m.liquidity_buffer(rows[:1],D(0))==(D(0),'NONE')
    with pytest.raises(ValueError,match='Empty'):
        m.liquidity_buffer([],D(0))
