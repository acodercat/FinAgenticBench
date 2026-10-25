"""Cash bridge and cent-settled allocation contract regression checks."""
import importlib
import json
from decimal import Decimal
from itertools import product
from pathlib import Path
import pytest

IDS = ['corporate_payout_cash_buffer_capacity', 'tips_indexed_settlement_equal_face_budget']
# Regression anchors independently rederived in the author Fraction/dict audit;
# blind A/B numerical evidence is kept separately from these tests.
VALUES = [
    [60, '0000021344', '0000021344-25-000011', '2025-02-20T20:17:12',
     5.413, 10.828, 4.588, 3.244, -0.957, 4.054, 0.435, 6.804],
    [5, '91282CNS6', 62332.69052218, 2, '2025-07-31', '2025-11-28',
     3035759.10, '2025-09-30', 18370.0, 988200, 62.86],
]

@pytest.mark.parametrize('case_id,expected', list(zip(IDS, VALUES)))
def test_numeric_and_turn_contracts(case_id, expected):
    module = importlib.import_module('evals.scenario_analysis.' + case_id)
    assert module.ground_truth() == tuple(expected)
    values = dict(zip([v.name for v in module.variables], expected))
    assert module.validate(values).success
    spec = json.loads(Path('evals/scenario_analysis', case_id+'.json').read_text())
    turns = spec['conversations'][0]['turns']
    assert [n for t in turns for n in t['stores']] == list(values)
    for i, turn in enumerate(turns, 1):
        local = {n: values[n] for n in turn['stores']}
        validator = getattr(module, f'validate_turn_{i}')
        assert validator(local).success
        for name in local:
            assert not validator({k:v for k,v in local.items() if k != name}).success
    for var, places in zip(module.variables, module.DECIMALS):
        value = values[var.name]
        invalid = ([value+' ', 'UNKNOWN'] if places is None else
                   [True, None, str(value), float('nan'), float('inf'), value+1, value*1000])
        for other in invalid:
            assert not module.validate(values | {var.name:other}).success
        if places is not None:
            assert module.validate(values | {var.name:round(value, places)}).success
            if places > 0:
                for sign in (-1, 1):
                    assert module.validate(values | {var.name:value+sign*0.4*10**-places}).success
                    assert not module.validate(values | {var.name:value+sign*0.8*10**-places}).success


def test_wrong_economic_paths_jointly_rejected():
    module = importlib.import_module('evals.scenario_analysis.'+IDS[0])
    correct = dict(zip([v.name for v in module.variables], VALUES[0]))
    for count, ending, repurchases in product([60,19], [4.054,10.929,-2.033], [0.435,1.795]):
        candidate = correct | dict(eligible_filer_count=count,
            full_plan_stressed_closing_cash_bn=ending, maximum_retained_repurchases_bn=repurchases)
        assert module.validate(candidate).success == (count==60 and ending==4.054 and repurchases==0.435)
    module = importlib.import_module('evals.scenario_analysis.'+IDS[1])
    correct = dict(zip([v.name for v in module.variables], VALUES[1]))
    for total, face in product([3035759.1,3054494.77,3017292.27,3023988.71],[988200,1000000]):
        assert module.validate(correct | dict(total_settlement_cash_usd=total,common_face_usd=face)).success == (total==3035759.1 and face==988200)


def test_cash_floor_feasibility():
    from evals.scenario_analysis.corporate_payout_cash_buffer_capacity import retained_repurchases
    assert retained_repurchases(10,5,2,3,4,5)==4
    assert retained_repurchases(10,1,2,3,4,5)==1
    with pytest.raises(ValueError, match='infeasible'):
        retained_repurchases(10,0,4,4,4,5)


def test_invoice_rounding_and_discrete_budget():
    from evals.scenario_analysis.tips_indexed_settlement_equal_face_budget import invoice, maximum_common_face
    from fractions import Fraction
    assert invoice(1, Decimal('1.005'))==Decimal('1.01')
    # Heterogeneous bid increments, exact rational cents and an exhaustive grid
    # challenge the binary search independently, including tight cent budgets.
    coeffs=[Decimal('0.99198443'),Decimal('1.02267031'),Decimal('1.02110436')]
    for budget in [Decimal('2000.01'),Decimal('3000.00'),Decimal('3035.76')]:
        for increments in [[100,100,100],[100,200,300]]:
            def rational_cost(face):
                total=0
                for c in coeffs:
                    x=Fraction(c)*face*100
                    total+=(2*x.numerator+x.denominator)//(2*x.denominator)
                return Decimal(total)/100
            choices=[f for f in range(100,2001,100) if all(f%i==0 for i in increments) and rational_cost(f)<=budget]
            face, remaining=maximum_common_face(coeffs,[100]*3,increments,budget)
            assert face==max(choices)
            assert remaining==budget-rational_cost(face)
    with pytest.raises(ValueError, match='minimum'):
        maximum_common_face(coeffs,[100]*3,[100]*3,Decimal('1'))
