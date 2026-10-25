"""Meaningful contract and financial-path checks for an unregistered case."""
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
from evals.temporal_path import fund_risk_scale_flow_profile_reassessment as case


def test_selection_uses_risk_intensity_not_dollar_risk_decline():
    records=[dict(series=str(i),drop=F(100-i),dv0=F(100),dv1=F(i),nav0=F(100),nav1=F(10*i)) for i in range(1,7)]
    cohort,selected=case.rank_profiles(records)
    assert [r['series'] for r in cohort]==['1','2','3','4','5']
    assert selected['series']=='1'  # exact intensity-change tie uses series ID
    records[3]['nav1']=F(1)
    _,selected=case.rank_profiles(records)
    assert selected['series']=='4'
    with pytest.raises(ValueError):case.rank_profiles(records[:4])


def test_scenario_scales_capital_preserves_signs_and_tenor_ties():
    result=case.scenario([F(2)]*5,[F(1)]*5,F(100),F(50),[F(-1)]*5)
    assert result[:4]==(F(5),F(5),'3M',F(0))
    a=case.scenario([F(1)]*5,[F(2)]*5,F(100),F(100),[F(1),F(-1),F(0),F(0),F(0)])
    assert a[2:4]==('3M',F(-1))
    with pytest.raises(ValueError):case.scenario([F(1)]*5,[F(1)]*5,F(0),F(100),[F(1)]*5)


def test_live_ground_truth_turns_and_unscaled_profile_rejection():
    answer,evidence=case.analysis()
    assert case.validate(answer).success
    runtime=SimpleNamespace(retrieve=lambda name:answer[name])
    for names,validator in zip(case.TURN_NAMES,case.validators.values()):
        result=validator('do not parse this prose',runtime,SimpleNamespace(stores=names))
        assert result.success and set(result.variable_results)==set(names)
    assert evidence['invariant_bridge_exact']
    wrong=dict(answer,june_profile_value_change_usd=evidence['unscaled_june_scenario'])
    assert not case.validate_profile_scenario(wrong).success
    # Inverting the redemption direction is a different balance bridge.
    wrong=dict(answer,net_share_transactions_usd=-answer['net_share_transactions_usd'])
    assert not case.validate_share_bridge(wrong).success


def test_ordered_list_contract_and_unset_diagnostics():
    answer,_=case.analysis()
    for value in [list(reversed(answer['decliner_series'])),tuple(answer['decliner_series']),np.array(answer['decliner_series']),[], 'S000004071']:
        assert not case.validate_cohort(dict(answer,decliner_series=value)).success
    for value in [None]:
        assert case.validate_cohort(dict(answer,decliner_series=value)).variables_not_set
    missing=dict(answer);del missing['decliner_series']
    assert case.validate_cohort(missing).variables_not_set


def test_numeric_types_units_and_rounding_boundary():
    answer,_=case.analysis()
    rounded={n:round(answer[n],p) if p is not None else answer[n] for n,_,p in case.SPECS}
    assert case.validate(rounded).success
    for v in [None,True,float('inf'),float('nan'),'432.7511']:
        assert not case.validate_intensity(dict(answer,june_unit_dv01=v)).success
    assert not case.validate_intensity(dict(answer,june_unit_dv01=answer['june_unit_dv01']/100)).success
    assert case.validate_intensity(dict(answer,june_unit_dv01=answer['june_unit_dv01']+0.000059)).success
    assert not case.validate_intensity(dict(answer,june_unit_dv01=answer['june_unit_dv01']+0.000061)).success


