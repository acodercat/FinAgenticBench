"""Observable delayed-state correctness; never infer internal model memory."""
import numpy as np
import pytest
from evals.scenario_analysis import hmda_pool_acquisition_state_preservation as m
from evals.scenario_analysis import nmfp_bridge_facility_state_preservation as f


def test_lending_conservation_order_and_delay():
    x,h,r=m.inputs();z=x[x.loan_type.eq('1')];a=m.pool(z,h);b=m.pool(z.iloc[::-1],h)
    assert abs(a['cf'].sum()-b['cf'].sum())<.01
    assert np.all(a['loss']>=0) and np.all(a['loss']<=a['b'])
    assert abs(a['cf'][-1]-a['a'].sum()-a['b'].sum()+a['loss'].sum())<.01
    correct=m.calculate()[0];wrong=m.calculate(contamination='fha')[0]
    assert {k:correct[k] for k in m.TURN_1_NAMES+m.TURN_2_NAMES}=={k:wrong[k] for k in m.TURN_1_NAMES+m.TURN_2_NAMES}
    assert not m.validate_turn_5({k:wrong[k] for k in m.TURN_5_NAMES}).success
    assert not m.validate_turn_6({k:wrong[k] for k in m.TURN_6_NAMES}).success

def test_bridge_cash_conservation_and_delayed_fee():
    a,e=f.calculate();c=e['prime_capped']
    assert abs(c['end_cash']-(500000-.02*c['sold']-c['interest']))<1e-7
    assert c['peak']<=e['facility']+1e-7
    assert e['unused_dollar_days']>0 and a['government_max_commitment_fee_bp']>25
    wrong=f.calculate(contamination='prime')[0]
    assert {k:a[k] for k in f.TURN_1_NAMES+f.TURN_2_NAMES}=={k:wrong[k] for k in f.TURN_1_NAMES+f.TURN_2_NAMES}
    assert not f.validate_turn_5({k:wrong[k] for k in f.TURN_5_NAMES}).success

@pytest.mark.parametrize('module,sources',[(m,['fhfa_hpi','nyfed_reference_rates']),(f,['sec_nmfp','sec_nmfp_liquidity','nyfed_reference_rates'])])
def test_source_removal(module,sources,monkeypatch):
    for source in sources:
        original=module.load_expansion_table
        with monkeypatch.context() as patch:
            def missing(name):
                if name==source:raise FileNotFoundError(source)
                return original(name)
            patch.setattr(module,'load_expansion_table',missing);module.inputs.cache_clear()
            with pytest.raises(FileNotFoundError):module.inputs()
        module.inputs.cache_clear()


def test_hmda_source_removal(monkeypatch):
    m.inputs.cache_clear()
    monkeypatch.setattr(m, 'load_catalog', lambda: {'tables': {}})
    with pytest.raises(KeyError, match='hmda.loan_applications'):
        m.inputs()
    m.inputs.cache_clear()
