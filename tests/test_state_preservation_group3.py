"""Observable delayed-state correctness; never infer internal model memory."""
import numpy as np
import pytest
from evals.scenario_analysis import industry_style_attribution_state_preservation as m
from evals.scenario_analysis import storm_payroll_policy_state_preservation as f


def test_original_model_rank_orthogonality_attribution_and_delay():
    out,e=m.calculate()
    for tag in ['original_fit','comparison_fit']:
        assert e[tag]['rank']==4 and e[tag]['condition']<300
        assert e[tag]['orthogonality']<1e-9
    for tag in ['original_path','comparison_path']:
        assert abs(e[tag]['attribution_residual'])<1e-8
        assert e[tag]['minimum_gross_return']>0
    wrong=m.calculate(coefficient_state='comparison')[0]
    early=sum((getattr(m,f'TURN_{i}_NAMES') for i in range(1,5)),[])
    assert all(out[n]==wrong[n] for n in early)
    assert not m.validate_turn_5({n:wrong[n] for n in m.TURN_5_NAMES}).success
    assert not m.validate_turn_6({n:wrong[n] for n in m.TURN_6_NAMES}).success


def test_interval_union_financial_equivalence_and_waterfall():
    import pandas as pd
    p,s=f.inputs();a,d=f.claims(p,s,'31-33',24)
    # Under the stated parametric coverage, repeated equal intervals add no insured time.
    b,_=f.claims(p,pd.concat([s,s.iloc[:3]],ignore_index=True),'31-33',24)
    c,_=f.claims(p,s.iloc[::-1],'31-33',24)
    assert a.equals(b) and a.equals(c)
    paid,ex=f.settle(a);ix=paid.index.get_loc(ex.name)
    assert abs(paid.paid.sum()-8e6)<1e-7
    assert np.allclose(paid.paid.iloc[:ix],paid.loss.iloc[:ix])
    assert (paid.paid.iloc[ix+1:]==0).all() and 0<ex.paid<ex.loss
    _,e=f.calculate();assert e['reinsurance_conservation']<1e-8 and e['retained_leader_margin']>0


def test_delayed_policy_pollution_preserves_early_reports():
    out=f.calculate()[0];early=sum((getattr(f,f'TURN_{i}_NAMES') for i in range(1,5)),[])
    for args in [dict(exposure_state='comparison'),dict(waiting_state='comparison')]:
        bad=f.calculate(**args)[0];assert all(out[k]==bad[k] for k in early)
        assert not f.validate_turn_5({k:bad[k] for k in f.TURN_5_NAMES}).success
        assert not f.validate_turn_6({k:bad[k] for k in f.TURN_6_NAMES}).success


@pytest.mark.parametrize('module,loader,sources',[(m,'load_expansion_table',['ofr_market_stress']),(m,'load_ken_french_table',['daily_factors','daily_industry_returns']),(f,'load_expansion_table',['bls_qcew','noaa_storm_events'])])
def test_source_removal(module,loader,sources,monkeypatch):
    for source in sources:
        original=getattr(module,loader)
        with monkeypatch.context() as patch:
            def missing(name):
                if name==source:raise FileNotFoundError(source)
                return original(name)
            patch.setattr(module,loader,missing);module.inputs.cache_clear()
            with pytest.raises(FileNotFoundError):module.inputs()
        module.inputs.cache_clear()
