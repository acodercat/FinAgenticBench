"""Observable delayed-state correctness; never infer internal model memory."""
import numpy as np
import pytest
from evals.scenario_analysis import wti_inventory_hedge_state_preservation as m
from evals.scenario_analysis import pension_contribution_basis_state_preservation as f


def test_hedge_cash_invariants_and_delayed_positions():
    out,e=m.calculate();months,a,b,k,actual,r=m.inputs()
    q=np.asarray(e['quantities']);h=e['hedge']
    assert np.max(np.abs(q*a-1e7))<1e-7
    assert max(e['stress_cash'])<=11e6+1e-7
    assert max(1.25*q*a-(h-1e-6)*q*(1.25*a-k))>11e6
    assert months[min(e['binding_constraints'])]==out['original_binding_month']
    assert abs(e['debt']+e['storage']-out['original_break_even_sale_usd_barrel']*q.sum())<1e-7
    for kw in [dict(quantity_state='proposal'),dict(hedge_state='proposal')]:
        wrong=m.calculate(**kw)[0]
        early=sum((getattr(m,f'TURN_{i}_NAMES') for i in range(1,5)),[])
        assert all(out[n]==wrong[n] for n in early)
        assert not m.validate_turn_5({n:wrong[n] for n in m.TURN_5_NAMES}).success
        assert not m.validate_turn_6({n:wrong[n] for n in m.TURN_6_NAMES}).success


def test_continuous_grant_uniqueness_and_delayed_basis():
    out,e=f.calculate();a=np.array(e['current']);l=np.array(e['target']);B=e['budget']
    for budget in [B*.1,B,B*10]:
        g,t=f.allocate(a,l,budget)
        assert min(g)>=0 and abs(g.sum()-budget)<1e-7
        assert min((a+g)/l)>=t-1e-12
        assert np.allclose(g,np.maximum(t*l-a,0),atol=1e-8,rtol=0)
        assert np.maximum((t+1e-7)*l-a,0).sum()>budget
        perm=np.array([2,0,3,1]);gp,tp=f.allocate(a[perm],l[perm],budget)
        assert abs(t-tp)<1e-12 and np.allclose(g[perm],gp,atol=1e-8,rtol=0)
    assert np.min(np.array(e['restored_assets'])-e['terminal_target'])>=-1e-7
    for kw in [dict(asset_state='actuarial'),dict(grant_state='actuarial')]:
        wrong=f.calculate(**kw)[0]
        early=sum((getattr(f,f'TURN_{i}_NAMES') for i in range(1,5)),[])
        assert all(out[n]==wrong[n] for n in early)
        assert not f.validate_turn_5({n:wrong[n] for n in f.TURN_5_NAMES}).success


def test_nullable_eligibility_requires_complete_financial_inputs():
    s=f.load_expansion_table('dol_form5500_schedule_sb');fin=f.load_expansion_table('dol_form5500_financials')
    s=s[s.plan_year_end.eq('2024-12-31')&s.valuation_date.eq('2024-01-01')&s.filing_status.eq('FILING_RECEIVED')&s.date_received.le('2025-12-31')]
    q=s.merge(fin[fin.filing_status.eq('FILING_RECEIVED')][['filing_id','total_expenses_usd']],on='filing_id',validate='one_to_one')
    q=q.sort_values(['date_received','filing_id']).drop_duplicates(['sponsor_ein','plan_number','plan_year_end'],keep='last')
    cols=['current_value_assets_usd','actuarial_value_assets_usd','total_funding_target_usd','total_expenses_usd']
    finite=q[cols].notna().all(axis=1);positive=q[cols].gt(0).all(axis=1)
    assert ((~finite)&positive).sum()>0
    z,count,_=f.inputs();assert count==int((finite&positive).sum())
    assert z[cols].notna().all().all()


@pytest.mark.parametrize('module,sources',[(m,['eia_steo_vintages','eia_bulk_wti','nyfed_reference_rates']),(f,['dol_form5500_schedule_sb','dol_form5500_financials','treasury_yield_curve'])])
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
