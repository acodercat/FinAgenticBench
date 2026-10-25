"""Case-local quantitative invariants and preregistered strict acceptance."""
import json
from pathlib import Path
import numpy as np
import pytest
from evals.scenario_analysis import wti_evt_cluster_maximum_layer as evt
from evals.scenario_analysis import industry_black_litterman_view_risk_budget as bl

@pytest.mark.parametrize('m',[evt,bl])
def test_strict_boundary_replay_and_stores(m):
    truth=dict(zip((v.name for v in m.variables),m.ground_truth()));assert m.validate(truth).success
    assert m.validate({n:round(v,d)for(n,v),d in zip(truth.items(),m.DECIMALS)}).success
    assert m.ground_truth()==m.ground_truth()
    spec=json.loads(Path(m.__file__).with_suffix('.json').read_text());ns=[]
    for i,t in enumerate(spec['conversations'][0]['turns'],1):
        ns.extend(t['stores']);subset={n:truth[n]for n in t['stores']};assert getattr(m,f'validate_turn_{i}')(subset).success
    assert ns==list(truth)
    for (n,v),d in zip(truth.items(),m.DECIMALS):
        for x in [None,True,str(v),float('nan'),float('inf'),-float('inf')]:assert not m.validate({**truth,n:x}).success
        if d:
            tol=.6*10**-d;assert m.validate({**truth,n:v+.99*tol}).success;assert not m.validate({**truth,n:v+1.01*tol}).success
            if 'residual' in n:assert not m.validate({**truth,n:-1e-12}).success
            elif v!=0:
                assert not m.validate({**truth,n:-v}).success
                assert not m.validate({**truth,n:v*100}).success
        else:assert not m.validate({**truth,n:v+.1}).success


def test_source_removal(monkeypatch):
    for alias in ['eia_bulk_wti_daily','ofr_market_stress']:
        original=evt.load_expansion_table
        with monkeypatch.context() as m:
            def absent(name):
                if name==alias:raise FileNotFoundError(alias)
                return original(name)
            m.setattr(evt,'load_expansion_table',absent);evt.evt_inputs.cache_clear()
            with pytest.raises(FileNotFoundError):evt.evt_inputs()
    evt.evt_inputs.cache_clear()
    for alias in ['daily_industry_returns','daily_factors']:
        original=bl.load_ken_french_table
        with monkeypatch.context() as m:
            def absent(name):
                if name==alias:raise FileNotFoundError(alias)
                return original(name)
            m.setattr(bl,'load_ken_french_table',absent);bl.bl_inputs.cache_clear()
            with pytest.raises(FileNotFoundError):bl.bl_inputs()
    bl.bl_inputs.cache_clear()


@pytest.mark.parametrize("module, calculate, limits", [
    (evt, evt.evt_a, {'lmoment_residual_usd': 1e-10, 'quantile_probability_residual': 1e-10, 'threshold_stability_residual': 1e-10, 'layer_additivity_residual_usd': 1e-10}),
    (bl, bl.bl_a, {'posterior_precision_residual': 1e-10, 'allocation_kkt_residual_bp': 1e-08, 'risk_budget_residual_bp': 1e-08}),
])
def test_computed_invariants_are_internal(module, calculate, limits):
    # These are computed solver diagnostics, never evidence from a submitted zero.
    diagnostics, _ = calculate()
    public = {v.name for v in module.variables}
    assert public.isdisjoint(limits)
    for name, bound in limits.items():
        assert np.isfinite(diagnostics[name]) and 0 <= diagnostics[name] < bound, name
    spec = json.loads(Path(module.__file__).with_suffix('.json').read_text())
    for turn in spec['conversations'][0]['turns']:
        assert len(turn['stores']) >= 2
        assert set(turn['stores']).isdisjoint(limits)
    assert not module.validate({name: 0 for name in public}).success
