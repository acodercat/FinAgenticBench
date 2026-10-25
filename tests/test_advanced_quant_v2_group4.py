"""Independent state-price/optimal-stopping and structural-credit checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from evals.scenario_analysis import mortgage_ho_lee_prepayment_oas as mortgage
from evals.scenario_analysis import treasury_leland_structural_debt_recapitalization as leland
MODULES=(mortgage,leland)
def truth(m):return dict(zip((v.name for v in m.variables),m.ground_truth()))
@pytest.mark.parametrize('m',MODULES)
def test_strict_boundary_replay(m):
    t=truth(m);d=json.loads(Path(m.__file__).with_suffix('.json').read_text());assert [n for q in d['conversations'][0]['turns']for n in q['stores']]==list(t)
    assert m.validate(t).success
    for i in range(1,5):
        names=getattr(m,f'TURN_{i}_NAMES');f=getattr(m,f'validate_turn_{i}');assert f({n:t[n]for n in names}).success;assert not f({}).success
    for n,dec in zip(t,m.DECIMALS):
        v=t[n]
        for w in [None,True,str(v),float('nan'),float('inf'),v+1,v*100 if abs(v)>1e-5 else 1]:assert not m.validate(t|{n:w}).success,(n,w)
        if dec:
            tol=.6*10**-dec
            assert m.validate(t|{n:round(v,dec)}).success
            assert m.validate(t|{n:v+.99*tol}).success
            assert not m.validate(t|{n:v+1.01*tol}).success
            if 'residual' in n:assert not m.validate(t|{n:-.01*tol}).success
            else:assert not m.validate(t|{n:-v}).success
        else:assert not m.validate(t|{n:v+.1}).success
    before=m.ground_truth();m.ground_truth.cache_clear();assert before==m.ground_truth()


@pytest.mark.parametrize('m',[mortgage,leland])
def test_treasury_source_removal(m,monkeypatch):
    original=m.load_expansion_table
    def without(n):
        if n=='treasury_yield_curve':raise FileNotFoundError('removed source')
        return original(n)
    if m is mortgage:m.mortgage_inputs.cache_clear()
    m.ground_truth.cache_clear()
    with monkeypatch.context() as ctx:
        ctx.setattr(m,'load_expansion_table',without)
        with pytest.raises(FileNotFoundError):m.ground_truth()
    m.ground_truth.cache_clear()

def test_hmda_source_removal(monkeypatch):
    mortgage.mortgage_inputs.cache_clear();mortgage.ground_truth.cache_clear();original=mortgage.load_catalog
    def without():
        d=original();del d['tables']['hmda.loan_applications'];return d
    with monkeypatch.context() as ctx:
        ctx.setattr(mortgage,'load_catalog',without)
        with pytest.raises(KeyError):mortgage.ground_truth()
    mortgage.ground_truth.cache_clear()


@pytest.mark.parametrize("module, calculate, limits", [
    (mortgage, mortgage.mortgage_a, {'state_price_residual': 1e-10, 'spread_repricing_residual': 1e-10}),
    (leland, leland.leland_a, {'debt_repricing_residual': 1e-10, 'smooth_pasting_residual': 1e-10, 'optimality_residual': 1e-08}),
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
