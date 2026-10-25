"""Independent GRS/KKT and integrated-Gaussian option verification."""
import json
from pathlib import Path
import numpy as np
import pytest
from evals.scenario_analysis import industry_grs_residual_alpha_overlay as grs
from evals.scenario_analysis import sofr_vasicek_jamshidian_bond_option as vas

MODULES=(grs,vas)
def truth(m):return dict(zip((v.name for v in m.variables),m.ground_truth()))

@pytest.mark.parametrize('m',MODULES)
def test_strict_contract_replay_boundaries(m):
    t=truth(m);s=json.loads(Path(m.__file__).with_suffix('.json').read_text())
    assert len(s['conversations'][0]['turns'])==4
    assert [n for q in s['conversations'][0]['turns'] for n in q['stores']]==list(t)
    assert m.validate(t).success
    for i in range(1,5):
        names=getattr(m,f'TURN_{i}_NAMES');check=getattr(m,f'validate_turn_{i}')
        assert check({n:t[n] for n in names}).success
        assert not check({}).success
    for n,d in zip(t,m.DECIMALS):
        v=t[n];invariant=n=='regression_orthogonality' or ('residual' in n and n!='residual_covariance_condition')
        for w in [None,True,float('nan'),float('inf'),str(v),v+max(1,abs(v)),v*100 if abs(v)>1e-5 else 1]:
            assert not m.validate(t|{n:w}).success,(n,w)
        if d:
            tol=.6*10**(-d)
            assert m.validate(t|{n:v+tol*.99}).success,n
            assert not m.validate(t|{n:v+tol*1.01}).success,n
            assert m.validate(t|{n:round(v,d)}).success,n
            if invariant:assert not m.validate(t|{n:-tol*.1}).success,n
            else:assert not m.validate(t|{n:-v}).success,n
        else:assert not m.validate(t|{n:v+.01}).success
    before=m.ground_truth();m.ground_truth.cache_clear();assert m.ground_truth()==before


@pytest.mark.parametrize('m,label,loader',[(grs,'daily_industry_returns','load_ken_french_table'),(grs,'daily_factors','load_ken_french_table'),(vas,'nyfed_reference_rates','load_expansion_table'),(vas,'treasury_yield_curve','load_expansion_table')])
def test_source_removal(m,label,loader,monkeypatch):
    original=getattr(m,loader)
    def without(name):
        if name==label:raise FileNotFoundError('removed family')
        return original(name)
    m.ground_truth.cache_clear()
    with monkeypatch.context() as ctx:
        ctx.setattr(m,loader,without)
        with pytest.raises(FileNotFoundError):m.ground_truth()
    m.ground_truth.cache_clear()


@pytest.mark.parametrize("module, calculate, limits", [
    (grs, grs.grs_a, {'regression_orthogonality': 1e-10, 'overlay_kkt_residual': 1e-08}),
    (vas, vas.vasicek_a, {'par_repricing_residual_per100': 1e-10, 'put_call_parity_residual_per100': 1e-10}),
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
