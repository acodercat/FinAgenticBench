"""Independent conditional-Gaussian and compound-mixture evidence."""
import json
from pathlib import Path
import numpy as np
import pytest
from evals.temporal_path import industry_kalman_beta_smoothing as beta
from evals.scenario_analysis import nfip_compound_poisson_stoploss_capacity as loss

MODULES=(beta,loss)
def truth(m):return dict(zip((v.name for v in m.variables),m.ground_truth()))

def test_contract_and_strict_pv():
    for m in MODULES:
        t=truth(m);spec=json.loads(Path(m.__file__).with_suffix('.json').read_text())
        assert len(spec['conversations'][0]['turns'])==4
        assert [n for x in spec['conversations'][0]['turns'] for n in x['stores']]==list(t)
        assert m.validate(t).success
        for i in range(1,5):
            ns=getattr(m,f'TURN_{i}_NAMES');assert getattr(m,f'validate_turn_{i}')({n:t[n] for n in ns}).success
        for n,v in t.items():
            for wrong in [None,True,float('nan'),float('inf'),str(v),v+max(1,abs(v)),v*100 if abs(v)>1e-5 else 1]:
                assert not m.validate(t|{n:wrong}).success,(n,wrong)
            if v>0:assert not m.validate(t|{n:-v}).success,n
        for n,d in zip(t,m.DECIMALS):
            if d==0:assert not m.validate(t|{n:t[n]+.01}).success
            else:
                tol=.6*10**(-d)
                assert m.validate(t|{n:t[n]+tol*.99}).success,n
                assert not m.validate(t|{n:t[n]+tol*1.01}).success,n
                if 'residual' in n or 'identity' in n:
                    assert not m.validate(t|{n:-tol*.1}).success
        first=m.ground_truth();m.ground_truth.cache_clear();assert first==m.ground_truth()


def test_runtime_source_removal_and_substitutes():
    # Loaded table identities, not hidden author fixtures, own each core input.
    from core.runtime_catalog import select_runtime_tables
    for m,sources in [(beta,['ken_french_10_industry_portfolios','ken_french_daily_factors']),(loss,['fema_nfip','treasury_yield_curve'])]:
        all_ids={x.name for x in select_runtime_tables(sources)}
        for removed in sources:
            kept={x.name for x in select_runtime_tables([x for x in sources if x!=removed])}
            assert kept<all_ids

@pytest.mark.parametrize('module,label,loader',[
    (beta,'daily_industry_returns','load_ken_french_table'),
    (beta,'daily_factors','load_ken_french_table'),
    (loss,'fema_nfip','load_expansion_table'),
    (loss,'treasury_yield_curve','load_expansion_table'),
])
def test_removed_source_prevents_core_answer(module,label,loader,monkeypatch):
    original=getattr(module,loader)
    def restricted(name):
        if name==label:raise FileNotFoundError('source removed')
        return original(name)
    module.ground_truth.cache_clear()
    with monkeypatch.context() as scoped:
        scoped.setattr(module,loader,restricted)
        with pytest.raises(FileNotFoundError):module.ground_truth()
    module.ground_truth.cache_clear()


@pytest.mark.parametrize("module, calculate, limits", [
    (beta, beta.kalman_a, {'smoother_terminal_identity': 1e-10}),
    (loss, loss.insurance, {'aggregate_mass_residual': 1e-10, 'loss_conservation_residual_thousand': 1e-10, 'capacity_budget_residual_thousand': 1e-08}),
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
