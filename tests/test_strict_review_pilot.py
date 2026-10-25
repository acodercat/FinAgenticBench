"""Independent financial-object checks and strict acceptance for the two-case pilot."""
import json
from pathlib import Path
import pytest
from evals.scenario_analysis import industry_entropy_pooling_tail_kelly as ep
from evals.scenario_analysis import bank_eisenberg_noe_recovery_allocation as en

@pytest.mark.parametrize('m',[ep,en])
def test_strict_outputs_and_turns(m):
    truth=dict(zip((v.name for v in m.variables),m.ground_truth()));assert m.validate(truth).success
    assert m.validate({k:round(v,d)if d is not None else v for (k,v),d in zip(truth.items(),m.DECIMALS)}).success
    m.ground_truth.cache_clear();assert tuple(truth.values())==m.ground_truth()
    spec=json.loads(Path(m.__file__).with_suffix('.json').read_text());stores=[]
    for i,t in enumerate(spec['conversations'][0]['turns'],1):
        stores+=t['stores'];assert len(t['stores'])>=3
        assert getattr(m,f'validate_turn_{i}')({n:truth[n]for n in t['stores']}).success
    assert stores==list(truth)
    for (k,v),d in zip(truth.items(),m.DECIMALS):
        assert not m.validate({n:z for n,z in truth.items()if n!=k}).success
        if d is None:
            for bad in [v.lower()+' ',int(v)if v.isdigit()else'Energy',None]:assert not m.validate({**truth,k:bad}).success
            continue
        for bad in [True,None,str(v),float('nan'),float('inf'),float('-inf')]:assert not m.validate({**truth,k:bad}).success
        if d==0:assert not m.validate({**truth,k:v+.1}).success
        else:
            tol=.6*10**-d
            assert m.validate({**truth,k:v+.999*tol}).success
            assert not m.validate({**truth,k:v+1.001*tol}).success
            if abs(v)>tol:
                for bad in [-v,100*v,v/100]:assert not m.validate({**truth,k:bad}).success
    assert not m.validate({n:0 for n in truth}).success


def test_source_removal(monkeypatch):
    cases=[(ep,'entropy_inputs','load_ken_french_table','daily_industry_returns'),(ep,'entropy_inputs','load_expansion_table','ofr_market_stress'),(en,'network_inputs','load_expansion_table','fdic_bankfind'),(en,'network_inputs','load_expansion_table','treasury_yield_curve')]
    for m,loader,access,alias in cases:
        original=getattr(m,access)
        with monkeypatch.context() as mm:
            def missing(name):
                if name==alias:raise FileNotFoundError(alias)
                return original(name)
            mm.setattr(m,access,missing);getattr(m,loader).cache_clear()
            with pytest.raises(FileNotFoundError):getattr(m,loader)()
        getattr(m,loader).cache_clear()
