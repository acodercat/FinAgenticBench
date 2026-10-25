import itertools
import numpy as np
from evals.calculation_reconciliation import equity_awards_convertible_antidilution_sequence as eps
from evals.comparison_ranking import sba_collateral_seniority_guarantee_loss_frontier as sba

def test_eps_control_number_sequence_and_actual_conversion():
    n,shares,d=eps.inputs();result=eps.analysis();inc=result['incremental_option_shares_mn']*1e6
    answers=[(n+sum(e for bit,e in zip(mask,[150e6,240e6,180e6]) if bit))/(shares+inc+sum(s for bit,s in zip(mask,[30e6,30e6,15e6]) if bit)) for mask in itertools.product([0,1],repeat=3)]
    assert abs(result['profitable_diluted_eps']-min(answers))<1e-12
    assert result['included_convertibles']=='A'
    assert result['loss_total_diluted_eps']>0 and result['loss_continuing_diluted_eps']<0
    assert result['loss_included_potential']=='NONE'
    assert result['converted_potential_set']=='NONE'
    assert abs(result['converted_basic_shares_mn']-(shares+30e6*184/365)/1e6)<1e-10
    assert abs(result['converted_continuing_diluted_eps']-(-.2*n+240e6*184/365)/(shares+30e6*184/365))<1e-12

def test_sba_loss_conservation_rank_reversal_and_frontier():
    result=sba.analysis();_,_,a,b,g,q=sba.inputs()
    for r in [.0,.35,.6,.85,1.]:
        cash=.95*r*(a+b)/.9
        sr=np.minimum(a,cash);jr=np.minimum(b,np.maximum(cash-a,0))
        assert np.allclose(sr+jr,np.minimum(cash,a+b))
    assert result['deterministic_lower_loss_book']=='SBA504'
    assert result['stochastic_lower_loss_book']=='SBA7A'
    assert result['stochastic_504_mean_loss_mn']>result['deterministic_senior_loss_mn']+10
    z=result['maximum_common_recovery_drop_pp']/100
    assert abs(sba.expected(z)[0]-35)<1e-8
    assert max(sba.expected(z-1e-5)[[0,3]])<35
    assert max(sba.expected(z+1e-5)[[0,3]])>35

def test_pair_strict_legal_representations_and_wrong_financial_paths():
    for case in [eps,sba]:
        values=case.analysis()
        assert case.validate(values).success
        rounded={v.name:round(values[v.name],dp) if dp is not None else values[v.name] for v,dp in zip(case.variables,case.DECIMALS)}
        assert case.validate(rounded).success
        for v,dp in zip(case.variables,case.DECIMALS):
            altered=dict(values);altered[v.name]=True if dp is not None else 'INVALID'
            assert not case.validate(altered).success
    values=dict(eps.analysis());n,s,d=eps.inputs();values['profitable_diluted_eps']=(n+390e6)/(s+values['incremental_option_shares_mn']*1e6+60e6)
    assert not eps.validate(values).success
    values=dict(sba.analysis());values['deterministic_senior_loss_mn']=100*(1-.95*.6/.9)
    assert not sba.validate(values).success
