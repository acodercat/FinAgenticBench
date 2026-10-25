import json
from pathlib import Path

import numpy as np

from evals.scenario_analysis import energy_garch_ofr_tail_forecast as garch
from evals.temporal_path import nmfp_hmm_liquidity_regime as hmm
from scripts.audit_case_design import has_closed_text_representation

ROOT=Path(__file__).resolve().parents[1]

def _truth(module): return dict(zip((v.name for v in module.variables),module.ground_truth()))

def _assert_declared_precision(module):
    for variable, decimals in zip(module.variables, module.DECIMALS):
        if isinstance(decimals, int) and decimals > 0:
            assert f"rounded to {decimals} decimals" in variable.description

def test_group2_four_turns_sources_and_store_partition():
    for relative,module in (("evals/scenario_analysis/energy_garch_ofr_tail_forecast.json",garch),("evals/temporal_path/nmfp_hmm_liquidity_regime.json",hmm)):
        spec=json.loads((ROOT/relative).read_text()); turns=spec["conversations"][0]["turns"]
        assert len(turns)==4 and len(set(spec["data_sources"]))>=2
        stores=[x for turn in turns for x in turn["stores"]]
        assert stores==[v.name for v in module.variables] and len(stores)==len(set(stores))

def test_group2_output_descriptions_expose_scored_precision_and_frequency():
    _assert_declared_precision(hmm)
    _assert_declared_precision(garch)
    by_name = {variable.name: variable.description for variable in garch.variables}
    for name in (
        "garch_unconditional_volatility_pct",
        "stress_tail_mean_conditional_volatility_pct",
        "other_days_mean_conditional_volatility_pct",
    ):
        assert "daily" in by_name[name]

def test_group2_truth_passes_all_validators():
    for module in (garch,hmm):
        truth=_truth(module); assert module.validate(truth).success
        assert all(has_closed_text_representation(v.description) for v,value in zip(module.variables,module.ground_truth()) if isinstance(value,str))
        for i in range(1,5):
            names=getattr(module,f"TURN_{i}_NAMES")
            assert getattr(module,f"validate_turn_{i}")({n:truth[n] for n in names}).success

def test_garch_constraints_recursion_and_tail_order():
    truth=_truth(garch); train,p,residual,variance,_=garch._fit(); mean,oil,omega,alpha,beta=p
    assert omega>0 and alpha>=0 and beta>=0 and alpha+beta<.999
    y=train.energy_excess.to_numpy(float); oil_values=train.oil_return.to_numpy(float)
    preliminary=np.linalg.lstsq(np.column_stack([np.ones(len(y)),oil_values]),y,rcond=None)[0]
    initial=np.var(y-preliminary[0]-preliminary[1]*oil_values,ddof=1)
    independent=np.empty(len(residual)); independent[0]=initial
    for i in range(1,len(residual)): independent[i]=omega+alpha*residual[i-1]**2+beta*independent[i-1]
    assert np.max(np.abs(independent-variance))<1e-12
    assert truth["conditional_gaussian_es_average_loss_pct"]>truth["conditional_gaussian_var_average_loss_pct"]

def test_garch_likelihood_beats_nearby_parameters_and_coverage_identity():
    train,p,residual,variance,ll=garch._fit()
    def loglike(omega,alpha,beta):
        path=garch._variance_path(residual,omega,alpha,beta,float(np.var(residual,ddof=1)))
        return float(-.5*np.sum(np.log(2*np.pi)+np.log(path)+residual**2/path))
    assert ll>loglike(p[2]*1.05,p[3],p[4])
    truth=_truth(garch)
    assert np.isclose(truth["conditional_var_breach_rate_pct"],truth["conditional_var_breach_count"]/truth["test_observation_count"]*100)

def test_hmm_forward_backward_mass_and_transition_invariants():
    panel,center,scale,initial,transition,means,variances,ll,gamma,_=hmm._fit()
    assert np.allclose(transition.sum(1),1) and np.all(transition>0)
    assert np.all(variances>=1e-6)
    assert np.max(np.abs(gamma.sum(1)-1))<1e-9
    assert means[1].sum()<means[0].sum()

def test_hmm_predictive_score_and_label_are_live():
    truth=_truth(hmm)
    assert truth["hmm_beats_single_gaussian"]==(truth["hmm_average_predictive_log_score"]>truth["single_gaussian_average_log_score"])
    assert truth["maximum_filtered_stress_probability"]>.5

def test_group2_replays_exactly_and_wrong_values_fail():
    for module in (garch,hmm):
        first=module.ground_truth(); module.ground_truth.cache_clear(); second=module.ground_truth(); assert first==second
        truth=_truth(module)
        for i in range(1,5):
            names=getattr(module,f"TURN_{i}_NAMES"); candidate={n:truth[n] for n in names}
            key=next(n for n in names if isinstance(candidate[n],(int,float)) and not isinstance(candidate[n],bool))
            candidate[key]=float(candidate[key])+1
            assert not getattr(module,f"validate_turn_{i}")(candidate).success
