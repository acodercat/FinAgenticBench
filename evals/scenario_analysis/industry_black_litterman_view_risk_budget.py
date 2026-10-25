"""Advanced quantitative formula-free baseline with a single strict PV channel.
Pinned variants remain offline regression probes, not live traps.
Ground truth uses governed runtime; raw admission is recorded independently.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import solve
from core.data import load_ken_french_table

@lru_cache(None)
def bl_inputs(excess=True,start='2018-01-01'):
    i=load_ken_french_table('daily_industry_returns');f=load_ken_french_table('daily_factors');d=i.merge(f,on='date',validate='one_to_one').sort_values('date');cols=['hitec_pct','utils_pct','enrgy_pct','nodur_pct'];tr=d[d.date.between(start,'2022-12-31')];te=d[d.date.between('2023-01-01','2023-12-31')]
    def values(x):return (x[cols].to_numpy(float)-(x.rf_pct.to_numpy(float)[:,None] if excess else 0))/100
    return values(tr),values(te),tr,te

def bl_state(S,pi,k=1,predictive=True,view_sign=1):
    if k<=0 or np.min(np.linalg.eigvalsh(S))<=0:raise ValueError('invalid BL domain')
    P=np.array([[1.,-1,0,0],[0,0,1,-1]]);q=np.array([.03,.01])*view_sign;T=.05*S;O=k*np.diag([.02**2,.03**2]);K=solve(P@T@P.T+O,P@T,assume_a='pos').T
    mu=pi+K@(q-P@pi);M=T-K@P@T;V=S+M if predictive else S
    block=np.block([[4*V,np.ones((4,1))],[np.ones((1,4)),np.zeros((1,1))]])
    ans=solve(block,np.r_[mu,1]);w=ans[:4];lam=ans[4];ref=np.ones(4)/4
    return mu,M,w,float(np.sqrt((w-ref)@S@(w-ref))),dict(V=V,lambda_budget=lam,kkt=float(max(abs(mu-4*V@w-lam)))*10000,budget=float(abs(w.sum()-1)),posterior_precision_residual=float(np.max(abs(M@(np.linalg.inv(T)+P.T@solve(O,P))-np.eye(4)))))

def bl_a(excess=True,ddof=1,predictive=True,view_sign=1,start='2018-01-01'):
    R,test,tr,te=bl_inputs(excess,start);S=np.cov(R,rowvar=False,ddof=ddof)*252;ref=np.ones(4)/4;pi=4*S@ref
    mu,M,w,tracking,d=bl_state(S,pi,predictive=predictive,view_sign=view_sign)
    def f(k):return bl_state(S,pi,k,predictive,view_sign)[3]-.5*tracking
    k,info=brentq(f,1,100,xtol=1e-12,rtol=1e-13,full_output=True)
    mus,Ms,ws,tracks,ds=bl_state(S,pi,k,predictive,view_sign);ret=test@ws
    out=dict(training_count=len(R),equilibrium_hitec_mean_pct=pi[0]*100,benchmark_volatility_pct=np.sqrt(ref@S@ref)*100,covariance_condition=np.linalg.cond(S),posterior_hitec_mean_pct=mu[0]*100,posterior_enrgy_mean_pct=mu[2]*100,posterior_hitec_mean_sd_pct=np.sqrt(M[0,0])*100,posterior_precision_residual=d['posterior_precision_residual'],hitec_weight_pct=w[0]*100,utils_weight_pct=w[1]*100,enrgy_weight_pct=w[2]*100,nodur_weight_pct=w[3]*100,tracking_error_pct=tracking*100,predictive_volatility_pct=np.sqrt(w@d['V']@w)*100,allocation_kkt_residual_bp=d['kkt'],view_variance_multiplier=k,recalibrated_enrgy_weight_pct=ws[2]*100,heldout_mean_excess_pct=ret.mean()*252*100,heldout_volatility_pct=ret.std(ddof=1)*np.sqrt(252)*100,risk_budget_residual_bp=abs(tracks-.5*tracking)*10000)
    return out,dict(S=S,pi=pi,mu=mu,M=M,w=w,adjusted_w=ws,baseline=d,adjusted=ds,root_converged=bool(info.converged),root_iterations=info.iterations,root_slope=(f(k+1e-4)-f(k-1e-4))/2e-4,root_endpoints=[f(1),f(100)],test_count=len(test),test=test,R=R)


from cave_agent import Variable
from core.validation import validate_ordered_outputs,turn_validator

TURN_1_NAMES = ['training_count', 'equilibrium_hitec_mean_pct', 'benchmark_volatility_pct', 'covariance_condition']

TURN_2_NAMES = ['posterior_hitec_mean_pct', 'posterior_enrgy_mean_pct', 'posterior_hitec_mean_sd_pct']

TURN_3_NAMES = ['hitec_weight_pct', 'utils_weight_pct', 'enrgy_weight_pct', 'nodur_weight_pct', 'tracking_error_pct', 'predictive_volatility_pct']

TURN_4_NAMES = ['view_variance_multiplier', 'recalibrated_enrgy_weight_pct', 'heldout_mean_excess_pct', 'heldout_volatility_pct']

variables = [
    Variable('training_count',None,'Store the common training count as an integer.'),
    Variable('equilibrium_hitec_mean_pct',None,'Store the implied HiTec annual excess mean in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('benchmark_volatility_pct',None,'Store the benchmark annual volatility in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('covariance_condition',None,'Store the covariance spectral condition number, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('posterior_hitec_mean_pct',None,'Store the posterior HiTec annual excess mean in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('posterior_enrgy_mean_pct',None,'Store the posterior Enrgy annual excess mean in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('posterior_hitec_mean_sd_pct',None,'Store the posterior standard deviation of HiTec annual mean in percentage points, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('hitec_weight_pct',None,'Store the signed HiTec allocation in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('utils_weight_pct',None,'Store the signed Utils allocation in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('enrgy_weight_pct',None,'Store the signed Enrgy allocation in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('nodur_weight_pct',None,'Store the signed NoDur allocation in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('tracking_error_pct',None,'Store the annual training-covariance tracking error in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('predictive_volatility_pct',None,'Store the posterior predictive portfolio volatility in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('view_variance_multiplier',None,'Store the common view-error variance multiplier as a dimensionless number, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('recalibrated_enrgy_weight_pct',None,'Store the revised signed Enrgy allocation in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('heldout_mean_excess_pct',None,'Store the annualized arithmetic held-out excess mean in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('heldout_volatility_pct',None,'Store the annualized held-out sample volatility in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    out=bl_a()[0]
    return tuple(out[v.name] for v in variables)

def _validate_subset(outputs,names):
    truth=dict(zip((v.name for v in variables),ground_truth())); by_name={v.name:v for v in variables};places=dict(zip(by_name,DECIMALS))
    result=validate_ordered_outputs(outputs,[by_name[n] for n in names],[truth[n] for n in names],[places[n] for n in names])
    for n in names:
        v=outputs.get(n)
        if 'residual' in n and isinstance(v,(int,float,np.number)) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v) and v<0:
            result.success=False;result.message='negative absolute invariant residual'
    return result

def validate_turn_1(outputs):return _validate_subset(outputs,TURN_1_NAMES)
def validate_turn_2(outputs):return _validate_subset(outputs,TURN_2_NAMES)
def validate_turn_3(outputs):return _validate_subset(outputs,TURN_3_NAMES)
def validate_turn_4(outputs):return _validate_subset(outputs,TURN_4_NAMES)
def validate(outputs):return _validate_subset(outputs,[v.name for v in variables])
validators={f'validate_turn_{i}':turn_validator(globals()[f'validate_turn_{i}']) for i in range(1,5)}
