"""Advanced quantitative formula-free baseline with a single strict PV channel.
Pinned variants remain offline regression probes, not live traps.
Ground truth uses governed runtime; raw admission is recorded independently.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.integrate import quad
from core.data import load_expansion_table

@lru_cache(None)
def evt_inputs(mark=True,start='2010-01-01'):
    w=load_expansion_table('eia_bulk_wti_daily').sort_values('observation_date').copy()
    prices=pd.to_numeric(w.wti_spot_price_usd_per_barrel,errors='raise').to_numpy(float)
    cents=np.rint(prices*100)
    if not np.allclose(prices*100,cents,atol=1e-8,rtol=0):raise ValueError('non-cent quote')
    w['loss']=np.r_[np.nan,-np.diff(cents)/100]
    o=load_expansion_table('ofr_market_stress')[['date','ofr_fsi']]
    d=w.merge(o,left_on='observation_date',right_on='date',validate='one_to_one')
    d=d[d.date.between(start,'2024-12-31')].dropna(subset=['loss','ofr_fsi']).sort_values('date')
    z=np.maximum(d.loss.to_numpy(float),0)
    if mark:z=np.where(d.ofr_fsi.to_numpy(float)>0,z,0)
    return z,d

def gpd_sf(x,u,b,xi):
    if b<=0 or not np.isfinite([u,b,xi]).all():raise ValueError('invalid GPD domain')
    h=1+xi*(np.asarray(x)-u)/b
    if np.any(h<=0):raise ValueError('outside GPD support')
    return np.exp(-(np.asarray(x)-u)/b) if abs(xi)<1e-12 else np.exp(-np.log(h)/xi)

def evt_a(mark=True,estimator='lmom',run=3,annual=252,start='2010-01-01'):
    z,d=evt_inputs(mark,start);n=len(z);u=float(np.quantile(z,.95,method='inverted_cdf'));e=np.sort(z[z>u]-u);k=len(e)
    l1=e.mean();l2=np.dot(2*np.arange(1,k+1)-k-1,e)/(k*(k-1));xi=2-l1/l2;beta=l1*(1-xi)
    if estimator=='mom':xi=(1-l1*l1/e.var(ddof=1))/2;beta=l1*(1-xi)
    if not (0<xi<1 and beta>0):raise ValueError('finite-mean heavy-tail domain')
    p=k/n;sf=lambda x:p*gpd_sf(x,u,beta,xi)
    var=brentq(lambda x:sf(x)-.01,u,u+10000,xtol=1e-12)
    es=var+(beta+xi*(var-u))/(1-xi)
    shifted=beta+xi*(var-u);ix=np.flatnonzero(z>var);k2=len(ix);cl=1+int(np.sum(np.diff(ix)>run));theta=cl/k2;p2=k2/n
    fs=lambda x,th:np.exp(-th*annual*p2*gpd_sf(x,var,shifted,xi))
    q95=brentq(lambda x:fs(x,theta)-.95,var,var+10000,xtol=1e-11)
    q99=brentq(lambda x:fs(x,theta)-.99,var,var+10000,xtol=1e-11)
    survival=lambda x,th:-np.expm1(-th*annual*p2*gpd_sf(x,var,shifted,xi))
    layer,err=quad(lambda x:survival(x,theta),q95,q99,epsabs=1e-11,epsrel=1e-11)
    upper,err2=quad(lambda x:survival(x,theta),q99,np.inf,epsabs=1e-10,epsrel=1e-11)
    whole,err3=quad(lambda x:survival(x,theta),q95,np.inf,epsabs=1e-10,epsrel=1e-11)
    iid=quad(lambda x:survival(x,1),q95,q99,epsabs=1e-11,epsrel=1e-11)[0]
    stab=abs(gpd_sf(q99,u,beta,xi)/gpd_sf(var,u,beta,xi)-gpd_sf(q99,var,shifted,xi))
    out=dict(observation_count=n,threshold_exceedance_count=k,initial_threshold_usd=u,gpd_shape=xi,gpd_scale_usd=beta,lmoment_residual_usd=max(abs(beta/(1-xi)-l1),abs(beta/((1-xi)*(2-xi))-l2)),marginal_var99_usd=var,marginal_es99_usd=es,quantile_probability_residual=abs(sf(var)-.01),raised_exceedance_count=k2,runs_cluster_count=cl,extremal_index=theta,annual_max95_usd=q95,annual_max99_usd=q99,threshold_stability_residual=stab,maximum_layer_premium_usd=layer,upper_stoploss_usd=upper,iid_layer_premium_usd=iid,layer_additivity_residual_usd=abs(whole-layer-upper))
    return out,dict(l1=l1,l2=l2,beta_raised=shifted,arrival=p2,tail_probability=p,theta=theta,threshold_margin=float(np.min(abs(z-var))),quadrature_error=max(err,err2,err3),annual_cdf_residual=max(abs(fs(q95,theta)-.95),abs(fs(q99,theta)-.99)),maximum_loss=float(z.max()),dates=[str(d.date.iloc[0]),str(d.date.iloc[-1])])


from cave_agent import Variable
from core.validation import validate_ordered_outputs,turn_validator

TURN_1_NAMES = ['observation_count', 'threshold_exceedance_count', 'initial_threshold_usd', 'gpd_shape', 'gpd_scale_usd']

TURN_2_NAMES = ['marginal_var99_usd', 'marginal_es99_usd']

TURN_3_NAMES = ['raised_exceedance_count', 'extremal_index', 'annual_max95_usd', 'annual_max99_usd']

TURN_4_NAMES = ['maximum_layer_premium_usd', 'upper_stoploss_usd', 'iid_layer_premium_usd']

variables = [
    Variable('observation_count',None,'Store the common observation count as an integer.'),
    Variable('threshold_exceedance_count',None,'Store the initial strict exceedance count as an integer.'),
    Variable('initial_threshold_usd',None,'Store the initial marked-loss threshold in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('gpd_shape',None,'Store the generalized Pareto shape as a dimensionless number, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('gpd_scale_usd',None,'Store the initial excess scale in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('marginal_var99_usd',None,'Store the positive-loss marginal 99-percent VaR in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('marginal_es99_usd',None,'Store the marginal 99-percent expected shortfall in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('raised_exceedance_count',None,'Store the raised-threshold strict exceedance count as an integer.'),
    Variable('extremal_index',None,'Store the extremal index as a dimensionless number, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('annual_max95_usd',None,'Store the 252-observation maximum 95th percentile in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('annual_max99_usd',None,'Store the 252-observation maximum 99th percentile in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('maximum_layer_premium_usd',None,'Store the undiscounted expected maximum-loss layer payment in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('upper_stoploss_usd',None,'Store the unlimited expected excess above exhaustion in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('iid_layer_premium_usd',None,'Store the iid-extremal-index fixed-layer expectation in USD per barrel, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
]
DECIMALS = [0, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    out=evt_a()[0]
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
