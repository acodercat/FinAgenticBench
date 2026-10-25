"""Formula-free quantitative baseline; private invariants, one strict PV channel."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from core.data import load_ken_french_table, load_expansion_table

TOKENS = ["HiTec", "Enrgy", "Utils", "NoDur"]
CERTS = ["628", "3510", "7213"]

@lru_cache(None)
def entropy_inputs(start='2018-01-01', scope='four'):
    r=load_ken_french_table('daily_industry_returns')
    o=load_expansion_table('ofr_market_stress')
    cols=['hitec_pct','enrgy_pct','utils_pct','nodur_pct']
    d=r.merge(o[['date','ofr_fsi']],on='date',validate='one_to_one')
    d=d[d.date.between(start,'2022-12-31')].dropna(subset=cols+['ofr_fsi']).sort_values('date')
    R=d[cols].to_numpy(float)/100
    if scope=='equal':w=np.full(4,.25)
    else:w=np.array([.4,.2,.2,.2])
    return R,d.ofr_fsi.to_numpy(float),w,d

def tail_atoms(loss,p,alpha=.975):
    order=np.argsort(loss);cum=np.cumsum(p[order]);q=loss[order[np.searchsorted(cum,alpha)]]
    mass=np.where(loss>q,p,0.);ties=loss==q
    mass[ties]=p[ties]*((1-alpha)-mass.sum())/p[ties].sum()
    return q,mass

def entropy_a(start='2018-01-01',scope='four',confidence=1.,tail='fractional',view_sign=1):
    R,fsi,w,d=entropy_inputs(start,scope);x=R@w;n=len(x);q=np.quantile(fsi,.95,method='inverted_cdf');s=fsi>q
    def probs(t):
        z=t*x
        return np.where(s,.2*np.exp(z-logsumexp(z[s])),.8*np.exp(z-logsumexp(z[~s])))
    theta,info=brentq(lambda t:probs(t)@x-view_sign*.001,-1000,1000,xtol=1e-13,full_output=True)
    pp=probs(theta);p=confidence*pp+(1-confidence)/n
    var,mass=tail_atoms(-x,p)
    if tail=='whole_atom':mass=np.where(-x>=var,p,0.);mass*=.025/mass.sum()
    alloc=(-R*w).T@mass/.025;winner=min(range(4),key=lambda j:(-alloc[j],TOKENS[j]));a=R[:,winner]
    der=lambda l:float(p@(a/(1+l*a)))
    if np.min(1+4*a)<=0:raise ValueError('Kelly wealth domain')
    lev=brentq(der,0,4,xtol=1e-13) if der(0)>0>der(4) else (4. if der(4)>=0 else 0.)
    _,mm=tail_atoms(-lev*a,p)
    out=dict(scenario_count=n,prior_daily_volatility_pct=np.std(x,ddof=0)*100,stress_threshold=q,entropy_divergence=float(p@np.log(p*n)),effective_scenario_count=float(1/(p@p)),posterior_daily_volatility_pct=float(np.sqrt(p@((x-p@x)**2)))*100,posterior_var975_pct=var*100,posterior_es975_pct=float((-x)@mass/.025)*100,largest_tail_contributor=TOKENS[winner],kelly_leverage=lev,kelly_daily_log_growth_bp=float(p@np.log1p(lev*a))*10000,kelly_es975_pct=float((-lev*a)@mm/.025)*100)
    out.update({t.lower()+'_es_contribution_pct':float(v)*100 for t,v in zip(TOKENS,alloc)})
    return out,dict(R=R,fsi=fsi,w=w,x=x,p=p,stress=s,theta=theta,root_converged=bool(info.converged),root_iterations=info.iterations,moment_residual=max(abs(pp.sum()-1),abs(pp@s-.2),abs(pp@x-view_sign*.001)),mean_slope=float(sum(pp[g]@((x[g]-pp[g]@x[g]/pp[g].sum())**2)for g in [s,~s])),tail_mass_residual=abs(mass.sum()-.025),euler_residual=abs(alloc.sum()-(-x)@mass/.025),winner_margin=float(np.sort(alloc)[-1]-np.sort(alloc)[-2]),kelly_derivative=der(lev),kelly_curvature=float(-p@((a/(1+lev*a))**2)),minimum_wealth=float(min(1+lev*a)),stress_cutoff_ties=int(sum(fsi==q)),tail_cutoff_ties=int(sum(-x==var)),dates=[str(d.date.iloc[0]),str(d.date.iloc[-1])])

from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ['scenario_count', 'prior_daily_volatility_pct', 'stress_threshold']
TURN_2_NAMES = ['entropy_divergence', 'effective_scenario_count', 'posterior_daily_volatility_pct']
TURN_3_NAMES = ['posterior_var975_pct', 'posterior_es975_pct', 'hitec_es_contribution_pct', 'enrgy_es_contribution_pct', 'utils_es_contribution_pct', 'nodur_es_contribution_pct', 'largest_tail_contributor']
TURN_4_NAMES = ['kelly_leverage', 'kelly_daily_log_growth_bp', 'kelly_es975_pct']

variables = [
    Variable('scenario_count', None, 'Store common complete scenario count as an integer.'),
    Variable('prior_daily_volatility_pct', None, 'Store prior daily portfolio population volatility in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('stress_threshold', None, 'Store OFR FSI stress threshold in index units, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('entropy_divergence', None, 'Store posterior-relative-to-prior Kullback-Leibler divergence using natural logs, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('effective_scenario_count', None, 'Store posterior inverse-Herfindahl effective number of scenarios, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('posterior_daily_volatility_pct', None, 'Store posterior daily portfolio population volatility in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('posterior_var975_pct', None, 'Store posterior daily 97.5 percent VaR loss in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('posterior_es975_pct', None, 'Store posterior daily 97.5 percent expected shortfall loss in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('hitec_es_contribution_pct', None, 'Store HiTec signed component expected-shortfall contribution in portfolio-return percentage points, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('enrgy_es_contribution_pct', None, 'Store Enrgy signed component expected-shortfall contribution in portfolio-return percentage points, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('utils_es_contribution_pct', None, 'Store Utils signed component expected-shortfall contribution in portfolio-return percentage points, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('nodur_es_contribution_pct', None, 'Store NoDur signed component expected-shortfall contribution in portfolio-return percentage points, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('largest_tail_contributor', None, 'Store largest signed component contributor as exactly one token from HiTec, Enrgy, Utils, NoDur.'),
    Variable('kelly_leverage', None, 'Store optimal selected-industry notional per unit initial wealth, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('kelly_daily_log_growth_bp', None, 'Store posterior expected natural-log daily wealth growth in basis points, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('kelly_es975_pct', None, 'Store posterior daily 97.5 percent expected shortfall of the Kelly position in percent of initial wealth, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    out=entropy_a()[0]
    return tuple(out[v.name] for v in variables)

def _validate_subset(outputs,names):
    truth=dict(zip((v.name for v in variables),ground_truth()))
    by_name={v.name:v for v in variables}; places=dict(zip(by_name,DECIMALS))
    return validate_ordered_outputs(outputs,[by_name[n] for n in names],[truth[n] for n in names],[places[n] for n in names])

def validate_turn_1(outputs):return _validate_subset(outputs,TURN_1_NAMES)
def validate_turn_2(outputs):return _validate_subset(outputs,TURN_2_NAMES)
def validate_turn_3(outputs):return _validate_subset(outputs,TURN_3_NAMES)
def validate_turn_4(outputs):return _validate_subset(outputs,TURN_4_NAMES)
def validate(outputs):return _validate_subset(outputs,[v.name for v in variables])
validators={f'validate_turn_{i}':turn_validator(globals()[f'validate_turn_{i}'])for i in range(1,5)}
