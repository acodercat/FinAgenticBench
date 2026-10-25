"""Quantitative-hard baseline with independently measured model-variant probes.
The exact model conventions are pinned; alternatives are sensitivity tests, not live traps.
All states use unrounded data. Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator

IND=['nodur', 'durbl', 'manuf', 'enrgy', 'hitec', 'telcm', 'shops', 'hlth', 'utils', 'other']

def rate_inputs(start='2021-01-01',tenor='5 Yr'):
    d=load_expansion_table('nyfed_reference_rates');d=d[(d['Rate Type']=='SOFR') & d['Effective Date'].between(start,'2024-12-31')].sort_values('Effective Date')
    assert not d['Effective Date'].duplicated().any() and d['Rate (%)'].notna().all()
    rates=d['Rate (%)'].to_numpy(float)/100
    t=load_expansion_table('treasury_yield_curve');v=t.loc[t.Date=='2024-12-31',tenor]
    assert len(v)==1
    return rates,float(v.iloc[0])/100

def vasicek_a(start='2021-01-01',variant='exact',tenor='5 Yr',pricing='calibrated'):
    rates,coupon=rate_inputs(start,tenor);x=rates[:-1];y=rates[1:];n=len(x)
    coef=np.linalg.lstsq(np.c_[np.ones(n),x],y,rcond=None)[0];a,b=coef;e=y-a-b*x;v=float(e@e/n)
    if not 0<b<1:raise ValueError('mean reversion domain')
    k=-252*np.log(b) if variant=='exact' else 252*(1-b)
    theta=a/(1-b);sig=np.sqrt(v*2*k/(1-b*b)) if variant=='exact' else np.sqrt(v*252)
    r0=rates[-1]
    def B(t):return -np.expm1(-k*np.asarray(t))/k
    def price(t,r,th):
        bt=B(t);return np.exp((th-sig*sig/(2*k*k))*(bt-t)-sig*sig*bt*bt/(4*k)-bt*r)
    times=np.arange(.5,5.01,.5);cash=np.full(10,coupon*50);cash[-1]+=100
    tq=brentq(lambda th:cash@price(times,r0,th)-100,-.5,.5,xtol=1e-14,rtol=1e-14)
    if pricing=='physical':tq=theta
    # Contract: deferred coupons at1.5,2,...5, USD1.5 each plus100 principal.
    pay=np.arange(1.5,5.01,.5);cf=np.full(8,1.5);cf[-1]+=100;expiry=1.;strike=100.
    star=brentq(lambda r:cf@price(pay-expiry,r,tq)-strike,-.5,.5,xtol=1e-14,rtol=1e-14)
    ks=price(pay-1,star,tq);p0=price(pay,r0,tq);p1=float(price(1,r0,tq));vol=sig*B(pay-1)*np.sqrt(-np.expm1(-2*k)/(2*k))
    h=np.log(p0/(ks*p1))/vol+vol/2
    call=float(cf@(p0*norm.cdf(h)-ks*p1*norm.cdf(h-vol)))
    put=float(cf@(ks*p1*norm.cdf(-h+vol)-p0*norm.cdf(-h)))
    var=sig*sig*(-np.expm1(-2*k))/(2*k);sd=np.sqrt(var)
    mp=theta+(r0-theta)*np.exp(-k);mq=tq+(r0-tq)*np.exp(-k)
    cov_ir=sig*sig/(2*k*k)*(1-np.exp(-k))**2;mf=mq-cov_ir
    pq=float(norm.cdf((star-mq)/sd));pp=float(norm.cdf((star-mp)/sd))
    dig=float(100*p1*norm.cdf((star-mf)/sd));conditional=mq-sd*norm.pdf((star-mq)/sd)/pq
    out={'transition_count':n,'physical_kappa_per_year':float(k),'physical_theta_pct':float(theta*100),'diffusion_pct_per_sqrt_year':float(sig*100),
         'risk_neutral_theta_pct':float(tq*100),'five_year_discount_factor':float(price(5,r0,tq)),
         'par_repricing_residual_per100':float(abs(cash@price(times,r0,tq)-100)),
         'critical_short_rate_pct':float(star*100),'coupon_bond_call_per100':call,'coupon_bond_put_per100':put,
         'put_call_parity_residual_per100':float(abs(call-put-(cf@p0-strike*p1))),
         'physical_exercise_probability':pp,'risk_neutral_exercise_probability':pq,'exercise_digital_value_per100':dig,
         'conditional_exercise_short_rate_pct':float(conditional*100)}
    diag={'ar_coefficient':float(b),'physical_design_condition':float(np.linalg.cond(np.c_[np.ones(n),x])),
          'critical_rate_slope':float(-cf@(B(pay-1)*ks)),'critical_rate_strike_residual':float(abs(cf@ks-100)),
          'theta_slope':float(cash@(price(times,r0,tq)*(B(times)-times))), 'r0':float(r0),'coupon':float(coupon)}
    return out,diag

TURN_1_NAMES = ['transition_count', 'physical_kappa_per_year', 'physical_theta_pct', 'diffusion_pct_per_sqrt_year']

TURN_2_NAMES = ['risk_neutral_theta_pct', 'five_year_discount_factor']

TURN_3_NAMES = ['critical_short_rate_pct', 'coupon_bond_call_per100', 'coupon_bond_put_per100']

TURN_4_NAMES = ['physical_exercise_probability', 'risk_neutral_exercise_probability', 'exercise_digital_value_per100', 'conditional_exercise_short_rate_pct']

variables = [
    Variable('transition_count', None, 'Store the fitted transition count as an integer.'),
    Variable('physical_kappa_per_year', None, 'Store the physical mean-reversion speed per model-year, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('physical_theta_pct', None, 'Store the physical long-run short rate in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('diffusion_pct_per_sqrt_year', None, 'Store the diffusion volatility in percentage points per square root model-year, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('risk_neutral_theta_pct', None, 'Store the fitted risk-neutral long-run short rate in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('five_year_discount_factor', None, 'Store the model five-year discount factor as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('critical_short_rate_pct', None, 'Store the common critical expiry short rate in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('coupon_bond_call_per100', None, 'Store the European call value per 100 principal, rounded to 6 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('coupon_bond_put_per100', None, 'Store the European put value per 100 principal, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('physical_exercise_probability', None, 'Store the physical call exercise probability as a decimal probability, rounded to 7 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('risk_neutral_exercise_probability', None, 'Store the risk-neutral call exercise probability as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('exercise_digital_value_per100', None, 'Store the exercise-contingent cash claim present value per 100 payoff, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('conditional_exercise_short_rate_pct', None, 'Store the risk-neutral conditional mean exercise short rate in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 6, 4, 7, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    result=vasicek_a()[0]
    return tuple(result[v.name] for v in variables)

def _validate_subset(outputs,names):
    truth=dict(zip((v.name for v in variables),ground_truth()))
    by_name={v.name:v for v in variables}; places=dict(zip(by_name,DECIMALS))
    result=validate_ordered_outputs(outputs,[by_name[n] for n in names],[truth[n] for n in names],[places[n] for n in names])
    for name in names:
        value=outputs.get(name)
        if ('residual' in name and name not in ('residual_covariance_condition',)) or name=='regression_orthogonality':
            if isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value) and value<0:
                result.success=False; result.message='negative absolute invariant residual'
                if result.variable_results and name in result.variable_results:result.variable_results[name].update(correct=False,message='negative absolute invariant residual')
    return result
def validate_turn_1(outputs):return _validate_subset(outputs,TURN_1_NAMES)
def validate_turn_2(outputs):return _validate_subset(outputs,TURN_2_NAMES)
def validate_turn_3(outputs):return _validate_subset(outputs,TURN_3_NAMES)
def validate_turn_4(outputs):return _validate_subset(outputs,TURN_4_NAMES)
def validate(outputs):return _validate_subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1,5)}
