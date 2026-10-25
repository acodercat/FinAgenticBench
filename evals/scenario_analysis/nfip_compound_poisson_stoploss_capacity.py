"""Hard baseline: compound-Poisson aggregate-loss and premium-constrained reinsurance capacity.

Alternative conventions are measured against the pinned reading.
Pinned variants are baseline sensitivity probes, not live traps.
Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
All downstream states use unrounded calculations; only reported outputs round.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator, finite_number

def severity(scope='total',lattice='upper',start='2019-01-01'):
    d=load_expansion_table('fema_nfip');d=d.loc[d.state_usps.eq('FL')&d.date_of_loss.between(start,'2022-12-31')]
    cols=['net_building_payment_usd','net_contents_payment_usd','net_icc_payment_usd'] if scope=='total' else ['net_building_payment_usd']
    v=d[cols].sum(axis=1).to_numpy(float);v=v[v>0];j=(np.ceil(v/10000) if lattice=='upper' else np.floor(v/10000+.5)).clip(0,50).astype(int)
    f=np.bincount(j,minlength=51)/len(j);return f,len(j)

def panjer(f,lam,limit=2000):
    if lam<=0 or not np.isfinite(lam):raise ValueError('positive finite intensity required')
    g=np.zeros(limit+1);g[0]=np.exp(lam*(f[0]-1));z=np.arange(1,len(f))*f[1:]
    for k in range(1,len(g)):
        m=min(k,len(z));g[k]=lam/k*np.dot(z[:m],g[k-np.arange(1,m+1)])
    return g

def insurance(engine=panjer,scope='total',lattice='upper',discount=True):
    f,n=severity(scope,lattice);z=np.arange(len(f));g=engine(f,3.);s=np.arange(len(g))*10.;mean=3*np.dot(z,f)*10
    cdf=np.cumsum(g);idx=int(np.searchsorted(cdf,.99));attachment=s[idx]
    es=attachment+np.dot(np.maximum(s-attachment,0),g)/.01
    row=load_expansion_table('treasury_yield_curve');rate=float(row.loc[row.Date.eq('2024-12-31'),'1 Yr'].iloc[0])/100;df=1/(1+rate) if discount else 1.
    ceded=np.dot(np.maximum(s-attachment,0),g);retained=np.dot(np.minimum(s,attachment),g)
    budget=2*ceded*df
    fn=lambda lam:float(np.dot(np.maximum(s-attachment,0),engine(f,lam))*df-budget)
    if engine==panjer:root=brentq(fn,3,10,xtol=1e-12,rtol=1e-14)
    else:
        a,b=3.,10.
        for _ in range(48):
            mid=(a+b)/2
            if fn(mid)>0:b=mid
            else:a=mid
        root=(a+b)/2
    gstar=engine(f,root)
    return dict(severity_record_count=n,severity_mean_thousand=float(np.dot(z,f)*10),severity_second_moment_million=float(np.dot(z*z,f)*100),aggregate_var99_thousand=float(attachment),aggregate_es99_thousand=float(es),aggregate_mean_thousand=float(mean),aggregate_mass_residual=float(abs(g.sum()-1)),stoploss_present_value_thousand=float(ceded*df),retained_mean_thousand=float(retained),loss_conservation_residual_thousand=float(abs(retained+ceded-mean)),capacity_frequency=float(root),capacity_exhaustion_probability=float(gstar[idx+1:].sum()),capacity_budget_residual_thousand=float(abs(fn(root)))),dict(f=f,g=g,gstar=gstar,attachment=attachment,df=df,budget=budget,quantile_lower=float(.99-cdf[idx-1]),quantile_upper=float(cdf[idx]-.99),slope=(fn(root+1e-4)-fn(root-1e-4))/2e-4)

TURN_1_NAMES = ['severity_record_count', 'severity_mean_thousand', 'severity_second_moment_million']

TURN_2_NAMES = ['aggregate_var99_thousand', 'aggregate_es99_thousand', 'aggregate_mean_thousand']

TURN_3_NAMES = ['stoploss_present_value_thousand', 'retained_mean_thousand']

TURN_4_NAMES = ['capacity_frequency', 'capacity_exhaustion_probability']

variables = [
    Variable('severity_record_count', None, 'Store the eligible physical claim-record count as an integer.'),
    Variable('severity_mean_thousand', None, 'Store the discrete severity mean in thousands of USD, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('severity_second_moment_million', None, 'Store the discrete second raw severity moment in units of 10^6 USD^2 (the USD^2 moment divided by 1,000,000), rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('aggregate_var99_thousand', None, 'Store the lower 99-percent aggregate loss quantile in thousands of USD, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('aggregate_es99_thousand', None, 'Store the integrated-quantile 99-percent expected shortfall in thousands of USD, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('aggregate_mean_thousand', None, 'Store the aggregate expected loss in thousands of USD, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('stoploss_present_value_thousand', None, 'Store the ceded expected-loss present value in thousands of USD, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('retained_mean_thousand', None, 'Store the undiscounted retained expected loss in thousands of USD, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('capacity_frequency', None, 'Store the capacity mean annual claim count as a positive dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('capacity_exhaustion_probability', None, 'Store the capacity probability of strictly exceeding the deductible as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    result=insurance()[0]
    return tuple(result[v.name] for v in variables)

def _validate_subset(outputs,names):
    truth=dict(zip((v.name for v in variables),ground_truth()))
    by_name={v.name:v for v in variables};places=dict(zip(by_name,DECIMALS))
    result=validate_ordered_outputs(outputs,[by_name[n] for n in names],[truth[n] for n in names],[places[n] for n in names])
    for n in names:
        if ('residual' in n or 'identity' in n) and finite_number(outputs.get(n)) and outputs[n] < 0:
            result.success=False
            result.message='absolute residual must be nonnegative'
            result.variable_results[n]['correct']=False
            result.variable_results[n]['message']=result.message
    return result
def validate_turn_1(outputs):return _validate_subset(outputs,TURN_1_NAMES)
def validate_turn_2(outputs):return _validate_subset(outputs,TURN_2_NAMES)
def validate_turn_3(outputs):return _validate_subset(outputs,TURN_3_NAMES)
def validate_turn_4(outputs):return _validate_subset(outputs,TURN_4_NAMES)
def validate(outputs):return _validate_subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1,5)}
