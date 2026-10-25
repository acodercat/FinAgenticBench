"""Hard baseline: dynamic CAPM filtering, predictive hedging and retrospective RTS smoothing.

Alternative conventions are measured against the pinned reading.
Pinned variants are baseline sensitivity probes, not live traps.
Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
All downstream states use unrounded calculations; only reported outputs round.
"""
from functools import lru_cache
import numpy as np
from cave_agent import Variable
from core.data import load_ken_french_table
from core.validation import validate_ordered_outputs, turn_validator, finite_number

def kalman_inputs(excess=True, start='2019-01-01'):
    a=load_ken_french_table('daily_industry_returns')[['date','hitec_pct']]
    b=load_ken_french_table('daily_factors')[['date','mkt_rf_pct','rf_pct']]
    p=a.merge(b,on='date',validate='one_to_one').sort_values('date')
    p['y']=p.hitec_pct-(p.rf_pct if excess else 0)
    train=p.loc[p.date.between(start,'2021-12-31')]
    later=p.loc[p.date.between('2022-01-01','2023-01-31')]
    return train,later

def kalman_a(excess=True,q=0.0001,start='2019-01-01',hedge='predictive'):
    tr,d=kalman_inputs(excess,start);x=tr.mkt_rf_pct.to_numpy(float);y=tr.y.to_numpy(float)
    design=np.column_stack([np.ones(len(x)),x]);alpha,beta=np.linalg.lstsq(design,y,rcond=None)[0]
    e=y-design@np.array([alpha,beta]);r=e@e/(len(x)-2);p0=r/np.sum((x-x.mean())**2)
    m=beta;p=p0;ms=[];ps=[];pri=[];ivs=[];ll=[]
    for xx,yy in zip(d.mkt_rf_pct.astype(float),d.y.astype(float)):
        v=p+q;err=yy-alpha-xx*m;s=r+xx*xx*v;gain=v*xx/s
        pri.append(m);ivs.append(err/np.sqrt(s));ll.append(-.5*(np.log(2*np.pi*s)+err*err/s))
        m=m+gain*err;p=(1-gain*xx)**2*v+gain*gain*r;ms.append(m);ps.append(p)
    ms=np.array(ms);ps=np.array(ps);prior=np.array(pri);idx=int(np.sum(d.date<='2022-12-31'))-1
    h=d.date.to_numpy()>'2022-12-31';hedged=d.y.to_numpy(float)[h]-(prior if hedge=='predictive' else ms)[h]*d.mkt_rf_pct.to_numpy(float)[h]
    smooth=ms.copy();sv=ps.copy()
    for i in range(len(ms)-2,-1,-1):
        g=ps[i]/(ps[i]+q);smooth[i]+=g*(smooth[i+1]-ms[i]);sv[i]+=g*g*(sv[i+1]-ps[i]-q)
    out=dict(calibration_count=len(tr),initial_alpha_pp=float(alpha),initial_beta=float(beta),observation_variance_pp2=float(r),initial_beta_variance=float(p0),filtered_beta=float(ms[idx]),filtered_beta_variance=float(ps[idx]),filter_log_likelihood=float(sum(ll[:idx+1])),holdout_count=int(h.sum()),hedged_mean_pp=float(hedged.mean()),hedged_volatility_pct=float(hedged.std(ddof=1)*np.sqrt(252)),holdout_terminal_beta=float(ms[-1]),smoothed_yearend_beta=float(smooth[idx]),smoothed_yearend_variance=float(sv[idx]),smoother_terminal_identity=float(max(abs(smooth[-1]-ms[-1]),abs(sv[-1]-ps[-1]))))
    return out,dict(ms=ms,ps=ps,smooth=smooth,sv=sv,p0=p0,r=r,alpha=alpha,beta=beta,q=q,d=d)

TURN_1_NAMES = ['calibration_count', 'initial_alpha_pp', 'initial_beta', 'observation_variance_pp2', 'initial_beta_variance']

TURN_2_NAMES = ['filtered_beta', 'filtered_beta_variance', 'filter_log_likelihood']

TURN_3_NAMES = ['holdout_count', 'hedged_mean_pp', 'hedged_volatility_pct', 'holdout_terminal_beta']

TURN_4_NAMES = ['smoothed_yearend_beta', 'smoothed_yearend_variance']

variables = [
    Variable('calibration_count', None, 'Store the calibration observation count as an integer.'),
    Variable('initial_alpha_pp', None, 'Store the fitted daily intercept in percentage points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('initial_beta', None, 'Store the fitted market beta as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('observation_variance_pp2', None, 'Store the fitted observation variance in squared percentage points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('initial_beta_variance', None, 'Store the initial beta variance in squared beta units, rounded to 7 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('filtered_beta', None, 'Store the final filtered market beta as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('filtered_beta_variance', None, 'Store the final filtered beta variance in squared beta units, rounded to 6 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('filter_log_likelihood', None, 'Store the total innovation log likelihood using percentage-point observations, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('holdout_count', None, 'Store the held-out observation count as an integer.'),
    Variable('hedged_mean_pp', None, 'Store the mean daily hedged excess return in percentage points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('hedged_volatility_pct', None, 'Store the annualized hedged sample volatility in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('holdout_terminal_beta', None, 'Store the last held-out filtered beta as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('smoothed_yearend_beta', None, 'Store the smoothed last-2022 market beta as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('smoothed_yearend_variance', None, 'Store the smoothed last-2022 beta variance in squared beta units, rounded to 6 decimals. Compute from the unrounded carried model or source inputs.'),
]
DECIMALS = [0, 4, 4, 4, 7, 4, 6, 4, 0, 4, 4, 4, 4, 6]

@lru_cache(maxsize=1)
def ground_truth():
    result=kalman_a()[0]
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
