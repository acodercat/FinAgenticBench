"""Quantitative-hard baseline; exact named-model conventions are public.
Private dual methods and wrong-path evidence are archived in group3.
All state is unrounded; strict PV only. Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.integrate import quad
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator

def sba_inputs(cutoff='2024-12-31',origin='disbursement'):
    d=load_expansion_table('sba_7a');d=d[d.ProjectState.eq('CA') & d.ApprovalFY.between(2020,2022)].copy()
    for c in ['FirstDisbursementDate','ApprovalDate','PaidInFullDate','ChargeOffDate']:d[c]=pd.to_datetime(d[c])
    d=d[d.FirstDisbursementDate.notna() & d.FirstDisbursementDate.le(cutoff)].copy()
    start=d.FirstDisbursementDate if origin=='disbursement' else d.ApprovalDate
    assert not ((d.PaidInFullDate<start)|(d.ChargeOffDate<start)).any()
    co=d.ChargeOffDate.fillna(pd.Timestamp.max);pa=d.PaidInFullDate.fillna(pd.Timestamp.max)
    end=pd.concat([co,pa,pd.Series(pd.Timestamp(cutoff),index=d.index)],axis=1).min(axis=1)
    event=np.where((co<=pa)&(co<=pd.Timestamp(cutoff)),1,np.where(pa<=pd.Timestamp(cutoff),2,0))
    age=(end-start).dt.days.to_numpy(int);assert min(age)>=0
    r=load_expansion_table('nyfed_reference_rates');r=r[(r['Effective Date']=='2024-12-31')&(r['Rate Type']=='SOFR')]
    assert len(r)==1
    return age,event,float(r['Rate (%)'].iloc[0])/100

def aj_path(age,event,multiplier=1.,km=False,ij=False):
    t=np.unique(age);p=np.array([1.,0.,0.]);cov=np.zeros((3,3));out=[];IF=np.zeros((len(age),3)) if ij else None
    for day in t:
        risk=age>=day;y=int(risk.sum());e1=(age==day)&(event==1);e2=(age==day)&(event==2)
        q=np.array([e1.sum()*multiplier,e2.sum()*(not km)])/y
        assert q.sum()<1
        T=np.eye(3);T[0]=[1-q.sum(),*q];prev=p.copy()
        if ij:
            dq=np.c_[(e1*multiplier-q[0]*risk)/y,(e2*(not km)-q[1]*risk)/y]
            dT=np.c_[-dq.sum(1),dq];IF=IF@T+prev[0]*dT
        cov=T.T@cov@T+prev[0]**2/y*(np.diag(T[0])-np.outer(T[0],T[0]))
        p=p@T;out.append((float(day),*p,float(cov[1,1]),float(np.sum(IF[:,1]**2)) if ij else 0.,*q))
    return np.asarray(out)

def aj_state(path,t):
    rows=path[path[:,0]<=t];return rows[-1,1:4] if len(rows) else np.array([1.,0.,0.])

def credit_cashflows(path,rate,start=730.,end=1460.,independent=False):
    s0=aj_state(path,start)[0];times=np.r_[start,path[(path[:,0]>start)&(path[:,0]<end),0],end]
    surv=np.array([aj_state(path,t)[0]/s0 for t in times[:-1]])
    rho=np.log1p(rate)/365
    if independent:
        ann=sum(s*quad(lambda z:np.exp(-rho*(z-start))/365,a,b,epsabs=1e-13)[0] for s,a,b in zip(surv,times[:-1],times[1:]))
    else:ann=float(np.sum(surv*(np.exp(-rho*(times[:-1]-start))-np.exp(-rho*(times[1:]-start)))/(rho*365)))
    old=np.r_[0,path[:-1,2]];inc=path[:,2]-old;m=(path[:,0]>start)&(path[:,0]<=end);u=(path[m,0]-start)/365
    mass=inc[m]/s0;weights=mass*np.exp(-np.log1p(rate)*u)
    loss=.6*weights.sum();spread=loss/ann;duration=float(weights@u/weights.sum())
    rm=float(surv@np.diff(times));return spread,ann,loss,duration,rm

def sba_a(cutoff='2024-12-31',origin='disbursement',variant='competing',discount='effective'):
    age,event,r=sba_inputs(cutoff,origin);path=aj_path(age,event,km=variant=='km');s2=aj_state(path,730);s4=aj_state(path,1460)
    rr=r if discount=='effective' else np.expm1(r)
    prem,ann,loss,dur,rm=credit_cashflows(path,rr)
    stress=aj_path(age,event,1.5,km=variant=='km');sp,an,lo,du,_=credit_cashflows(stress,rr)
    out={'cohort_count':len(age),'active_probability_day730':float(s2[0]),'chargeoff_probability_day730':float(s2[1]),
         'chargeoff_standard_error_day730':float(np.sqrt(path[path[:,0]<=730][-1,4])),
         'state_probability_residual':float(np.max(abs(path[:,1:4].sum(1)-1))),
         'conditional_chargeoff_probability':float((s4[1]-s2[1])/s2[0]),'conditional_paid_probability':float((s4[2]-s2[2])/s2[0]),
         'restricted_active_days':rm,'fair_protection_spread_bp':float(prem*10000),'premium_balance_residual_per100':float(abs(prem*ann-loss)*100),
         'default_payment_duration_years':dur,'stressed_spread_bp':float(sp*10000),'stressed_payment_duration_years':du,
         'stressed_conditional_chargeoff_probability':float((aj_state(stress,1460)[1]-aj_state(stress,730)[1])/aj_state(stress,730)[0])}
    return out,{'minimum_risk_set_to1460':int(np.sum(age>=1460)),'maximum_exit_increment':float(path[:,6:8].sum(1).max()),'annuity':ann,'cohort_event_counts':np.bincount(event).tolist(),'identical_rows_preserved':True,'all_state_covariances_finite':bool(np.isfinite(path).all())}

TURN_1_NAMES = ['cohort_count', 'active_probability_day730', 'chargeoff_probability_day730', 'chargeoff_standard_error_day730']

TURN_2_NAMES = ['conditional_chargeoff_probability', 'conditional_paid_probability', 'restricted_active_days']

TURN_3_NAMES = ['fair_protection_spread_bp', 'default_payment_duration_years']

TURN_4_NAMES = ['stressed_conditional_chargeoff_probability', 'stressed_spread_bp', 'stressed_payment_duration_years']

variables = [
    Variable('cohort_count', None, 'Store the included public-row count as an integer.'),
    Variable('active_probability_day730', None, 'Store the day-730 active-state probability as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('chargeoff_probability_day730', None, 'Store the day-730 charge-off cumulative incidence as a decimal probability, rounded to 6 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('chargeoff_standard_error_day730', None, 'Store the day-730 charge-off probability standard error as a dimensionless number, rounded to 7 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('conditional_chargeoff_probability', None, 'Store the conditional charge-off probability as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('conditional_paid_probability', None, 'Store the conditional paid-in-full probability as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('restricted_active_days', None, 'Store the conditional restricted mean active time in actual days, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('fair_protection_spread_bp', None, 'Store the fair annual protection premium in basis points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('default_payment_duration_years', None, 'Store the Macaulay duration of discounted default-benefit payments in Actual/365 years from inception, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('stressed_conditional_chargeoff_probability', None, 'Store the stressed conditional charge-off probability as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('stressed_spread_bp', None, 'Store the stressed fair annual protection premium in basis points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('stressed_payment_duration_years', None, 'Store the stressed default-payment Macaulay duration in Actual/365 years from inception, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
]
DECIMALS = [0, 4, 6, 7, 4, 4, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    result=sba_a()[0]
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
