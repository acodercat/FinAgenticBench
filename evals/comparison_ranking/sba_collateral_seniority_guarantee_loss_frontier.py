"""Hypothetical secured loss frontier over retained public SBA report rows."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [
    ["rows_504", "rows_7a", "normalized_junior_principal_mn"],
    ["deterministic_senior_loss_mn", "deterministic_junior_loss_mn", "deterministic_7a_shortfall_mn"],
    ["deterministic_7a_pv_loss_mn", "deterministic_lower_loss_book"],
    ["stochastic_504_mean_loss_mn", "stochastic_7a_mean_pv_loss_mn", "stochastic_lower_loss_book", "stochastic_504_worst_loss_mn"],
    ["maximum_common_recovery_drop_pp", "limiting_book", "boundary_7a_mean_pv_loss_mn"],
]
variables = [
    Variable("rows_504",None,"Store the eligible 504 reported-row count as an integer."),
    Variable("rows_7a",None,"Store the eligible 7(a) reported-row count as an integer."),
    Variable("normalized_junior_principal_mn",None,"Store normalized 504 junior principal in USD millions, rounded to 4 decimals."),
    Variable("deterministic_senior_loss_mn",None,"Store the 504 senior principal loss at deterministic recovery in USD millions, rounded to 4 decimals."),
    Variable("deterministic_junior_loss_mn",None,"Store the 504 junior principal loss at deterministic recovery in USD millions, rounded to 4 decimals."),
    Variable("deterministic_7a_shortfall_mn",None,"Store 7(a) collateral shortfall before guarantee in USD millions, rounded to 4 decimals."),
    Variable("deterministic_7a_pv_loss_mn",None,"Store bank 7(a) present-value principal loss in USD millions, rounded to 4 decimals."),
    Variable("deterministic_lower_loss_book",None,"Store the lower-loss bank book as the exact string SBA504 or SBA7A, with SBA504 breaking an exact tie."),
    Variable("stochastic_504_mean_loss_mn",None,"Store expected bank 504 principal loss under the two states in USD millions, rounded to 4 decimals."),
    Variable("stochastic_7a_mean_pv_loss_mn",None,"Store expected bank 7(a) present-value principal loss under the two states in USD millions, rounded to 4 decimals."),
    Variable("stochastic_lower_loss_book",None,"Store the lower expected-loss bank book as the exact string SBA504 or SBA7A, with SBA504 breaking an exact tie."),
    Variable("stochastic_504_worst_loss_mn",None,"Store worst-state bank 504 principal loss in USD millions, rounded to 4 decimals."),
    Variable("maximum_common_recovery_drop_pp",None,"Store maximum common recovery reduction in percentage points, rounded to 4 decimals."),
    Variable("limiting_book",None,"Store the binding bank book at the maximum reduction as the exact string SBA504, SBA7A, BOTH, or DOMAIN."),
    Variable("boundary_7a_mean_pv_loss_mn",None,"Store expected bank 7(a) present-value loss at the maximum reduction in USD millions, rounded to 4 decimals."),
]
DECIMALS = [0,0,4,4,4,4,4,None,4,4,None,4,4,None,4]

@lru_cache(maxsize=1)
def inputs():
    x=load_expansion_table("sba_504");y=load_expansion_table("sba_7a")
    def cohort(d):
        return d.loc[d.ApprovalFY.eq(2024)&d.ProjectState.astype(str).eq("CA")
            &d.FirstDisbursementDate.notna()&d.FirstDisbursementDate.astype(str).le("2024-12-31")
            &d.GrossApproval.gt(0)]
    x=cohort(x);x=x.loc[x.ThirdPartyDollars.gt(0)];y=cohort(y);y=y.loc[y.SBAGuaranteedApproval.gt(0)]
    if y.SBAGuaranteedApproval.gt(y.GrossApproval).any():raise ValueError("Guarantee exceeds principal")
    a=x.ThirdPartyDollars.to_numpy(float);b=x.GrossApproval.to_numpy(float);g=y.GrossApproval.to_numpy(float);q=y.SBAGuaranteedApproval.to_numpy(float)/g
    return x,y,a,b,g,q

def losses(recovery):
    _,_,a,b,g,q=inputs();cash=.95*recovery*(a+b)/.9
    senior=np.minimum(a,cash);junior=np.minimum(b,np.maximum(cash-a,0));short=np.maximum(g-.95*recovery*1.1*g,0)
    bank7=short-short*q/1.05**2
    return np.array([(a-senior).sum()*100/a.sum(),(b-junior).sum()*100/a.sum(),short.sum()*100/g.sum(),bank7.sum()*100/g.sum()])

def expected(drop):return (losses(.35-drop)+losses(.85-drop))/2

@lru_cache(maxsize=1)
def analysis():
    x,y,a,b,_,_=inputs();det=losses(.6);mean=expected(0)
    z=.35 if max(expected(.35)[[0,3]])<=35 else brentq(lambda z:max(expected(z)[[0,3]])-35,0,.35,xtol=1e-13)
    bound=expected(z);binding=["SBA504","SBA7A"]
    active=[binding[i] for i,k in enumerate([0,3]) if abs(bound[k]-35)<1e-8]
    return {"rows_504":len(x),"rows_7a":len(y),"normalized_junior_principal_mn":b.sum()*100/a.sum(),
        "deterministic_senior_loss_mn":det[0],"deterministic_junior_loss_mn":det[1],"deterministic_7a_shortfall_mn":det[2],
        "deterministic_7a_pv_loss_mn":det[3],"deterministic_lower_loss_book":"SBA504" if det[0]<=det[3] else "SBA7A",
        "stochastic_504_mean_loss_mn":mean[0],"stochastic_7a_mean_pv_loss_mn":mean[3],
        "stochastic_lower_loss_book":"SBA504" if mean[0]<=mean[3] else "SBA7A",
        "stochastic_504_worst_loss_mn":losses(.35)[0],"maximum_common_recovery_drop_pp":z*100,
        "limiting_book":"BOTH" if len(active)==2 else active[0] if active else "DOMAIN",
        "boundary_7a_mean_pv_loss_mn":bound[3]}

@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
