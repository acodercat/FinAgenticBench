"""Quantitative-hard baseline; exact named-model conventions are public.
Private dual methods and wrong-path evidence are archived in group3.
All state is unrounded; strict PV only. Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator

def ac_inputs(start='2024-01-01',scope='futures_only',ddof=1):
    t=load_expansion_table('treasury_yield_curve');t=t[t.Date.between(start,'2024-12-31')].sort_values('Date');assert t['10 Yr'].notna().all() and not t.Date.duplicated().any()
    delta=np.diff(t['10 Yr'].to_numpy(float));sigma=8*np.std(delta,ddof=ddof)/100
    c=load_expansion_table('cftc_cot');c=c[(c['Report_Date_as_YYYY-MM-DD']=='2024-12-31')&(c.CFTC_Contract_Market_Code=='043602')&(c.report_scope==scope)];assert len(c)==1
    X=float(c.Open_Interest_All.iloc[0])*.001*.1
    return len(delta),sigma,X

def ac_schedule(X,sigma,lam,N=10,eta=.00002,gamma=.000002,tau=.5,variant='discrete',method='linear'):
    a=eta/tau-gamma/2 if variant=='discrete' else eta/tau
    assert a>0 and lam>=0 and sigma>0
    diag=2*a+lam*sigma*sigma*tau;A=np.diag(np.full(N-1,diag))+np.diag(np.full(N-2,-a),1)+np.diag(np.full(N-2,-a),-1)
    rhs=np.zeros(N-1);rhs[0]=a*X
    if method=='linear':x=np.r_[X,np.linalg.solve(A,rhs),0.]
    else:
        k=np.arccosh(1+lam*sigma*sigma*tau/(2*a));x=X*np.sinh(k*np.arange(N,-1,-1))/np.sinh(k*N) if k else X*np.arange(N,-1,-1)/N
    n=-np.diff(x);cost=.5*gamma*X*X+.00005*X+a*sum(n*n);var=sigma*sigma*tau*sum(x[1:]**2)
    kkt=float(np.max(abs(2*(A@x[1:-1]-rhs))))
    return x,float(cost),float(var),kkt,float(np.linalg.eigvalsh(A)[0])

def ac_a(start='2024-01-01',scope='futures_only',ddof=1,variant='discrete'):
    n,sig,X=ac_inputs(start,scope,ddof);x,e,v,kkt,mine=ac_schedule(X,sig,1.,variant=variant)
    target=.75*np.sqrt(v)
    def budget(lam):return np.sqrt(ac_schedule(X,sig,lam,variant=variant)[2])-target
    lam=brentq(budget,1.,100.,xtol=1e-13,rtol=1e-14)
    z,ce,cv,ck,_=ac_schedule(X,sig,lam,variant=variant)
    rem=z[4];w,we,wv,wk,_=ac_schedule(rem,sig,lam,N=6,eta=.00008,variant=variant)
    out={'price_increment_count':n,'daily_price_sigma_bp':sig*10000,'initial_face_million':X,
         'initial_first_trade_million':float(x[0]-x[1]),'initial_expected_cost_million':e,'initial_cost_sd_million':np.sqrt(v),'initial_kkt_residual':kkt*10000,
         'frontier_risk_aversion_per_million':float(lam),'frontier_expected_cost_million':ce,'frontier_inventory_after_four_million':float(z[4]),
         'frontier_risk_budget_residual_million':float(abs(np.sqrt(cv)-target)),
         'revised_next_trade_million':float(w[0]-w[1]),'revised_remaining_cost_million':we,'revised_remaining_sd_million':np.sqrt(wv),
         'revised_kkt_residual':wk*10000}
    return out,{'minimum_hessian_half_eigenvalue':mine,'risk_root_bracket':[budget(1),budget(100)],'minimum_trade_million':float(min((-np.diff(x)).min(),(-np.diff(z)).min(),(-np.diff(w)).min())),'inventory_endpoint':float(w[-1]),'original_planned_next_trade':float(z[4]-z[5]),'risk_root_derivative':float((budget(lam+1e-4)-budget(lam-1e-4))/2e-4)}

TURN_1_NAMES = ['price_increment_count', 'daily_price_sigma_bp', 'initial_face_million']

TURN_2_NAMES = ['initial_first_trade_million', 'initial_expected_cost_million', 'initial_cost_sd_million']

TURN_3_NAMES = ['frontier_risk_aversion_per_million', 'frontier_expected_cost_million', 'frontier_inventory_after_four_million']

TURN_4_NAMES = ['revised_next_trade_million', 'revised_remaining_cost_million', 'revised_remaining_sd_million']

variables = [
    Variable('price_increment_count', None, 'Store the within-window yield-increment count as an integer.'),
    Variable('daily_price_sigma_bp', None, 'Store the daily price-increment standard deviation in basis points of par per square root trading day, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('initial_face_million', None, 'Store the hypothetical initial USD face amount in millions, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('initial_first_trade_million', None, 'Store the first optimal sale in USD million face, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('initial_expected_cost_million', None, 'Store the expected implementation shortfall in USD millions, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('initial_cost_sd_million', None, 'Store the implementation-shortfall standard deviation in USD millions, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('frontier_risk_aversion_per_million', None, 'Store the efficient-frontier variance-penalty multiplier per USD million, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('frontier_expected_cost_million', None, 'Store the efficient-frontier expected shortfall in USD millions, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('frontier_inventory_after_four_million', None, 'Store the inventory after four efficient-frontier executions in USD million face, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('revised_next_trade_million', None, 'Store the first revised sale in USD million face, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('revised_remaining_cost_million', None, 'Store the revised remaining expected implementation shortfall in USD millions, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('revised_remaining_sd_million', None, 'Store the revised remaining shortfall standard deviation in USD millions, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    result=ac_a()[0]
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
