"""Formula-free quantitative baseline; private invariants, one strict PV channel."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, linprog
from core.data import load_expansion_table

TOKENS = ["HiTec", "Enrgy", "Utils", "NoDur"]
CERTS = ["628", "3510", "7213"]

@lru_cache(None)
def network_inputs(date='2023-12-31',tenor='10 Yr',duration=6.,transpose=False):
    b=load_expansion_table('fdic_bankfind');rows=b[b.REPDTE.eq(date)&b.CERT.isin(CERTS)]
    if len(rows)!=3 or rows.CERT.duplicated().any():raise ValueError('bank key')
    b=rows.set_index('CERT').loc[CERTS];A=b.ASSET.to_numpy(float)/1e6;L=A-b.EQ.to_numpy(float)/1e6
    P=np.array([[0,.12,.08],[.09,0,.21],[.24,.16,0]])
    if transpose:P=P.T
    ext=A-P.T@L
    c=load_expansion_table('treasury_yield_curve').set_index('Date');change=(float(c.loc['2022-12-30',tenor])-float(c.loc['2021-12-31',tenor]))/100
    e=ext*(1-duration*change)
    if min(e)<=0 or max(P.sum(1))>=1:raise ValueError('external asset/network domain')
    return A,L,P,ext,e,b

def clearing_a(e,L,P):
    p=L.copy();seen=[]
    for _ in range(5):
        default=e+P.T@p<L-1e-10;idx=np.flatnonzero(default);solv=np.flatnonzero(~default);q=L.copy()
        if len(idx):q[idx]=np.linalg.solve(np.eye(len(idx))-P[np.ix_(idx,idx)].T,e[idx]+P[np.ix_(solv,idx)].T@L[solv])
        seen.append(default.tolist())
        if np.max(abs(q-np.minimum(L,e+P.T@q)))<1e-9:return q,seen
        p=q
    raise RuntimeError('fictitious-default convergence')

def allocation_lp(e,L,P,B):
    M=np.eye(3)-P.T
    U=np.vstack([np.c_[M,-np.eye(3),np.zeros(3)],np.c_[-np.eye(3),np.zeros((3,3)),L]])
    rhs=np.r_[e,np.zeros(3)];eq=np.array([[0,0,0,1,1,1,0.]])
    result=linprog(np.r_[np.zeros(6),-1.],A_ub=U,b_ub=rhs,A_eq=eq,b_eq=[B],bounds=[(0,float(z))for z in L]+[(0,None)]*3+[(0,1)],method='highs-ds',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError(result.message)
    return result,U,rhs,eq

def network_a(date='2023-12-31',tenor='10 Yr',duration=6.,transpose=False,scope='clearing'):
    A,L,P,ext,e,b=network_inputs(date,tenor,duration,transpose);p,steps=clearing_a(e,L,P)
    if scope=='first_round':p=np.minimum(L,e+P.T@L)
    B=.01*A.sum();lp,U,rhs,eq=allocation_lp(e,L,P,B);u=lp.x[3:6];floor=lp.x[6];funded,steps3=clearing_a(e+u,L,P)
    f=lambda h:min(clearing_a((1-h)*e+u,L,P)[0]/L)-.95*floor
    h,info=brentq(f,0,.5,xtol=1e-13,full_output=True);terminal,steps4=clearing_a((1-h)*e+u,L,P);binding=min(range(3),key=lambda j:(terminal[j]/L[j],CERTS[j]));outside=1-P.sum(1)
    out={f'external_assets_{c}_usd_bn':float(v)for c,v in zip(CERTS,ext)}
    out['nominal_outside_liabilities_usd_bn']=float(outside@L)
    out.update({f'initial_recovery_{c}_pct':float(v)*100 for c,v in zip(CERTS,p/L)})
    out['initial_outside_shortfall_usd_bn']=float(outside@(L-p))
    out.update({f'injection_{c}_usd_bn':float(v)for c,v in zip(CERTS,u)})
    out.update(optimal_minimum_recovery_pct=float(floor)*100,additional_haircut_pct=h*100,binding_certificate=CERTS[binding],terminal_outside_shortfall_usd_bn=float(outside@(L-terminal)))
    stationarity=lp.c if hasattr(lp,'c') else np.r_[np.zeros(6),-1.]
    dual=U.T@lp.ineqlin.marginals+eq.T@lp.eqlin.marginals+lp.lower.marginals+lp.upper.marginals
    return out,dict(A=A,L=L,P=P,ext=ext,e=e,p=p,u=u,floor=floor,terminal=terminal,funded=funded,default_steps=[steps,steps3,steps4],root_converged=bool(info.converged),root_residual=abs(f(h)),clearing_residual=max(abs(terminal-np.minimum(L,(1-h)*e+u+P.T@terminal))),balance_residual=max(abs(ext+P.T@L-A)),budget_residual=abs(u.sum()-B),lp_stationarity=max(abs(stationarity-dual)),lp_primal=max(0,float(max(U@lp.x-rhs))),lp_complementarity=max(abs(lp.ineqlin.residual*lp.ineqlin.marginals)),lp_dual=lp.ineqlin.marginals,lp_lower=lp.lower.marginals,contraction=max(P.sum(1)),condition=np.linalg.cond(np.eye(3)-P.T),terminal_recovery_margin=float(np.sort(terminal/L)[1]-np.sort(terminal/L)[0]))

from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ['external_assets_628_usd_bn', 'external_assets_3510_usd_bn', 'external_assets_7213_usd_bn', 'nominal_outside_liabilities_usd_bn']
TURN_2_NAMES = ['initial_recovery_628_pct', 'initial_recovery_3510_pct', 'initial_recovery_7213_pct', 'initial_outside_shortfall_usd_bn']
TURN_3_NAMES = ['injection_628_usd_bn', 'injection_3510_usd_bn', 'injection_7213_usd_bn', 'optimal_minimum_recovery_pct']
TURN_4_NAMES = ['additional_haircut_pct', 'binding_certificate', 'terminal_outside_shortfall_usd_bn']

variables = [
    Variable('external_assets_628_usd_bn', None, 'Store pre-stress external asset value of certificate 628 in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('external_assets_3510_usd_bn', None, 'Store pre-stress external asset value of certificate 3510 in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('external_assets_7213_usd_bn', None, 'Store pre-stress external asset value of certificate 7213 in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('nominal_outside_liabilities_usd_bn', None, 'Store total nominal liabilities owed to outside creditors in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('initial_recovery_628_pct', None, 'Store initial stressed total creditor repayment fraction of certificate 628 in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('initial_recovery_3510_pct', None, 'Store initial stressed total creditor repayment fraction of certificate 3510 in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('initial_recovery_7213_pct', None, 'Store initial stressed total creditor repayment fraction of certificate 7213 in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('initial_outside_shortfall_usd_bn', None, 'Store initial stressed aggregate outside-creditor nominal shortfall in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('injection_628_usd_bn', None, 'Store optimal cash grant to certificate 628 in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('injection_3510_usd_bn', None, 'Store optimal cash grant to certificate 3510 in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('injection_7213_usd_bn', None, 'Store optimal cash grant to certificate 7213 in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('optimal_minimum_recovery_pct', None, 'Store maximum achievable minimum bank creditor repayment fraction in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('additional_haircut_pct', None, 'Store largest additional proportional haircut of post-initial-stress external assets in percent, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
    Variable('binding_certificate', None, 'Store binding bank certificate as exactly one string from 628, 3510, 7213.'),
    Variable('terminal_outside_shortfall_usd_bn', None, 'Store aggregate outside-creditor nominal shortfall at the capacity boundary in USD billions, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None, 4]

@lru_cache(maxsize=1)
def ground_truth():
    out=network_a()[0]
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
