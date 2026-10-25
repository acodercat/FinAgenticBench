"""Quantitative-hard formula-free baseline; alternatives are pinned-convention regressions.
Single strict PV contract. Full model states are retained unrounded.
Exact source-native raw admission is tracked separately from runtime validation.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import norm
from core.data import load_expansion_table

def leland_values(V,C,r,sigma,tax=.21,loss=.5,protected=False):
    if V<=0 or C<=0 or r<=0 or sigma<=0:raise ValueError('domain')
    x=2*r/sigma**2;b=(1-tax)*C/(r+sigma**2/2)
    if protected:b=C/r
    if b>=V:raise ValueError('initial default')
    z=(b/V)**x;D=C/r*(1-z)+(1-loss)*b*z;tb=tax*C/r*(1-z);bc=loss*b*z;firm=V+tb-bc;E=firm-D
    return dict(barrier=b,laplace=z,debt=D,equity=E,firm=firm,spread=(C/D-r)*10000,x=x,tax_benefit=tb,bankruptcy_cost=bc)

def passage(V,B,r,sigma,T=5):
    a=np.log(V/B);mu=r-sigma*sigma/2
    prob=norm.cdf((-a-mu*T)/(sigma*np.sqrt(T)))+np.exp(-2*mu*a/sigma**2)*norm.cdf((-a+mu*T)/(sigma*np.sqrt(T)))
    def density(t):return a/(sigma*np.sqrt(2*np.pi*t**3))*np.exp(-(a+mu*t)**2/(2*sigma*sigma*t)) if t>0 else 0
    z,err=quad(lambda t:np.exp(-r*t)*density(t),0,T,epsabs=1e-13,epsrel=1e-12)
    inf,erri=quad(lambda t:np.exp(-r*t)*density(t),0,np.inf,epsabs=1e-13,epsrel=1e-12)
    return float(prob),z,inf,max(err,erri)

def leland_a(tenor='30 Yr',tax=.21,variant='endogenous',probability='passage',loss=.5):
    d=load_expansion_table('treasury_yield_curve');v=d.loc[d.Date=='2024-12-31',tenor];assert len(v)==1;r=float(v.iloc[0])/100;V=100.;sigma=.25;target=60.
    protected=variant=='protected';k=(1-tax)/(r+sigma*sigma/2) if not protected else 1/r;x=2*r/sigma**2
    coef=(1/r-(1-loss)*k)*(k/V)**x
    cmax=(1/(r*(x+1)*coef))**(1/x);hi=min(cmax,V/k*(1-1e-12))
    C=brentq(lambda c:leland_values(V,c,r,sigma,tax,loss,protected)['debt']-target,1e-10,hi,xtol=1e-13,rtol=1e-14)
    a=leland_values(V,C,r,sigma,tax,loss,protected);p,z,inf,err=passage(V,a['barrier'],r,sigma)
    if probability=='terminal':p=float(norm.cdf((np.log(a['barrier']/V)-(r-sigma*sigma/2)*5)/(sigma*np.sqrt(5))))
    smooth=abs(1-a['x']*((1-tax)*C/r-a['barrier'])/a['barrier'])
    vs=V*.75;ss=sigma*1.5;b=leland_values(vs,C,r,ss,tax,loss,protected);pp=passage(vs,b['barrier'],r,ss)[0]
    if probability=='terminal':pp=float(norm.cdf((np.log(b['barrier']/vs)-(r-ss*ss/2)*5)/(ss*np.sqrt(5))))
    xx=2*r/ss**2;kk=(1-tax)/(r+ss*ss/2) if not protected else 1/r;g=(tax/r+loss*kk)*(kk/vs)**xx
    def score(c):return tax/r-(xx+1)*g*c**xx
    opt=brentq(score,1e-10,vs/kk*(1-1e-12),xtol=1e-13,rtol=1e-14);o=leland_values(vs,opt,r,ss,tax,loss,protected)
    out=dict(calibrated_coupon_per_year=C,debt_repricing_residual=abs(a['debt']-target),debt_capacity_coupon_per_year=hi,
      endogenous_barrier=a['barrier'],five_year_default_probability=p,five_year_recovery_pv=(1-loss)*a['barrier']*z,smooth_pasting_residual=smooth,
      stressed_barrier=b['barrier'],stressed_debt_value=b['debt'],stressed_equity_value=b['equity'],stressed_default_probability=pp,
      optimal_coupon_per_year=opt,optimal_market_leverage_pct=o['debt']/o['firm']*100,optimal_credit_spread_bp=o['spread'],optimality_residual=abs(score(opt)),recapitalization_value_gain=o['firm']-b['firm'])
    return out,dict(rate=r,base=a,stress=b,optimum=o,laplace_error=abs(inf-a['laplace']),quadrature_error=err,root_slope=1/r-(x+1)*coef*C**x,root_capacity=leland_values(V,hi,r,sigma,tax,loss,protected)['debt'],optimum_second_derivative=-xx*(xx+1)*g*opt**(xx-1),optimal_firm_value=o['firm'],current_stressed_firm_value=b['firm'])

from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ['calibrated_coupon_per_year', 'debt_capacity_coupon_per_year']

TURN_2_NAMES = ['endogenous_barrier', 'five_year_default_probability', 'five_year_recovery_pv']

TURN_3_NAMES = ['stressed_barrier', 'stressed_debt_value', 'stressed_equity_value', 'stressed_default_probability']

TURN_4_NAMES = ['optimal_coupon_per_year', 'optimal_market_leverage_pct', 'optimal_credit_spread_bp', 'recapitalization_value_gain']

variables = [
    Variable('calibrated_coupon_per_year', None, 'Store the calibrated annual coupon in value units per year, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('debt_capacity_coupon_per_year', None, 'Store the debt-value-maximizing annual coupon in value units per year, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('endogenous_barrier', None, 'Store the endogenous bankruptcy barrier in asset-value units, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('five_year_default_probability', None, 'Store the five-year first-bankruptcy probability as a decimal probability, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('five_year_recovery_pv', None, 'Store the five-year default-recovery present value in value units, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('stressed_barrier', None, 'Store the stressed bankruptcy barrier in asset-value units, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('stressed_debt_value', None, 'Store the stressed debt market value in value units, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('stressed_equity_value', None, 'Store the stressed equity market value in value units, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('stressed_default_probability', None, 'Store the stressed five-year first-bankruptcy probability as a decimal probability, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('optimal_coupon_per_year', None, 'Store the optimal annual coupon in value units per year, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('optimal_market_leverage_pct', None, 'Store the optimal debt-to-total-firm market leverage in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('optimal_credit_spread_bp', None, 'Store the optimal credit spread in basis points per year, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('recapitalization_value_gain', None, 'Store the optimal-minus-carried total firm-value gain in value units, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    out=leland_a()[0]
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
