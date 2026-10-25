"""Legal-netting exposures and sharp finite-marginal default-loss bounds.

Baseline with pinned hypothetical legal sets, collateral and marginals.
Whole-book cross-netting gives independent CVA0.1957245m rather than
0.2087624m; averaging signed exposures before positive-part and collateral
gives0.0581008m. Fixed marginals permit sharp upper CVA0.6345785m, not the
independence value. At the robust collateral boundary3.5234063m per set,
independent CVA0.0291473m differs from the worst-case0.10m budget.
The25% default marginal requires fractional mass across exposure states,
unlike a single highest-state allocation. Independent LP/finite-vertex
certificates establish sharpness and feasible default/nondefault laws.
Four-decimal tolerance0.00006m excludes the measured wrong conventions;
publicly excluded alternatives are regression probes, not a live trap.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
    b=load_runtime_table('bis.otc_derivatives')
    b=b[b['TIME_PERIOD:Time period or range'].astype(str).eq('2024-S2')&b['DER_INSTR:Instrument'].astype(str).str.startswith('A:')&b['DER_SECTOR_CPY:Counterparty sector'].astype(str).str.startswith('A:')&b['DER_BASIS:Basis'].astype(str).str.startswith('C:')]
    notionals=[];values=[]
    for category in ['D: Interest rate','B: Foreign exchange','E: Equity']:
        z=b[b['DER_RISK:Risk category'].astype(str).eq(category)]
        n=z[z['DER_TYPE:Measure'].astype(str).str.startswith('A:')];m=z[z['DER_TYPE:Measure'].astype(str).str.startswith('D:')]
        if len(n)!=1 or len(m)!=1:raise ValueError('Nonunique BIS template')
        notionals.append(float(n['OBS_VALUE:Observation Value'].iloc[0]));values.append(float(m['OBS_VALUE:Observation Value'].iloc[0]))
    scale=100*np.array(values)/sum(notionals)
    y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2024-12-31')]
    if len(y)!=1:raise ValueError('Nonunique closeout curve')
    discount=(1+float(y.iloc[0]['1 Yr'])/200)**-2
    legs=np.array([[-3,1,-2],[-1,2,1],[0,-1,2],[1,-2,-1],[2,1,-3],[4,-3,4]])*scale
    netting=np.c_[legs[:,0]+legs[:,1],legs[:,2]]
    return scale,netting,discount

def exposures(netting,margin):return np.maximum(netting-np.array([.2,.1])-margin,0).sum(axis=1)

def extremal_loss(exposure,discount,maximum):
    total=0.;remaining=.25
    for value in sorted(exposure,reverse=maximum):
        mass=min(remaining,1/6);total+=mass*value;remaining-=mass
        if remaining<=1e-15:break
    return .6*discount*total

@lru_cache(maxsize=1)
def analysis():
    scale,netting,discount=inputs();base=exposures(netting,0)
    margin=brentq(lambda x:extremal_loss(exposures(netting,x),discount,True)-.10,0,5,xtol=1e-12)
    return {'interest_mark_scale_mn':scale[0],'fx_mark_scale_mn':scale[1],'equity_mark_scale_mn':scale[2],'state5_exposure_mn':base[4],'state6_exposure_mn':base[5],'independent_cva_mn':.6*discount*.25*float(np.mean(base)),'sharp_cva_lower_mn':extremal_loss(base,discount,False),'sharp_cva_upper_mn':extremal_loss(base,discount,True),'minimum_extra_collateral_each_mn':margin,'boundary_independent_cva_mn':.6*discount*.25*float(np.mean(exposures(netting,margin)))}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['interest_mark_scale_mn', 'fx_mark_scale_mn', 'equity_mark_scale_mn'], ['state5_exposure_mn', 'state6_exposure_mn'], ['independent_cva_mn'], ['sharp_cva_lower_mn', 'sharp_cva_upper_mn'], ['minimum_extra_collateral_each_mn', 'boundary_independent_cva_mn']]
variables = [
    Variable('interest_mark_scale_mn',None,'Store interest-rate reference mark scale in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('fx_mark_scale_mn',None,'Store foreign-exchange reference mark scale in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('equity_mark_scale_mn',None,'Store equity reference mark scale in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('state5_exposure_mn',None,'Store total collateral-adjusted positive closeout exposure in state 5 in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('state6_exposure_mn',None,'Store total collateral-adjusted positive closeout exposure in state 6 in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('independent_cva_mn',None,'Store independent-market-default CVA in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('sharp_cva_lower_mn',None,'Store sharp lower CVA bound in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('sharp_cva_upper_mn',None,'Store sharp upper CVA bound in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('minimum_extra_collateral_each_mn',None,'Store minimum extra cash collateral per netting set in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('boundary_independent_cva_mn',None,'Store independent-market-default CVA at the collateral boundary in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
