"""Sharp information bounds and nested least-cost liquidation."""
from functools import lru_cache
import itertools
import numpy as np
from scipy.optimize import brentq
from core.dataset_layers import load_runtime_table
RESET=np.array([1,7,30,90,180,1,1,1.]);FINAL=np.array([1,7,30,90,180,90,180,365.]);SPREAD=np.array([2,5,12,35,65,18,28,45])/10000

@lru_cache(maxsize=1)
def inputs():
 f=load_runtime_table('sec_nmfp.series_filings');f=f[f.report_date.astype(str).eq('2025-06-30')&f.series_id.astype(str).eq('S000001111')].sort_values(['filing_date','accession']);r=f.iloc[-1]
 y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2024-12-31')].iloc[0];nodes=np.array([1/12,2/12,3/12,.5,1,2,3,5,7,10,20,30]);rates=np.array([y[c] for c in ['1 Mo','2 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']],float)/100
 t=RESET/365;zero=np.interp(t,nodes,rates);q=((1+(zero+.01)/2)/(1+zero/2))**(-2*t)
 return np.array([1,float(r.weighted_average_maturity_days),float(r.weighted_average_life_days)]),float(r.portfolio_securities_usd)/1e6,float(r.cash_usd)/1e6,q

@lru_cache(maxsize=2)
def vertices(capped=False):
 b,B,cash,q=inputs();ix=[0,1,2,3,5,6] if capped else list(range(8));A=np.vstack([np.ones(8),RESET,FINAL]);E=A[:,ix];result=[]
 for k in [3,4] if capped else [3]:
  mat=np.vstack([E,[float(j>=5) for j in ix]]) if k==4 else E;rhs=np.r_[b,.2] if k==4 else b
  for chosen in itertools.combinations(range(len(ix)),k):
   try:sol=np.linalg.solve(mat[:,chosen],rhs)
   except np.linalg.LinAlgError:continue
   if min(sol)<-1e-9:continue
   w=np.zeros(8);w[[ix[j] for j in chosen]]=np.maximum(sol,0)
   if capped and sum(w[5:])>.2+1e-9:continue
   if np.max(abs(A@w-b))>1e-8:raise ValueError('Invalid moment witness')
   if not any(np.max(abs(w-old))<1e-9 for old in result):result.append(w)
 if not result:raise ValueError('Empty compatible set')
 return np.array(result)

def liquidation(w,R):
 b,B,cash,q=inputs();need=max(0,R*(B+cash)-cash);loss=0.;sales=np.zeros(8)
 # Fractional knapsack: cost per net executable dollar, not quoted mark discount.
 for i in np.argsort(SPREAD/(1-SPREAD),kind='stable'):
  net=q[i]*(1-SPREAD[i]);amount=min(w[i]*B,need/net);sales[i]=amount;need-=amount*net;loss+=amount*q[i]*SPREAD[i]
  if need<=1e-10:break
 if need>1e-8:return float('inf'),float('-inf'),sales
 nav=(cash+B*w@q-R*(B+cash)-loss)/((1-R)*(B+cash)/100)
 return float(loss),float(nav),sales

@lru_cache(maxsize=1)
def analysis():
 b,B,cash,q=inputs();pool=B+cash;v=vertices();vc=vertices(True);mature=(FINAL<=7).astype(float);base=[liquidation(w,.7) for w in v];cap=[liquidation(w,.7) for w in vc];budget=.0009*pool
 worst=lambda R:max(liquidation(w,R)[0] for w in vc)
 if worst(.95)<=budget:capacity=.95;binding='DOMAIN'
 else:capacity=brentq(lambda R:worst(R)-budget,0,.95,xtol=1e-12);binding='COST'
 return {'compatible_set':'FEASIBLE','opening_pool_mn':pool,'liquidity_lower_pct':min(v@mature)*100,'liquidity_upper_pct':max(v@mature)*100,'worst_70pct_liquidation_cost_mn':max(x[0] for x in base),'worst_70pct_remaining_nav':min(x[1] for x in base),'capped_liquidity_upper_pct':max(vc@mature)*100,'capped_70pct_cost_mn':max(x[0] for x in cap),'budget_guarantee':'YES' if max(x[0] for x in cap)<=budget else 'NO','robust_redemption_capacity_pct':capacity*100,'capacity_binding':binding}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['opening_pool_mn', 'compatible_set'], ['liquidity_lower_pct', 'liquidity_upper_pct'], ['worst_70pct_liquidation_cost_mn', 'worst_70pct_remaining_nav'], ['capped_liquidity_upper_pct', 'capped_70pct_cost_mn', 'budget_guarantee'], ['robust_redemption_capacity_pct', 'capacity_binding']]
variables = [
    Variable('opening_pool_mn',None,'Store the hypothetical opening cash-plus-securities pool in USD millions, rounded to 4 decimals.'),
    Variable('compatible_set',None,'Store the compatible-holdings-set existence result as exactly FEASIBLE or INFEASIBLE.'),
    Variable('liquidity_lower_pct',None,'Store the sharp lower seven-day final-maturity opening-value share as a percentage, rounded to 4 decimals.'),
    Variable('liquidity_upper_pct',None,'Store the sharp upper seven-day final-maturity opening-value share as a percentage, rounded to 4 decimals.'),
    Variable('worst_70pct_liquidation_cost_mn',None,'Store the maximum optimized execution cost across the original compatible set in USD millions, rounded to 4 decimals.'),
    Variable('worst_70pct_remaining_nav',None,'Store the minimum post-redemption remaining-unit NAV across the original compatible set in USD per unit, rounded to 4 decimals.'),
    Variable('capped_liquidity_upper_pct',None,'Store the sharp upper seven-day final-maturity opening-value share under the additional mandate as a percentage, rounded to 4 decimals.'),
    Variable('capped_70pct_cost_mn',None,'Store maximum optimized execution cost under the additional mandate in USD millions, rounded to 4 decimals.'),
    Variable('budget_guarantee',None,'Store whether every compatible mandated book satisfies the stated execution-cost budget as exactly YES or NO.'),
    Variable('robust_redemption_capacity_pct',None,'Store maximum robust redemption capacity as a percentage of opening units, rounded to 4 decimals.'),
    Variable('capacity_binding',None,'Store the limiting capacity condition as exactly COST, PROCEEDS, DOMAIN, or MULTIPLE.'),
]
DECIMALS = [4, None, 4, 4, 4, 4, 4, 4, None, 4, None]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
