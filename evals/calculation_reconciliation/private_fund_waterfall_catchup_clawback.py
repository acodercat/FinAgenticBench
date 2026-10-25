"""Hypothetical private-fund contractual distribution and protection rights."""
from functools import lru_cache
import numpy as np
import pandas as pd
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
 f=load_runtime_table('sec_form_d.filings');r=f[f.ACCESSIONNUMBER.astype(str).eq('0000895527-24-000002')]
 if len(r)!=1:raise ValueError('Nonunique original offering')
 r=r.iloc[0];capital=float(r.TOTALAMOUNTSOLD)/1e6;fee=float(r.SALESCOMM_DOLLARAMOUNT+r.FINDERSFEE_DOLLARAMOUNT)/1e6
 y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2024-12-31')].iloc[0]
 times=np.array([(pd.Timestamp(x)-pd.Timestamp('2025-01-01')).days/365 for x in ['2027-01-01','2028-01-01','2030-01-01']])
 nodes=np.array([1/12,2/12,3/12,.5,1,2,3,5,7,10,20,30]);rates=np.array([y[c] for c in ['1 Mo','2 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']],float)/100
 return capital,fee,times,(1+np.interp(times,nodes,rates)/2)**(-2*times)

def waterfall(capital,times,cash):
 unpaid=0.;unreturned=capital;last=0.;lp_profit=0.;gp_profit=0.;lp=[];gp=[];ledger=[]
 for time,amount in zip(times,cash):
  unpaid+=.08*unreturned*(time-last);last=time
  repaid=min(amount,unreturned);amount-=repaid;unreturned-=repaid
  pref=min(amount,unpaid);amount-=pref;unpaid-=pref;lp_profit+=pref
  catch=min(amount,max(0,.25*lp_profit-gp_profit));amount-=catch
  g=catch+.2*amount;l=repaid+pref+.8*amount;gp_profit+=g;lp_profit+=.8*amount
  lp.append(l);gp.append(g);ledger.append((repaid,pref,catch,amount,unreturned,unpaid))
 return np.array(lp),np.array(gp),ledger

@lru_cache(maxsize=1)
def analysis():
 capital,fee,times,df=inputs();weights=np.array([.4,.35,.25]);calls=capital*weights;invest=(capital-fee)*weights;flows=invest*np.array([1.8,1.4,.4])
 lp,gp,_=waterfall(capital,times,flows)
 early=[waterfall(calls[i],[times[i]],[flows[i]])[1][0] for i in [0,1]];paid=sum(early)
 claw=max(0,paid-sum(gp));deal_lp=flows.copy();deal_lp[:2]-=early;deal_lp[-1]+=claw
 wpv=float(lp@df);dpv=float(deal_lp@df)
 stress=flows.copy();stress[-1]=invest[-1]*.1;entitled=sum(waterfall(capital,times,stress)[1]);stress_claw=max(0,paid-entitled)
 state_claws=[]
 for m in [0,.4,1.2]:
  x=flows.copy();x[-1]=invest[-1]*m;state_claws.append(max(0,paid-sum(waterfall(capital,times,x)[1])))
 proportion=max(state_claws)/paid
 return {'lp_contribution_mn':capital,'investable_mn':capital-fee,'year3_lp_mn':lp[1],'year3_gp_mn':gp[1],'final_whole_gp_mn':sum(gp),'early_deal_carry_mn':paid,'deal_lp_pv_mn':dpv,'lp_preferred_contract':'DEAL' if dpv>wpv else ('WHOLE' if wpv>dpv else 'TIE'),'stress_clawback_mn':stress_claw,'stress_final_gp_mn':entitled,'stress_final_lp_cash_mn':stress[-1]+stress_claw,'minimum_early_escrow_pct':proportion*100,'binding_last_multiple':['0.0','0.4','1.2'][int(np.argmax(state_claws))],'gp_escrow_pv_cost_mn':float(proportion*(np.dot(early,df[:2])-paid*df[-1]))}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['lp_contribution_mn', 'investable_mn'], ['year3_lp_mn', 'year3_gp_mn', 'final_whole_gp_mn'], ['early_deal_carry_mn', 'deal_lp_pv_mn', 'lp_preferred_contract'], ['stress_clawback_mn', 'stress_final_gp_mn', 'stress_final_lp_cash_mn'], ['minimum_early_escrow_pct', 'binding_last_multiple', 'gp_escrow_pv_cost_mn']]
variables = [
    Variable('lp_contribution_mn',None,'Store hypothetical LP contributed capital in USD millions, rounded to 4 decimals.'),
    Variable('investable_mn',None,'Store capital remaining for project investment after the initial expense in USD millions, rounded to 4 decimals.'),
    Variable('year3_lp_mn',None,'Store LP cash distribution at the second whole-fund exit in USD millions, rounded to 4 decimals.'),
    Variable('year3_gp_mn',None,'Store GP cash distribution at the second whole-fund exit in USD millions, rounded to 4 decimals.'),
    Variable('final_whole_gp_mn',None,'Store cumulative terminal whole-fund GP entitlement in USD millions, rounded to 4 decimals.'),
    Variable('early_deal_carry_mn',None,'Store carry actually paid at the first two deal-by-deal exits in USD millions, rounded to 4 decimals.'),
    Variable('deal_lp_pv_mn',None,'Store present value of all baseline deal-by-deal LP distributions including terminal true-up in USD millions, rounded to 4 decimals.'),
    Variable('lp_preferred_contract',None,'Store the higher LP-present-value contract as exactly WHOLE, DEAL, or TIE.'),
    Variable('stress_clawback_mn',None,'Store terminal carry clawback in the adverse comparison in USD millions, rounded to 4 decimals.'),
    Variable('stress_final_gp_mn',None,'Store cumulative terminal GP entitlement in the adverse comparison in USD millions, rounded to 4 decimals.'),
    Variable('stress_final_lp_cash_mn',None,'Store LP cash received at final exit including clawback in the adverse comparison in USD millions, rounded to 4 decimals.'),
    Variable('minimum_early_escrow_pct',None,'Store the minimum uniform escrow fraction of early carry as a percentage, rounded to 4 decimals.'),
    Variable('binding_last_multiple',None,'Store the binding final-project investment multiple as a string using exactly one token from "0.0", "0.4", "1.2".'),
    Variable('gp_escrow_pv_cost_mn',None,'Store GP opportunity cost in present value from withholding early carry until terminal escrow release in USD millions, rounded to 4 decimals.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, None, 4, 4, 4, 4, None, 4]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
