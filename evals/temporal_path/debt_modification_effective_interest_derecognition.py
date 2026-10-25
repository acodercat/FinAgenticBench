"""Initial EIR, fee-adjusted modification accretion and subsequent exchanges.

Baseline IFRS liability recognition and successive legal-contract changes,
under frozen and explicitly pinned conventions. Initial semiannual EIR2.2962%
differs from A's fee-amortizing2.4371%. The next10% test is13.3085%; using the
fee-adjusted book denominator gives12.6245%, and substituting the fee rate
gives13.3031%. The B comparison cannot overwrite A's actual carried liability.
These measured paths test financial rates and rights rather than a live trap.
All source dollars become USDmillions;4dp strict comparison rejects the errors.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
    f=load_runtime_table('sec.financial_statements');f=f[f.accession.astype(str).eq('0000935036-25-000006')]
    if len(f)!=1:raise ValueError('Nonunique original liability template')
    face=float(f.iloc[0].LongTermDebtNoncurrent)/1e6
    y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2024-12-31')]
    if len(y)!=1 or face<=0:raise ValueError('Invalid pricing input')
    nodes=np.array([1/12,2/12,3/12,.5,1,2,3,5,7,10,20,30]);rates=np.array([y.iloc[0][c] for c in ['1 Mo','2 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']],float)/100
    return face,nodes,rates

def market_pv(flows,shift_bp=0):
    _,nodes,rates=inputs();time=np.arange(1,len(flows)+1)/2
    return float(np.array(flows)@(1+(np.interp(time,nodes,rates)+shift_bp/10000)/2)**(-2*time))

def pv(flows,rate):return float(np.array(flows)@(1+rate)**(-np.arange(1,len(flows)+1)))

def effective_rate(flows,carrying):return brentq(lambda rate:pv(flows,rate)-carrying,0,.2,xtol=1e-14)

def accrue(carrying,rate,payments):
    interest=[]
    for cash in payments:
        current=carrying*rate;interest.append(current);carrying+=current-cash
    return carrying,interest

@lru_cache(maxsize=1)
def analysis():
    face,*_=inputs();original=np.full(10,.025*face);original[-1]+=face
    issue=market_pv(original)-.01*face;original_rate=effective_rate(original,issue);carrying,interest=accrue(issue,original_rate,original[:4])
    lender=.004*face;legal=.006*face
    a=np.full(8,.02*face);a[-1]+=face;new_pv=pv(a,original_rate)
    test_a=abs(new_pv+lender-pv(original[4:],original_rate))/pv(original[4:],original_rate)
    gain_a=carrying-new_pv;adjusted=new_pv-lender-legal;a_rate=effective_rate(a,adjusted)
    b=np.full(6,.025*.8*face);b[-1]+=.8*face;test_b=abs(pv(b,original_rate)+lender-pv(original[4:],original_rate))/pv(original[4:],original_rate)
    fair_b=market_pv(b,200)
    classification_b='DERECOGNIZE' if test_b>=.1 else 'MODIFY'
    if classification_b=='DERECOGNIZE':
        book_b=fair_b;gain_b=carrying-fair_b-lender-legal
    else:
        modified_pv_b=pv(b,original_rate)
        book_b=modified_pv_b-lender-legal;gain_b=carrying-modified_pv_b
    next_carrying,_=accrue(adjusted,a_rate,a[:2]);next_flows=a[2:].copy();next_flows[-1]-=.15*face
    reference=pv(a[2:],original_rate);next_test=abs(pv(next_flows,original_rate)-reference)/reference;next_fair=market_pv(next_flows,200)
    return {'net_issuance_mn':issue,'original_semiannual_eir_pct':100*original_rate,'pre_modification_carrying_mn':carrying,'fourth_coupon_interest_mn':interest[-1],'A_ten_percent_test_pct':100*test_a,'A_modification_gain_mn':gain_a,'A_adjusted_carrying_mn':adjusted,'A_fee_amortizing_semiannual_eir_pct':100*a_rate,'B_classification':classification_b,'B_postchange_carrying_mn':book_b,'B_accounting_gain_mn':gain_b,'second_prechange_carrying_mn':next_carrying,'second_ten_percent_test_pct':100*next_test,'second_gain_mn':next_carrying-next_fair}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['net_issuance_mn', 'original_semiannual_eir_pct'], ['pre_modification_carrying_mn', 'fourth_coupon_interest_mn'], ['A_ten_percent_test_pct', 'A_modification_gain_mn', 'A_adjusted_carrying_mn', 'A_fee_amortizing_semiannual_eir_pct'], ['B_classification', 'B_postchange_carrying_mn', 'B_accounting_gain_mn'], ['second_prechange_carrying_mn', 'second_ten_percent_test_pct', 'second_gain_mn']]
variables = [
    Variable('net_issuance_mn',None,'Store initial net carrying amount in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('original_semiannual_eir_pct',None,'Store initial-recognition effective interest rate per half year in percent, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('pre_modification_carrying_mn',None,'Store carrying amount immediately after the fourth original coupon in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('fourth_coupon_interest_mn',None,'Store effective-interest expense in the fourth original half-year period in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('A_ten_percent_test_pct',None,'Store absolute contractual present-value change for proposal A in percent, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('A_modification_gain_mn',None,'Store positive modification gain for A before modification fees in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('A_adjusted_carrying_mn',None,'Store post-A carrying amount after both modification fees in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('A_fee_amortizing_semiannual_eir_pct',None,'Store post-A fee-amortizing effective rate per half year in percent, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('B_classification',None,'Store exactly MODIFY or DERECOGNIZE.'),
    Variable('B_postchange_carrying_mn',None,'Store liability carrying amount immediately after proposal B in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('B_accounting_gain_mn',None,'Store current-period accounting gain or loss for B including the applicable fee treatment, positive for a gain and negative for a loss, in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('second_prechange_carrying_mn',None,'Store actual A carrying amount immediately before the second exchange in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('second_ten_percent_test_pct',None,'Store absolute contractual present-value change for the second exchange in percent, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('second_gain_mn',None,'Store positive gain for the second exchange under its applicable accounting treatment in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, None, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
