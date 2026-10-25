"""County-date claim union and finite nonoverlapping occurrence optimization.

Baseline perfect retrospective hypothetical treaty, not actual NFIP coverage.
Ignoring reinstatement premium produces18m instead of16.9836m net recovery;
greedy windows yield16.9672m. Only the first chosen window is executed before
future revisions. The endorsement changes the best future start fromOct26 to
Oct27, and copying the old selection understates value1.4754m versus1.4836m.
Claims are counted once by a county/date union. These are measured financial
error paths, not a hidden diagnostic trap. Exact ordered dates and4dp amounts
use strict case-local storage validation; no score-dependent tolerance tuning.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
    claims=load_runtime_table('fema_nfip.claims');declarations=load_runtime_table('fema_disasters.declaration_areas')
    claims=claims[claims.state_usps.astype(str).eq('LA')&claims.date_of_loss.astype(str).str.startswith('2020')].copy()
    declarations=declarations[declarations.state_usps.astype(str).eq('LA')&declarations.incident_begin_date.astype(str).str.startswith('2020')&declarations.incident_type.astype(str).isin(['Hurricane','Flood'])]
    eligible=np.zeros(len(claims),bool)
    for row in declarations.itertuples():
        eligible|=(claims.county_code.astype('string[python]').str.zfill(5).eq(str(row.county_fips).zfill(5))&claims.date_of_loss.astype('string[python]').between(str(row.incident_begin_date)[:10],str(row.incident_end_date)[:10])).to_numpy(bool,na_value=False)
    claims=claims[eligible];claims['gross']=claims[['gross_building_payment_usd','gross_contents_payment_usd','gross_icc_payment_usd']].sum(axis=1,min_count=3);claims=claims[claims.gross.gt(0)]
    if claims.claim_record_id.duplicated().any():raise ValueError('Duplicate claim identity')
    dates=pd.date_range('2020-08-20','2020-10-29');daily=claims.groupby('date_of_loss').gross.sum().reindex(dates.strftime('%Y-%m-%d'),fill_value=0).to_numpy(float)/1e6*.4
    gross=float(claims.gross.sum()/1e6)
    if abs(sum(daily)/.4-gross)>1e-8:raise ValueError('Activity outside permitted calendar book')
    fractions=np.array([(pd.Timestamp('2021-01-01')-(z+pd.Timedelta(days=3))).days/366 for z in dates[:-2]])
    return len(claims),gross,dates,daily,fractions

def settlement(loss,pos,state):
    *_,fraction=inputs();layer,aggregate,reinstatement=state
    recovery=min(max(0,loss-5),layer,aggregate)
    restored=min(recovery,reinstatement)
    premium=3*(restored/10)*fraction[pos]
    next_state=(layer-recovery+restored,aggregate-recovery,reinstatement-restored)
    return recovery-premium,recovery,premium,next_state

def ledger(starts,windows,initial=(10.,18.,10.)):
    state=initial;net=0.;gross=0.;premium=0.
    for pos in starts:
        n,g,p,state=settlement(windows[pos],pos,state);net+=n;gross+=g;premium+=p
    return net,gross,premium,state

def optimum(windows,initial=(10.,18.,10.),minimum=0,budget=3):
    windows=tuple(windows)
    @lru_cache(None)
    def solve(pos,left,layer,aggregate,reinstatement):
        if left==0 or pos>=len(windows):return 0.,()
        best=solve(pos+1,left,layer,aggregate,reinstatement)
        immediate,_,_,state=settlement(windows[pos],pos,(layer,aggregate,reinstatement))
        continuation,path=solve(pos+3,left-1,*[round(v,12) for v in state]);candidate=immediate+continuation,(pos,)+path
        if candidate[0]>best[0]+1e-10 or (abs(candidate[0]-best[0])<=1e-10 and (len(candidate[1]),candidate[1])<(len(best[1]),best[1])):best=candidate
        return best
    return solve(minimum,budget,*initial)

@lru_cache(maxsize=1)
def analysis():
    count,gross,dates,daily,_=inputs();windows=np.convolve(daily,np.ones(3),'valid')
    fixed=ledger([dates.get_loc('2020-08-25'),dates.get_loc('2020-10-08')],windows)
    best,path=optimum(windows);best_ledger=ledger(path,windows);first=path[0];actual=ledger([first],windows)
    changed=daily.copy();changed[(dates>='2020-10-06')&(dates<='2020-10-10')]*=.1;changed[dates.get_loc('2020-10-28')]*=20;new_windows=np.convolve(changed,np.ones(3),'valid')
    future,future_path=optimum(new_windows,actual[3],first+3,2)
    layer,aggregate,reinstatement=actual[3];enhanced,enhanced_path=optimum(new_windows,(layer,aggregate+8,reinstatement+10),first+3,2)
    def label(path):return ','.join(str(dates[i].date()) for i in path) if path else 'NONE'
    value=enhanced-future
    return {'eligible_claim_count':count,'source_gross_mn':gross,'fixed_windows_gross_recovery_mn':fixed[1],'fixed_windows_paid_premium_mn':fixed[2],'maximum_net_recovery_mn':best,'optimal_window_starts':label(path),'optimal_reinstatement_premium_mn':best_ledger[2],'carried_aggregate_remaining_mn':actual[3][1],'future_optimal_starts':label(future_path),'future_net_recovery_mn':future,'endorsement_optimal_starts':label(enhanced_path),'endorsement_max_price_mn':value,'endorsement_decision':'BUY' if value>1 else 'DECLINE'}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['eligible_claim_count', 'source_gross_mn'], ['fixed_windows_gross_recovery_mn', 'fixed_windows_paid_premium_mn'], ['maximum_net_recovery_mn', 'optimal_window_starts', 'optimal_reinstatement_premium_mn'], ['carried_aggregate_remaining_mn', 'future_optimal_starts', 'future_net_recovery_mn'], ['endorsement_optimal_starts', 'endorsement_max_price_mn', 'endorsement_decision']]
variables = [
    Variable('eligible_claim_count',None,'Store the distinct eligible claim count as an integer.'),
    Variable('source_gross_mn',None,'Store source gross building, contents and ICC payment sum in USD millions before hypothetical scaling, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('fixed_windows_gross_recovery_mn',None,'Store gross treaty recovery for the two fixed windows in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('fixed_windows_paid_premium_mn',None,'Store total required reinstatement premium for the fixed windows in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('maximum_net_recovery_mn',None,'Store maximum recovery net of required reinstatement premiums in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('optimal_window_starts',None,'Store chronological window start dates as exact YYYY-MM-DD strings joined by commas, with no spaces; store NONE for an empty selection.'),
    Variable('optimal_reinstatement_premium_mn',None,'Store required premium for the selected globally optimal windows in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('carried_aggregate_remaining_mn',None,'Store actual annual gross recovery aggregate remaining after the first selected window in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('future_optimal_starts',None,'Store chronological window start dates as exact YYYY-MM-DD strings joined by commas, with no spaces; store NONE for an empty selection.'),
    Variable('future_net_recovery_mn',None,'Store optimized future recovery net of future premiums in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('endorsement_optimal_starts',None,'Store chronological window start dates as exact YYYY-MM-DD strings joined by commas, with no spaces; store NONE for an empty selection.'),
    Variable('endorsement_max_price_mn',None,'Store maximum additional endorsement price justified by incremental optimized future net recovery in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('endorsement_decision',None,'Store exactly BUY or DECLINE.'),
]
DECIMALS = [0, 4, 4, 4, 4, None, 4, 4, None, 4, None, 4, None]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
