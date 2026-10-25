"""Financed fixed-date delivery boundary with a changing cheapest security.

This is a baseline hypothetical contract, not a diagnostic futures trap.
Omitting interim coupon carry yields a boundary of114.7273 rather than113.9520
quote points. The lowest cash clean price at+200bp selects91282CBL4, while
converted financed delivery cost selects91282CJJ1. The continuous envelope
switches at20.3511bp and has different left/right one-bp quote losses.
Those are measured financial error probes; public contract fixes calendars,
financing and conversion factors. Exact CUSIPs and4dp quantities use the local
strict scoring contract, without acceptance-band tuning.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
    ids=['91282CBL4','91282CJJ1','91282CKC4']
    a=load_runtime_table('treasury_auctions.auctions');a=a[a.cusip.astype(str).isin(ids)&a.reopening.astype(str).eq('No')]
    if len(a)!=3 or a.cusip.duplicated().any():raise ValueError('Nonunique original bond templates')
    y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2024-12-31')]
    s=load_runtime_table('nyfed.reference_rates');s=s[s['Effective Date'].astype(str).eq('2024-12-31')&s['Rate Type'].astype(str).eq('SOFR')]
    if len(y)!=1 or len(s)!=1:raise ValueError('Nonunique curve/funding fixing')
    nodes=np.array([1/12,2/12,3/12,.5,1,2,3,5,7,10,20,30]);rates=np.array([y.iloc[0][c] for c in ['1 Mo','2 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']],float)/100
    return ids,a.set_index('cusip'),nodes,rates,float(s['Rate (%)'].iloc[0])/100

def coupon_dates(maturity,value_date):
    maturity=pd.Timestamp(maturity);dates=[]
    for k in range(100):
        date=maturity-pd.DateOffset(months=6*k)
        if maturity.is_month_end:date+=pd.offsets.MonthEnd(0)
        if date<=value_date:return sorted(dates),date
        dates.append(date)
    raise ValueError('Coupon calendar exceeds domain')

def prices(cusip,shift_bp=0):
    _,a,nodes,rates,funding_base=inputs();row=a.loc[cusip];funding=funding_base+.0025*float(row.bid_to_cover_ratio);v=pd.Timestamp('2024-12-31');d=pd.Timestamp('2025-03-31')
    dates,last=coupon_dates(row.maturity_date,v);coupon=float(row.int_rate)/2
    cash=np.full(len(dates),coupon);cash[-1]+=100;time=np.array([(x-v).days/365 for x in dates])
    discount=(1+(np.interp(time,nodes,rates)+shift_bp/10000)/2)**(-2*time)
    dirty=float(cash@discount);ai=coupon*(v-last).days/(dates[0]-last).days
    carried=sum(coupon*(1+funding*(d-x).days/360) for x in dates if x<=d)
    previous=max([last]+[x for x in dates if x<=d]);next_date=min(x for x in dates if x>d)
    delivery_ai=coupon*(d-previous).days/(next_date-previous).days
    forward=dirty*(1+funding*(d-v).days/360)-carried-delivery_ai
    return dirty,dirty-ai,carried,forward

def quotes(bp):
    ids,*_=inputs();return np.array([prices(id,bp)[3]/factor for id,factor in zip(ids,[.73,.874893,.80])])

@lru_cache(maxsize=1)
def analysis():
    ids,*_=inputs();base=quotes(0);switch=brentq(lambda bp:quotes(bp)[0]-quotes(bp)[1],-200,200,xtol=1e-10)
    return {'cash_price_lowest_cusip':ids[int(np.argmin([prices(id)[1] for id in ids]))],'bond2_dirty_price':prices(ids[1])[0],'bond1_carried_coupon_value':prices(ids[0])[2],'bond3_carried_coupon_value':prices(ids[2])[2],'bond1_forward_clean_price':prices(ids[0])[3],'base_ctd_cusip':ids[int(np.argmin(base))],'base_boundary_quote':float(min(base)),'first_switch_bp':switch,'post_switch_cusip':ids[int(np.argmin(quotes(switch+1e-4)))],'up200_boundary_quote':float(min(quotes(200))),'left_switch_dv01':float(min(quotes(switch-1))-min(quotes(switch))),'right_switch_dv01':float(min(quotes(switch))-min(quotes(switch+1)))}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['cash_price_lowest_cusip', 'bond2_dirty_price'], ['bond1_carried_coupon_value', 'bond3_carried_coupon_value', 'bond1_forward_clean_price'], ['base_ctd_cusip', 'base_boundary_quote'], ['first_switch_bp', 'post_switch_cusip', 'up200_boundary_quote'], ['left_switch_dv01', 'right_switch_dv01']]
variables = [
    Variable('cash_price_lowest_cusip',None,'Store the selected Treasury CUSIP as its exact nine-character source string.'),
    Variable('bond2_dirty_price',None,'Store second bond dirty price in USD per 100 face, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('bond1_carried_coupon_value',None,'Store delivery-date value of first bond interim coupons in USD per 100 face, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('bond3_carried_coupon_value',None,'Store delivery-date value of third bond interim coupons in USD per 100 face, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('bond1_forward_clean_price',None,'Store first bond financed forward clean price in USD per 100 face, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('base_ctd_cusip',None,'Store the selected Treasury CUSIP as its exact nine-character source string.'),
    Variable('base_boundary_quote',None,'Store base delivery boundary in futures quote points, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('first_switch_bp',None,'Store first positive curve shift changing the cheapest deliverable in basis points, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('post_switch_cusip',None,'Store the selected Treasury CUSIP as its exact nine-character source string.'),
    Variable('up200_boundary_quote',None,'Store delivery boundary at a positive 200 basis-point curve shift in futures quote points, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('left_switch_dv01',None,'Store left one-basis-point interval price loss in futures quote points, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('right_switch_dv01',None,'Store right one-basis-point interval price loss in futures quote points, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
]
DECIMALS = [None, 4, 4, 4, 4, None, 4, 4, None, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
