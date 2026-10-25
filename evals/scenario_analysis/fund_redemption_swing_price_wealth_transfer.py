"""Nondiluting redemption price with carried holdings and cumulative sale tiers.

Baseline: cash-first gross sale costs belong to redeeming units, while the
remaining-unit NAV stays unchanged in a fair redemption. Paying pretrade NAV
instead reduces survivors' NAV from95.4309 to95.3703. Resetting consumed tiers
incorrectly raises the second price from91.9270 to92.7870. The actual first
redemption survives the comparison branches; the120bp swing cap binds at
1.9233 million next units. These are regression probes, not hidden trap cues.
Fullprecision cash/face/unit conservation precedes4dp storage validation.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
    n=load_runtime_table('sec_nport.fund_reports');n=n[n.accession.astype(str).eq('0001752724-25-122382')]
    if len(n)!=1:raise ValueError('Nonunique original fund report')
    row=n.iloc[0];nav=float(row.net_assets_usd)/1e6;cash=float(row.cash_not_elsewhere_reported_usd)/1e6;requested=float(row.redemptions_month_3_usd)/1e6/100
    y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2025-03-31')]
    if len(y)!=1:raise ValueError('Nonunique pricing curve')
    nodes=np.array([1/12,2/12,3/12,.5,1,2,3,5,7,10,20,30]);rates=np.array([y.iloc[0][c] for c in ['1 Mo','2 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']],float)/100
    return nav,cash,requested,nodes,rates

def discounts(time,bp=0):
    *_,nodes,rates=inputs();time=np.array(time)
    return (1+(np.interp(time,nodes,rates)+bp/10000)/2)**(-2*time)

@lru_cache(maxsize=1)
def coupons():
    return [2*(1-discounts(t))/sum(discounts(np.arange(1,2*t+1)/2)) for t in [2,5,10]]

def marks(bp):
    return np.array([c/2*sum(discounts(np.arange(1,2*t+1)/2,bp))+discounts(t,bp) for t,c in zip([2,5,10],coupons())])

def cumulative_cost(gross):
    nav,*_=inputs()
    return .002*min(gross,.05*nav)+.008*min(max(0,gross-.05*nav),.10*nav)+.02*max(0,gross-.15*nav)

def sale_cost(gross,previous):return cumulative_cost(previous+gross)-cumulative_cost(previous)

def transaction(units,faces,cash,request,bp,previous=0,fair=True):
    marked=faces*marks(bp);assets=cash+sum(marked);price_before=assets/units
    if request<=0 or request>=units:raise ValueError('Request outside nonterminal domain')
    if fair:gross=max(0,request*price_before-cash)
    elif request*price_before<=cash:gross=0.
    else:gross=brentq(lambda g:cash+g-sale_cost(g,previous)-request*price_before,0,sum(marked),xtol=1e-10)
    cost=sale_cost(gross,previous);price=price_before-cost/request if fair else price_before
    fraction=gross/sum(marked);remaining_faces=faces*(1-fraction);remaining_cash=cash+gross-cost-request*price
    if gross>sum(marked)+1e-8 or remaining_cash<-1e-8:raise ValueError('Unfunded transaction')
    post=(remaining_cash+sum(remaining_faces*marks(bp)))/(units-request)
    return {'price':price,'pre_nav':price_before,'gross':gross,'cost':cost,'faces':remaining_faces,'cash':remaining_cash,'units':units-request,'previous':previous+gross,'post_nav':post}

@lru_cache(maxsize=1)
def analysis():
    nav,cash,request,*_=inputs();units=nav/100;faces=(nav-cash)*np.array([.3,.4,.3])
    naive=transaction(units,faces,cash,request,100,fair=False);first=transaction(units,faces,cash,request,100)
    second=transaction(first['units'],first['faces'],first['cash'],.1*units,150,first['previous']);simultaneous=transaction(units,faces,cash,request+.1*units,150)
    pre=second['pre_nav']
    def swing_fraction(u):return sale_cost(max(0,u*pre-first['cash']),first['previous'])/(u*pre)
    upper=.95*first['units'];capacity=brentq(lambda u:swing_fraction(u)-.012,1e-6,upper,xtol=1e-10)
    return {'opening_units_mn':units,'first_redemption_units_mn':request,'shock_pretrade_nav':first['pre_nav'],'unswung_post_nav':naive['post_nav'],'fair_first_price':first['price'],'fair_first_gross_sales_mn':first['gross'],'second_fair_price':second['price'],'second_post_nav':second['post_nav'],'simultaneous_150bp_price':simultaneous['price'],'maximum_next_request_units_mn':capacity,'cap_binding':'SWING'}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['opening_units_mn', 'first_redemption_units_mn'], ['shock_pretrade_nav', 'unswung_post_nav'], ['fair_first_price', 'fair_first_gross_sales_mn'], ['second_fair_price', 'second_post_nav', 'simultaneous_150bp_price'], ['maximum_next_request_units_mn', 'cap_binding']]
variables = [
    Variable('opening_units_mn',None,'Store opening units in millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('first_redemption_units_mn',None,'Store first redemption request in millions of units, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('shock_pretrade_nav',None,'Store post-shock pre-redemption NAV in USD per unit, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('unswung_post_nav',None,'Store remaining-unit NAV after an unswung first redemption in USD per unit, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('fair_first_price',None,'Store first nondiluting redemption price in USD per unit, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('fair_first_gross_sales_mn',None,'Store gross asset sales for the nondiluting first redemption in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('second_fair_price',None,'Store sequential second nondiluting redemption price in USD per unit, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('second_post_nav',None,'Store remaining-unit NAV after the sequential second redemption in USD per unit, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('simultaneous_150bp_price',None,'Store common nondiluting price for the simultaneous comparison in USD per unit, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('maximum_next_request_units_mn',None,'Store maximum next redemption request in millions of units, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('cap_binding',None,'Store exactly one canonical token: SWING, ASSET or DOMAIN.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, None]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
