"""Quantitative-hard formula-free baseline; alternatives are pinned-convention regressions.
Single strict PV contract. Full model states are retained unrounded.
Exact source-native raw admission is tracked separately from runtime validation.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from core.data import load_expansion_table
from core.dataset_layers import load_catalog, DATASETS_DIR

@lru_cache(None)
def mortgage_inputs(state='CA',ddof=1):
    meta=load_catalog()['tables']['hmda.loan_applications']
    columns=['activity_year','state_code','action_taken','lien_status','loan_type','interest_rate','occupancy_type','total_units']
    values=[]; eligible=0
    for d in pd.read_csv(DATASETS_DIR/meta['layers']['runtime']['path'],usecols=columns,dtype=str,chunksize=300000):
        d=d[d.activity_year.eq('2024')&d.state_code.eq(state)&d.action_taken.eq('1')&d.lien_status.eq('1')&d.loan_type.eq('1')&d.occupancy_type.eq('1')&d.total_units.eq('1')]
        eligible+=len(d);x=pd.to_numeric(d.interest_rate,errors='coerce');values.extend(x[x.between(.1,20)].tolist())
    rates=load_expansion_table('treasury_yield_curve');y=rates[rates.Date.between('2024-01-01','2024-12-31')].sort_values('Date');assert y.Date.is_unique
    sigma=np.diff(y['1 Yr'].to_numpy(float)/100).std(ddof=ddof)*np.sqrt(252)
    row=rates[rates.Date=='2024-12-31'];assert len(row)==1
    knots=np.array([.5,1,2,3,5]);par=np.array([float(row[k].iloc[0])/100 for k in ['6 Mo','1 Yr','2 Yr','3 Yr','5 Yr']])
    return float(np.median(values))/100,sigma,np.interp(np.arange(.5,4.01,.5),knots,par),len(values),eligible,len(y)-1

def bootstrap(par,zero=False):
    ds=[]
    for i,c in enumerate(par):ds.append((1+c/2)**(-(i+1)) if zero else (1-c/2*sum(ds))/(1+c/2))
    ds=np.array(ds);assert np.all(ds>0)
    return ds

def ho_tree(discounts,sigma):
    if sigma<=0 or np.any(discounts<=0):raise ValueError('domain')
    q=np.ones(1);tree=[];stateprices=[q];res=[]
    for i,d in enumerate(discounts):
        x=(2*np.arange(i+1)-i)*sigma*np.sqrt(.5)
        a=2*np.log(np.dot(q,np.exp(-x*.5))/d);r=a+x;tree.append(r)
        nq=np.zeros(i+2);z=q*np.exp(-r*.5)/2;nq[:-1]+=z;nq[1:]+=z;q=nq;stateprices.append(q);res.append(abs(q.sum()-d))
    return tree,stateprices,max(res)

def amortizer(coupon):
    c=coupon/2;payment=100*c/(1-(1+c)**-8);b=[100.]
    for _ in range(8):b.append(b[-1]*(1+c)-payment)
    assert abs(b[-1])<1e-10;b[-1]=0
    return payment,np.array(b)

def mortgage_value(tree,coupon,spread=0,call=True,wrong_discount=False):
    pay,bal=amortizer(coupon);v=np.zeros(9);choices={};margins=[]
    for i in range(7,-1,-1):
        disc=1/(1+(tree[i]+spread)*.5) if wrong_discount else np.exp(-(tree[i]+spread)*.5)
        continuation=disc*(pay+(v[:-1]+v[1:])/2)
        if i>0 and call:
            choices[i]=continuation>=bal[i];margins.extend(abs(continuation-bal[i]));v=np.minimum(continuation,bal[i])
        else:v=continuation
    return float(v[0]),choices,float(min(margins)) if margins else 0

def mortgage_a(state='CA',ddof=1,curve='par',discount='continuous'):
    coupon,sigma,par,n,eligible,nd= mortgage_inputs(state,ddof);ds=bootstrap(par,curve=='zero');tree,q,res=ho_tree(ds,sigma)
    pay,b=amortizer(coupon);plain=mortgage_value(tree,coupon,call=False)[0];call=mortgage_value(tree,coupon)[0]
    wrong=discount=='simple';fn=lambda s:mortgage_value(tree,coupon,s,wrong_discount=wrong)[0]-95
    spread=brentq(fn,-.1,.2,xtol=1e-14,rtol=1e-14);p,choices,margin=mortgage_value(tree,coupon,spread,wrong_discount=wrong)
    # Actual stopping distribution under the spread-dependent optimal policy.
    probs=np.ones(1);prepay=0
    for i in range(8):
        if i>0:
            ex=choices[i];prepay+=probs[ex].sum();probs=probs.copy();probs[ex]=0
        nxt=np.zeros(i+2);nxt[:-1]+=probs/2;nxt[1:]+=probs/2;probs=nxt
    bump=.0001;times=np.arange(.5,4.01,.5);ps=[]
    for shift in [-bump,bump]:
        tr,_,_=ho_tree(ds*np.exp(-shift*times),sigma);ps.append(mortgage_value(tr,coupon,spread,wrong_discount=wrong)[0])
    duration=(ps[0]-ps[1])/(2*bump*p);convexity=(ps[0]+ps[1]-2*p)/(bump*bump*p)
    out=dict(mortgage_rate_count=n,median_coupon_pct=coupon*100,short_rate_sigma_bp=sigma*10000,yield_increment_count=nd,
      fourth_year_discount=ds[-1],year_two_low_short_rate_pct=tree[4][0]*100,state_price_residual=res,
      noncallable_value_per100=plain,prepayable_value_per100=call,option_adjusted_spread_bp=spread*10000,spread_repricing_residual=abs(p-95),
      prepayment_probability=prepay,effective_duration_years=duration,effective_convexity_years2=convexity)
    return out,dict(eligible=eligible,payment=pay,balances=b.tolist(),discounts=ds.tolist(),tree=[v.tolist()for v in tree],prepayment_margin=margin,probability_residual=abs(prepay+probs.sum()-1),spread_slope=(fn(spread+1e-6)-fn(spread-1e-6))/2e-6,prices_bumped=ps,call_policy={str(k):v.tolist()for k,v in choices.items()})


from cave_agent import Variable
from core.validation import validate_ordered_outputs, turn_validator

TURN_1_NAMES = ['mortgage_rate_count', 'median_coupon_pct', 'short_rate_sigma_bp', 'yield_increment_count']

TURN_2_NAMES = ['fourth_year_discount', 'year_two_low_short_rate_pct']

TURN_3_NAMES = ['noncallable_value_per100', 'prepayable_value_per100', 'option_adjusted_spread_bp']

TURN_4_NAMES = ['prepayment_probability', 'effective_duration_years', 'effective_convexity_years2']

variables = [
    Variable('mortgage_rate_count', None, 'Store the retained mortgage-rate count as an integer.'),
    Variable('median_coupon_pct', None, 'Store the estimated median nominal annual coupon in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('short_rate_sigma_bp', None, 'Store the normal short-rate diffusion volatility in basis points per square root year, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('yield_increment_count', None, 'Store the yield-increment count as an integer.'),
    Variable('fourth_year_discount', None, 'Store the four-year discount factor as a dimensionless number, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('year_two_low_short_rate_pct', None, 'Store the lowest year-two continuously compounded short rate in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('noncallable_value_per100', None, 'Store the nonprepayable present value per 100 original principal, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('prepayable_value_per100', None, 'Store the zero-spread prepayable present value per 100 original principal, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('option_adjusted_spread_bp', None, 'Store the fitted continuously compounded option-adjusted spread in basis points per year, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('prepayment_probability', None, 'Store the pre-maturity prepayment probability as a decimal probability, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('effective_duration_years', None, 'Store the effective duration in years, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
    Variable('effective_convexity_years2', None, 'Store the effective convexity in squared years, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.'),
]
DECIMALS = [0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    out=mortgage_a()[0]
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
