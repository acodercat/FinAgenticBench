"""Hypothetical acquisition, full-goodwill attribution and internal capital.

Baseline: the public conversation pins full goodwill, ownership attribution,
the accounting standards and the hypothetical internal capital policy.
Adding original target equity incorrectly makes initial internal capital
8535.0475m rather than7998.6295m. Deducting the parent impairment again after
allowing for goodwill deductibility incorrectly produces7939.4080m.
The control-retaining NCI purchase uses its impaired carrying amount and
raises the goodwill-deduction ownership factor to90%; final capital7967.9497m.
These measured alternatives are regression probes, not a claimed live trap.
Inherited subsidiary NCI is separately carried unchanged at book/fair value;
the new markdown, goodwill impairment and buyout concern direct bank ownership.
All source amounts are converted from USD thousands to USD millions;
four-decimal tolerance is0.00006m and cannot absorb these financial errors.
"""
from functools import lru_cache
import numpy as np
from core.dataset_layers import load_runtime_table

@lru_cache(maxsize=1)
def inputs():
    balances=load_runtime_table('ffiec_call_reports.bank_balance_sheets')
    capital=load_runtime_table('ffiec_call_reports.bank_capital')
    rows=[]
    for rssd in ['197478','141556']:
        b=balances[balances.bank_rssd_id.astype(str).eq(rssd)&balances.report_date.astype(str).eq('2025-09-30')]
        c=capital[capital.bank_id.astype(str).eq(rssd)&capital.report_date.astype(str).eq('2025-09-30')]
        if len(b)!=1 or len(c)!=1:raise ValueError('Nonunique bank template')
        b=b.iloc[0];c=c.iloc[0]
        if b.balance_reporting_basis!='consolidated' or c.capital_reporting_basis!='consolidated' or bool(c.cblr_election):raise ValueError('Invalid reporting basis/election')
        rows.append({'assets':float(b.consolidated_total_assets_thousand_usd)/1000,'equity':float(b.consolidated_total_equity_thousand_usd)/1000,'liabilities':float(b.consolidated_total_liabilities_thousand_usd)/1000,'inherited_nci':(float(b.consolidated_total_assets_thousand_usd)-float(b.consolidated_total_liabilities_thousand_usd)-float(b.consolidated_total_equity_thousand_usd))/1000,'cet1':float(c.common_equity_tier1_capital_thousand_usd)/1000,'rwa':float(c.standardized_risk_weighted_assets_thousand_usd)/1000})
    y=load_runtime_table('treasury_rates.yield_curve');y=y[y.Date.astype(str).eq('2025-09-30')]
    if len(y)!=1:raise ValueError('Nonunique appraisal curve')
    nodes=np.array([1/12,2/12,3/12,.5,1,2,3,5,7,10,20,30])
    rates=np.array([y.iloc[0][c] for c in ['1 Mo','2 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']],float)/100
    return *rows,nodes,rates

@lru_cache(maxsize=1)
def analysis():
    buyer,target,nodes,rates=inputs();stake=.2*target['equity'];time=np.arange(1,6)
    payments=np.repeat(.05*stake,5);payments[-1]+=1.4*stake
    nci=float(payments@(1+np.interp(time,nodes,rates)/2)**(-2*time))
    price=.8*1.4*target['equity'];issued=.5*price;transaction_fee=.01*price;issuance_fee=.02*issued
    markdown=.01*target['assets'];goodwill=price+nci-(target['equity']-markdown)
    parent=buyer['equity']+issued-transaction_fee-issuance_fee
    rwa=buyer['rwa']+target['rwa']-markdown
    policy_adjustment=buyer['cet1']-buyer['equity']
    cap=parent+policy_adjustment-.8*goodwill
    impairment=.3*goodwill;parent_after=parent-.8*impairment;nci_after=nci-.2*impairment;gw_after=goodwill-impairment
    cap_after=parent_after+policy_adjustment-.8*gw_after
    transferred=.5*nci_after;paid=1.2*transferred;parent_final=parent_after+transferred-paid
    cap_final=parent_final+policy_adjustment-.9*gw_after
    return {'consideration_mn':price,'nci_fair_value_mn':nci,'new_goodwill_mn':goodwill,'postdeal_parent_equity_mn':parent,'postdeal_internal_capital_mn':cap,'new_loan_rwa_capacity_mn':max(0,cap/.12-rwa),'impairment_parent_equity_mn':parent_after,'impairment_internal_capital_mn':cap_after,'buyout_internal_capital_mn':cap_final,'buyout_capital_ratio_pct':100*cap_final/rwa}

from cave_agent import Variable
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [['consideration_mn', 'nci_fair_value_mn'], ['new_goodwill_mn', 'postdeal_parent_equity_mn'], ['postdeal_internal_capital_mn', 'new_loan_rwa_capacity_mn'], ['impairment_parent_equity_mn', 'impairment_internal_capital_mn'], ['buyout_internal_capital_mn', 'buyout_capital_ratio_pct']]
variables = [
    Variable('consideration_mn',None,'Store acquisition consideration in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('nci_fair_value_mn',None,'Store independently appraised noncontrolling interest in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('new_goodwill_mn',None,'Store newly recognized full goodwill in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('postdeal_parent_equity_mn',None,'Store post-acquisition equity attributable to the parent in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('postdeal_internal_capital_mn',None,'Store post-acquisition internal capital in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('new_loan_rwa_capacity_mn',None,'Store additional loan risk-weighted-asset capacity in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('impairment_parent_equity_mn',None,'Store equity attributable to the parent after impairment in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('impairment_internal_capital_mn',None,'Store internal capital after impairment in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('buyout_internal_capital_mn',None,'Store internal capital after the subsequent NCI purchase in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
    Variable('buyout_capital_ratio_pct',None,'Store final internal capital divided by carried risk-weighted assets in percent, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.'),
]
DECIMALS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
