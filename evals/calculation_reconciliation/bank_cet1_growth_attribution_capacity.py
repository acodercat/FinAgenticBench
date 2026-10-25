"""Exact endpoint Shapley attribution and funded lending capacity.

Measured alternatives: capital-first contribution 171.2222990533470837604726265 bp;
capital-last 188.5209409424931727026252020 bp versus canonical 179.87161999792013.
Ignoring retained earnings yields 14.22545687024791685439250524 billion capacity.
Business-day-only funding costs 183.0166666666666666666666666 million.
These are pinned-method regression challenges, not live traps. Exact finite
Shapley, no statistical conditional SHAP, sampling, or iterative solver."""
from functools import lru_cache
from decimal import Decimal
import pandas as pd
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs

def D(value):
    """Decimal from the source repr, so no float rounding enters the arithmetic."""
    return Decimal(str(value))

variables = [
    Variable('eligible_bank_count', None, 'Store the eligible bank count as an integer.'),
    Variable('selected_bank_rssd', None, 'Store the selected RSSD identifier as its exact source digit string.'),
    Variable('ending_cet1_pct', None, 'Store the ending standardized CET1 ratio in percent as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('capital_contribution_bp', None, 'Store the CET1-capital contribution in basis points as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('rwa_contribution_bp', None, 'Store the risk-weighted-assets contribution in basis points as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('funding_expense_mn', None, 'Store the total funding expense in USD millions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('proforma_cet1_pct', None, 'Store the pro forma year-end standardized CET1 ratio in percent as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('maximum_new_loans_bn', None, 'Store the maximum additional loan amount in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('additional_cet1_for_plan_bn', None, 'Store the minimum additional CET1 for the stated plan in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
]
DECIMALS = [0, None, 4, 4, 4, 4, 4, 4, 4]
TURN_1_NAMES = ['eligible_bank_count', 'selected_bank_rssd', 'ending_cet1_pct']
TURN_2_NAMES = ['capital_contribution_bp', 'rwa_contribution_bp']
TURN_3_NAMES = ['funding_expense_mn', 'proforma_cet1_pct']
TURN_4_NAMES = ['maximum_new_loans_bn', 'additional_cet1_for_plan_bn']


START, END = '2024-03-31', '2025-09-30'
CAP = 'common_equity_tier1_capital_thousand_usd'
RWA = 'standardized_risk_weighted_assets_thousand_usd'


def analysis():
    capital = load_expansion_table('ffiec_call_reports_capital')
    selected = capital[capital.report_date.isin([START, END])
        & capital.capital_reporting_basis.eq('consolidated')
        & capital.cblr_election.eq(False)].copy()
    for column in [CAP, RWA]:
        selected = selected[selected[column].notna() & selected[column].gt(0)]
    panel = selected.pivot(index='bank_id', columns='report_date', values=[CAP, RWA]).dropna()
    panel = panel[(panel[CAP, START] >= 10000000) & (panel[CAP, END] >= 10000000)]
    states = {str(i): tuple(D(row[c,t]) for c in [CAP,RWA] for t in [START,END]) for i,row in panel.iterrows()}
    if not states or not all(v.is_finite() for state in states.values() for v in state):
        raise ValueError('Invalid bank panel')
    changes = {i:s[1]/s[3]-s[0]/s[2] for i,s in states.items()}
    leader = min(states, key=lambda i:(-changes[i],int(i)))
    c0,c1,r0,r1 = states[leader]
    cap_effect = (c1-c0)*(1/r0+1/r1)/2
    rwa_effect = (c0+c1)*(1/r1-1/r0)/2
    rates = load_expansion_table('nyfed_reference_rates')
    rates = rates[rates['Rate Type'].eq('SOFR')].copy()
    if rates['Effective Date'].duplicated().any():
        raise ValueError('Duplicate SOFR effective date')
    fixings = rates.set_index(pd.to_datetime(rates['Effective Date']))['Rate (%)'].sort_index()
    days = pd.date_range('2025-10-01','2025-12-31',freq='D')
    daily = fixings.reindex(days,method='ffill')
    if daily.isna().any():
        raise ValueError('Missing SOFR coverage')
    funding = sum(((D(rate)/100+D('0.01'))/360 for rate in daily),D(0))
    net_margin = (D('0.06')*len(days)/360-funding)*D('0.75')
    floor = c0/r0
    if floor <= net_margin or c1 < floor*r1:
        raise ValueError('No finite positive lending capacity under scenario')
    capacity = (c1-floor*r1)/(floor-net_margin)
    plan = D(20000000)
    required = max(D(0),floor*(r1+plan)-c1-net_margin*plan)
    answer = dict(eligible_bank_count=len(states),selected_bank_rssd=leader,
        ending_cet1_pct=float(100*c1/r1),capital_contribution_bp=float(cap_effect*10000),
        rwa_contribution_bp=float(rwa_effect*10000),funding_expense_mn=float(plan*funding/1000),
        proforma_cet1_pct=float(100*(c1+plan*net_margin)/(r1+plan)),
        maximum_new_loans_bn=float(capacity/10**6),additional_cet1_for_plan_bn=float(required/10**6))
    return answer

@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names],
        [truth[n] for n in names], [precision[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])

validators = {
    'validate_turn_1': turn_validator(validate_turn_1),
    'validate_turn_2': turn_validator(validate_turn_2),
    'validate_turn_3': turn_validator(validate_turn_3),
    'validate_turn_4': turn_validator(validate_turn_4),
}
