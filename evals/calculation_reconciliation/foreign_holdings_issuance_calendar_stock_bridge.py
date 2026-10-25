"""Foreign-holdings changes versus calendar-aligned reported net issuance.
Baseline: country breadth, net issuance and issued maturity mix do not identify
primary-market foreign purchases. Auction-date instead of issue-date aggregation
changes accepted principal (2403.4334 versus 2552.7346 USD billions).
Numeric fields use shared display tolerances; source-verbatim strings are exact.
"""

from functools import lru_cache
from decimal import Decimal
import numpy as np
import pandas as pd
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


def _sum(values):
    return sum((Decimal(str(x)) for x in values), Decimal(0))

TURN_1_NAMES = ['selected_month', 'increasing_country_count', 'decreasing_country_count']
TURN_2_NAMES = ['debt_transaction_day_count', 'net_marketable_issues_bn']
TURN_3_NAMES = ['holdings_to_net_issuance_pct', 'increasing_country', 'increasing_country_change_bn', 'decreasing_country', 'decreasing_country_change_bn']
TURN_4_NAMES = ['issued_accepted_principal_bn', 'selected_short_maturity_share_pct', 'previous_short_maturity_share_pct']

variables = [
    Variable('selected_month', None, 'Store the selected month in YYYY-MM format as a string.'),
    Variable('increasing_country_count', None, 'Store countries with strictly positive holdings changes as an integer.'),
    Variable('decreasing_country_count', None, 'Store countries with strictly negative holdings changes as an integer.'),
    Variable('debt_transaction_day_count', None, 'Store the number of distinct DTS observation dates as an integer.'),
    Variable('net_marketable_issues_bn', None, 'Store reported marketable issues net of redemptions in USD billions, as a number rounded to 4 decimals.'),
    Variable('holdings_to_net_issuance_pct', None, 'Store foreign-holdings increase divided by net reported marketable issuance, in percent, as a number rounded to 4 decimals.'),
    Variable('increasing_country', None, 'Store the increasing country label verbatim from TIC as a string.'),
    Variable('increasing_country_change_bn', None, 'Store its signed holdings change in USD billions, as a number rounded to 4 decimals.'),
    Variable('decreasing_country', None, 'Store the decreasing country label verbatim from TIC as a string.'),
    Variable('decreasing_country_change_bn', None, 'Store its signed holdings change in USD billions, as a number rounded to 4 decimals.'),
    Variable('issued_accepted_principal_bn', None, 'Store total accepted principal issued in the selected month in USD billions, as a number rounded to 4 decimals.'),
    Variable('selected_short_maturity_share_pct', None, 'Store selected-month accepted principal with issue-to-maturity distance at most 366 days, as a percent of all accepted principal, as a number rounded to 4 decimals.'),
    Variable('previous_short_maturity_share_pct', None, 'Store the corresponding previous-month share in percent, as a number rounded to 4 decimals.'),
]
DECIMALS = [None, 0, 0, 0, 4, 4, None, 4, None, 4, 4, 4, 4]

def analysis():
    holdings = load_expansion_table('treasury_tic')
    debt = load_expansion_table('treasury_dts_public_debt')
    auctions = load_expansion_table('treasury_auctions')
    if holdings.duplicated(['country_name', 'observation_month']).any():
        raise ValueError('Duplicate TIC country/month keys')
    totals = holdings[holdings.country_name.eq('Grand Total')].set_index('observation_month').treasury_holdings_usd_billions
    changes = {str(m): Decimal(str(totals[str(m)])) - Decimal(str(totals[str(m-1)]))
               for m in pd.period_range('2024-01', '2024-12', freq='M')}
    month = min(changes, key=lambda m: (-changes[m], m))
    prior = str(pd.Period(month, freq='M') - 1)
    countries = holdings[holdings.row_type.eq('country') & holdings.observation_month.isin([prior, month])]
    paired = countries.pivot(index='country_name', columns='observation_month', values='treasury_holdings_usd_billions').dropna()
    moves = {name: Decimal(str(row[month])) - Decimal(str(row[prior])) for name, row in paired.iterrows()}
    up = min(moves, key=lambda n: (-moves[n], n))
    down = min(moves, key=lambda n: (moves[n], n))
    daily = debt[debt.record_date.str.startswith(month) & debt.security_market.eq('Marketable')]
    net = (_sum(daily.loc[daily.transaction_type.eq('Issues'), 'transaction_today_amount_million_usd'])
           - _sum(daily.loc[daily.transaction_type.eq('Redemptions'), 'transaction_today_amount_million_usd'])) / 1000
    if net == 0:
        raise ValueError('Undefined holdings/net-issuance ratio')
    def maturity_mix(period):
        issues = auctions[auctions.issue_date.str.startswith(period)]
        if issues.empty or issues.duplicated(['cusip', 'auction_date']).any():
            raise ValueError('Missing or duplicated auction issues')
        days = (pd.to_datetime(issues.maturity_date) - pd.to_datetime(issues.issue_date)).dt.days
        if days.isna().any() or days.le(0).any() or not np.isfinite(issues.total_accepted).all():
            raise ValueError('Invalid auction dates or accepted amounts')
        total = _sum(issues.total_accepted)
        return float(total / 10**9), float(100 * _sum(issues.loc[days.le(366), 'total_accepted']) / total)
    principal, share = maturity_mix(month)
    _, previous_share = maturity_mix(prior)
    return dict(selected_month=month, increasing_country_count=sum(v > 0 for v in moves.values()),
        decreasing_country_count=sum(v < 0 for v in moves.values()),
        debt_transaction_day_count=int(daily.record_date.nunique()), net_marketable_issues_bn=float(net),
        holdings_to_net_issuance_pct=float(100 * changes[month] / net),
        increasing_country=str(up), increasing_country_change_bn=float(moves[up]),
        decreasing_country=str(down), decreasing_country_change_bn=float(moves[down]),
        issued_accepted_principal_bn=principal, selected_short_maturity_share_pct=share,
        previous_short_maturity_share_pct=previous_share)

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
    "validate_turn_1": turn_validator(validate_turn_1),
    "validate_turn_2": turn_validator(validate_turn_2),
    "validate_turn_3": turn_validator(validate_turn_3),
    "validate_turn_4": turn_validator(validate_turn_4),
}
