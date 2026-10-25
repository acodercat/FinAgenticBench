"""Indexed Treasury invoices and discrete equal-face allocation.
Measured alternatives at USD 1 million face: double indexation costs 3054494.77;
unadjusted clean price costs 3017292.27; omitting accrual costs 3023988.71.
Published adjusted inputs cost 3035759.10. Reconstructing adjusted prices agrees
to cents at that face only; do not assume equivalence at other face amounts.
Quant Method Gate: settlement arithmetic and integer lot feasibility; not applicable."""
from functools import lru_cache
from decimal import Decimal, ROUND_HALF_UP
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs

variables = [
    Variable('eligible_security_count', None, 'Store the count of eligible distinct CUSIPs as an integer.'),
    Variable('selected_tips_cusip', None, 'Store the selected CUSIP as its exact nine-character source string.'),
    Variable('reported_outstanding_mn', None, 'Store the published CUSIP-level outstanding balance in USD millions as a finite number at its source precision, rounded to 8 decimals.'),
    Variable('reopening_count', None, 'Store the number of reopening auctions as an integer.'),
    Variable('original_settlement_date', None, 'Store the original issue date as an ISO YYYY-MM-DD string.'),
    Variable('last_settlement_date', None, 'Store the final settlement date as an ISO YYYY-MM-DD string.'),
    Variable('total_settlement_cash_usd', None, 'Store the sum of cent-settled purchase invoices in USD as a finite number, rounded to 4 decimals; invoice amounts use the stipulated cent settlement convention.'),
    Variable('largest_invoice_date', None, 'Store the settlement date with the largest invoice as an ISO YYYY-MM-DD string.'),
    Variable('settlement_inflation_uplift_usd', None, 'Store the sum of inflation-adjusted principal less original principal at each respective settlement date in USD as a finite number, rounded to 4 decimals.'),
    Variable('common_face_usd', None, 'Store the maximum common original-face amount per auction in USD as an integer.'),
    Variable('unused_reserve_usd', None, 'Store the reserve remaining after all cent-settled invoices in USD as a finite number, rounded to 4 decimals; invoice amounts use the stipulated cent settlement convention.'),
]
DECIMALS = [0, None, 8, 0, None, None, 4, None, 4, 0, 4]
TURN_1_NAMES = ['eligible_security_count', 'selected_tips_cusip', 'reported_outstanding_mn']
TURN_2_NAMES = ['reopening_count', 'original_settlement_date', 'last_settlement_date']
TURN_3_NAMES = ['total_settlement_cash_usd', 'largest_invoice_date', 'settlement_inflation_uplift_usd']
TURN_4_NAMES = ['common_face_usd', 'unused_reserve_usd']

def invoice(face, coefficient):
    return (Decimal(face) * coefficient).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def maximum_common_face(coefficients, minimums, increments, budget):
    # Treasury bid sizes are positive integer-dollar multiples, subject to minima.
    from math import lcm
    lot = lcm(*increments)
    lower = max((n + lot - 1) // lot for n in minimums)
    cost = lambda n: sum((invoice(n * lot, c) for c in coefficients), Decimal(0))
    if lower < 1 or not coefficients or any(c <= 0 for c in coefficients):
        raise ValueError('Invalid auction terms')
    if cost(lower) > budget:
        raise ValueError('Reserve cannot fund minimum common face')
    upper = lower + 1
    while cost(upper) <= budget:
        upper *= 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if cost(middle) <= budget:
            lower = middle
        else:
            upper = middle
    return lower * lot, budget - cost(lower)


def analysis():
    auctions = load_expansion_table('treasury_auctions')
    stocks = load_expansion_table('treasury_marketable_securities')
    auctions = auctions[auctions.inflation_index_security.eq('Yes')
                        & auctions.issue_date.between('2025-01-01', '2025-12-31')]
    stocks = stocks[stocks.record_date.eq('2025-12-31')
                    & stocks.security_class.eq('Inflation-Protected Securities')
                    & stocks.security_identifier_or_total_label.isin(auctions.cusip)]
    grouped = stocks.groupby('security_identifier_or_total_label').outstanding_million_usd
    if not grouped.count().eq(1).all():
        raise ValueError('Missing or duplicate CUSIP outstanding balance')
    ranked = grouped.sum()
    ranked = ranked[ranked.gt(0)].sort_index().sort_values(ascending=False, kind='stable')
    if ranked.empty:
        raise ValueError('No eligible securities')
    cusip = ranked.index[0]
    lineage = auctions[auctions.cusip.eq(cusip)].sort_values('issue_date')
    original = lineage[lineage.reopening.eq('No')]
    reopenings = lineage[lineage.reopening.eq('Yes')]
    if (len(original) != 1 or not lineage.issue_date.is_unique
            or len(original) + len(reopenings) != len(lineage)
            or not reopenings.original_issue_date.eq(original.iloc[0].issue_date).all()):
        raise ValueError('Invalid original/reopening lineage')
    dec = lambda value: Decimal(str(value))
    coefficients = [dec(row.adj_price) / 100 + dec(row.adj_accrued_int_per1000) / 1000
                    for row in lineage.itertuples()]
    amounts = [invoice(1000000, c) for c in coefficients]
    def integral_terms(values):
        if any(not dec(v).is_finite() or dec(v) <= 0 or dec(v) != int(v) for v in values):
            raise ValueError('Invalid minimum/increment')
        return [int(v) for v in values]
    face, reserve = maximum_common_face(coefficients, integral_terms(lineage.min_bid_amt),
                                        integral_terms(lineage.multiples_to_bid), dec(3000000))
    return dict(eligible_security_count=len(ranked), selected_tips_cusip=str(cusip),
        reported_outstanding_mn=float(ranked.iloc[0]), reopening_count=len(reopenings),
        original_settlement_date=str(original.iloc[0].issue_date),
        last_settlement_date=str(lineage.iloc[-1].issue_date),
        total_settlement_cash_usd=float(sum(amounts)),
        largest_invoice_date=str(lineage.iloc[max(range(len(amounts)), key=amounts.__getitem__)].issue_date),
        settlement_inflation_uplift_usd=float(sum(dec(1000000) * (dec(v) - 1) for v in lineage.index_ratio_on_issue_date)),
        common_face_usd=face, unused_reserve_usd=float(reserve))

@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


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
