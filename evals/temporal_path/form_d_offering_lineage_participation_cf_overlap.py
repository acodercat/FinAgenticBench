"""Annual Form D offering lineages, missing participation and same-issuer Form C context.
Baseline: public cutoff and lineage rules exclude notice-level double counting.
Investor counts missing in 3147 selected-industry offerings are not zero.
Offering ceilings in Form C are not observed proceeds.
Numeric fields use shared display tolerances; source-verbatim strings are exact.
The disclosure cutoff is 2024-12-31, which the query states as a date: "by year-end"
also read as the end of the data, 2025-12-31, and that cutoff gives 16978 offerings
from 15226 issuers with 197.5581 USD billions sold, against 17001, 15230 and 176.2729.
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

TURN_1_NAMES = ['offering_count', 'issuer_count', 'reported_sold_bn']
TURN_2_NAMES = ['selected_industry', 'finite_sold_to_ceiling_pct', 'indefinite_offering_share_pct']
TURN_3_NAMES = ['positive_participation_count', 'zero_participation_count', 'missing_participation_count', 'positive_group_median_sold_usd', 'zero_group_median_sold_usd']
TURN_4_NAMES = ['matched_issuer_count', 'matched_cf_offering_count', 'median_issuer_cross_regime_ratio']

variables = [
    Variable('offering_count', None, 'Store the number of retained offering lineages as an integer.'),
    Variable('issuer_count', None, 'Store the number of distinct SEC issuer identifiers as an integer.'),
    Variable('reported_sold_bn', None, 'Store aggregate selected-notice reported sold amount in USD billions, as a number rounded to 4 decimals.'),
    Variable('selected_industry', None, 'Store the selected industry label verbatim from Form D as a string.'),
    Variable('finite_sold_to_ceiling_pct', None, 'Store aggregate sold divided by aggregate finite positive offering ceilings, in percent, as a number rounded to 4 decimals.'),
    Variable('indefinite_offering_share_pct', None, 'Store Indefinite-ceiling offerings as a percentage of all selected-industry cohort offerings, as a number rounded to 4 decimals.'),
    Variable('positive_participation_count', None, 'Store offerings reporting a positive non-accredited investor count as an integer.'),
    Variable('zero_participation_count', None, 'Store offerings explicitly reporting zero non-accredited investors as an integer.'),
    Variable('missing_participation_count', None, 'Store offerings without a reported non-accredited investor count as an integer.'),
    Variable('positive_group_median_sold_usd', None, 'Store median sold amount among the positive-count group in USD, as a number rounded to 4 decimals.'),
    Variable('zero_group_median_sold_usd', None, 'Store median sold amount among the explicit-zero group in USD, as a number rounded to 4 decimals.'),
    Variable('matched_issuer_count', None, 'Store the distinct matched issuer count as an integer.'),
    Variable('matched_cf_offering_count', None, 'Store the retained issuer-and-file-number crowdfunding offering count as an integer.'),
    Variable('median_issuer_cross_regime_ratio', None, 'Store the median of issuer-level selected-industry cohort Form D reported sold amounts divided by summed Form C maximum amounts, as a dimensionless ratio, as a number rounded to 4 decimals.'),
]
DECIMALS = [0, 0, 4, None, 4, 4, 0, 0, 0, 4, 4, 0, 0, 4]

def offering_endpoints(filings):
    """Resolve complete, acyclic same-issuer chains to 2024 original notices."""
    frame = filings[filings.FILING_DATE.le('2024-12-31')].copy()
    if frame.ACCESSIONNUMBER.duplicated().any():
        raise ValueError('Duplicate Form D accessions')
    rows = frame.set_index('ACCESSIONNUMBER').to_dict('index')
    resolved = {}
    for accession in rows:
        current, path, seen = accession, [], set()
        while current not in resolved:
            if current not in rows or current in seen:
                root = None
                break
            seen.add(current)
            path.append(current)
            row = rows[current]
            if pd.isna(row['CIK']):
                root = None
                break
            if row['SUBMISSIONTYPE'] == 'D' and not row['ISAMENDMENT']:
                root = current
                break
            parent = row['PREVIOUSACCESSIONNUMBER']
            if pd.isna(parent) or parent not in rows or rows[parent]['CIK'] != row['CIK']:
                root = None
                break
            current = parent
        else:
            root = resolved[current]
        for node in path:
            resolved[node] = root
    frame['root'] = frame.ACCESSIONNUMBER.map(resolved)
    roots = {a for a, r in rows.items() if r['SUBMISSIONTYPE'] == 'D' and not r['ISAMENDMENT']
             and r['FILING_DATE'].startswith('2024') and resolved.get(a) == a}
    linked = frame[frame.root.isin(roots)]
    return linked.sort_values(['FILING_DATE', 'ACCESSIONNUMBER']).drop_duplicates('root', keep='last')


def analysis():
    endpoints = offering_endpoints(load_expansion_table('sec_form_d'))
    cohort = endpoints[endpoints.INDUSTRYGROUPTYPE.notna()
        & endpoints.INDUSTRYGROUPTYPE.ne('Pooled Investment Fund')
        & endpoints.TOTALAMOUNTSOLD.ge(0)].copy()
    cohort['cap'] = pd.to_numeric(cohort.TOTALOFFERINGAMOUNT, errors='coerce')
    finite = cohort[cohort.cap.gt(0) & np.isfinite(cohort.cap)]
    industries = []
    for industry, group in finite.groupby('INDUSTRYGROUPTYPE'):
        if len(group) >= 100:
            industries.append((industry, _sum(group.TOTALAMOUNTSOLD), _sum(group.cap)))
    industry, sold, ceiling = min(industries, key=lambda r: (-r[1], r[0]))
    selected = cohort[cohort.INDUSTRYGROUPTYPE.eq(industry)]
    positive = selected[selected.NUMBERNONACCREDITEDINVESTORS.gt(0)]
    zero = selected[selected.NUMBERNONACCREDITEDINVESTORS.eq(0)]
    missing = selected[selected.NUMBERNONACCREDITEDINVESTORS.isna()]
    if len(positive) + len(zero) + len(missing) != len(selected):
        raise ValueError('Unclassified investor counts')
    crowdfunding = load_expansion_table('sec_form_c')
    originals = crowdfunding[crowdfunding.SUBMISSION_TYPE.eq('C')
        & crowdfunding.FILING_DATE.between('2024-01-01', '2024-12-31')
        & crowdfunding.CIK.notna() & crowdfunding.FILE_NUMBER.notna()]
    first = originals.sort_values(['FILING_DATE', 'ACCESSION_NUMBER']).drop_duplicates(['CIK', 'FILE_NUMBER'], keep='first')
    retained = first[first.MAXIMUMOFFERINGAMOUNT.gt(0) & np.isfinite(first.MAXIMUMOFFERINGAMOUNT)
        & first.CIK.isin(selected.CIK)]
    c_max = retained.groupby('CIK').MAXIMUMOFFERINGAMOUNT.sum()
    d_sold = selected.groupby('CIK').TOTALAMOUNTSOLD.sum()
    ratios = d_sold.loc[c_max.index] / c_max
    if ratios.empty or positive.empty or zero.empty:
        raise ValueError('Undefined median in frozen cohort')
    return dict(offering_count=int(len(cohort)), issuer_count=int(cohort.CIK.nunique()),
        reported_sold_bn=float(_sum(cohort.TOTALAMOUNTSOLD) / 10**9), selected_industry=str(industry),
        finite_sold_to_ceiling_pct=float(100 * sold / ceiling),
        indefinite_offering_share_pct=float(100 * Decimal(int(selected.TOTALOFFERINGAMOUNT.eq('Indefinite').sum())) / len(selected)),
        positive_participation_count=int(len(positive)), zero_participation_count=int(len(zero)),
        missing_participation_count=int(len(missing)), positive_group_median_sold_usd=float(positive.TOTALAMOUNTSOLD.median()),
        zero_group_median_sold_usd=float(zero.TOTALAMOUNTSOLD.median()),
        matched_issuer_count=int(len(c_max)), matched_cf_offering_count=int(len(retained)),
        median_issuer_cross_regime_ratio=float(ratios.median()))

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
