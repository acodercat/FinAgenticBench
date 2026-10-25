"""Cash-flow bridge and constrained distributions.
Measured alternatives: filtering to positive gaps gives 19 instead of 60 issuers;
carrying historical other cash movements gives 10.929 instead of 4.054 billion.
Ignoring the cash floor retains 1.795 instead of 0.435 billion in repurchases.
Quant Method Gate: ordinary accounting and a linear cash constraint; not applicable."""
from functools import lru_cache
from decimal import Decimal
import pandas as pd
from cave_agent import Variable
from core.data import load_facts, load_filings
from core.validation import turn_validator, validate_ordered_outputs

variables = [
    Variable('eligible_filer_count', None, 'Store the eligible issuer count as an integer.'),
    Variable('selected_issuer_cik', None, 'Store the selected CIK as an exact zero-padded 10-digit string.'),
    Variable('selected_filing_accession', None, 'Store the selected accession as its exact hyphenated source string.'),
    Variable('selected_acceptance_timestamp', None, 'Store the source-recorded acceptance timestamp as an exact YYYY-MM-DDTHH:MM:SS string, preserving its recorded clock.'),
    Variable('payout_cash_shortfall_bn', None, 'Store the signed distribution shortfall in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('closing_cash_bn', None, 'Store the reported closing cash-and-equivalents balance in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('other_investing_inflow_bn', None, 'Store net investing inflow excluding PP&E purchases in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('other_financing_inflow_bn', None, 'Store net financing inflow excluding dividend and common-stock repurchase payments in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('cash_change_residual_bn', None, 'Store cash-balance change less the three reported cash-flow totals in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('full_plan_stressed_closing_cash_bn', None, 'Store projected ending cash under the unchanged distribution plan in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('maximum_retained_repurchases_bn', None, 'Store the maximum repurchase amount permitted by the scenario in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('minimum_operating_cash_full_plan_bn', None, 'Store the minimum operating cash flow supporting the original distribution plan and cash floor in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.'),
]
DECIMALS = [0, None, None, None, 4, 4, 4, 4, 4, 4, 4, 4]
TURN_1_NAMES = ['eligible_filer_count', 'selected_issuer_cik', 'selected_filing_accession', 'selected_acceptance_timestamp', 'payout_cash_shortfall_bn']
TURN_2_NAMES = ['closing_cash_bn', 'other_investing_inflow_bn', 'other_financing_inflow_bn', 'cash_change_residual_bn']
TURN_3_NAMES = ['full_plan_stressed_closing_cash_bn']
TURN_4_NAMES = ['maximum_retained_repurchases_bn', 'minimum_operating_cash_full_plan_bn']

CFO = 'NetCashProvidedByUsedInOperatingActivities'
CFI = 'NetCashProvidedByUsedInInvestingActivities'
CFF = 'NetCashProvidedByUsedInFinancingActivities'
CAP = 'PaymentsToAcquirePropertyPlantAndEquipment'
REP = 'PaymentsForRepurchaseOfCommonStock'
DIV = 'PaymentsOfDividends'
CASH = 'CashAndCashEquivalentsAtCarryingValue'


def retained_repurchases(opening, operating, capex, dividends, plan, floor):
    capacity = opening + operating - capex - dividends - floor
    if capacity < 0:
        raise ValueError('Cash floor infeasible even with zero repurchases')
    return min(plan, capacity)


def analysis():
    filings = load_filings()
    filings = filings[filings.form.eq('10-K') & filings.report_date.between('2024-01-01', '2024-12-31')
                      & filings.acceptance_datetime.le('2025-12-31T23:59:59')]
    filings = filings.sort_values(['acceptance_datetime', 'accession']).drop_duplicates('cik', keep='last')
    facts = load_facts()
    facts = facts[facts.concept.isin([CFO, CFI, CFF, CAP, REP, DIV, CASH])
                  & facts.taxonomy.eq('us-gaap') & facts.unit.eq('USD')]
    facts = facts.merge(filings[['cik', 'accession', 'report_date', 'acceptance_datetime']],
                        on=['cik', 'accession'], validate='many_to_one')
    span = (pd.to_datetime(facts.end_date) - pd.to_datetime(facts.start_date)).dt.days
    annual = facts[facts.end_date.eq(facts.report_date) & span.between(350, 380)]
    if (annual.groupby(['cik', 'accession', 'concept']).value.nunique().gt(1).any()
            or annual.groupby(['cik', 'accession']).start_date.nunique().gt(1).any()):
        raise ValueError('Conflicting annual facts or fiscal windows')
    cohort = annual.pivot_table(index=['cik', 'accession', 'report_date'], columns='concept', values='value', aggfunc='last')
    cohort = cohort.dropna(subset=[CFO, CAP, REP, DIV])
    cohort = cohort[cohort[CFO].gt(0) & cohort[[CAP, REP, DIV]].ge(0).all(axis=1)].copy()
    cohort['gap'] = cohort[CAP] + cohort[REP] + cohort[DIV] - cohort[CFO]
    cohort = cohort.reset_index().sort_values(['gap', 'cik'], ascending=[False, True])
    if cohort.empty:
        raise ValueError('No eligible issuers')
    selected = cohort.iloc[0]
    rows = facts[facts.cik.eq(selected.cik) & facts.accession.eq(selected.accession)]
    selected_annual = annual[annual.cik.eq(selected.cik) & annual.accession.eq(selected.accession)]
    start = selected_annual.start_date.iloc[0]
    opening_date = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    def amount(concept, end=None):
        source = rows if concept == CASH else selected_annual
        values = source[source.concept.eq(concept) & source.end_date.eq(end or selected.report_date)].value.dropna().unique()
        if len(values) != 1:
            raise ValueError(f'Missing or conflicting {concept}')
        result = Decimal(str(values[0]))
        if not result.is_finite():
            raise ValueError('Nonfinite monetary fact')
        return result

    opening, closing = amount(CASH, opening_date), amount(CASH)
    operating, investing, financing = amount(CFO), amount(CFI), amount(CFF)
    capex, repurchases, dividends = amount(CAP), amount(REP), amount(DIV)
    stressed = operating * Decimal('0.8')
    floor = closing * Decimal('0.5')
    bn = lambda value: float(value / Decimal(10**9))
    return dict(
        eligible_filer_count=len(cohort), selected_issuer_cik=str(selected.cik).zfill(10),
        selected_filing_accession=str(selected.accession),
        selected_acceptance_timestamp=str(rows.acceptance_datetime.iloc[0]),
        payout_cash_shortfall_bn=bn(capex + repurchases + dividends - operating),
        closing_cash_bn=bn(closing), other_investing_inflow_bn=bn(investing + capex),
        other_financing_inflow_bn=bn(financing + repurchases + dividends),
        cash_change_residual_bn=bn(closing - opening - operating - investing - financing),
        full_plan_stressed_closing_cash_bn=bn(closing + stressed - capex - dividends - repurchases),
        maximum_retained_repurchases_bn=bn(retained_repurchases(closing, stressed, capex, dividends, repurchases, floor)),
        minimum_operating_cash_full_plan_bn=bn(floor + capex + dividends + repurchases - closing))

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
