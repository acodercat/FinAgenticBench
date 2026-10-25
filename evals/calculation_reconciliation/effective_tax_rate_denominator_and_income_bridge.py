"""Compare effective tax rates on a defined positive-pretax cohort.

Tax expense over pretax income is the standard measure for this cohort. Of the
5,524 annual filings in this window,
3,515 report pretax income from continuing operations, income tax expense and
net income together, and only 1,995 of those earned a positive pretax figure.
The other 1,520 lost money before tax. Their ratios can be defined, but their
signs and comparability need loss-year interpretation rather than inclusion in
the requested positive-pretax median.

On the 1,995 that can carry it the median rate is 21.2760%, which is where the
21% federal statutory rate sits, but the distribution is not tight around it.
139 of them report a tax benefit rather than an expense against positive pretax
income, so their rate is negative, and 32 pay more than their whole pretax
income. 965 sit below the statutory rate. Weighting by size rather than by
filer gives 18.1859%, three percentage points under the median, because the
largest profitable filers pay less than the typical one.

The three figures do not close. Pretax income less tax expense equals net
income for 2,405 of the 3,515 and not for the other 1,110, whose residuals sum
to 17.190644 USD billions. The pretax tag is continuing income before
noncontrolling-interest attribution: it includes the minority share but excludes
discontinued operations. The gap need not be a filing error. The
largest gap is Walgreens Boots Alliance at -6.829000 USD billions, a year it
reported a 14.2 billion pretax loss and an 8.6 billion net loss.

The residual has a name, and the fact file can put one on it. Of the 1,110
filers with a residual, 140 appear in the separate fact file with an annual
figure for discontinued operations or for income attributable to
noncontrolling interests. For 129 of those the residual is exactly
noncontrolling interest less discontinued operations, to the dollar. The
eleven that do not close may carry additional income or attribution items that
these two concepts do not measure.

Five conventions the data forces:

1. The rate is struck only on filers whose pretax income is above zero. No
   filer here reports exactly zero, so the boundary is strict or inclusive to
   the same effect, and the case test says so.
2. The median is taken with linear interpolation. The 1,995 rates are an odd
   count so nothing turns on it today, and the rule is stated because a later
   file need not be odd.
3. The weighted rate divides summed tax by summed pretax rather than averaging
   the filer-level rates, which would let a filer with a tiny denominator
   dominate.
4. A fact counts as annual when its period runs 300 to 430 days to the filer's
   own period end, and a filer reporting the same concept twice for that year
   is taken as first filed. A filer reporting only one of the two concepts is
   read as zero on the other for this two-concept screen; absence of a fact
   alone does not certify the economic amount is zero.
5. Residuals are measured against a one dollar threshold, which changes
   nothing here: every residual is a whole number of dollars, so the same
   1,110 filers fail exact equality.

Two readings the query closes. Each of the three concepts appears twice in
this table, once for the fiscal year and once for a single quarter, and the
quarterly columns are sparse on an annual filing: taking all three of them
leaves 46 filers rather than 3,515. And no filer's rate lands exactly on the
statutory 21%, the nearest being 20.9950 and 20.9917, so counting below it
strictly or inclusively gives the same 965 either way.
"""

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2024-01-01", "2024-12-31")
STATUTORY_PCT = 21.0
CLOSE_USD = Decimal(1)
ANNUAL_FACT_DAYS = (300, 430)
PRETAX = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest_ytd"
TAX = "IncomeTaxExpenseBenefit_ytd"
NET = "NetIncomeLoss_ytd"
DISCONTINUED = "IncomeLossFromDiscontinuedOperationsNetOfTax"
MINORITY = "NetIncomeLossAttributableToNoncontrollingInterest"
BILLION = Decimal(10) ** 9

OUTPUTS = (
    ("annual_filer_count", 0, "the count of filers in the annual population as an integer"),
    ("complete_filer_count", 0, "how many report all three income figures, as an integer"),
    ("positive_pretax_count", 0, "how many of those earned pretax income above zero, as an integer"),
    ("median_effective_tax_rate_pct", 4, "the median rate across those filers, in percent"),
    ("tax_benefit_count", 0, "how many of them report a benefit rather than an expense, as an integer"),
    ("above_hundred_count", 0, "how many pay more than their pretax income, as an integer"),
    ("below_statutory_count", 0, "how many sit below the federal statutory rate, as an integer"),
    ("aggregate_effective_tax_rate_pct", 4, "their summed tax over their summed pretax income, in percent"),
    ("residual_filer_count", 0, "how many filers the three figures fail to close for, as an integer"),
    ("residual_total_bn", 6, "their residuals summed, in USD billions"),
    ("largest_residual_cik", None, "the filer with the largest residual in absolute terms, by its source-verbatim CIK"),
    ("largest_residual_bn", 6, "that filer's residual in USD billions"),
    ("facts_matched_count", 0, "how many of those filers the fact file covers, as an integer"),
    ("decomposition_holds_count", 0, "for how many of them the residual is fully named, as an integer"),
    ("minority_interest_total_bn", 6, "the matched filers' income attributable to minority holders, summed, in USD billions"),
    ("discontinued_total_bn", 6, "their discontinued-operations income summed, in USD billions"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:8]),
    tuple(n for n, _, _ in OUTPUTS[8:12]),
    tuple(n for n, _, _ in OUTPUTS[12:]),
)


@lru_cache(maxsize=1)
def _annual():
    statements = load_financial_statements()
    frame = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
    ].copy()
    frame["period_end"] = frame.report_date.astype(str)
    frame = frame.loc[frame.period_end.between(*WINDOW)]
    if frame.empty:
        raise ValueError("no annual filing has a period end in the window")
    ordered = frame.sort_values(["cik", "period_end", "filed_date", "accession"])
    kept = ordered.drop_duplicates("cik", keep="last").copy()
    kept["cik"] = kept.cik.astype(str)
    return kept


@lru_cache(maxsize=1)
def _complete():
    frame = _annual().dropna(subset=[PRETAX, TAX, NET]).copy()
    if frame.empty:
        raise ValueError("no filer reports all three income figures")
    frame["residual"] = frame[PRETAX] - frame[TAX] - frame[NET]
    return frame


@lru_cache(maxsize=1)
def _named_components():
    """One annual figure per filer and concept, as that year was first filed."""
    residual = _complete()
    residual = residual.loc[residual.residual.abs() >= float(CLOSE_USD)]
    facts = load_facts()
    facts = facts.loc[
        facts.form.eq("10-K")
        & facts.unit.eq("USD")
        & facts.concept.astype(str).isin([DISCONTINUED, MINORITY])
    ].copy()
    facts["cik"] = facts.cik.astype(str)
    facts["concept"] = facts.concept.astype(str)
    facts["fact_end"] = facts.end_date.astype(str)
    facts["fact_days"] = (
        pd.to_datetime(facts.fact_end) - pd.to_datetime(facts.start_date.astype(str))
    ).dt.days
    facts = facts.loc[facts.fact_days.between(*ANNUAL_FACT_DAYS)]
    ordered = facts.sort_values(["cik", "concept", "fact_end", "filed_date", "accession"])
    ordered = ordered.drop_duplicates(["cik", "concept", "fact_end"], keep="first")
    wide = ordered.pivot_table(
        index=["cik", "fact_end"], columns="concept", values="value", aggfunc="first"
    ).reset_index()
    matched = residual[["cik", "period_end", "residual"]].merge(
        wide, left_on=["cik", "period_end"], right_on=["cik", "fact_end"]
    )
    if matched.empty:
        raise ValueError("no residual filer appears in the fact file")
    # A filer reporting only one of the two concepts reports nothing on the other.
    for column in (DISCONTINUED, MINORITY):
        if column not in matched:
            matched[column] = 0.0
        matched[column] = matched[column].fillna(0.0)
    matched["named"] = matched[MINORITY] - matched[DISCONTINUED]
    return matched


def _sum_bn(series):
    return float(sum((Decimal(str(x)) for x in series), Decimal(0)) / BILLION)


@lru_cache(maxsize=1)
def analysis():
    annual = _annual()
    complete = _complete()
    positive = complete.loc[complete[PRETAX] > 0].copy()
    if positive.empty:
        raise ValueError("no filer reports positive pretax income")
    positive["rate"] = positive[TAX] / positive[PRETAX] * 100

    residual = complete.loc[complete.residual.abs() >= float(CLOSE_USD)]
    if residual.empty:
        raise ValueError("the three figures close for every filer")
    # The query breaks a tie on this ranking by the numerically lower CIK; the
    # case test asserts that nothing currently ties.
    ranked = residual.assign(
        magnitude=residual.residual.abs(), identifier=residual.cik.astype(int)
    ).sort_values(["magnitude", "identifier"], ascending=[False, True])
    leader = ranked.iloc[0]

    matched = _named_components()
    closes = (matched.residual - matched.named).abs() < float(CLOSE_USD)

    return {
        "annual_filer_count": len(annual),
        "complete_filer_count": len(complete),
        "positive_pretax_count": len(positive),
        "median_effective_tax_rate_pct": float(positive.rate.median()),
        "tax_benefit_count": int((positive[TAX] < 0).sum()),
        "above_hundred_count": int((positive.rate > 100).sum()),
        "below_statutory_count": int((positive.rate < STATUTORY_PCT).sum()),
        "aggregate_effective_tax_rate_pct": float(
            sum((Decimal(str(x)) for x in positive[TAX]), Decimal(0))
            / sum((Decimal(str(x)) for x in positive[PRETAX]), Decimal(0))
            * 100
        ),
        "residual_filer_count": len(residual),
        "residual_total_bn": _sum_bn(residual.residual),
        "largest_residual_cik": str(leader.cik),
        "largest_residual_bn": float(Decimal(str(leader.residual)) / BILLION),
        "facts_matched_count": len(matched),
        "decomposition_holds_count": int(closes.sum()),
        "minority_interest_total_bn": _sum_bn(matched[MINORITY]),
        "discontinued_total_bn": _sum_bn(matched[DISCONTINUED]),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
