"""Diluted earnings per share is two conventions, not one division.

The denominator is governed by a rule and the numerator is not net income,
and a reader who knows neither can still divide and get a plausible answer.

On the denominator: potential common shares enter the diluted count where
they reduce earnings per share on income from continuing operations. A net-loss
year alone does not establish the result of that control test. Of the 4,257
filers here 2,111 report a loss, and 2,042 of those do set the two counts
equal. 67 report more diluted shares than basic ones despite the loss, which
the rule allows only where income from continuing operations, the figure the
test is applied to, was positive; the statement file carries no after-tax
continuing-operations line, so the case does not call all 67 errors. A
further two report fewer. The same rule read from
the other side says a diluted count can never fall below a basic one, and five
profitable filers report exactly that.

On the numerator: basic earnings per share starts from parent income available
to common after preferred dividends and participating allocations. The diluted
numerator may also adjust for assumed conversions. Dividing reported net income by the
reported diluted share count reproduces the reported diluted figure for most
filers and misses it by two cents or more for 689, with a median miss of
0.1998 USD. The same division on the basic side misses for 625, suggesting a
shared numerator or tagging issue; the counts alone do not isolate a cause.

The fact set covering the largest filers separates the income total struck
before the noncontrolling share from the figure attributed to the parent. 340
of the 4,257 appear there; 201 report the former, and for 166 it differs from
the net income in hand by more than a dollar. That net income is already the
parent figure for all 340, so the noncontrolling share is not what opens the
third turn's gap; the filers whose two totals differ miss by two cents or
more at 29.5 percent against 37.1 percent for the rest. Preferred dividends,
participating allocations, diluted-numerator adjustments and tagging anomalies
are possible contributors; the declared facts do not measure every component. Of
the 625 basic-side misses, 48 sit within 0.01 in log terms of a power of ten
other than one and 31 have the opposite sign, share or per-share figures
tagged at the wrong scale or sign rather than numerator differences.

The excluded securities are disclosed, and they are not small. Among the 158
filers in the overlap disclosing an antidilutive amount, the median is 0.278%
of basic shares, but 39 of them excluded more shares than the entire gap
between their diluted and basic counts. These excluded shares represent
potential dilution outside the current denominator, not assured future entry.

Six things the data forces:

1. Nine filers close two fiscal years inside calendar 2024. Keeping both puts
   the population at 4,264 rather than 4,257 and shifts every count.

   One filer reports a basic count of 4.17 billion shares against a diluted
   count of zero, which no per-share figure can be struck on. Requiring the
   diluted count to be positive as well as the basic one removes it; leaving
   it in makes its recomputed figure infinite, which a floating-point division
   produces silently rather than refusing, and lifts the population to 4,258,
   the loss count to 2,112, the two-cent count to 690 and its median to
   0.2005.
2. ``form`` carries no amendments in this window; all 5,533 annual rows are
   10-K, so restricting to originals removes nothing and the rule is stated
   rather than relied on.
3. The threshold is stated as two cents or more. Reported earnings per share
   is itself rounded to the cent, so the difference from an exact division
   piles up on whole-cent multiples: at one cent a filer sits on the boundary
   to within 2e-16 and the count turns on whether the comparison is strict. At two cents the nearest filer is 3.5e-04 away on the
   diluted side and 7.0e-05 on the basic side, so strict and non-strict
   readings give the same 689 and 625. The case test asserts that.
4. ``CommonStockSharesOutstanding`` is a balance at the period end, not the
   weighted average the per-share figures are struck on. Using it in place of
   the diluted count puts 2,312 of the 3,422 filers that report it two cents
   or more away from their own reported figure, which measures the substitution
   rather than the numerator.
5. Facts in the large-filer set repeat across comparative periods, so a fact is
   taken only where its period ends on the filer's own period end and runs
   between 350 and 380 days. Without that pin every one of the four concepts
   resolves to more than one value per filing.
6. Every median is taken with linear interpolation. The dilution increment and
   the antidilutive share are both even counts whose two middle values differ,
   at 0.9454 against 0.9495 and at 0.2715 against 0.2846. The median miss sits
   on an odd count of 689, so there the rule is stated and does not bind, and
   the case test asserts that.

7. Three comparisons are stated as strict and the ties behind them are not
   small. Four filers disclose an antidilutive amount exactly equal to their
   whole diluted-to-basic gap, so admitting equality would read 43 rather than
   39. 128 of the 163 filers disclosing the share-based increment report it
   equal to that whole gap, meaning share-based payments are the entire
   dilution for them, so admitting equality would read 135 rather than 7. No
   filer's consolidated total sits one dollar from its parent figure: 35 agree
   exactly and the rest differ by at least two, so that threshold reads the
   same either way.

8. The fact set carries only undimensioned facts, so a concept broken out by
   security class does not compete with its own total here. Three competing
   readings of the two income measures were measured rather than assumed away.
   Reading the pre-attribution total as the figure attributable to the parent
   puts 340 filers in that count rather than 201 and leaves every difference
   against the net income already in hand at zero. Reading it as the total
   struck before only the nonredeemable portion is attributed gives 37, of
   which 34 differ; all 37 also report the full pre-attribution figure, so the
   narrower concept adds no filer and only subtracts. And the noncontrolling
   amount is reported undivided by 152 filers while a further 11 report only
   its redeemable or nonredeemable parts, so counting any of the three gives
   163. Redeemable noncontrolling interest sits outside permanent equity,
   which is why the split exists and why the undivided figure is the one
   asked for.

9. A report filed jointly under two filers counts once for each filer. Two
   such reports enter the population; counting each report once instead moves
   the filer count and the four outputs of the third turn.

10. The quotient in the third turn is compared unrounded. Rounding it to the
   cent before the comparison reads 730 and 655 rather than 689 and 625.
11. A disclosed antidilutive amount of zero counts as a disclosure. 26 of the
   158 are zero; counting only positive amounts reads 132 and a median of
   0.3643 percent rather than 158 and 0.2780.

One relation worth naming so it is not mistaken for an oversight: the count of
loss filers is not the sum of the two loss-side counts reported beside it,
because two loss filers report fewer diluted shares than basic ones and that
count is not asked for.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2024-01-01", "2024-12-31")
GAP_THRESHOLD_USD = 0.02
DURATION_DAYS = (350, 380)
NET_INCOME = "NetIncomeLoss_ytd"
BASIC_SHARES = "WeightedAverageNumberOfSharesOutstandingBasic_ytd"
DILUTED_SHARES = "WeightedAverageNumberOfDilutedSharesOutstanding_ytd"
BASIC_EPS = "EarningsPerShareBasic_ytd"
DILUTED_EPS = "EarningsPerShareDiluted_ytd"
REQUIRED = (NET_INCOME, BASIC_SHARES, DILUTED_SHARES, BASIC_EPS, DILUTED_EPS)

CONSOLIDATED = "ProfitLoss"
NONCONTROLLING = "NetIncomeLossAttributableToNoncontrollingInterest"
ANTIDILUTIVE = "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount"
SHARE_BASED_INCREMENT = "IncrementalCommonSharesAttributableToShareBasedPaymentArrangements"

OUTPUTS = (
    ("eps_filer_count", 0, "the count of filers carrying all five per-share fields as an integer"),
    ("loss_filer_count", 0, "how many of them report a loss, as an integer"),
    ("loss_diluted_equals_basic_count", 0, "how many loss filers set the two share counts equal, as an integer"),
    ("loss_diluted_above_basic_count", 0, "how many loss filers report more diluted shares than basic, as an integer"),
    ("profit_diluted_below_basic_count", 0, "how many profitable filers report fewer diluted shares than basic, as an integer"),
    ("profit_diluted_equals_basic_count", 0, "how many profitable filers set the two share counts equal, as an integer"),
    ("diluted_eps_above_basic_eps_count", 0, "how many filers report a diluted per-share figure above the basic one, as an integer"),
    ("median_dilution_increment_pct", 4, "the median increment from basic to diluted shares in percent"),
    ("diluted_numerator_gap_count", 0, "how many filers miss their reported diluted figure by the stated amount, as an integer"),
    ("median_diluted_numerator_gap_usd", 4, "the median size of that miss in USD"),
    ("recomputed_above_reported_count", 0, "how many of those misses land above the reported figure, as an integer"),
    ("basic_numerator_gap_count", 0, "how many filers miss their reported basic figure by the stated amount, as an integer"),
    ("overlap_filer_count", 0, "how many of the population appear in the large-filer fact set, as an integer"),
    ("consolidated_income_reported_count", 0, "how many of those report the income total the query names, as an integer"),
    ("consolidated_differs_count", 0, "how many of those totals differ from the net income figure in hand by more than a dollar, as an integer"),
    ("noncontrolling_reported_count", 0, "how many of the overlap report the undivided noncontrolling amount, as an integer"),
    ("antidilutive_disclosure_count", 0, "how many of the overlap disclose an antidilutive amount, as an integer"),
    ("median_antidilutive_share_of_basic_pct", 4, "the median of that amount as a percentage of basic shares"),
    ("antidilutive_above_increment_count", 0, "how many excluded more shares than their whole diluted-to-basic gap, as an integer"),
    ("share_based_increment_above_total_count", 0, "how many report a share-based increment above that whole gap, as an integer"),
)
variables = [
    Variable(name, None, f"Store {text}" + ("." if p == 0 else f", rounded to {p} decimals from unrounded source rows."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:8]),
    tuple(n for n, _, _ in OUTPUTS[8:12]),
    tuple(n for n, _, _ in OUTPUTS[12:16]),
    tuple(n for n, _, _ in OUTPUTS[16:]),
)


@lru_cache(maxsize=1)
def _population():
    """One annual filing per filer, carrying every per-share field the case needs."""
    frame = load_financial_statements()
    annual = frame.loc[
        frame.form.astype(str).eq("10-K") & frame.fiscal_period.astype(str).eq("FY")
    ].copy()
    annual["period_end"] = annual.report_date.astype(str)
    annual = annual.loc[annual.period_end.between(*WINDOW)]
    annual["cik"] = annual.cik.astype(str)
    annual["accession"] = annual.accession.astype(str)
    annual = annual.sort_values(
        ["cik", "period_end", "filed_date", "accession"]
    ).drop_duplicates("cik", keep="last")

    kept = annual.loc[
        annual[list(REQUIRED)].notna().all(axis=1)
        & annual[BASIC_SHARES].gt(0)
        & annual[DILUTED_SHARES].gt(0)
    ].copy()
    if kept.empty:
        raise ValueError("no annual filing carries every per-share field")
    kept["increment"] = kept[DILUTED_SHARES] - kept[BASIC_SHARES]
    kept["diluted_gap"] = (kept[NET_INCOME] / kept[DILUTED_SHARES] - kept[DILUTED_EPS]).abs()
    kept["basic_gap"] = (kept[NET_INCOME] / kept[BASIC_SHARES] - kept[BASIC_EPS]).abs()
    kept["loss"] = kept[NET_INCOME].lt(0)
    return kept


@lru_cache(maxsize=1)
def _overlap():
    """Population filings that appear in the large-filer fact set at all.

    Taken before any concept is chosen, so the count states how far the two
    sources overlap rather than how many carry the four concepts this case
    happens to read.
    """
    population = _population()
    facts = load_facts()
    present = set(facts.accession.astype(str))
    return set(population.accession) & present


@lru_cache(maxsize=1)
def _facts():
    """Large-filer facts pinned to each filing's own fiscal year.

    A filing carries the same concept for several comparative periods, so the
    period end and the length of the period are both part of the key. Anything
    left resolving to more than one value is a defect rather than a choice.
    """
    population = _population()
    facts = load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    facts = facts.loc[
        facts.accession.isin(set(population.accession)) & facts.form.astype(str).eq("10-K")
    ]
    facts = facts.loc[
        facts.concept.astype(str).isin(
            [CONSOLIDATED, NONCONTROLLING, ANTIDILUTIVE, SHARE_BASED_INCREMENT]
        )
    ].copy()
    facts["end_date"] = facts.end_date.astype(str)
    facts["start_date"] = facts.start_date.astype(str)
    days = pd.to_datetime(facts.end_date, errors="coerce") - pd.to_datetime(
        facts.start_date, errors="coerce"
    )
    facts["days"] = days.dt.days

    anchor = population[["accession", "period_end", BASIC_SHARES, DILUTED_SHARES, NET_INCOME]]
    facts = facts.merge(anchor, on="accession")
    return facts.loc[
        facts.end_date.eq(facts.period_end) & facts.days.between(*DURATION_DAYS)
    ]


def _one_per_filing(concept, unit):
    rows = _facts()
    rows = rows.loc[rows.concept.astype(str).eq(concept) & rows.unit.astype(str).eq(unit)]
    if rows.groupby("accession").value.nunique().gt(1).any():
        raise ValueError(f"{concept} resolves to more than one value for a filing")
    return rows.drop_duplicates("accession")


@lru_cache(maxsize=1)
def analysis():
    population = _population()
    loss = population.loc[population.loss]
    profit = population.loc[~population.loss]
    threshold = GAP_THRESHOLD_USD

    dilutive_profit = profit.loc[profit.increment > 0]
    increment_pct = dilutive_profit.increment / dilutive_profit[BASIC_SHARES] * 100

    diluted_missed = population.loc[population.diluted_gap >= threshold]
    basic_missed = population.loc[population.basic_gap >= threshold]
    above = diluted_missed.loc[
        diluted_missed[NET_INCOME] / diluted_missed[DILUTED_SHARES] > diluted_missed[DILUTED_EPS]
    ]

    consolidated = _one_per_filing(CONSOLIDATED, "USD")
    differs = consolidated.loc[(consolidated.value - consolidated[NET_INCOME]).abs() > 1.0]
    noncontrolling = _one_per_filing(NONCONTROLLING, "USD")

    antidilutive = _one_per_filing(ANTIDILUTIVE, "shares").copy()
    antidilutive["share_pct"] = antidilutive.value / antidilutive[BASIC_SHARES] * 100
    antidilutive["increment"] = antidilutive[DILUTED_SHARES] - antidilutive[BASIC_SHARES]

    share_based = _one_per_filing(SHARE_BASED_INCREMENT, "shares").copy()
    share_based["increment"] = share_based[DILUTED_SHARES] - share_based[BASIC_SHARES]

    return {
        "eps_filer_count": len(population),
        "loss_filer_count": len(loss),
        "loss_diluted_equals_basic_count": int(loss.increment.eq(0).sum()),
        "loss_diluted_above_basic_count": int(loss.increment.gt(0).sum()),
        "profit_diluted_below_basic_count": int(profit.increment.lt(0).sum()),
        "profit_diluted_equals_basic_count": int(profit.increment.eq(0).sum()),
        "diluted_eps_above_basic_eps_count": int(
            population[DILUTED_EPS].gt(population[BASIC_EPS]).sum()
        ),
        "median_dilution_increment_pct": float(increment_pct.median()),
        "diluted_numerator_gap_count": len(diluted_missed),
        "median_diluted_numerator_gap_usd": float(diluted_missed.diluted_gap.median()),
        "recomputed_above_reported_count": len(above),
        "basic_numerator_gap_count": len(basic_missed),
        "overlap_filer_count": len(_overlap()),
        "consolidated_income_reported_count": len(consolidated),
        "consolidated_differs_count": len(differs),
        "noncontrolling_reported_count": len(noncontrolling),
        "antidilutive_disclosure_count": len(antidilutive),
        "median_antidilutive_share_of_basic_pct": float(antidilutive.share_pct.median()),
        "antidilutive_above_increment_count": int(
            antidilutive.value.gt(antidilutive.increment).sum()
        ),
        "share_based_increment_above_total_count": int(
            share_based.value.gt(share_based.increment).sum()
        ),
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


def validate_turn_5(outputs):
    return _check(outputs, TURNS[4])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 6)
}
