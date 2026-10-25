"""The cash flow statement reconciles cash including restricted cash.

Since the 2016 update to the cash flow standard, the three activity totals and
the effect of exchange rates reconcile the change in cash, cash equivalents
and restricted cash together, not the balance-sheet cash line alone. A check
written against that cash line fails for any filer that holds restricted cash,
however clean its statement is.

Across the 3,642 annual reports for 2024 that carry all three activity totals
and a cash balance at both ends of the year, the three totals reproduce the
change in the cash line to within three dollars for 1,162. The other 2,480 are not
all errors. Among the filers that also appear in the fact set covering the
largest filers, reading both balances from the same annual report and adding
the effect of exchange rates, the reconciliation against cash including
restricted cash closes for 354 of 402, and against the cash line alone for 93
of 308. 183 of the 284 filers that report both closing balances hold restricted
cash, and 185 report an effect of exchange rates other than zero.

Eight conventions the query states because each moves an output:

1. The opening balance in the fact-set comparison is the comparative column of
   the same annual report. The prior year's own report can differ where the
   figure was later revised: of the 393 filers carrying both combined opening
   figures, 16 differ, and among those the reconciliation closes for 10 from
   the comparative figure and for 1 from the prior report's own. On the cash
   line alone 13 of 300 differ, and reconciling the cash line from the prior
   report instead closes 88 of 300 rather than 93 of 308.
2. An effect of exchange rates the filer does not report is taken as zero. Of
   the 402 filers checkable on the inclusive balance, 223 report none and 190
   of those close on that reading; dropping them instead leaves 179 checkable
   and 164 closing.
3. A reconciliation closes within three dollars. It rests on five figures,
   three totals and two balances, each rounded to the dollar at worst, so a
   correct statement can miss by up to two and a half. The threshold is not a
   boundary any figure sits on: five first-turn reports miss by exactly one
   dollar and the next smallest miss is six, while in the fact-set comparison
   the smallest miss that is not zero is 27,000 dollars. A threshold of one
   dollar would sit exactly on those five and read 1,162 or 1,157 according
   to whether the comparison is strict.
4. The combined balance and its exchange effect are both the plain lines,
   not the variants that add cash held in disposal groups and discontinued
   operations. 38 filers report the exchange effect only on the variant line,
   all of them nonzero; the plain reading takes their plain effect as zero.
   Taking the effect from whichever line carries it instead reads 374 of 402
   closing on the combined balance, 103 of 308 on the cash line alone and 223
   nonzero effects. Using the variant family throughout, balances and effect
   alike, wherever a filer reports it reads 404 of 430. The plain family is
   the one every filer in the comparison can be checked against the same way.
   The balance variant on its own: 41 of the filers in
   the comparison report that variant and 24 report it without the plain
   total; preferring it wherever it appears gives 426 checkable and 367
   closing rather than 402 and 354, a lower closure rate, so the variant is
   stated out rather than chosen.
   The cash-only balance is likewise the same line the first turn reads.
   Among the filers in the comparison that do not report it, 49 report a
   bank's cash and amounts due from banks and 18 report cash excluding
   equivalents; substituting either would change who is checkable.
5. A report filed jointly under two filers, a parent and its operating
   partnership, is counted once. Three such reports appear in 2024 and two of
   them enter the first-turn population, so counting by filer would read
   3,644 rather than 3,642 and credit the same statement twice.
6. The first-turn opening balance comes from the filer's own report for a
   fiscal year ending between 350 and 380 days earlier, which leaves out 11
   filers whose fiscal year moved; admitting them gives 3,653 reports and the
   same 1,162 that close.
7. The second turn returns to every 2024 report, not only the first turn's
   final population. Carrying the first turn's population forward instead
   leaves 277 checkable on the combined balance, 238 closing and 147 nonzero
   exchange effects, rather than 402, 354 and 185.
8. Amendments are outside the population. Admitting them, the later filing
   then winning the tie, reads 3,615 filers and 1,162 falls to 1,161.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_financial_statements
from core.validation import turn_validator, validate_ordered_outputs


WINDOW = ("2024-01-01", "2024-12-31")
PRIOR_WINDOW = ("2023-01-01", "2023-12-31")
YEAR_DAYS = (350, 380)
TOLERANCE_USD = 3.0
CASH = "CashAndCashEquivalentsAtCarryingValue"
FLOWS = (
    "NetCashProvidedByUsedInOperatingActivities_ytd",
    "NetCashProvidedByUsedInInvestingActivities_ytd",
    "NetCashProvidedByUsedInFinancingActivities_ytd",
)
RESTRICTED_INCLUSIVE = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
EXCHANGE_EFFECT = "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"

OUTPUTS = (
    ("statement_population_count", "the count of filers carrying all three totals and both cash balances as an integer"),
    ("statement_closing_count", "how many of them close against the cash line, as an integer"),
    ("inclusive_checkable_count", "how many overlapping filers carry both balances including restricted cash, as an integer"),
    ("inclusive_closing_count", "how many of those close, as an integer"),
    ("cash_only_checkable_count", "how many overlapping filers carry both cash-line balances, as an integer"),
    ("cash_only_closing_count", "how many of those close, as an integer"),
    ("both_closing_balances_count", "how many overlapping filers report both closing balances, as an integer"),
    ("restricted_cash_holder_count", "how many of those hold restricted cash, as an integer"),
    ("nonzero_exchange_effect_count", "how many overlapping filers report an exchange-rate effect other than zero, as an integer"),
    ("opening_comparable_count", "how many overlapping filers carry both opening combined balances, as an integer"),
    ("opening_revised_count", "how many of those differ between the two, as an integer"),
    ("revised_closing_with_comparative_count", "how many revised filers close from the comparative opening, as an integer"),
    ("revised_closing_with_prior_report_count", "how many revised filers close from the prior report's closing balance, as an integer"),
)
variables = [Variable(name, None, f"Store {text}.") for name, text in OUTPUTS]
PRECISION = {name: 0 for name, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _ in OUTPUTS[:2]),
    tuple(n for n, _ in OUTPUTS[2:6]),
    tuple(n for n, _ in OUTPUTS[6:9]),
    tuple(n for n, _ in OUTPUTS[9:]),
)


def _annual(bounds):
    frame = load_financial_statements()
    annual = frame.loc[
        frame.form.astype(str).eq("10-K") & frame.fiscal_period.astype(str).eq("FY")
    ].copy()
    annual["period_end"] = annual.report_date.astype(str)
    annual["cik"] = annual.cik.astype(str)
    annual["accession"] = annual.accession.astype(str)
    annual = annual.loc[annual.period_end.between(*bounds)]
    return annual.sort_values(["cik", "period_end", "filed_date", "accession"]).drop_duplicates(
        "cik", keep="last"
    )


@lru_cache(maxsize=1)
def _statements():
    """Current filings with the prior year's own report attached by filer."""
    current = _annual(WINDOW)
    prior = _annual(PRIOR_WINDOW)[["cik", "period_end", "accession", CASH]].rename(
        columns={"period_end": "prior_end", "accession": "prior_accession", CASH: "prior_cash"}
    )
    joined = current.merge(prior, on="cik", how="left", validate="one_to_one")
    gap = pd.to_datetime(joined.period_end) - pd.to_datetime(joined.prior_end)
    joined["prior_gap_days"] = gap.dt.days
    joined["flows"] = joined[list(FLOWS)].sum(axis=1, min_count=len(FLOWS))
    return joined


@lru_cache(maxsize=1)
def _facts():
    """Balances and the exchange effect read from each current filing itself.

    Returns the 2024 accessions the fact set lists at all, taken before any
    concept is chosen, the rows this case reads from those filings, and each
    filer's prior-report combined closing balance keyed by the 2024 accession.
    """
    statements = _statements()
    facts = load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    listed = frozenset(facts.accession) & frozenset(statements.accession)
    prior_listed = frozenset(statements.prior_accession.dropna())
    facts = facts.loc[
        facts.accession.isin(listed | prior_listed)
        & facts.unit.astype(str).eq("USD")
        & facts.concept.astype(str).isin([CASH, RESTRICTED_INCLUSIVE, EXCHANGE_EFFECT])
    ].copy()
    facts["concept"] = facts.concept.astype(str)
    facts["end_date"] = facts.end_date.astype(str)
    start = pd.to_datetime(facts.start_date, errors="coerce")
    facts["instant"] = start.isna()
    facts["days"] = (pd.to_datetime(facts.end_date) - start).dt.days
    rows = facts.merge(
        statements[["accession", "period_end"]].drop_duplicates("accession"), on="accession"
    )

    attached = statements.loc[
        statements.prior_gap_days.between(*YEAR_DAYS),
        ["accession", "prior_accession", "prior_end"],
    ].drop_duplicates("accession")
    prior_rows = facts.loc[facts.concept.eq(RESTRICTED_INCLUSIVE) & facts.instant].merge(
        attached, left_on=["accession", "end_date"], right_on=["prior_accession", "prior_end"],
        suffixes=("_prior", ""),
    )
    if prior_rows.groupby("accession").value.nunique().gt(1).any():
        raise ValueError("a prior report resolves to more than one combined closing balance")
    prior_combined = prior_rows.drop_duplicates("accession").set_index("accession").value
    return listed, rows, prior_combined


def _pick(concept, when):
    _, facts, _ = _facts()
    rows = facts.loc[facts.concept.eq(concept)]
    if when == "closing":
        rows = rows.loc[rows.instant & rows.end_date.eq(rows.period_end)]
    elif when == "opening":
        before = pd.to_datetime(rows.period_end) - pd.to_datetime(rows.end_date)
        rows = rows.loc[rows.instant & before.dt.days.between(*YEAR_DAYS)]
    else:
        rows = rows.loc[~rows.instant & rows.end_date.eq(rows.period_end) & rows.days.between(*YEAR_DAYS)]
    if rows.groupby("accession").value.nunique().gt(1).any():
        raise ValueError(f"{concept} resolves to more than one {when} value for a filing")
    return rows.drop_duplicates("accession").set_index("accession").value


def _closes(flows, opening, closing, exchange):
    difference = flows + exchange - (closing - opening)
    return difference.abs().le(TOLERANCE_USD)


@lru_cache(maxsize=1)
def analysis():
    statements = _statements()
    population = statements.loc[
        statements.flows.notna()
        & statements[CASH].notna()
        & statements.prior_cash.notna()
        & statements.prior_gap_days.between(*YEAR_DAYS)
    ].drop_duplicates("accession")
    statement_close = _closes(
        population.flows, population.prior_cash, population[CASH], 0.0
    )

    overlap = statements.loc[
        statements.flows.notna() & statements.accession.isin(_facts()[0])
    ].drop_duplicates("accession").set_index("accession")
    exchange = _pick(EXCHANGE_EFFECT, "duration").reindex(overlap.index)
    frame = pd.DataFrame({
        "flows": overlap.flows,
        "exchange": exchange.fillna(0.0),
        "inclusive_open": _pick(RESTRICTED_INCLUSIVE, "opening"),
        "inclusive_close": _pick(RESTRICTED_INCLUSIVE, "closing"),
        "cash_open": _pick(CASH, "opening"),
        "cash_close": _pick(CASH, "closing"),
    }, index=overlap.index)

    inclusive = frame.dropna(subset=["inclusive_open", "inclusive_close"])
    cash_only = frame.dropna(subset=["cash_open", "cash_close"])
    both = frame.dropna(subset=["inclusive_close", "cash_close"])
    frame["prior_combined"] = _facts()[2].reindex(frame.index)
    opening = frame.dropna(subset=["inclusive_open", "inclusive_close", "prior_combined"])
    revised = opening.loc[
        (opening.inclusive_open - opening.prior_combined).abs().gt(TOLERANCE_USD)
    ]

    return {
        "statement_population_count": len(population),
        "statement_closing_count": int(statement_close.sum()),
        "inclusive_checkable_count": len(inclusive),
        "inclusive_closing_count": int(_closes(
            inclusive.flows, inclusive.inclusive_open, inclusive.inclusive_close, inclusive.exchange
        ).sum()),
        "cash_only_checkable_count": len(cash_only),
        "cash_only_closing_count": int(_closes(
            cash_only.flows, cash_only.cash_open, cash_only.cash_close, cash_only.exchange
        ).sum()),
        "both_closing_balances_count": len(both),
        "restricted_cash_holder_count": int(
            (both.inclusive_close - both.cash_close).abs().gt(TOLERANCE_USD).sum()
        ),
        "nonzero_exchange_effect_count": int(exchange.fillna(0.0).ne(0).sum()),
        "opening_comparable_count": len(opening),
        "opening_revised_count": len(revised),
        "revised_closing_with_comparative_count": int(_closes(
            revised.flows, revised.inclusive_open, revised.inclusive_close, revised.exchange
        ).sum()),
        "revised_closing_with_prior_report_count": int(_closes(
            revised.flows, revised.prior_combined, revised.inclusive_close, revised.exchange
        ).sum()),
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


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
