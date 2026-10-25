"""Boundaries for the cash flow statement restricted-cash closure case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import pandas as pd
import pytest

from evals.calculation_reconciliation import cash_flow_statement_restricted_cash_closure as case


def _clear():
    for cached in (case._statements, case._facts, case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    return dict(case.analysis())


def _comparison():
    """The fact-set comparison frame as the case builds it, before any count."""
    statements = case._statements()
    overlap = statements.loc[
        statements.flows.notna() & statements.accession.isin(case._facts()[0])
    ].drop_duplicates("accession").set_index("accession")
    return pd.DataFrame({
        "flows": overlap.flows,
        "exchange": case._pick(case.EXCHANGE_EFFECT, "duration").reindex(overlap.index),
        "inclusive_open": case._pick(case.RESTRICTED_INCLUSIVE, "opening"),
        "inclusive_close": case._pick(case.RESTRICTED_INCLUSIVE, "closing"),
        "cash_open": case._pick(case.CASH, "opening"),
        "cash_close": case._pick(case.CASH, "closing"),
    }, index=overlap.index)


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["statement_population_count"] == 3642
    assert answers["statement_closing_count"] == 1162
    assert answers["inclusive_checkable_count"] == 402
    assert answers["inclusive_closing_count"] == 354
    assert answers["cash_only_checkable_count"] == 308
    assert answers["cash_only_closing_count"] == 93
    assert answers["both_closing_balances_count"] == 284
    assert answers["restricted_cash_holder_count"] == 183
    assert answers["nonzero_exchange_effect_count"] == 185
    assert answers["opening_comparable_count"] == 393
    assert answers["opening_revised_count"] == 16
    assert answers["revised_closing_with_comparative_count"] == 10
    assert answers["revised_closing_with_prior_report_count"] == 1


# --- properties (no literals) ----------------------------------------------

def test_the_cash_line_check_fails_for_most_reports():
    answers = case.analysis()
    assert answers["statement_closing_count"] < 0.5 * answers["statement_population_count"]


def test_the_combined_balance_closes_where_the_cash_line_does_not():
    """The premise: the statement reconciles the combined balance."""
    answers = case.analysis()
    inclusive = answers["inclusive_closing_count"] / answers["inclusive_checkable_count"]
    cash_only = answers["cash_only_closing_count"] / answers["cash_only_checkable_count"]
    assert inclusive > 0.8
    assert cash_only < 0.5
    assert inclusive > 2 * cash_only


def test_restricted_cash_is_held_by_most_reporters_of_both():
    answers = case.analysis()
    assert 0.5 < answers["restricted_cash_holder_count"] / answers["both_closing_balances_count"] < 1


def test_revisions_to_the_opening_balance_are_a_small_minority():
    answers = case.analysis()
    assert 0 < answers["opening_revised_count"] < 0.1 * answers["opening_comparable_count"]


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert answers["inclusive_closing_count"] < answers["inclusive_checkable_count"]
    assert answers["cash_only_closing_count"] < answers["cash_only_checkable_count"]
    assert answers["restricted_cash_holder_count"] < answers["both_closing_balances_count"]
    assert answers["inclusive_checkable_count"] < answers["statement_population_count"]


def test_the_comparative_opening_closes_where_the_stale_one_does_not():
    answers = case.analysis()
    assert answers["revised_closing_with_comparative_count"] > 5 * answers["revised_closing_with_prior_report_count"]
    assert answers["revised_closing_with_comparative_count"] < answers["opening_revised_count"]


# --- the conventions the query states ----------------------------------------

def test_the_threshold_sits_clear_of_every_difference():
    """Five first-turn reports miss by exactly one dollar, so a one-dollar
    threshold would sit on a boundary; three dollars sits clear of all."""
    frame = _comparison().dropna(subset=["inclusive_open", "inclusive_close"])
    difference = (
        frame.flows + frame.exchange.fillna(0.0) - (frame.inclusive_close - frame.inclusive_open)
    ).abs()
    assert not difference.between(case.TOLERANCE_USD - 0.5, case.TOLERANCE_USD + 0.5).any()
    assert not difference.between(1, 1000).any()

    statements = case._statements()
    population = statements.loc[
        statements.flows.notna() & statements[case.CASH].notna()
        & statements.prior_cash.notna() & statements.prior_gap_days.between(*case.YEAR_DAYS)
    ]
    first = (population.flows - (population[case.CASH] - population.prior_cash)).abs()
    assert not first.between(case.TOLERANCE_USD - 1.5, case.TOLERANCE_USD + 2.5).any()
    assert int(first.eq(1).sum()) > 0


def test_a_missing_exchange_effect_is_read_as_zero():
    """Dropping the filers that report no exchange effect is the alternative
    reading; it removes more than half the checkable filers."""
    frame = _comparison().dropna(subset=["inclusive_open", "inclusive_close"])
    kept = frame.loc[frame.exchange.notna()]
    assert len(kept) < 0.5 * len(frame)
    outside = _reported()
    outside["inclusive_checkable_count"] = len(kept)
    assert not case.validate(outside).success


def test_the_disposal_group_variant_is_a_different_answer():
    """Preferring the total that adds cash held in disposal groups wherever it
    appears is the competing reading the query states out."""
    variant = (
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
        "IncludingDisposalGroupAndDiscontinuedOperations"
    )
    facts = case.load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    statements = case._statements().drop_duplicates("accession")
    rows = facts.loc[
        facts.accession.isin(case._facts()[0])
        & facts.concept.astype(str).eq(variant)
        & facts.unit.astype(str).eq("USD")
    ].merge(statements[["accession", "period_end"]], on="accession")
    rows["end_date"] = rows.end_date.astype(str)
    instant = pd.to_datetime(rows.start_date, errors="coerce").isna()
    closing = rows.loc[instant & rows.end_date.eq(rows.period_end)].groupby("accession").value.first()
    before = (pd.to_datetime(rows.period_end) - pd.to_datetime(rows.end_date)).dt.days
    opening = rows.loc[instant & before.between(*case.YEAR_DAYS)].groupby("accession").value.first()

    frame = _comparison()
    swap = pd.DataFrame({"open": opening, "close": closing}).reindex(frame.index).dropna()
    frame.loc[swap.index, "inclusive_open"] = swap.open
    frame.loc[swap.index, "inclusive_close"] = swap.close
    checkable = frame.dropna(subset=["inclusive_open", "inclusive_close"])
    assert len(checkable) > case.analysis()["inclusive_checkable_count"]
    outside = _reported()
    outside["inclusive_checkable_count"] = len(checkable)
    assert not case.validate(outside).success


def test_the_variant_exchange_line_is_a_different_answer():
    """38 filers give the exchange effect only on the disposal-group variant
    line; taking it from there instead of reading zero moves three outputs."""
    variant = case.EXCHANGE_EFFECT + "IncludingDisposalGroupAndDiscontinuedOperations"
    frame = _comparison()
    facts = case.load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    facts["end_date"] = facts.end_date.astype(str)
    rows = facts.loc[
        facts.accession.isin(frame.index) & facts.concept.astype(str).eq(variant)
        & facts.unit.astype(str).eq("USD")
    ].merge(case._statements().drop_duplicates("accession")[["accession", "period_end"]], on="accession")
    days = (pd.to_datetime(rows.end_date) - pd.to_datetime(rows.start_date, errors="coerce")).dt.days
    effect = rows.loc[rows.end_date.eq(rows.period_end) & days.between(*case.YEAR_DAYS)].groupby("accession").value.first()
    filled = frame.exchange.fillna(effect.reindex(frame.index))
    assert int((frame.exchange.isna() & filled.notna()).sum()) > 0
    outside = _reported()
    outside["nonzero_exchange_effect_count"] = int(filled.fillna(0.0).ne(0).sum())
    assert not case.validate(outside).success


def test_carrying_the_first_turn_population_forward_is_a_different_answer():
    statements = case._statements()
    first = statements.loc[
        statements.flows.notna() & statements[case.CASH].notna()
        & statements.prior_cash.notna() & statements.prior_gap_days.between(*case.YEAR_DAYS)
    ]
    frame = _comparison()
    narrowed = frame.loc[frame.index.isin(set(first.accession))].dropna(
        subset=["inclusive_open", "inclusive_close"]
    )
    assert len(narrowed) < case.analysis()["inclusive_checkable_count"]
    outside = _reported()
    outside["inclusive_checkable_count"] = len(narrowed)
    assert not case.validate(outside).success


def test_a_joint_report_is_counted_once():
    statements = case._statements()
    population = statements.loc[
        statements.flows.notna() & statements[case.CASH].notna()
        & statements.prior_cash.notna() & statements.prior_gap_days.between(*case.YEAR_DAYS)
    ]
    assert population.accession.duplicated().any()
    outside = _reported()
    outside["statement_population_count"] = len(population)
    assert not case.validate(outside).success


def test_the_prior_report_opening_is_a_different_answer():
    """Reconciling the cash line from the prior report's own closing balance
    instead of the comparative column moves the count."""
    frame = _comparison()
    prior = case._statements().drop_duplicates("accession").set_index("accession").prior_cash
    frame["prior_cash"] = prior.reindex(frame.index)
    usable = frame.dropna(subset=["prior_cash", "cash_close"])
    closes = (
        usable.flows + usable.exchange.fillna(0.0) - (usable.cash_close - usable.prior_cash)
    ).abs().le(case.TOLERANCE_USD)
    assert int(closes.sum()) != case.analysis()["cash_only_closing_count"]


def test_every_fact_resolves_to_one_value_per_report():
    _, facts, _ = case._facts()
    for concept in (case.CASH, case.RESTRICTED_INCLUSIVE, case.EXCHANGE_EFFECT):
        for when in ("opening", "closing", "duration"):
            if concept == case.EXCHANGE_EFFECT and when != "duration":
                continue
            if concept != case.EXCHANGE_EFFECT and when == "duration":
                continue
            assert case._pick(concept, when).index.is_unique
    assert not facts.empty


# --- rejections -------------------------------------------------------------

def test_strict_display_boundary_both_sides():
    reported = _reported()
    for variable in case.variables:
        for direction in (1, -1):
            outside = dict(reported)
            outside[variable.name] = reported[variable.name] + direction
            assert not case.validate(outside).success, f"{variable.name} {direction:+d}"


# --- source dependence ------------------------------------------------------

@pytest.mark.parametrize("missing", ["load_financial_statements", "load_facts"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    _clear()

    def guarded(*args, **kwargs):
        raise RuntimeError(f"{missing} unavailable")

    try:
        monkeypatch.setattr(case, missing, guarded)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        _clear()
        case.analysis()
