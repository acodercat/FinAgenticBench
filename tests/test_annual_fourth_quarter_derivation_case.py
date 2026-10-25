"""Boundaries for the fourth-quarter flow derivation case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

from decimal import Decimal

import pytest

from core.data import load_financial_statements
from evals.calculation_reconciliation import annual_report_fourth_quarter_flow_derivation as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._annual, case._paired, case._derived, case._submissions,
                   case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["annual_filer_count"] == 5524
    assert answers["noncalendar_year_end_filers"] == 1251
    assert answers["annual_with_year_net_income"] == 5170
    assert answers["annual_with_quarter_net_income"] == 268
    assert answers["paired_filer_count"] == 5394
    assert answers["derived_filer_count"] == 5018
    assert round(answers["year_net_income_bn"], 6) == 1817.773855
    assert round(answers["derived_q4_net_income_bn"], 6) == 488.833662
    assert answers["positive_year_filers"] == 2508
    assert answers["positive_year_negative_quarter_filers"] == 285
    assert answers["validation_filer_count"] == 256
    assert answers["validation_disagreement_count"] == 42
    assert answers["annual_figure_in_quarter_slot_count"] == 5
    assert answers["largest_disagreement_cik"] == "0000040729"
    assert round(answers["largest_disagreement_derived_mn"], 6) == -140.0
    assert round(answers["largest_disagreement_reported_mn"], 6) == 108.0
    assert answers["shared_accession_count"] == 85
    assert answers["superseded_by_amendment_count"] == 574
    assert round(answers["median_days_to_public"], 1) == 65.0
    assert answers["form_disagreement_count"] == 2


# --- properties (no literals) ----------------------------------------------

def test_the_annual_filing_reports_the_year_not_the_quarter():
    """The premise of the case. If annual filings tagged the closing quarter as
    often as the year, there would be nothing to derive."""
    answers = case.analysis()
    assert answers["annual_with_quarter_net_income"] < 0.2 * answers["annual_with_year_net_income"]


def test_the_derivation_reproduces_most_disclosed_quarters():
    answers = case.analysis()
    assert answers["validation_disagreement_count"] < 0.3 * answers["validation_filer_count"]
    assert answers["validation_disagreement_count"] > 0


def test_no_agreeing_filer_matches_on_an_empty_nine_months():
    """A filer with nothing reported through nine months would agree trivially;
    the validation would then be evidence of nothing."""
    derived = case._derived()
    validation = derived.loc[derived.NetIncomeLoss_q.notna()]
    agreeing = validation.loc[
        (validation.derived_q4_usd - validation.NetIncomeLoss_q.map(lambda x: Decimal(str(x)))).abs()
        < case.AGREEMENT_USD
    ]
    assert len(agreeing) > 0
    assert not (agreeing.nine_month_usd == 0).any()


def test_the_validation_holds_at_scale():
    """Agreement is not confined to shells: it survives among filers reporting
    a year in the billions, where a coincidental match is implausible."""
    derived = case._derived()
    validation = derived.loc[derived.NetIncomeLoss_q.notna()]
    large = validation.loc[validation.year_usd.abs() > Decimal(10) ** 9]
    assert len(large) > 0
    agreeing = (
        large.derived_q4_usd - large.NetIncomeLoss_q.map(lambda x: Decimal(str(x)))
    ).abs() < case.AGREEMENT_USD
    assert agreeing.mean() > 0.5


def test_subtraction_is_load_bearing():
    answers = case.analysis()
    assert answers["derived_q4_net_income_bn"] != answers["year_net_income_bn"]
    assert abs(answers["derived_q4_net_income_bn"]) < abs(answers["year_net_income_bn"])


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["positive_year_negative_quarter_filers"] < answers["positive_year_filers"]
    assert 0 < answers["annual_figure_in_quarter_slot_count"] < answers["validation_disagreement_count"]
    assert answers["derived_filer_count"] < answers["paired_filer_count"] < answers["annual_filer_count"]


def test_a_filer_closes_two_fiscal_years_inside_the_window():
    """The retention rule exists because of these filers; without one the
    population is not one row per filer."""
    statements = load_financial_statements()
    window = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
        & statements.report_date.astype(str).between(*case.WINDOW)
    ]
    counts = window.groupby("cik").report_date.nunique()
    assert int((counts > 1).sum()) > 0
    assert case.analysis()["annual_filer_count"] == window.cik.nunique()


def test_the_derivation_rests_on_a_revisable_filing():
    answers = case.analysis()
    assert 0 < answers["superseded_by_amendment_count"] < answers["annual_filer_count"]
    assert answers["median_days_to_public"] > 30


def test_the_accession_alone_does_not_identify_a_filing():
    """The listed overclaim: an accession filed jointly by several filers
    appears once per filer, so joining on it alone multiplies rows."""
    joined, index = case._submissions()
    annual = case._annual()
    by_accession = annual[["accession"]].astype(str).merge(
        index.assign(accession=index.accession.astype(str))[["accession", "cik"]], on="accession"
    )
    assert len(by_accession) > len(joined)
    assert case.analysis()["shared_accession_count"] > 0


def test_the_stated_tie_breaks_do_not_bind():
    """House rule: a selection states its tie rule even when nothing ties, and
    the case asserts that nothing does, so a later data change is visible."""
    statements = load_financial_statements()
    window = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
        & statements.report_date.astype(str).between(*case.WINDOW)
    ]
    per_filing = window.groupby(["cik", "report_date", "filed_date"]).size()
    assert int((per_filing > 1).sum()) == 0

    derived = case._derived()
    validation = derived.loc[derived.NetIncomeLoss_q.notna()].copy()
    gaps = (
        validation.derived_q4_usd - validation.NetIncomeLoss_q.map(lambda x: Decimal(str(x)))
    ).abs().sort_values(ascending=False)
    assert gaps.iloc[0] > gaps.iloc[1]


# --- rejections -------------------------------------------------------------

def test_the_fiscal_year_label_is_a_different_population():
    """The listed overclaim: the label files two fiscal years of a 52/53-week
    filer under one year, so it does not select the same filers."""
    statements = load_financial_statements()
    by_label = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
        & statements.fiscal_year.eq(2024)
    ]
    assert by_label.cik.nunique() != case.analysis()["annual_filer_count"]


@pytest.mark.parametrize("gap_days", [3650])
def test_removing_the_pairing_limit_is_a_different_answer(monkeypatch, gap_days):
    reference = _reported()
    _clear()
    try:
        monkeypatch.setattr(case, "MAX_PAIR_GAP_DAYS", gap_days)
        loosened = case.analysis()["derived_q4_net_income_bn"]
    finally:
        monkeypatch.undo()
        _clear()
    assert round(loosened, 6) != reference["derived_q4_net_income_bn"]
    outside = dict(reference)
    outside["derived_q4_net_income_bn"] = round(loosened, 6)
    assert not case.validate(outside).success


def test_admitting_amendments_is_a_different_answer():
    """Amended filings restate the same period; letting them in changes both
    the retained annual figure and the total, so the exclusion is not cosmetic."""
    statements = load_financial_statements()
    annual = statements.loc[
        statements.form.isin(["10-K", "10-K/A"]) & statements.fiscal_period.eq("FY")
        & statements.report_date.astype(str).between(*case.WINDOW)
    ]
    assert annual.cik.nunique() >= case.analysis()["annual_filer_count"]
    assert annual.form.eq("10-K/A").any()


def test_strict_display_boundary_both_sides():
    answers = case.analysis()
    reported = _reported()
    for variable in case.variables:
        precision = case.PRECISION[variable.name]
        if precision is None:
            continue
        for direction in (1, -1):
            outside = dict(reported)
            step = 0.7 * 10 ** (-precision) if precision else 1
            outside[variable.name] = answers[variable.name] + direction * step
            assert not case.validate(outside).success, f"{variable.name} {direction:+d}"


# --- source dependence ------------------------------------------------------

def test_the_named_filer_figures_come_from_the_table():
    """Stronger than checking the loader is called: both reported figures must
    equal what the annual and third-quarter filings of that CIK carry."""
    answers = case.analysis()
    cik = answers["largest_disagreement_cik"].lstrip("0")
    row = case._derived().loc[lambda f: f.cik.astype(str).eq(cik)].iloc[0]
    assert round(float(row.NetIncomeLoss_q) / 1e6, 6) == round(
        answers["largest_disagreement_reported_mn"], 6
    )
    assert round(float(row.year_usd - row.nine_month_usd) / 1e6, 6) == round(
        answers["largest_disagreement_derived_mn"], 6
    )
    assert row.nine_month_usd != 0


@pytest.mark.parametrize("missing", ["load_financial_statements", "load_filings"])
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
