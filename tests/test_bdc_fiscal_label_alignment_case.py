"""Boundaries for the BDC fiscal-label cross-section alignment case.

Anchor tests pin the literals. Property tests assert the relations the case
rests on and deliberately name no literal, so that each can fail on its own
rather than being decided by an equality asserted earlier in the same function.
"""

import pytest

from core.data import load_expansion_table
from evals.temporal_path import bdc_fiscal_label_cross_section_alignment as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["index_filer_count"] == 167
    assert answers["noncalendar_fiscal_year_end_filers"] == 22
    assert answers["label_filing_count"] == 164
    assert answers["label_distinct_period_ends"] == 5
    assert answers["label_period_span_days"] == 305
    assert answers["label_filer_count"] == 163
    assert answers["label_dominant_period_end"] == "2024-09-30"
    assert answers["label_off_date_filers"] == 20
    assert answers["label_filers_with_schedule_rows"] == 135
    assert answers["calendar_filer_count"] == 162
    assert round(answers["label_fair_value_mn"], 3) == 1287095.147
    assert round(answers["off_date_fair_value_mn"], 3) == 79739.295
    assert round(answers["calendar_fair_value_mn"], 3) == 1290044.225
    assert round(answers["earliest_period_sofr_pct"], 2) == 5.33
    assert round(answers["dominant_period_sofr_pct"], 2) == 4.96


def test_filer_count_is_by_cik_not_by_name():
    """167 by CIK against 169 by reported name; the query pins CIK, and the
    output carries zero tolerance, so the two readings are not interchangeable."""
    filings = load_expansion_table("sec_bdc_filings")
    assert case.analysis()["index_filer_count"] == filings.cik.nunique()
    assert filings.filer_name.nunique() != filings.cik.nunique()


# --- properties (no literals) ----------------------------------------------

def test_label_is_not_one_as_of_date():
    answers = case.analysis()
    assert answers["label_distinct_period_ends"] > 1
    assert answers["label_period_span_days"] > 180
    assert 0 < answers["label_off_date_filers"] < answers["label_filer_count"]


def test_retention_collapses_a_duplicate_filing():
    answers = case.analysis()
    assert answers["label_filer_count"] < answers["label_filing_count"]


def test_value_and_filer_counts_rest_on_different_denominators():
    answers = case.analysis()
    assert answers["label_filers_with_schedule_rows"] < answers["label_filer_count"]


def test_off_date_weight_is_material():
    answers = case.analysis()
    share = answers["off_date_fair_value_mn"] / answers["label_fair_value_mn"]
    assert 0.02 < share < 0.15


def test_span_brackets_a_rate_move():
    answers = case.analysis()
    assert answers["earliest_period_sofr_pct"] > answers["dominant_period_sofr_pct"]


# --- rejections -------------------------------------------------------------

def test_dropping_incomplete_accessions_is_a_different_answer():
    """Discarding a whole filing over one absent mark is the listed overclaim;
    it removes most of the cross-section, so it must not validate."""
    schedule = load_expansion_table("sec_bdc")
    filings = load_expansion_table("sec_bdc_filings")
    answers = case.analysis()
    labelled = filings.loc[
        filings.fiscal_year.eq(case.FISCAL_YEAR) & filings.fiscal_period.eq(case.FISCAL_PERIOD)
    ]
    retained = set(case._latest_per_filer(labelled).accession)
    rows = schedule.loc[schedule.accession.isin(retained)]
    incomplete = set(rows.loc[rows.fair_value_usd.isna()].accession)
    naive = float(rows.loc[~rows.accession.isin(incomplete), "fair_value_usd"].dropna().sum() / 1e6)
    assert naive < 0.5 * answers["label_fair_value_mn"]
    outside = _reported()
    outside["label_fair_value_mn"] = round(naive, 3)
    assert not case.validate(outside).success


def test_excluding_negative_marks_is_a_different_answer():
    """The other listed overclaim: negative fair values are reported marks."""
    schedule = load_expansion_table("sec_bdc")
    filings = load_expansion_table("sec_bdc_filings")
    labelled = filings.loc[
        filings.fiscal_year.eq(case.FISCAL_YEAR) & filings.fiscal_period.eq(case.FISCAL_PERIOD)
    ]
    retained = set(case._latest_per_filer(labelled).accession)
    values = schedule.loc[schedule.accession.isin(retained), "fair_value_usd"].dropna()
    positive_only = float(values.loc[values > 0].sum() / 1e6)
    outside = _reported()
    outside["label_fair_value_mn"] = round(positive_only, 3)
    assert not case.validate(outside).success


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

def test_sofr_outputs_come_from_the_nyfed_table():
    """Stronger than checking the loader is called: the two reported rates must
    equal what the NY Fed table publishes on those dates."""
    answers = case.analysis()
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR")]

    def published(as_of):
        window = rates.loc[rates["Effective Date"].astype(str).le(as_of)]
        return float(window.sort_values("Effective Date")["Rate (%)"].iloc[-1])

    assert round(answers["dominant_period_sofr_pct"], 2) == round(
        published(answers["label_dominant_period_end"]), 2
    )
    assert answers["earliest_period_sofr_pct"] != answers["dominant_period_sofr_pct"]


@pytest.mark.parametrize("missing", ["sec_bdc", "nyfed_reference_rates"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    case.analysis.cache_clear()
    case.ground_truth.cache_clear()
    real = case.load_expansion_table

    def guarded(name):
        if name == missing or (missing == "sec_bdc" and name == "sec_bdc_filings"):
            raise RuntimeError(f"{name} unavailable")
        return real(name)

    try:
        monkeypatch.setattr(case, "load_expansion_table", guarded)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        case.analysis.cache_clear()
        case.ground_truth.cache_clear()
        case.analysis()
