"""Boundaries for the HMDA debt-to-income encoding case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

import pytest

from evals.comparison_ranking import hmda_debt_to_income_encoding_and_denial_reason as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._applications, case._certificate_lenders, case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["record_count"] == 12043028
    assert answers["distinct_dti_values"] == 21
    assert answers["ordered_class_count"] == 19
    assert answers["unorderable_record_count"] == 4320765
    assert answers["bank_linked_lei_count"] == 1711
    assert answers["bank_linked_record_count"] == 2999048
    assert round(answers["bank_exempt_pct"], 4) == 5.2955
    assert round(answers["nonbank_exempt_pct"], 4) == 1.3361
    assert answers["dti_denied_count"] == 668880
    assert answers["dti_denied_above_50_count"] == 545342
    assert answers["dti_denied_below_36_count"] == 25287
    assert answers["dti_denied_unreported_count"] == 21101
    assert answers["exact_integer_record_count"] == 3740552
    assert answers["dti_denied_exact_integer_count"] == 77150
    assert answers["all_record_median_class"] == "41"
    assert answers["dti_denied_median_class"] == ">60%"


# --- properties (no literals) ----------------------------------------------

def test_the_denied_records_sit_higher_on_the_scale():
    """The comparison the case draws. If the two medians landed in the same
    class the two rankings would say nothing against each other."""
    answers = case.analysis()
    order = list(case.ORDERED)
    assert order.index(answers["dti_denied_median_class"]) > order.index(
        answers["all_record_median_class"]
    )


def test_the_exemption_code_separates_the_two_lender_groups():
    answers = case.analysis()
    assert answers["bank_exempt_pct"] > 2 * answers["nonbank_exempt_pct"]
    assert 0 < answers["bank_linked_record_count"] < answers["record_count"]


def test_a_large_share_of_the_column_is_off_the_scale():
    """If nearly every record sat on the scale, the encoding problem would be
    a footnote rather than the case."""
    answers = case.analysis()
    assert answers["unorderable_record_count"] > 0.25 * answers["record_count"]


def test_the_reason_mostly_agrees_with_the_field_but_not_always():
    answers = case.analysis()
    assert answers["dti_denied_above_50_count"] > 0.7 * answers["dti_denied_count"]
    assert answers["dti_denied_below_36_count"] > 0
    assert answers["dti_denied_unreported_count"] > 0


def test_the_encodings_partition_the_column():
    answers = case.analysis()
    ratio = case._applications().debt_to_income_ratio
    bins = int(ratio.isin(case.LOW_BINS + case.HIGH_BINS).sum())
    assert bins + answers["exact_integer_record_count"] + answers["unorderable_record_count"] == answers["record_count"]


def test_every_exact_value_falls_inside_one_band():
    ratio = case._applications().debt_to_income_ratio
    present = {v for v in ratio.unique() if str(v).isdigit()}
    assert present == set(case.EXACT)


# --- rejections -------------------------------------------------------------

def test_sorting_the_column_as_text_is_a_different_ranking():
    """The listed overclaim: a text sort lifts the lowest range above every
    integer, so the median class moves."""
    answers = case.analysis()
    textual = sorted(case.ORDERED)
    assert textual != list(case.ORDERED)
    assert textual.index("<20%") > textual.index("49")
    ratio = case._applications().debt_to_income_ratio
    counts = ratio.value_counts()
    total = int(counts.reindex(list(case.ORDERED)).fillna(0).sum())
    running = 0
    for name in textual:
        running += int(counts.get(name, 0))
        if running * 2 >= total:
            break
    assert name != answers["all_record_median_class"]
    outside = _reported()
    outside["all_record_median_class"] = name
    assert not case.validate(outside).success


def test_dropping_the_off_scale_records_is_a_different_answer():
    answers = case.analysis()
    ratio = case._applications().debt_to_income_ratio
    on_scale = int(ratio.isin(case.ORDERED).sum())
    assert on_scale < answers["record_count"]
    naive = answers["dti_denied_count"] - answers["dti_denied_unreported_count"]
    assert naive != answers["dti_denied_count"]
    outside = _reported()
    outside["dti_denied_count"] = naive
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

def test_the_exempt_percentages_rest_on_the_certificate_match():
    """Stronger than checking the loader is called: the split must be the one
    the institution file draws, and it must not be an empty or total set."""
    answers = case.analysis()
    lenders = case._certificate_lenders()
    filers = set(case._applications().lei.astype(str).unique())
    assert answers["bank_linked_lei_count"] == len(lenders & filers)
    assert 0 < len(lenders & filers) < len(filers)


@pytest.mark.parametrize("missing", ["hmda", "ffiec_nic_institutions"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    _clear()
    real = case.load_expansion_table

    def guarded(name):
        if name == missing:
            raise RuntimeError(f"{name} unavailable")
        return real(name)

    try:
        monkeypatch.setattr(case, "load_expansion_table", guarded)
        with pytest.raises(RuntimeError):
            case.analysis()
    finally:
        monkeypatch.undo()
        _clear()
        case.analysis()
