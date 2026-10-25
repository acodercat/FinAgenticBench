"""Boundaries for the SBA 7(a) withheld-status denominator case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

import pytest

from evals.calculation_reconciliation import sba_loan_status_withheld_chargeoff_denominator as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._loans, case._disbursed, case._insured_certificates, case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["loan_row_count"] == 347514
    assert answers["cancelled_count"] == 44921
    assert answers["committed_not_disbursed_count"] == 8681
    assert answers["charged_off_count"] == 6893
    assert answers["disbursed_loan_count"] == 293909
    assert round(answers["chargeoff_pct_all_rows"], 4) == 1.9835
    assert round(answers["chargeoff_pct_disbursed"], 4) == 2.3453
    assert round(answers["chargeoff_pct_disclosed_only"], 4) == 9.1897
    assert answers["status_withheld_count"] == 218901
    assert round(answers["withheld_pct_earliest_fy"], 4) == 37.2223
    assert round(answers["withheld_pct_latest_fy"], 4) == 96.3493
    assert round(answers["withheld_to_disclosed_median_ratio"], 4) == 1.3933
    assert round(answers["chargeoff_upper_bound_pct"], 4) == 76.8245
    assert answers["status_date_inconsistency_count"] == 6
    assert answers["paid_in_full_without_disbursement_count"] == 3
    assert answers["exact_duplicate_row_count"] == 513
    assert answers["lender_without_certificate_count"] == 31585
    assert round(answers["withheld_pct_uninsured_lender"], 4) == 78.7304
    assert round(answers["withheld_pct_insured_lender"], 4) == 73.9673
    assert answers["unmatched_certificate_loan_count"] == 5


# --- properties (no literals) ----------------------------------------------

def test_the_denominator_not_the_numerator_moves_the_rate():
    """The point of the case. All three rates carry the same charge-off count,
    so any gap between them is the population and nothing else."""
    answers = case.analysis()
    implied = [
        answers["chargeoff_pct_all_rows"] / 100 * answers["loan_row_count"],
        answers["chargeoff_pct_disbursed"] / 100 * answers["disbursed_loan_count"],
    ]
    for value in implied:
        assert abs(value - answers["charged_off_count"]) < 1e-6
    assert answers["chargeoff_pct_disclosed_only"] > 3 * answers["chargeoff_pct_all_rows"]


def test_the_withholding_rises_with_every_vintage():
    disbursed = case._disbursed()
    by_year = disbursed.groupby("ApprovalFY").LoanStatus.apply(
        lambda s: float(s.eq(case.WITHHELD).mean())
    ).sort_index()
    assert len(by_year) > 2
    assert by_year.is_monotonic_increasing
    assert by_year.iloc[-1] - by_year.iloc[0] > 0.3


def test_the_withheld_loans_are_the_larger_ones():
    assert case.analysis()["withheld_to_disclosed_median_ratio"] > 1


def test_the_interval_is_too_wide_to_be_a_rate():
    answers = case.analysis()
    assert answers["chargeoff_upper_bound_pct"] > 10 * answers["chargeoff_pct_disbursed"]
    assert answers["chargeoff_pct_disbursed"] < answers["chargeoff_pct_disclosed_only"] < answers["chargeoff_upper_bound_pct"]


def test_disbursement_and_status_agree_without_being_the_same_test():
    """Both single out the loans that never put money at risk, but they do not
    select the identical set, so which one the query names matters."""
    answers = case.analysis()
    by_status = (
        answers["loan_row_count"]
        - answers["cancelled_count"]
        - answers["committed_not_disbursed_count"]
    )
    assert by_status != answers["disbursed_loan_count"]
    assert abs(by_status - answers["disbursed_loan_count"]) == answers["paid_in_full_without_disbursement_count"]


def test_no_cancelled_or_committed_row_was_ever_disbursed():
    loans = case._loans()
    for code in (case.CANCELLED, case.COMMITTED):
        rows = loans.loc[loans.LoanStatus.eq(code)]
        assert len(rows) > 0
        assert rows.FirstDisbursementDate.isna().all()


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["status_withheld_count"] < answers["disbursed_loan_count"] < answers["loan_row_count"]
    assert 0 < answers["charged_off_count"] < answers["disbursed_loan_count"]
    assert 0 < answers["exact_duplicate_row_count"] < answers["loan_row_count"]


def test_the_withholding_tracks_the_lender_too():
    """The fifth turn only carries information if the two lender groups differ;
    an equal split would make the bank file decoration."""
    answers = case.analysis()
    assert answers["withheld_pct_uninsured_lender"] > answers["withheld_pct_insured_lender"]
    assert 0 < answers["lender_without_certificate_count"] < answers["disbursed_loan_count"]


def test_a_named_certificate_can_be_absent_from_the_bank_file():
    disbursed = case._disbursed()
    named = disbursed.fdic_certificate.dropna().astype(str)
    assert named.nunique() > 1
    missing = set(named) - set(case._insured_certificates())
    assert len(missing) > 0
    assert int((~named.isin(case._insured_certificates())).sum()) == case.analysis()["unmatched_certificate_loan_count"]


# --- rejections -------------------------------------------------------------

def test_a_tidied_status_spelling_matches_nothing():
    """One code carries interior spaces; a screen written against the obvious
    spelling silently selects an empty set rather than failing."""
    loans = case._loans()
    assert loans.LoanStatus.eq(case.PAID_IN_FULL).sum() > 0
    assert loans.LoanStatus.eq("PIF").sum() == 0


def test_deduplicating_is_a_different_answer():
    """The listed overclaim: the extract has no unique loan identifier and
    preserves exact-looking rows, so collapsing them changes the file."""
    answers = case.analysis()
    loans = case._loans()
    deduplicated = len(loans.drop_duplicates())
    assert deduplicated == answers["loan_row_count"] - answers["exact_duplicate_row_count"]
    naive = answers["charged_off_count"] / deduplicated * 100
    assert round(naive, 4) != round(answers["chargeoff_pct_all_rows"], 4)
    outside = _reported()
    outside["chargeoff_pct_all_rows"] = round(naive, 4)
    assert not case.validate(outside).success


def test_counting_every_repeated_row_is_a_different_answer():
    loans = case._loans()
    answers = case.analysis()
    all_involved = int(loans.duplicated(keep=False).sum())
    assert all_involved > answers["exact_duplicate_row_count"]
    outside = _reported()
    outside["exact_duplicate_row_count"] = all_involved
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

def test_the_contradicted_rows_are_the_two_named_kinds():
    """Stronger than checking the count: each contradicted row must be one of
    the two dispositions the query names."""
    loans = case._loans()
    missing_date = loans.LoanStatus.eq(case.CHARGED_OFF) & loans.ChargeOffDate.isna()
    stray_date = loans.LoanStatus.ne(case.PAID_IN_FULL) & loans.PaidInFullDate.notna()
    assert int(missing_date.sum()) > 0 and int(stray_date.sum()) > 0
    assert int((missing_date | stray_date).sum()) == case.analysis()["status_date_inconsistency_count"]


@pytest.mark.parametrize("missing", ["sba_7a", "fdic_bankfind"])
def test_both_sources_are_load_bearing(monkeypatch, missing):
    _clear()
    case._insured_certificates.cache_clear()
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
        case._insured_certificates.cache_clear()
        case.analysis()
