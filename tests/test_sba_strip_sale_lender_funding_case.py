"""Boundaries for the SBA guaranteed-strip sale case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

import pytest

from evals.comparison_ranking import sba_guaranteed_strip_sale_and_lender_funding as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._disbursed, case._bank_assets, case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["sold_loan_count"] == 108175
    assert round(answers["sold_share_pct"], 4) == 36.8056
    assert answers["lender_count"] == 2116
    assert answers["never_selling_lender_count"] == 1465
    assert answers["revolver_sold_count"] == 5
    assert round(answers["nonrevolver_sold_share_pct"], 4) == 49.5602
    assert round(answers["sold_variable_rate_pct"], 4) == 98.2796
    assert round(answers["retained_variable_rate_pct"], 4) == 74.5302
    assert round(answers["noncertificate_sold_share_pct"], 4) == 70.9007
    assert round(answers["certificate_sold_share_pct"], 4) == 32.7004
    assert round(answers["large_bank_sold_share_pct"], 4) == 16.0144
    assert round(answers["small_bank_sold_share_pct"], 4) == 55.9638
    assert round(answers["top_ten_seller_share_pct"], 4) == 45.3108
    assert round(answers["largest_seller_share_pct"], 4) == 7.7957
    assert round(answers["sold_chargeoff_pct"], 4) == 1.6464
    assert round(answers["retained_chargeoff_pct"], 4) == 2.7523


# --- properties (no literals) ----------------------------------------------

def test_funding_model_separates_the_cohorts():
    """The finding of the case. If the two lender groups sold at similar rates
    the funding argument would have nothing behind it."""
    answers = case.analysis()
    assert answers["noncertificate_sold_share_pct"] > 2 * answers["certificate_sold_share_pct"]
    assert answers["small_bank_sold_share_pct"] > 3 * answers["large_bank_sold_share_pct"]


def test_revolving_lines_are_effectively_never_sold():
    loans = case._disbursed()
    revolving = loans.RevolverStatus.astype(str).eq(case.REVOLVER)
    assert int(revolving.sum()) > 1000
    assert case.analysis()["revolver_sold_count"] < 0.001 * int(revolving.sum())


def test_the_sold_side_is_floating_rate():
    answers = case.analysis()
    assert answers["sold_variable_rate_pct"] > answers["retained_variable_rate_pct"]
    assert answers["sold_variable_rate_pct"] > 90


def test_the_sellers_are_concentrated():
    answers = case.analysis()
    assert answers["top_ten_seller_share_pct"] > 40
    assert answers["largest_seller_share_pct"] < answers["top_ten_seller_share_pct"]
    assert 0 < answers["never_selling_lender_count"] < answers["lender_count"]


def test_the_charge_off_comparison_is_confounded():
    """The caveat the last turn asks about has to be there in the data, or the
    claim is asking the response to invent one."""
    loans = case._disbursed()
    withheld = loans.LoanStatus.astype(str).eq("EXEMPT")
    sold = loans.sold
    assert withheld[sold].mean() > withheld[~sold].mean()
    answers = case.analysis()
    assert answers["sold_chargeoff_pct"] < answers["retained_chargeoff_pct"]


def test_the_disclosed_only_denominator_would_differ():
    """The query pins every loan in the cohort as the denominator; the
    disclosed-only reading is a different number, so the clause is load-bearing."""
    loans = case._disbursed()
    answers = case.analysis()
    disclosed = loans.loc[loans.LoanStatus.astype(str).ne("EXEMPT")]
    naive = float(
        disclosed.loc[disclosed.sold, "LoanStatus"].eq("CHGOFF").mean() * 100
    )
    assert round(naive, 4) != round(answers["sold_chargeoff_pct"], 4)
    outside = _reported()
    outside["sold_chargeoff_pct"] = round(naive, 4)
    assert not case.validate(outside).success


# --- the stated rules that do not bind --------------------------------------

def test_no_lender_sits_on_the_asset_threshold():
    """House rule: the boundary is stated strictly and the case asserts that
    strict and inclusive give the same split."""
    loans = case._disbursed()
    matched = loans.loc[loans.fdic_certificate.notna()].copy()
    matched["assets"] = matched.fdic_certificate.astype(str).map(case._bank_assets())
    matched = matched.loc[matched.assets.notna()]
    assert int((matched.assets == case.LARGE_BANK_ASSETS_THOUSAND_USD).sum()) == 0


def test_the_top_ten_cut_has_no_tie():
    loans = case._disbursed()
    sellers = loans.loc[loans.sold].groupby("BankName").size().sort_values(ascending=False)
    assert sellers.iloc[case.TOP_SELLERS - 1] > sellers.iloc[case.TOP_SELLERS]


def test_the_size_split_is_loan_weighted_and_the_lender_reading_reverses_it():
    """The sharpest pin in the case: weighting by lender instead of by loan
    does not just move the figures, it turns the finding round."""
    loans = case._disbursed()
    matched = loans.loc[loans.fdic_certificate.notna()].copy()
    matched["assets"] = matched.fdic_certificate.astype(str).map(case._bank_assets())
    matched = matched.loc[matched.assets.notna()]
    per_lender = matched.groupby("fdic_certificate").agg(
        sold=("sold", "sum"), assets=("assets", "first")
    )
    large = per_lender.assets > case.LARGE_BANK_ASSETS_THOUSAND_USD
    answers = case.analysis()
    assert answers["large_bank_sold_share_pct"] < answers["small_bank_sold_share_pct"]
    assert (per_lender.loc[large, "sold"] > 0).mean() > (
        per_lender.loc[~large, "sold"] > 0
    ).mean()


def test_the_matched_scope_is_narrower_than_the_certificate_scope():
    loans = case._disbursed()
    answers = case.analysis()
    carried = loans.loc[loans.fdic_certificate.notna()]
    matched = carried.loc[carried.fdic_certificate.astype(str).isin(case._bank_assets().index)]
    assert len(matched) < len(carried)
    assert round(float(matched.sold.mean() * 100), 4) != round(
        answers["certificate_sold_share_pct"], 4
    )


def test_unreported_rate_type_sits_in_the_denominator():
    loans = case._disbursed()
    answers = case.analysis()
    rate = loans.FixedorVariableInterestInd
    assert int(rate.isna().sum()) > 0
    excluded = float(rate[loans.sold].dropna().eq(case.VARIABLE).mean() * 100)
    assert round(excluded, 4) != round(answers["sold_variable_rate_pct"], 4)


# --- rejections -------------------------------------------------------------

def test_counting_lenders_by_certificate_is_a_different_answer():
    """The listed overclaim: the two keys do not give the same lender count."""
    loans = case._disbursed()
    answers = case.analysis()
    by_certificate = int(loans.fdic_certificate.nunique())
    assert by_certificate != answers["lender_count"]
    outside = _reported()
    outside["lender_count"] = by_certificate
    assert not case.validate(outside).success


def test_the_indicator_is_positive_only():
    loans = case._disbursed()
    assert set(loans.SoldSecMrktInd.dropna().unique()) == {case.SOLD}
    assert int(loans.SoldSecMrktInd.isna().sum()) > 0


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

def test_the_size_split_rests_on_the_bank_file():
    """Stronger than checking the loader is called: the split must use assets
    the bank file reports at the stated date, and must be non-degenerate."""
    assets = case._bank_assets()
    loans = case._disbursed()
    matched = loans.loc[loans.fdic_certificate.notna()]
    covered = matched.fdic_certificate.astype(str).isin(assets.index)
    assert covered.mean() > 0.99
    large = assets > case.LARGE_BANK_ASSETS_THOUSAND_USD
    assert 0 < int(large.sum()) < len(assets)


@pytest.mark.parametrize("missing", ["sba_7a", "fdic_bankfind"])
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
