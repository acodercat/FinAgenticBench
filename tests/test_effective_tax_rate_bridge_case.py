"""Boundaries for the effective tax rate and income bridge case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""


import pytest

from evals.calculation_reconciliation import effective_tax_rate_denominator_and_income_bridge as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._annual, case._complete, case._named_components,
                   case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["annual_filer_count"] == 5524
    assert answers["complete_filer_count"] == 3515
    assert answers["positive_pretax_count"] == 1995
    assert round(answers["median_effective_tax_rate_pct"], 4) == 21.276
    assert answers["tax_benefit_count"] == 139
    assert answers["above_hundred_count"] == 32
    assert answers["below_statutory_count"] == 965
    assert round(answers["aggregate_effective_tax_rate_pct"], 4) == 18.1859
    assert answers["residual_filer_count"] == 1110
    assert round(answers["residual_total_bn"], 6) == 17.190644
    assert answers["largest_residual_cik"] == "1618921"
    assert round(answers["largest_residual_bn"], 6) == -6.829
    assert answers["facts_matched_count"] == 140
    assert answers["decomposition_holds_count"] == 129
    assert round(answers["minority_interest_total_bn"], 6) == 19.573048
    assert round(answers["discontinued_total_bn"], 6) == 1.547843


# --- properties (no literals) ----------------------------------------------

def test_the_denominator_rules_out_a_large_share_of_filers():
    """The first finding. If nearly every filer earned positive pretax income
    the denominator caveat would be a footnote."""
    answers = case.analysis()
    excluded = answers["complete_filer_count"] - answers["positive_pretax_count"]
    assert excluded > 0.3 * answers["complete_filer_count"]


def test_the_median_sits_near_the_statutory_rate_over_a_wide_spread():
    answers = case.analysis()
    assert abs(answers["median_effective_tax_rate_pct"] - case.STATUTORY_PCT) < 2
    assert answers["tax_benefit_count"] > 0
    assert answers["above_hundred_count"] > 0


def test_the_size_weighted_rate_is_lower_than_the_median():
    """The second finding: the largest profitable filers bear less."""
    answers = case.analysis()
    assert answers["aggregate_effective_tax_rate_pct"] < answers["median_effective_tax_rate_pct"]


def test_the_identity_fails_for_a_substantial_minority():
    answers = case.analysis()
    share = answers["residual_filer_count"] / answers["complete_filer_count"]
    assert 0.15 < share < 0.6


def test_the_decomposition_mostly_closes_but_not_entirely():
    """The fourth turn only says something if the two named items explain most
    of the gap and yet leave some of it standing."""
    answers = case.analysis()
    closed = answers["decomposition_holds_count"] / answers["facts_matched_count"]
    assert 0.8 < closed < 1.0


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["positive_pretax_count"] < answers["complete_filer_count"] < answers["annual_filer_count"]
    assert 0 < answers["tax_benefit_count"] < answers["positive_pretax_count"]
    assert 0 < answers["facts_matched_count"] < answers["residual_filer_count"]


def test_the_two_named_items_differ_in_size():
    answers = case.analysis()
    assert abs(answers["minority_interest_total_bn"]) > abs(answers["discontinued_total_bn"])


# --- the stated rules that do not bind --------------------------------------

def test_no_filer_reports_exactly_zero_pretax_income():
    """The boundary is stated strictly; the case asserts strict and inclusive
    give the same population."""
    complete = case._complete()
    assert int((complete[case.PRETAX] == 0).sum()) == 0


def test_the_largest_residual_is_a_strict_leader():
    complete = case._complete()
    residual = complete.loc[complete.residual.abs() >= 1]
    ordered = residual.residual.abs().sort_values(ascending=False)
    assert ordered.iloc[0] > ordered.iloc[1]


def test_no_filer_sits_exactly_on_the_statutory_rate():
    """The count below the statutory rate is stated strictly; the case asserts
    strict and inclusive give the same population."""
    complete = case._complete()
    positive = complete.loc[complete[case.PRETAX] > 0]
    rate = positive[case.TAX] / positive[case.PRETAX] * 100
    assert int((rate == case.STATUTORY_PCT).sum()) == 0


def test_the_quarterly_columns_are_a_different_population():
    """The table carries each concept for the year and for a quarter; the
    query pins the year, and the quarterly reading is far smaller."""
    annual = case._annual()
    quarterly = annual.dropna(
        subset=[c.replace("_ytd", "_q") for c in (case.PRETAX, case.TAX, case.NET)]
    )
    answers = case.analysis()
    assert 0 < len(quarterly) < 0.1 * answers["complete_filer_count"]
    outside = _reported()
    outside["complete_filer_count"] = len(quarterly)
    assert not case.validate(outside).success


# --- rejections -------------------------------------------------------------

def test_including_loss_making_filers_is_a_different_median():
    """The listed overclaim: the ratio inverts its meaning below zero."""
    complete = case._complete()
    answers = case.analysis()
    naive = float((complete[case.TAX] / complete[case.PRETAX] * 100).median())
    assert round(naive, 4) != round(answers["median_effective_tax_rate_pct"], 4)
    outside = _reported()
    outside["median_effective_tax_rate_pct"] = round(naive, 4)
    assert not case.validate(outside).success


def test_averaging_filer_rates_is_a_different_aggregate():
    complete = case._complete()
    positive = complete.loc[complete[case.PRETAX] > 0]
    answers = case.analysis()
    naive = float((positive[case.TAX] / positive[case.PRETAX] * 100).mean())
    assert round(naive, 4) != round(answers["aggregate_effective_tax_rate_pct"], 4)
    outside = _reported()
    outside["aggregate_effective_tax_rate_pct"] = round(naive, 4)
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

def test_the_decomposition_uses_the_fact_file_values():
    """Stronger than checking the loader is called: every closing filer's gap
    must equal the two fact-file figures combined."""
    matched = case._named_components()
    closing = matched.loc[(matched.residual - matched.named).abs() < 1]
    assert len(closing) == case.analysis()["decomposition_holds_count"]
    assert (closing.named - (closing[case.MINORITY] - closing[case.DISCONTINUED])).abs().max() < 1e-6


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
