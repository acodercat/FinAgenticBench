"""Boundaries for the diluted-share antidilution and EPS-numerator case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import pandas as pd
import pytest

from core.data import load_financial_statements
from evals.calculation_reconciliation import diluted_share_antidilution_and_eps_numerator as case


def _clear():
    for cached in (case._population, case._overlap, case._facts, case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] else answers[v.name]
        for v in case.variables
    }


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["eps_filer_count"] == 4257
    assert answers["loss_filer_count"] == 2111
    assert answers["loss_diluted_equals_basic_count"] == 2042
    assert answers["loss_diluted_above_basic_count"] == 67
    assert answers["profit_diluted_below_basic_count"] == 5
    assert answers["profit_diluted_equals_basic_count"] == 319
    assert answers["diluted_eps_above_basic_eps_count"] == 16
    assert round(answers["median_dilution_increment_pct"], 4) == 0.9474
    assert answers["diluted_numerator_gap_count"] == 689
    assert round(answers["median_diluted_numerator_gap_usd"], 4) == 0.1998
    assert answers["recomputed_above_reported_count"] == 529
    assert answers["basic_numerator_gap_count"] == 625
    assert answers["overlap_filer_count"] == 340
    assert answers["consolidated_income_reported_count"] == 201
    assert answers["consolidated_differs_count"] == 166
    assert answers["noncontrolling_reported_count"] == 152
    assert answers["antidilutive_disclosure_count"] == 158
    assert round(answers["median_antidilutive_share_of_basic_pct"], 4) == 0.278
    assert answers["antidilutive_above_increment_count"] == 39
    assert answers["share_based_increment_above_total_count"] == 7


# --- properties (no literals) ----------------------------------------------

def test_the_antidilution_rule_is_the_rule_and_not_the_exception():
    """The case only says something if the great majority of loss filers do set
    the two counts equal; a corpus split near the middle would carry no rule."""
    answers = case.analysis()
    assert answers["loss_diluted_equals_basic_count"] > 0.9 * answers["loss_filer_count"]
    assert 0 < answers["loss_diluted_above_basic_count"] < 0.1 * answers["loss_filer_count"]


def test_losses_are_roughly_half_the_population():
    answers = case.analysis()
    assert 0.3 < answers["loss_filer_count"] / answers["eps_filer_count"] < 0.7


def test_breaches_of_the_share_count_ordering_are_rare():
    answers = case.analysis()
    assert 0 < answers["profit_diluted_below_basic_count"] < 0.01 * answers["eps_filer_count"]
    assert 0 < answers["diluted_eps_above_basic_eps_count"] < 0.01 * answers["eps_filer_count"]


def test_dilution_is_small_for_the_typical_filer():
    assert 0 < case.analysis()["median_dilution_increment_pct"] < 5


def test_the_basic_side_also_shows_a_gap():
    """A basic-side gap rules out dilution mechanics as the sole explanation."""
    answers = case.analysis()
    assert answers["basic_numerator_gap_count"] > 0.5 * answers["diluted_numerator_gap_count"]


def test_the_gap_is_far_larger_than_reporting_precision():
    """Reported per-share figures are rounded to the cent, so a median miss of
    this size cannot be an artefact of that rounding."""
    assert case.analysis()["median_diluted_numerator_gap_usd"] > 0.05


def test_the_consolidated_total_usually_differs_where_it_is_reported():
    answers = case.analysis()
    assert answers["consolidated_differs_count"] > 0.5 * answers["consolidated_income_reported_count"]
    assert answers["consolidated_income_reported_count"] < answers["overlap_filer_count"]


def test_the_overhang_exceeds_the_admitted_dilution_for_a_real_minority():
    answers = case.analysis()
    share = answers["antidilutive_above_increment_count"] / answers["antidilutive_disclosure_count"]
    assert 0.1 < share < 0.5


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert answers["loss_filer_count"] < answers["eps_filer_count"]
    assert answers["overlap_filer_count"] < answers["eps_filer_count"]
    assert answers["antidilutive_disclosure_count"] < answers["overlap_filer_count"]
    assert answers["noncontrolling_reported_count"] < answers["consolidated_income_reported_count"]


def test_the_loss_counts_do_not_partition_the_loss_population():
    """Two loss filers report fewer diluted shares than basic ones and that
    count is not asked for, so no reported count is the sum of two others."""
    answers = case.analysis()
    assert (
        answers["loss_diluted_equals_basic_count"] + answers["loss_diluted_above_basic_count"]
        < answers["loss_filer_count"]
    )


# --- the boundary the threshold sits on -------------------------------------

def test_the_threshold_is_clear_of_the_cent_grid():
    """Reported per-share figures are rounded to the cent, so exact differences
    pile up on whole-cent multiples. At two cents no filer sits on the boundary
    and the strict and non-strict readings agree; at one cent they do not."""
    population = case._population()
    for column in ("diluted_gap", "basic_gap"):
        gaps = population[column]
        assert int(gaps.gt(0.02).sum()) == int(gaps.ge(0.02).sum())
        assert float((gaps - 0.02).abs().min()) > 1e-5
    assert float((population.diluted_gap - 0.01).abs().min()) < 1e-10


def test_every_retained_filer_can_be_divided_by():
    """One filing reports a diluted count of zero against billions of basic
    shares; a floating-point division would return infinity rather than refuse."""
    population = case._population()
    assert population[case.DILUTED_SHARES].gt(0).all()
    assert population[case.BASIC_SHARES].gt(0).all()
    assert population.diluted_gap.notna().all()
    assert not population.diluted_gap.eq(float("inf")).any()


def test_dropping_the_zero_diluted_filer_is_load_bearing():
    frame = load_financial_statements()
    annual = frame.loc[
        frame.form.astype(str).eq("10-K") & frame.fiscal_period.astype(str).eq("FY")
    ].copy()
    annual["period_end"] = annual.report_date.astype(str)
    annual = annual.loc[annual.period_end.between(*case.WINDOW)]
    annual = annual.loc[
        annual[list(case.REQUIRED)].notna().all(axis=1) & annual[case.BASIC_SHARES].gt(0)
    ]
    assert int(annual[case.DILUTED_SHARES].le(0).sum()) > 0


# --- the selections that would change the answer ----------------------------

def test_one_filing_per_filer_is_load_bearing():
    frame = load_financial_statements()
    annual = frame.loc[
        frame.form.astype(str).eq("10-K") & frame.fiscal_period.astype(str).eq("FY")
    ].copy()
    annual["period_end"] = annual.report_date.astype(str)
    annual = annual.loc[annual.period_end.between(*case.WINDOW)]
    assert annual.cik.astype(str).nunique() < len(annual)

    outside = _reported()
    outside["eps_filer_count"] = case.analysis()["eps_filer_count"] + 7
    assert not case.validate(outside).success


def test_the_period_pin_is_load_bearing():
    """Without pinning the period a filing resolves the same concept to more
    than one value, which the case refuses rather than picks among."""
    population = case._population()
    facts = case.load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    unpinned = facts.loc[
        facts.accession.isin(set(population.accession))
        & facts.concept.astype(str).eq(case.CONSOLIDATED)
        & facts.unit.astype(str).eq("USD")
    ]
    assert int(unpinned.groupby("accession").value.nunique().gt(1).sum()) > 0
    assert not case._facts().empty


def test_period_end_shares_are_a_different_answer():
    """The listed substitution: a balance at the period end in place of the
    weighted average the per-share figures are struck on."""
    population = case._population()
    with_balance = population.loc[population.CommonStockSharesOutstanding.notna()]
    assert len(with_balance) > 0
    substituted = (
        with_balance[case.NET_INCOME] / with_balance.CommonStockSharesOutstanding
        - with_balance[case.DILUTED_EPS]
    ).abs()
    naive = int(substituted.ge(case.GAP_THRESHOLD_USD).sum())
    assert naive > case.analysis()["diluted_numerator_gap_count"]
    outside = _reported()
    outside["diluted_numerator_gap_count"] = naive
    assert not case.validate(outside).success


def test_the_lower_and_upper_middle_values_are_rejected():
    """Both reported medians sit on an even count whose two middle values
    differ, so the interpolation rule the query states decides them."""
    population = case._population()
    profitable = population.loc[~population.loss & population.increment.gt(0)]
    increments = (profitable.increment / profitable[case.BASIC_SHARES] * 100).sort_values()
    assert len(increments) % 2 == 0
    lower = float(increments.iloc[len(increments) // 2 - 1])
    upper = float(increments.iloc[len(increments) // 2])
    assert lower != upper
    for candidate in (lower, upper):
        outside = _reported()
        outside["median_dilution_increment_pct"] = round(candidate, 4)
        assert not case.validate(outside).success


def test_the_interpolation_rule_binds_twice_and_not_a_third_time():
    """House rule: a stated rule that does not bind is stated anyway and the
    case asserts it does not, so a later data change is visible."""
    import numpy as np

    population = case._population()
    profitable = population.loc[~population.loss & population.increment.gt(0)]
    increments = np.sort(
        (profitable.increment / profitable[case.BASIC_SHARES] * 100).to_numpy(dtype=float)
    )
    antidilutive = case._one_per_filing(case.ANTIDILUTIVE, "shares")
    shares = np.sort(
        (antidilutive.value / antidilutive[case.BASIC_SHARES] * 100).to_numpy(dtype=float)
    )
    for sample in (increments, shares):
        assert len(sample) % 2 == 0
        assert sample[len(sample) // 2 - 1] != sample[len(sample) // 2]

    missed = population.loc[population.diluted_gap.ge(case.GAP_THRESHOLD_USD)]
    assert len(missed) % 2 == 1


def test_the_strict_comparisons_have_ties_behind_them():
    """Each of the three is stated as strictly greater and each has filings
    sitting exactly on it, so the strict reading is doing work."""
    antidilutive = case._one_per_filing(case.ANTIDILUTIVE, "shares").copy()
    antidilutive["increment"] = (
        antidilutive[case.DILUTED_SHARES] - antidilutive[case.BASIC_SHARES]
    )
    assert int(antidilutive.value.eq(antidilutive.increment).sum()) > 0

    share_based = case._one_per_filing(case.SHARE_BASED_INCREMENT, "shares").copy()
    share_based["increment"] = (
        share_based[case.DILUTED_SHARES] - share_based[case.BASIC_SHARES]
    )
    ties = int(share_based.value.eq(share_based.increment).sum())
    assert ties > 0.5 * len(share_based)

    consolidated = case._one_per_filing(case.CONSOLIDATED, "USD")
    difference = (consolidated.value - consolidated[case.NET_INCOME]).abs()
    assert int(difference.gt(1.0).sum()) == int(difference.ge(1.0).sum())


def test_the_parent_figure_is_a_different_answer():
    """Reading the consolidated total as the amount attributable to the parent
    counts every overlapping filer and leaves no difference to measure."""
    population = case._population()
    facts = case.load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    facts["end_date"] = facts.end_date.astype(str)
    anchored = facts.merge(
        population[["accession", "period_end"]], on="accession"
    )
    parent = anchored.loc[
        anchored.concept.astype(str).eq("NetIncomeLoss")
        & anchored.unit.astype(str).eq("USD")
        & anchored.end_date.eq(anchored.period_end)
    ]
    naive = int(parent.accession.nunique())
    assert naive > case.analysis()["consolidated_income_reported_count"]
    outside = _reported()
    outside["consolidated_income_reported_count"] = naive
    assert not case.validate(outside).success


def test_the_competing_income_and_noncontrolling_concepts_are_rejected():
    """Three competing readings exist in the fact set and each gives a
    different count, so the query pins which one is asked for."""
    population = case._population().drop_duplicates("accession")
    facts = case.load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    facts["end_date"] = facts.end_date.astype(str)
    facts["start_date"] = facts.start_date.astype(str)
    anchored = facts.loc[
        facts.accession.isin(set(population.accession)) & facts.form.astype(str).eq("10-K")
    ].merge(population[["accession", "period_end"]], on="accession")
    days = pd.to_datetime(anchored.end_date, errors="coerce") - pd.to_datetime(
        anchored.start_date, errors="coerce"
    )
    anchored = anchored.loc[
        anchored.end_date.eq(anchored.period_end) & days.dt.days.between(*case.DURATION_DAYS)
    ]

    def filers(concept):
        return set(
            anchored.loc[
                anchored.concept.astype(str).eq(concept) & anchored.unit.astype(str).eq("USD")
            ].accession
        )

    answers = case.analysis()
    nonredeemable_only = filers(
        "NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest"
    )
    assert 0 < len(nonredeemable_only) < answers["consolidated_income_reported_count"]
    outside = _reported()
    outside["consolidated_income_reported_count"] = len(nonredeemable_only)
    assert not case.validate(outside).success

    split = filers("NetIncomeLossAttributableToRedeemableNoncontrollingInterest") | filers(
        "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest"
    )
    union = filers(case.NONCONTROLLING) | split
    assert len(union) > answers["noncontrolling_reported_count"]
    outside = _reported()
    outside["noncontrolling_reported_count"] = len(union)
    assert not case.validate(outside).success


def test_counting_each_joint_report_once_is_a_different_answer():
    population = case._population()
    single = population.drop_duplicates("accession")
    assert len(single) < len(population)
    outside = _reported()
    outside["eps_filer_count"] = len(single)
    assert not case.validate(outside).success


def test_excluding_filers_that_also_split_the_noncontrolling_amount_is_a_different_answer():
    """Some filers report the undivided amount and its redeemable or
    nonredeemable parts; dropping them is the competing reading the query
    rules out."""
    population = case._population()
    facts = case.load_facts().copy()
    facts["accession"] = facts.accession.astype(str)
    facts["end_date"] = facts.end_date.astype(str)
    rows = facts.loc[
        facts.accession.isin(set(population.accession))
        & facts.concept.astype(str).isin([
            "NetIncomeLossAttributableToRedeemableNoncontrollingInterest",
            "NetIncomeLossAttributableToNonredeemableNoncontrollingInterest",
        ])
        & facts.unit.astype(str).eq("USD")
    ].merge(population[["accession", "period_end"]], on="accession")
    days = (pd.to_datetime(rows.end_date) - pd.to_datetime(rows.start_date, errors="coerce")).dt.days
    split = set(rows.loc[rows.end_date.eq(rows.period_end) & days.between(*case.DURATION_DAYS), "accession"])
    whole = set(case._one_per_filing(case.NONCONTROLLING, "USD").accession)
    assert whole & split
    outside = _reported()
    outside["noncontrolling_reported_count"] = len(whole - split)
    assert not case.validate(outside).success


def test_rounding_the_quotient_first_is_a_different_answer():
    population = case._population()
    rounded = ((population[case.NET_INCOME] / population[case.DILUTED_SHARES]).round(2)
               - population[case.DILUTED_EPS]).abs()
    outside = _reported()
    outside["diluted_numerator_gap_count"] = int((rounded >= case.GAP_THRESHOLD_USD - 1e-9).sum())
    assert not case.validate(outside).success


def test_counting_only_positive_antidilutive_amounts_is_a_different_answer():
    disclosed = case._one_per_filing(case.ANTIDILUTIVE, "shares")
    assert disclosed.value.eq(0).any()
    outside = _reported()
    outside["antidilutive_disclosure_count"] = int(disclosed.value.gt(0).sum())
    assert not case.validate(outside).success


# --- rejections -------------------------------------------------------------

def test_strict_display_boundary_both_sides():
    answers = case.analysis()
    reported = _reported()
    for variable in case.variables:
        precision = case.PRECISION[variable.name]
        for direction in (1, -1):
            outside = dict(reported)
            step = 0.7 * 10 ** (-precision) if precision else 1
            outside[variable.name] = answers[variable.name] + direction * step
            assert not case.validate(outside).success, f"{variable.name} {direction:+d}"


# --- source dependence ------------------------------------------------------

def test_both_sources_are_load_bearing(monkeypatch):
    for missing in ("load_financial_statements", "load_facts"):
        _clear()

        def guarded(*args, **kwargs):
            raise RuntimeError("unavailable")

        try:
            monkeypatch.setattr(case, missing, guarded)
            with pytest.raises(RuntimeError):
                case.analysis()
        finally:
            monkeypatch.undo()
            _clear()
    case.analysis()
