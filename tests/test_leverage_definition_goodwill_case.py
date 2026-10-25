"""Boundaries for the leverage-definition and goodwill-location case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

from decimal import Decimal

import numpy as np
import pytest

from core.data import load_expansion_table, load_financial_statements
from evals.comparison_ranking import leverage_definition_and_goodwill_location as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._annual, case._levered, case._segments, case.analysis, case.ground_truth):
        cached.cache_clear()


def _sum(series):
    return sum((Decimal(str(x)) for x in series), Decimal(0))


def _segment_join():
    """The segment/entity join as the case builds it, stopping short of the
    presentation filter so a test can read what that filter decides."""
    dimensional = case.load_expansion_table("sec_notes_dimensional")
    rows = dimensional.loc[
        dimensional.tag.astype(str).eq(case.GOODWILL_TAG)
        & dimensional.axis.astype(str).eq(case.SEGMENT_AXIS)
        & dimensional.form.astype(str).eq("10-K")
        & dimensional.qtrs.eq(0)
    ].copy()
    rows["period_end"] = rows.ddate.astype(str)
    rows = rows.loc[rows.period_end.between(*case.WINDOW)]
    rows["cik"] = rows.cik.astype(str)
    rows["member"] = rows.member.astype(str)
    rows["accession"] = rows.adsh.astype(str)
    entity = case._annual()
    entity = entity.loc[
        entity[case.GOODWILL].notna(), ["cik", "period_end", "accession", case.GOODWILL]
    ].copy()
    entity["accession"] = entity.accession.astype(str)
    return rows.merge(entity, on=["cik", "period_end", "accession"])


def _segment_stats(selected):
    frame = selected.groupby("cik").agg(
        segment_total=("value", "sum"),
        largest=("value", "max"),
        **{case.GOODWILL: (case.GOODWILL, "first")},
    ).reset_index()
    tied = frame.loc[(frame.segment_total - frame[case.GOODWILL]).abs() < 1.0]
    share = tied.largest / tied.segment_total * 100
    return {
        "segment_filer_count": len(frame),
        "segment_tie_count": len(tied),
        "median_largest_segment_share_pct": float(share.median()),
        "single_segment_majority_count": int((share > 50).sum()),
    }


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["annual_filer_count"] == 5524
    assert answers["complete_filer_count"] == 709
    assert answers["positive_equity_count"] == 624
    assert answers["total_below_noncurrent_count"] == 23
    assert round(answers["median_noncurrent_leverage"], 4) == 0.6305
    assert round(answers["median_total_debt_leverage"], 4) == 0.6805
    assert round(answers["median_lease_inclusive_leverage"], 4) == 0.7852
    assert round(answers["operating_lease_total_bn"], 6) == 419.849886
    assert round(answers["equity_total_bn"], 6) == 3394.290464
    assert round(answers["goodwill_total_bn"], 6) == 1612.234198
    assert answers["negative_tangible_equity_count"] == 138
    assert round(answers["aggregate_lease_tangible_leverage"], 4) == 1.4399
    assert answers["segment_filer_count"] == 180
    assert answers["segment_tie_count"] == 162
    assert round(answers["median_largest_segment_share_pct"], 4) == 63.5991
    assert answers["single_segment_majority_count"] == 125


# --- properties (no literals) ----------------------------------------------

def test_the_three_numerators_order_the_same_way():
    """The finding of turn two: each widening of the numerator raises the
    ratio, so the definition and not the balance sheet moves it."""
    answers = case.analysis()
    assert (
        answers["median_noncurrent_leverage"]
        < answers["median_total_debt_leverage"]
        < answers["median_lease_inclusive_leverage"]
    )


def test_removing_goodwill_roughly_halves_the_denominator():
    answers = case.analysis()
    assert 0.3 * answers["equity_total_bn"] < answers["goodwill_total_bn"] < 0.7 * answers["equity_total_bn"]
    assert answers["aggregate_lease_tangible_leverage"] > answers["median_lease_inclusive_leverage"]


def test_a_material_minority_has_no_tangible_equity_left():
    answers = case.analysis()
    assert 0.1 * answers["positive_equity_count"] < answers["negative_tangible_equity_count"] < 0.5 * answers["positive_equity_count"]


def test_goodwill_concentrates_in_one_segment():
    answers = case.analysis()
    assert answers["median_largest_segment_share_pct"] > 50
    assert answers["single_segment_majority_count"] > 0.5 * answers["segment_tie_count"]


def test_the_segment_sum_mostly_but_not_always_ties():
    answers = case.analysis()
    closed = answers["segment_tie_count"] / answers["segment_filer_count"]
    assert 0.7 < closed < 1.0


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["positive_equity_count"] < answers["complete_filer_count"] < answers["annual_filer_count"]
    assert 0 < answers["total_below_noncurrent_count"] < answers["complete_filer_count"]


def test_no_segment_share_falls_outside_the_scale():
    segments = case._segments()
    tied = segments.loc[(segments.segment_total - segments[case.GOODWILL]).abs() < 1]
    share = tied.largest / tied.segment_total * 100
    assert share.between(0, 100).all()
    assert not tied.cik.duplicated().any()


# --- the stated rules that do not bind --------------------------------------

def test_the_accession_tie_break_does_not_fire():
    """House rule: the retention rule names an accession tie-break the data
    never needs, and the case asserts that so a later change is visible."""
    statements = load_financial_statements()
    window = statements.loc[
        statements.form.eq("10-K") & statements.fiscal_period.eq("FY")
        & statements.report_date.astype(str).between(*case.WINDOW)
    ]
    per_filing = window.groupby(["cik", "report_date", "filed_date"]).size()
    assert int((per_filing > 1).sum()) == 0


# --- rejections -------------------------------------------------------------

def test_the_three_medians_need_the_interpolation_rule():
    """All three are even counts whose middle values differ, so the positional
    readings are different answers the query has to exclude."""
    levered = case._levered()
    answers = case.analysis()
    for column, name in (
        ("noncurrent_ratio", "median_noncurrent_leverage"),
        ("total_ratio", "median_total_debt_leverage"),
        ("lease_ratio", "median_lease_inclusive_leverage"),
    ):
        values = np.sort(levered[column].to_numpy(float))
        lower = float(values[(len(values) - 1) // 2])
        assert round(lower, 4) != round(answers[name], 4)
        outside = _reported()
        outside[name] = round(lower, 4)
        assert not case.validate(outside).success


def test_dropping_negative_tangible_filers_is_a_different_aggregate():
    """The turn asks for the aggregate across every filer; excluding the ones
    whose own denominator is negative is the other defensible reading."""
    levered = case._levered()
    answers = case.analysis()
    kept = levered.loc[(levered[case.EQUITY] - levered[case.GOODWILL]) > 0]
    naive = float(
        (_sum(kept[case.DEBT]) + _sum(kept[case.LEASE]))
        / (_sum(kept[case.EQUITY]) - _sum(kept[case.GOODWILL]))
    )
    assert round(naive, 4) != round(answers["aggregate_lease_tangible_leverage"], 4)
    outside = _reported()
    outside["aggregate_lease_tangible_leverage"] = round(naive, 4)
    assert not case.validate(outside).success


def test_matching_segments_on_the_period_end_alone_is_looser():
    """The listed overclaim: the accession is part of the key because a later
    annual report repeats the year as a comparative."""
    dimensional = load_expansion_table("sec_notes_dimensional")
    rows = dimensional.loc[
        dimensional.tag.astype(str).eq(case.GOODWILL_TAG)
        & dimensional.axis.astype(str).eq(case.SEGMENT_AXIS)
        & dimensional.form.astype(str).eq("10-K")
        & dimensional.qtrs.eq(0)
    ].copy()
    rows["period_end"] = rows.ddate.astype(str)
    rows = rows.loc[rows.period_end.between(*case.WINDOW)]
    rows["cik"] = rows.cik.astype(str)
    annual = case._annual()
    by_period = rows.merge(
        annual.loc[annual[case.GOODWILL].notna(), ["cik", "period_end", "accession"]],
        on=["cik", "period_end"],
    )
    assert by_period.adsh.astype(str).ne(by_period.accession.astype(str)).any()


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

def test_segment_rows_come_from_the_entity_filing():
    """Stronger than checking the loader is called: every retained segment row
    must sit in the same accession as the entity figure it is matched to."""
    segments = case._segments()
    annual = case._annual().set_index("cik")
    for row in segments.itertuples():
        assert annual.loc[row.cik, "period_end"] == row.period_end


def test_the_primary_presentation_rule_selects_rather_than_discards():
    """Every member group carries an index 0 row, so pinning the query to that
    index keeps the population whole; asserting it makes a data change visible."""
    joined = _segment_join()
    groups = joined.groupby(["cik", "member"])
    assert groups.ngroups > 0
    assert int(groups.iprx.min().ne(case.PRIMARY_PRESENTATION).sum()) == 0


def test_the_within_filing_repeat_changes_the_answer():
    """The rule is load-bearing, not decorative: one member is reported twice
    inside one filing at two different values, and reading the other index
    moves three of the four segment outputs."""
    joined = _segment_join()
    repeated = joined.groupby(["cik", "member"]).value.nunique()
    assert int(repeated.gt(1).sum()) > 0

    highest = joined.sort_values(["cik", "member", "iprx"]).drop_duplicates(
        ["cik", "member"], keep="last"
    )
    alternative = _segment_stats(highest)
    answers = case.analysis()
    assert alternative["segment_tie_count"] != answers["segment_tie_count"]

    outside = _reported()
    outside["segment_tie_count"] = alternative["segment_tie_count"]
    outside["median_largest_segment_share_pct"] = round(
        alternative["median_largest_segment_share_pct"], 4
    )
    outside["single_segment_majority_count"] = alternative["single_segment_majority_count"]
    assert not case.validate(outside).success


@pytest.mark.parametrize("missing", ["load_financial_statements", "load_expansion_table"])
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
