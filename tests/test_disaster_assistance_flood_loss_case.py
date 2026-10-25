"""Boundaries for the disaster-assistance versus insured-loss case.

Anchors and properties are kept in separate tests so that a relational
assertion is never decided by a literal equality asserted earlier in the same
function.
"""

from decimal import Decimal

import pytest

from core.data import load_expansion_table
from evals.calculation_reconciliation import disaster_household_assistance_versus_insured_flood_loss as case


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else answers[v.name]
        for v in case.variables
    }


def _clear():
    for cached in (case._designations, case._counties, case._claims, case.analysis, case.ground_truth):
        cached.cache_clear()


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["designation_row_count"] == 1625
    assert answers["disaster_count"] == 49
    assert answers["designated_county_count"] == 877
    assert answers["major_declaration_row_count"] == 812
    assert answers["ihp_declared_row_count"] == 368
    assert answers["ihp_county_count"] == 319
    assert answers["public_assistance_only_county_count"] == 311
    assert answers["multiply_designated_county_count"] == 149
    assert answers["claim_record_count"] == 102009
    assert round(answers["claim_net_paid_mn"], 4) == 8798.2578
    assert answers["matched_county_count"] == 382
    assert answers["matched_ihp_county_count"] == 260
    assert round(answers["ihp_county_paid_mn"], 4) == 8390.4851
    assert round(answers["non_ihp_county_paid_mn"], 4) == 160.7216
    assert answers["undeclared_county_count"] == 811
    assert round(answers["undeclared_county_paid_mn"], 4) == 247.051


# --- properties (no literals) ----------------------------------------------

def test_the_counts_and_the_money_point_opposite_ways():
    """The finding of the case. If both readings agreed there would be no
    comparison to draw."""
    answers = case.analysis()
    assert answers["undeclared_county_count"] > answers["matched_county_count"]
    outside = answers["undeclared_county_paid_mn"] + answers["non_ihp_county_paid_mn"]
    assert answers["ihp_county_paid_mn"] > 10 * outside


def test_emergency_declarations_never_carry_the_household_programme():
    designations = case._designations()
    emergency = designations.loc[designations.declaration_type.ne(case.MAJOR_DECLARATION)]
    assert len(emergency) > 0
    assert not emergency[case.HOUSEHOLD_FLAG].any()


def test_the_similarly_named_flag_carries_nothing():
    """A screen written against the other household-sounding field selects an
    empty set rather than failing."""
    frame = load_expansion_table("fema_disasters")
    assert not frame.individual_assistance_declared.any()
    assert case.analysis()["ihp_declared_row_count"] > 0


def test_reported_subsets_are_strict():
    answers = case.analysis()
    assert 0 < answers["ihp_county_count"] < answers["designated_county_count"]
    assert 0 < answers["matched_ihp_county_count"] < answers["matched_county_count"]
    assert 0 < answers["major_declaration_row_count"] < answers["designation_row_count"]
    assert 0 < answers["multiply_designated_county_count"] < answers["ihp_county_count"] + answers["public_assistance_only_county_count"]


def test_the_net_columns_are_the_complete_ones():
    claims = case._claims()
    gross = ["gross_building_payment_usd", "gross_contents_payment_usd", "gross_icc_payment_usd"]
    assert claims[list(case.NET_PAYMENTS)].notna().all().all()
    assert claims[gross].isna().any().any()


def test_negative_and_zero_payments_are_kept():
    claims = case._claims()
    assert int((claims.paid_usd < 0).sum()) > 0
    assert int((claims.paid_usd == 0).sum()) > 0


def test_tribal_and_statewide_designations_are_outside_the_population():
    """A code ending 000 is not a county, and several distinct tribal areas in
    one state share it, so keeping them would both inflate and merge."""
    frame = load_expansion_table("fema_disasters")
    window = frame.loc[
        frame.incident_begin_date.astype(str).str[:4].eq(case.INCIDENT_YEAR)
        & frame.incident_type.isin(case.INCIDENT_TYPES)
    ].copy()
    window["county_fips"] = window.county_fips.astype(str)
    dropped = window.loc[window.county_fips.str.endswith("000")]
    assert len(dropped) > 0
    assert dropped.designated_area.nunique() > dropped.county_fips.nunique()
    assert case._designations().county_fips.str.endswith("000").sum() == 0
    assert window.county_fips.nunique() > case.analysis()["designated_county_count"]


def test_both_sides_screen_geography_alike():
    assert not case._claims().county_fips.str.endswith("000").any()
    assert not case._designations().county_fips.str.endswith("000").any()


# --- rejections -------------------------------------------------------------

def test_including_emergency_declarations_is_a_different_answer():
    designations = case._designations()
    answers = case.analysis()
    every_county = designations.county_fips.nunique()
    major_county = designations.loc[
        designations.declaration_type.eq(case.MAJOR_DECLARATION)
    ].county_fips.nunique()
    assert every_county > major_county
    assert answers["designated_county_count"] == every_county
    assert answers["ihp_county_count"] + answers["public_assistance_only_county_count"] <= major_county


def test_the_gross_total_is_a_different_answer():
    claims = case._claims()
    answers = case.analysis()
    gross = ["gross_building_payment_usd", "gross_contents_payment_usd", "gross_icc_payment_usd"]
    naive = float(
        sum((Decimal(str(x)) for x in claims[gross].sum(axis=1).dropna()), Decimal(0)) / case.MILLION
    )
    assert round(naive, 4) != round(answers["claim_net_paid_mn"], 4)
    outside = _reported()
    outside["claim_net_paid_mn"] = round(naive, 4)
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

def test_the_paid_amounts_split_the_claim_total():
    """Stronger than checking the loader is called: the three reported amounts
    must exhaust the total, so every paying county lands in exactly one group."""
    answers = case.analysis()
    parts = (
        answers["ihp_county_paid_mn"]
        + answers["non_ihp_county_paid_mn"]
        + answers["undeclared_county_paid_mn"]
    )
    assert abs(parts - answers["claim_net_paid_mn"]) < 1e-6


@pytest.mark.parametrize("missing", ["fema_disasters", "fema_nfip"])
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
