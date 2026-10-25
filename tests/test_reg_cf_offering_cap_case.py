"""Boundaries for the Regulation Crowdfunding cap and Form D follow-on case.

Anchors pin the literals. Property tests name no literal, so each can fail on
its own rather than being decided by an equality asserted earlier in the same
function.
"""

import numpy as np
import pytest

from evals.temporal_path import reg_cf_offering_cap_and_reg_d_follow_on as case


def _clear():
    for cached in (case._offerings, case._form_d_dates, case.analysis, case.ground_truth):
        cached.cache_clear()


def _reported():
    answers = case.analysis()
    return {
        v.name: round(answers[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] else answers[v.name]
        for v in case.variables
    }


def _window():
    offerings = case._offerings()
    window = offerings.loc[offerings.filed.between(*case.FOLLOW_WINDOW)].copy()
    window["followed"] = case._followed(window)
    return window


# --- anchors ---------------------------------------------------------------

def test_anchor_values():
    answers = case.analysis()
    assert answers["pre_change_filing_count"] == 2124
    assert answers["pre_change_highest_maximum_usd"] == 1070000
    assert answers["pre_change_at_highest_count"] == 636
    assert answers["post_change_filing_count"] == 6729
    assert answers["post_change_highest_maximum_usd"] == 5000000
    assert answers["post_change_at_highest_count"] == 391
    assert answers["first_filing_above_old_highest"] == "2021-05-11"
    assert round(answers["pre_change_capped_follow_on_pct"], 2) == 15.67
    assert round(answers["pre_change_uncapped_follow_on_pct"], 2) == 6.83
    assert round(answers["post_change_capped_follow_on_pct"], 2) == 12.50
    assert round(answers["post_change_uncapped_follow_on_pct"], 2) == 5.36
    assert answers["capped_issuer_count"] == 618
    assert answers["capped_issuers_returning_above_count"] == 81
    assert answers["uncapped_issuer_count"] == 1346
    assert answers["uncapped_issuers_returning_above_count"] == 95


# --- properties (no literals) ----------------------------------------------

def test_both_ceilings_are_hard():
    offerings = case._offerings()
    answers = case.analysis()
    before = offerings.loc[~offerings.after_change]
    after = offerings.loc[offerings.after_change]
    assert before.maximum.le(answers["pre_change_highest_maximum_usd"]).all()
    assert after.maximum.le(answers["post_change_highest_maximum_usd"]).all()
    assert answers["post_change_highest_maximum_usd"] > answers["pre_change_highest_maximum_usd"]


def test_filings_bunch_at_the_ceiling():
    answers = case.analysis()
    assert answers["pre_change_at_highest_count"] > 0.2 * answers["pre_change_filing_count"]
    assert answers["post_change_at_highest_count"] > 0.05 * answers["post_change_filing_count"]


def test_the_first_offering_above_the_old_ceiling_came_after_the_change():
    assert case.analysis()["first_filing_above_old_highest"] > case.CHANGE_DATE


def test_capped_issuers_follow_on_more_under_both_ceilings():
    answers = case.analysis()
    for label in ("pre", "post"):
        capped = answers[f"{label}_change_capped_follow_on_pct"]
        uncapped = answers[f"{label}_change_uncapped_follow_on_pct"]
        assert capped > 2 * uncapped > 0


def test_capped_issuers_return_above_the_old_ceiling_more_often():
    answers = case.analysis()
    capped = answers["capped_issuers_returning_above_count"] / answers["capped_issuer_count"]
    uncapped = answers["uncapped_issuers_returning_above_count"] / answers["uncapped_issuer_count"]
    assert capped > 1.5 * uncapped > 0


def test_the_two_issuer_groups_are_disjoint_and_cover_the_earlier_period():
    offerings = case._offerings()
    before = offerings.loc[~offerings.after_change]
    answers = case.analysis()
    assert answers["capped_issuer_count"] + answers["uncapped_issuer_count"] == before.cik.nunique()


# --- the conventions the query states ----------------------------------------

def test_the_window_end_sits_clear_of_the_365_day_boundary():
    window = _window()
    dates = case._form_d_dates()
    gaps = []
    for cik, filed in zip(window.cik, window.filed):
        later = dates.get(cik)
        if later is None:
            continue
        start = np.datetime64(filed)
        first = np.searchsorted(later, start)
        if first < len(later):
            gaps.append((later[first] - start) / np.timedelta64(1, "D"))
    gaps = np.array(gaps)
    assert not np.isin(gaps, [364, 365]).any()
    assert (gaps == 0).any()


def test_a_strictly_later_window_is_a_different_answer():
    """Same-day notices count; excluding them moves the capped share."""
    window = _window()
    dates = case._form_d_dates()

    def strictly_after(cik, filed):
        later = dates.get(cik)
        if later is None:
            return False
        start = np.datetime64(filed)
        first = np.searchsorted(later, start, side="right")
        return bool(first < len(later) and later[first] <= start + np.timedelta64(case.FOLLOW_DAYS, "D"))

    window["strict"] = [strictly_after(c, f) for c, f in zip(window.cik, window.filed)]
    answers = case.analysis()
    outside = _reported()
    changed = False
    for label, part, ceiling in (
        ("pre", window.loc[~window.after_change], answers["pre_change_highest_maximum_usd"]),
        ("post", window.loc[window.after_change], answers["post_change_highest_maximum_usd"]),
    ):
        for group, mask in (("capped", part.maximum.eq(ceiling)), ("uncapped", part.maximum.ne(ceiling))):
            value = round(float(part.loc[mask, "strict"].mean() * 100), 2)
            key = f"{label}_change_{group}_follow_on_pct"
            changed |= value != round(answers[key], 2)
            outside[key] = value
    assert changed
    assert not case.validate(outside).success


def test_counting_issuers_is_a_different_answer():
    window = _window()
    answers = case.analysis()
    outside = _reported()
    for label, part, ceiling in (
        ("pre", window.loc[~window.after_change], answers["pre_change_highest_maximum_usd"]),
        ("post", window.loc[window.after_change], answers["post_change_highest_maximum_usd"]),
    ):
        part = part.assign(capped=part.maximum.eq(ceiling))
        issuers = part.groupby(["cik", "capped"]).followed.any().reset_index()
        for group, flag in (("capped", True), ("uncapped", False)):
            outside[f"{label}_change_{group}_follow_on_pct"] = round(
                float(issuers.loc[issuers.capped.eq(flag), "followed"].mean() * 100), 2
            )
    assert not case.validate(outside).success


def test_collapsing_same_day_duplicates_is_a_different_answer():
    offerings = case._offerings()
    collapsed = offerings.drop_duplicates(["cik", "filed", "maximum"])
    assert len(collapsed) < len(offerings)
    outside = _reported()
    outside["pre_change_filing_count"] = int((~collapsed.after_change).sum())
    outside["post_change_filing_count"] = int(collapsed.after_change.sum())
    assert not case.validate(outside).success


def test_the_narrower_window_is_a_different_answer_for_returning_issuers():
    offerings = case._offerings()
    narrow = offerings.loc[offerings.filed.between(*case.FOLLOW_WINDOW)]
    answers = case.analysis()
    before, after = narrow.loc[~narrow.after_change], narrow.loc[narrow.after_change]
    ceiling = answers["pre_change_highest_maximum_usd"]
    capped = set(before.loc[before.maximum.eq(ceiling), "cik"])
    uncapped = set(before.cik) - capped
    returning = set(after.loc[after.maximum > ceiling, "cik"])
    outside = _reported()
    outside["capped_issuer_count"] = len(capped)
    outside["capped_issuers_returning_above_count"] = len(capped & returning)
    outside["uncapped_issuer_count"] = len(uncapped)
    outside["uncapped_issuers_returning_above_count"] = len(uncapped & returning)
    assert not case.validate(outside).success


def test_admitting_issuers_with_no_stated_maximum_is_a_different_answer():
    table = case.load_expansion_table("sec_form_c")
    originals = table.loc[table.SUBMISSION_TYPE.astype(str).eq("C")].copy()
    originals["filed"] = originals.FILING_DATE.astype(str)
    originals["cik"] = case._cik(originals.CIK)
    earlier = originals.loc[originals.filed.between(case.CAP_WINDOW[0], "2021-03-14")]
    offerings = case._offerings()
    before = offerings.loc[~offerings.after_change]
    ceiling = case.analysis()["pre_change_highest_maximum_usd"]
    capped = set(before.loc[before.maximum.eq(ceiling), "cik"])
    wider = set(earlier.cik.dropna()) - capped
    assert len(wider) > case.analysis()["uncapped_issuer_count"]
    outside = _reported()
    outside["uncapped_issuer_count"] = len(wider)
    assert not case.validate(outside).success


# --- rejections -------------------------------------------------------------

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

@pytest.mark.parametrize("missing", ["sec_form_c", "sec_form_d"])
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
