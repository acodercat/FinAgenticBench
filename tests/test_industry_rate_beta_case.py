"""Case-local gate for industry_rate_beta_regime_reversal.

The case rests on a sign change that has to stay in the data, on a cross-section
whose two extremes have to stay apart, and on four conventions that each move the
answer past the tolerance. All of them are recomputed here, and the closed-form
partial slope is checked against an independent least-squares fit.
"""

import json

import numpy as np
import pytest
import pandas as pd

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.comparison_ranking import industry_rate_beta_regime_reversal as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/comparison_ranking/industry_rate_beta_regime_reversal.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["paired_trading_day_count"] == 1746
    assert truth["first_positive_correlation_year"] == 2022
    assert truth["most_negative_rate_beta_industry"] == "utils_pct"
    assert truth["most_positive_rate_beta_industry"] == "enrgy_pct"
    assert truth["largest_beta_decline_industry"] == "enrgy_pct"
    assert truth["late_regime_negative_beta_count"] == 9
    assert truth["sign_change_industry_count"] == 6
    assert round(truth["early_regime_correlation"], 4) == -0.3864
    assert round(truth["late_regime_correlation"], 4) == 0.0731
    assert round(truth["beta_ordering_rank_correlation"], 4) == 0.8303


def test_the_sign_change_is_real_and_the_last_year_turns_back():
    truth = _truth()
    panel, _ = case._panel()
    yearly = {int(y): case._correlation(g) for y, g in panel.groupby(panel.index.year)}
    assert truth["early_regime_correlation"] < 0 < truth["late_regime_correlation"], yearly
    negative_years = [y for y, v in sorted(yearly.items()) if v < 0]
    positive_years = [y for y, v in sorted(yearly.items()) if v > 0]
    assert negative_years and positive_years
    assert min(positive_years) == truth["first_positive_correlation_year"]
    # The regime is not permanent: the window ends negative again.
    assert yearly[max(yearly)] < 0, yearly


def test_every_pinned_reading_clears_the_tolerance():
    """Each clause the query spends words on has to change the answer."""
    truth = _truth()
    from core.data import load_expansion_table, load_ken_french_table

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["observed_on"] = pd.to_datetime(curve.Date)
    curve = curve.set_index("observed_on").sort_index()
    factors = load_ken_french_table("daily_factors").copy()
    factors["observed_on"] = pd.to_datetime(factors.date)
    factors = factors.set_index("observed_on").sort_index()
    industries = load_ken_french_table("daily_industry_returns").copy()
    industries["observed_on"] = pd.to_datetime(industries.date)
    industries = industries.set_index("observed_on").sort_index()
    columns = [name for name in industries.columns if name.endswith("_pct")]

    def regimes(tenor="10 Yr", excess=False, calendar=False):
        yields = pd.to_numeric(curve[tenor], errors="coerce").dropna()
        market = pd.to_numeric(factors.mkt_rf_pct, errors="coerce")
        if not excess:
            market = market + pd.to_numeric(factors.rf_pct, errors="coerce")
        frame = (industries[columns]
                 .join(yields.rename("tenor_yield"), how="inner")
                 .join(market.rename("market"), how="inner").dropna())
        if calendar:
            daily = yields.reindex(
                pd.date_range(yields.index.min(), yields.index.max(), freq="D")
            ).ffill().diff()
            frame["yield_change"] = daily.reindex(frame.index)
        else:
            frame["yield_change"] = frame.tenor_yield.diff()
        frame = frame.dropna(subset=["yield_change"])
        split = truth["first_positive_correlation_year"]
        early = frame.loc[frame.index.year < split]
        late = frame.loc[frame.index.year >= split]
        return case._correlation(early), case._correlation(late)

    pinned = (truth["early_regime_correlation"], truth["late_regime_correlation"])
    assert (round(pinned[0], 4), round(pinned[1], 4)) == (-0.3864, 0.0731)
    for label, kwargs, expected in [
        ("excess market return", {"excess": True}, (-0.3866, 0.0728)),
        ("calendar differencing", {"calendar": True}, (-0.3859, 0.0732)),
        ("two-year yield", {"tenor": "2 Yr"}, (-0.3239, -0.0051)),
        ("five-year yield", {"tenor": "5 Yr"}, (-0.3440, 0.0438)),
        ("thirty-year yield", {"tenor": "30 Yr"}, (-0.3799, 0.0748)),
    ]:
        early, late = regimes(**kwargs)
        assert (round(early, 4), round(late, 4)) == expected, (label, early, late)
        assert not (numeric_equal(early, pinned[0], decimals=4)
                    and numeric_equal(late, pinned[1], decimals=4)), label
    # The short end never joins the reversal, which is why the tenor is pinned.
    assert regimes(tenor="2 Yr")[1] < 0


def test_the_flip_year_is_robust_to_those_readings():
    """The year is derived rather than stated, so it must not depend on a convention."""
    truth = _truth()
    panel, _ = case._panel()
    yearly = {int(y): case._correlation(g) for y, g in panel.groupby(panel.index.year)}
    assert min(y for y, v in yearly.items() if v > 0) == truth["first_positive_correlation_year"]
    # The year before the flip is clearly negative, so the boundary is not marginal.
    before = yearly[truth["first_positive_correlation_year"] - 1]
    after = yearly[truth["first_positive_correlation_year"]]
    assert before < -0.05 < 0.05 < after, (before, after)


def test_the_day_count_is_unambiguous_on_every_prompt_surface():
    """The join and the differencing give counts one apart, so the wording must choose."""
    truth = _truth()
    panel, columns = case._panel()
    from core.data import load_expansion_table, load_ken_french_table

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["observed_on"] = pd.to_datetime(curve.Date)
    yields = pd.to_numeric(curve[case.TENOR], errors="coerce").set_axis(
        curve.observed_on).dropna().sort_index()
    industries = load_ken_french_table("daily_industry_returns").copy()
    industries["observed_on"] = pd.to_datetime(industries.date)
    industries = industries.set_index("observed_on").sort_index()
    factors = load_ken_french_table("daily_factors").copy()
    factors["observed_on"] = pd.to_datetime(factors.date)
    factors = factors.set_index("observed_on").sort_index()
    market = (pd.to_numeric(factors.mkt_rf_pct, errors="coerce")
              + pd.to_numeric(factors.rf_pct, errors="coerce")).rename("market")
    joined = (industries[columns].join(yields.rename("tenor_yield"), how="inner")
              .join(market, how="inner").dropna())

    assert len(joined) == len(panel) + 1, "the two counts no longer differ"
    assert truth["paired_trading_day_count"] == len(panel)
    surfaces = [turn["query"] for turn in _turns()] + [
        v.description for v in case.variables]
    for surface in surfaces:
        assert "days remain" not in surface.lower(), surface
        assert "jointly support" not in surface.lower(), surface
    assert "how many days carry a yield change" in _turns()[0]["query"]
    by_name = {v.name: v for v in case.variables}
    assert "carry a yield change" in by_name["paired_trading_day_count"].description


def test_the_statistics_are_free_of_a_degrees_of_freedom_choice():
    """The query pins no ddof, which is only safe because none of the figures move."""
    panel, columns = case._panel()
    change = panel.yield_change
    for name in columns:
        one = float(np.cov(panel[name], change, ddof=1)[0, 1] / change.var(ddof=1))
        zero = float(np.cov(panel[name], change, ddof=0)[0, 1] / change.var(ddof=0))
        assert abs(one - zero) < 1e-12, name
    surfaces = [turn["query"] for turn in _turns()]
    for surface in surfaces:
        assert "degree of freedom" not in surface.lower()


def test_no_calendar_year_is_a_stub():
    """The flip year is derived per calendar year, so a short year could fake one."""
    panel, _ = case._panel()
    counts = panel.groupby(panel.index.year).size()
    assert counts.min() / counts.max() > 0.95, counts.to_dict()
    assert len(counts) == 7, counts.to_dict()
    # The flip year and the one before it are both full, so the boundary is not an artefact.
    flip = _truth()["first_positive_correlation_year"]
    assert counts[flip] > 240 and counts[flip - 1] > 240


def test_the_closed_form_partial_slope_matches_least_squares():
    """An independent fit of the same two-regressor model."""
    panel, columns = case._panel()
    for name in columns:
        design = np.column_stack(
            [np.ones(len(panel)), panel.yield_change.to_numpy(), panel.market.to_numpy()]
        )
        fitted = np.linalg.lstsq(design, panel[name].to_numpy(), rcond=None)[0][1]
        assert abs(fitted - case._partial_slope(panel, name)) < 1e-9, name
    # And the single-regressor slope against its own least-squares fit.
    for name in columns:
        design = np.column_stack([np.ones(len(panel)), panel.yield_change.to_numpy()])
        fitted = np.linalg.lstsq(design, panel[name].to_numpy(), rcond=None)[0][1]
        assert abs(fitted - case._slope(panel, name)) < 1e-9, name


def test_the_cross_section_findings_survive():
    truth = _truth()
    panel, columns = case._panel()
    whole = {name: case._slope(panel, name) for name in columns}
    negative = [name for name in columns if whole[name] < 0]
    assert negative == [truth["most_negative_rate_beta_industry"]], negative
    split = truth["first_positive_correlation_year"]
    early = panel.loc[panel.index.year < split]
    late = panel.loc[panel.index.year >= split]
    declines = {n: case._slope(late, n) - case._slope(early, n) for n in columns}
    assert all(value < 0 for value in declines.values()), declines
    assert min(declines, key=declines.get) == truth["largest_beta_decline_industry"]
    controlled = {name: case._partial_slope(panel, name) for name in columns}
    flips = [n for n in columns if np.sign(whole[n]) != np.sign(controlled[n])]
    assert len(flips) == truth["sign_change_industry_count"]
    # The two extremes hold under both specifications.
    assert min(controlled, key=controlled.get) == truth["most_negative_rate_beta_industry"]
    assert max(controlled, key=controlled.get) == truth["most_positive_rate_beta_industry"]


def test_no_selection_can_tie_and_the_margins_are_real():
    truth = _truth()
    panel, columns = case._panel()
    whole = sorted(case._slope(panel, name) for name in columns)
    controlled = sorted(case._partial_slope(panel, name) for name in columns)
    assert len(set(whole)) == len(set(controlled)) == len(columns), "a sensitivity is tied"
    assert round(whole[1] - whole[0], 4) == 0.8644
    assert round(whole[-1] - whole[-2], 4) == 4.1802
    assert round(controlled[1] - controlled[0], 4) == 0.7198
    assert round(controlled[-1] - controlled[-2], 4) == 4.2990
    split = truth["first_positive_correlation_year"]
    early = panel.loc[panel.index.year < split]
    late = panel.loc[panel.index.year >= split]
    falls = sorted(case._slope(late, n) - case._slope(early, n) for n in columns)
    assert round(falls[1] - falls[0], 4) == 2.0140
    # The rank correlation needs no tie rule because no two values coincide.
    assert "average rank" not in _turns()[3]["query"]


@pytest.mark.parametrize("removed", ["treasury_yield_curve", "daily_industry_returns",
                                     "daily_factors"])
def test_removing_any_declared_source_makes_the_case_impossible(removed, monkeypatch):
    """Strong source removal, enforced rather than asserted in prose.

    The curve supplies the regressor, the industry file the cross-section and the
    factor file the market return that turns one and four. Taking any of the three
    away has to stop the case, not merely thin it.
    """
    expansion, french = case.load_expansion_table, case.load_ken_french_table

    def blocked_expansion(name):
        if name == removed:
            raise FileNotFoundError(f"removed source {name}")
        return expansion(name)

    def blocked_french(name):
        if name == removed:
            raise FileNotFoundError(f"removed source {name}")
        return french(name)

    case._panel.cache_clear()
    case.ground_truth.cache_clear()
    try:
        with monkeypatch.context() as context:
            context.setattr(case, "load_expansion_table", blocked_expansion)
            context.setattr(case, "load_ken_french_table", blocked_french)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        case._panel.cache_clear()
        case.ground_truth.cache_clear()


def test_the_market_control_is_what_the_factor_file_buys():
    """The third source is not a duplicate: it changes six of the ten answers."""
    truth = _truth()
    panel, columns = case._panel()
    assert len(columns) == 10, columns
    univariate = {name: case._slope(panel, name) for name in columns}
    controlled = {name: case._partial_slope(panel, name) for name in columns}
    differing = [n for n in columns if not numeric_equal(univariate[n], controlled[n], decimals=4)]
    assert len(differing) == len(columns), "the control now changes nothing"
    assert truth["sign_change_industry_count"] == 6


def test_every_text_output_declares_a_closed_representation():
    truth = _truth()
    by_name = {variable.name: variable for variable in case.variables}
    text_outputs = [name for name, value in truth.items() if isinstance(value, str)]
    assert len(text_outputs) == 3
    for name in text_outputs:
        assert has_closed_text_representation(by_name[name].description), name
        assert truth[name].endswith("_pct")


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = getattr(case, turn["validator"])
        outputs = {name: truth[name] for name in turn["stores"]}
        assert checker(dict(outputs)).success
        broken = dict(outputs)
        first = turn["stores"][0]
        broken[first] = (
            "__WRONG__" if isinstance(broken[first], str) else float(broken[first]) + 1_000.0
        )
        assert not checker(broken).success, turn["validator"]


def test_claim_contract_is_well_formed():
    for turn in _turns():
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()
            if "overlap_waiver" in claim:
                assert claim["overlap_waiver"].strip()


def test_judge_reference_carries_the_full_contract():
    reference = SPEC["judge_reference"]
    assert set(reference) == {"source", "facts", "known_overclaims"}
    assert reference["source"].strip() and len(reference["known_overclaims"]) >= 4
    facts = " ".join(reference["facts"])
    truth = _truth()
    assert f"{truth['paired_trading_day_count']:,}" in facts
    assert "-0.3864" in facts and "0.0731" in facts
    assert "-0.0051" in facts
    assert str(truth["first_positive_correlation_year"]) in facts
