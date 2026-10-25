"""Case-local gate for hpi_estimation_error_and_thin_market_ranking.

The statistical point has to keep holding: the standard error must keep being
published only on the series the case pins, precision must keep tracking lending
volume, the two published series must keep disagreeing, and the top of the
ranking must keep being thin markets. Every assertion can fail.
"""

import json

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.data import load_expansion_table
from core.validation import numeric_equal
from evals.comparison_ranking import hpi_estimation_error_and_thin_market_ranking as case


SPEC = json.loads(
    (PROJECT_ROOT
     / "evals/comparison_ranking/hpi_estimation_error_and_thin_market_ranking.json"
     ).read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


@pytest.fixture(scope="module")
def published():
    return case._annual_growth(case.PUBLISHED_ERROR_FLAVOR)


@pytest.fixture(scope="module")
def counts():
    return case._originations()


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["msa_index_count"] == 410
    assert truth["median_annual_growth_pct"] == 2.978911
    assert truth["median_ending_standard_error"] == 2.380000
    assert truth["change_within_standard_error_msa_count"] == 41
    assert truth["dual_flavor_msa_count"] == 100
    assert truth["mean_absolute_flavor_difference_pp"] == 1.749336
    assert truth["largest_flavor_difference_pp"] == 5.957833
    assert truth["opposite_sign_flavor_msa_count"] == 23
    assert truth["matched_origination_count"] == 5395611
    assert truth["unplaced_origination_count"] == 783114
    assert truth["thin_market_median_standard_error"] == 5.900000
    assert truth["deep_market_median_standard_error"] == 0.590000
    assert truth["fastest_growth_msa_code"] == "21300"
    assert truth["fastest_growth_pct"] == 23.991991
    assert truth["fastest_ending_standard_error"] == 5.990000
    assert truth["fastest_ten_median_origination_count"] == 2037


def test_only_the_pinned_series_publishes_a_standard_error():
    """The flavor choice is not a preference; it is the only one carrying the error."""
    prices = load_expansion_table("fhfa_hpi")
    error = pd.to_numeric(prices.rstderr, errors="coerce")
    has_error = error.notna()
    assert int(has_error.sum()) == 57400
    combos = prices[has_error].groupby(
        [prices.level.astype(str), prices.frequency.astype(str), prices.hpi_flavor.astype(str)]
    ).size()
    assert list(combos.index) == [(case.LEVEL, case.FREQUENCY, case.PUBLISHED_ERROR_FLAVOR)], combos
    purchase = case._metropolitan_panel(case.PURCHASE_FLAVOR)
    assert int(purchase.error.notna().sum()) == 0, (
        "the purchase-only series now publishes a standard error, so the first turn's "
        "series choice is no longer forced"
    )


def test_the_two_published_series_still_disagree(published):
    truth = _truth()
    purchase = case._annual_growth(case.PURCHASE_FLAVOR)
    shared = published.index.intersection(purchase.index)
    assert len(shared) == truth["dual_flavor_msa_count"]
    difference = published.growth[shared] - purchase.growth[shared]
    assert round(float(difference.abs().mean()), 6) == truth["mean_absolute_flavor_difference_pp"]
    assert truth["opposite_sign_flavor_msa_count"] > 0.15 * len(shared), (
        "the two series now agree on direction almost everywhere, so the second turn's "
        "required claim overstates the disagreement"
    )
    # Reading the purchase-only series instead would change the answer.
    assert not numeric_equal(round(float(purchase.growth.median()), 6),
                             truth["median_annual_growth_pct"], decimals=6)


def test_precision_tracks_lending_volume(published, counts):
    truth = _truth()
    matched = published.join(counts.rename("loans"), how="inner").assign(code=lambda f: f.index)
    assert len(matched) == truth["msa_index_count"]
    correlation = float(matched.loans.corr(matched.error, method="spearman"))
    assert correlation < -0.7, correlation
    thinnest = matched.sort_values(["loans", "code"], ascending=[True, True])
    deepest = matched.sort_values(["loans", "code"], ascending=[False, True])
    assert round(float(thinnest.head(case.TAIL_SIZE).error.median()), 6) == truth["thin_market_median_standard_error"]
    assert round(float(deepest.head(case.TAIL_SIZE).error.median()), 6) == truth["deep_market_median_standard_error"]
    assert truth["thin_market_median_standard_error"] > 5 * truth["deep_market_median_standard_error"], (
        "the precision spread has closed, so the third turn's required claim is stale"
    )


def test_the_ranking_selects_thin_markets(published, counts):
    truth = _truth()
    matched = published.join(counts.rename("loans"), how="inner")
    fastest = matched.sort_values("growth", ascending=False)
    assert str(fastest.index[0]) == truth["fastest_growth_msa_code"]
    assert "Elmira" in str(fastest.name.iloc[0])
    assert round(float(fastest.growth.iloc[1]), 6) == 16.246396
    assert fastest.growth.iloc[0] - fastest.growth.iloc[1] > 5, "the leader's margin has narrowed"
    assert int(fastest.head(10).loans.median()) < 0.5 * int(matched.loans.median()), (
        "the fastest movers are no longer unusually thin markets"
    )
    assert int(matched.loans.median()) == 5536


def test_a_third_of_the_panel_moves_inside_its_own_error(published):
    truth = _truth()
    change = (published.ending - published.starting).abs()
    assert int(change.lt(published.error).sum()) == truth["change_within_standard_error_msa_count"]
    assert int(change.lt(2 * published.error).sum()) == 87
    assert int(change.lt(3 * published.error).sum()) == 135


def test_the_register_reaches_beyond_the_panel(counts, published):
    truth = _truth()
    outside = counts.drop(index=published.index, errors="ignore")
    assert int(counts[case.NO_METROPOLITAN_AREA]) == truth["unplaced_origination_count"]
    assert case.NO_METROPOLITAN_AREA not in set(published.index), (
        "the no-metropolitan-area code now appears in the index panel, which the "
        "third turn's two counts assume it does not"
    )
    assert int(outside.sum()) == 796214
    assert 100 * outside.sum() / counts.sum() > 10, (
        "the panel now covers nearly all lending, so the coverage boundary is thin"
    )


@pytest.mark.parametrize("removed", ["fhfa_hpi", "hmda"])
def test_both_sources_are_load_bearing(monkeypatch, removed):
    original = case.load_expansion_table

    def without(name):
        if name == removed:
            raise FileNotFoundError("removed source")
        return original(name)

    case.ground_truth.cache_clear()
    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(case, "load_expansion_table", without)
            with pytest.raises(FileNotFoundError):
                case.ground_truth()
    finally:
        case.ground_truth.cache_clear()


def test_the_code_comparison_accepts_leading_zeros():
    truth = _truth()
    outputs = {name: truth[name] for name in case.TURN_4_NAMES}
    padded = dict(outputs)
    padded["fastest_growth_msa_code"] = "0" + truth["fastest_growth_msa_code"]
    assert case._validate_subset(padded, case.TURN_4_NAMES).success
    wrong = dict(outputs)
    wrong["fastest_growth_msa_code"] = "99999"
    assert not case._validate_subset(wrong, case.TURN_4_NAMES).success


def test_stores_partition_the_module_variables():
    declared = [name for turn in _turns() for name in turn["stores"]]
    assert declared == [variable.name for variable in case.variables]
    assert len(declared) == len(set(declared))


def test_every_turn_validator_rejects_a_wrong_value():
    truth = _truth()
    for turn in _turns():
        checker = case.validators[turn["validator"]]
        outputs = {name: truth[name] for name in turn["stores"]}
        assert case._validate_subset(outputs, turn["stores"]).success
        assert checker is not None
        for name in turn["stores"]:
            broken = dict(outputs)
            broken[name] = ("__WRONG__" if isinstance(broken[name], str)
                            else float(broken[name]) + 1.0)
            assert not case._validate_subset(broken, turn["stores"]).success, (turn["validator"], name)


def test_claim_contract_is_well_formed():
    for turn in _turns():
        assert turn["claims"]
        for claim in turn["claims"]:
            if claim["mode"] == "conditional":
                assert claim["description"].startswith("If ")
            if claim["mode"] == "required":
                assert " does not " not in claim["description"].lower()
            if "overlap_waiver" in claim:
                assert claim["overlap_waiver"].strip()


def test_judge_reference_quotes_recomputed_figures():
    truth = _truth()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    assert f"{truth['msa_index_count']:,}" in facts
    assert f"{truth['median_annual_growth_pct']:.6f}" in facts
    assert f"{truth['matched_origination_count']:,}" in facts
    assert f"{truth['fastest_growth_pct']:.6f}" in facts
    assert "-0.879796" in facts
    assert set(SPEC["judge_reference"]) == {"source", "facts", "known_overclaims"}
    assert SPEC["judge_reference"]["known_overclaims"]


def test_no_tie_binds_either_selection(published, counts):
    """The query states a tie rule; on this extract it never has to be used."""
    matched = published.join(counts.rename("loans"), how="inner")
    by_volume = matched.sort_values("loans")
    assert int(by_volume.loans.iloc[case.TAIL_SIZE - 1]) == 2027
    assert int(by_volume.loans.iloc[case.TAIL_SIZE]) == 2033
    assert int(by_volume.loans.iloc[-case.TAIL_SIZE]) == 37120
    assert int(by_volume.loans.iloc[-case.TAIL_SIZE - 1]) == 34318
    fastest = matched.sort_values("growth", ascending=False)
    assert round(float(fastest.growth.iloc[9]), 6) == 10.214351
    assert round(float(fastest.growth.iloc[10]), 6) == 10.119635

    # The rule is stated because ties are possible, not because one binds today.
    assert int(matched.loans.duplicated().sum()) > 0, (
        "no two areas share an origination count any more, so the tie rule could be "
        "dropped from the third turn"
    )
    assert "breaking a tie by the lower area code" in _turns()[2]["query"]
    assert "breaking a tie by the lower area code as before" in _turns()[3]["query"]


def test_the_series_and_basis_choices_are_both_load_bearing():
    """Six readings of the same median; exactly one is the pinned answer."""
    truth = _truth()
    prices = load_expansion_table("fhfa_hpi")
    frame = pd.DataFrame({
        "area": prices.place_id.astype(str),
        "year": pd.to_numeric(prices.yr, errors="coerce"),
        "quarter": pd.to_numeric(prices.period, errors="coerce"),
        "nsa": pd.to_numeric(prices.index_nsa, errors="coerce"),
        "sa": pd.to_numeric(prices.index_sa, errors="coerce"),
        "level": prices.level.astype(str),
        "frequency": prices.frequency.astype(str),
        "kind": prices.hpi_type.astype(str),
        "flavor": prices.hpi_flavor.astype(str),
    })
    frame = frame.loc[frame.level.eq(case.LEVEL) & frame.frequency.eq(case.FREQUENCY)
                      & frame.kind.eq(case.HPI_TYPE)]
    medians = {}
    for flavor in (case.PUBLISHED_ERROR_FLAVOR, case.PURCHASE_FLAVOR, "all-transactions"):
        for basis in ("nsa", "sa"):
            one = frame.loc[frame.flavor.eq(flavor)]
            ending = one.loc[one.year.eq(case.END_YEAR) & one.quarter.eq(case.END_QUARTER)].set_index("area")[basis]
            starting = one.loc[one.year.eq(case.END_YEAR - 1) & one.quarter.eq(case.END_QUARTER)].set_index("area")[basis]
            shared = ending.index.intersection(starting.index)
            growth = (100.0 * (ending[shared] / starting[shared] - 1.0)).dropna()
            medians[(flavor, basis)] = round(float(growth.median()), 6) if len(growth) else None
    assert medians[(case.PUBLISHED_ERROR_FLAVOR, "nsa")] == truth["median_annual_growth_pct"]
    assert medians[(case.PUBLISHED_ERROR_FLAVOR, "sa")] == 2.978438
    assert medians[(case.PURCHASE_FLAVOR, "nsa")] == 1.605588
    assert medians[("all-transactions", "nsa")] == 3.700666
    assert medians[("all-transactions", "sa")] is None
    reproducing = sum(v == truth["median_annual_growth_pct"] for v in medians.values())
    assert reproducing == 1, medians
    # Seasonal adjustment moves the answer past tolerance even though it looks close.
    assert not numeric_equal(2.978438, truth["median_annual_growth_pct"], decimals=6)


def test_the_pinned_quarters_carry_no_missing_level():
    """Two levels are missing in 1991; neither is inside the window."""
    panel = case._metropolitan_panel(case.PUBLISHED_ERROR_FLAVOR)
    window = panel.loc[panel.year.isin([case.END_YEAR - 1, case.END_YEAR])
                       & panel.quarter.eq(case.END_QUARTER)]
    assert len(window) == 820
    assert int(window["index"].isna().sum()) == 0
    assert int(panel["index"].isna().sum()) == 2, (
        "the panel's missing levels have moved; check they are still outside the window"
    )


def test_the_no_metropolitan_code_is_named_in_the_query():
    """The sentinel was documented only in a source comment, so the natural
    null-based reading returned zero and failed at zero tolerance."""
    assert case.NO_METROPOLITAN_AREA in _turns()[2]["query"]
    register = load_expansion_table("hmda")
    assert int(register.derived_msa_md.isna().sum()) == 0, (
        "the register now carries nulls, so a null reading would no longer return zero "
        "and the query may need to cover both"
    )


def test_the_tie_rule_is_applied_the_same_way_at_both_tails(published, counts):
    """Reading the deep tail off the end of an ascending frame inverts the rule."""
    matched = published.join(counts.rename("loans"), how="inner").assign(code=lambda f: f.index)
    ascending = matched.sort_values(["loans", "code"], ascending=[True, True])
    descending = matched.sort_values(["loans", "code"], ascending=[False, True])
    # Ties exist in the panel, so the two ways of taking the deep tail can differ.
    assert int(matched.loans.duplicated().sum()) > 0
    assert list(descending.head(case.TAIL_SIZE).index) == list(ascending.tail(case.TAIL_SIZE).index[::-1]), (
        "a tie now straddles the deep-tail boundary, so reading the end of the ascending "
        "frame and ranking downward no longer agree -- the descending form is the correct one"
    )
    assert "breaking a tie by the lower area code" in _turns()[2]["query"]


def test_the_ten_fastest_median_is_rounded_not_truncated(published, counts):
    """head(10) is an even count, so the median can land on a half."""
    matched = published.join(counts.rename("loans"), how="inner").assign(code=lambda f: f.index)
    fastest = matched.sort_values(["growth", "code"], ascending=[False, True])
    median = float(fastest.head(10).loans.median())
    assert int(round(median)) == _truth()["fastest_ten_median_origination_count"]
    assert median == int(median), (
        "the median now lands on a half; truncation and rounding would differ by one"
    )


def test_the_fourth_turn_ranks_by_growth_not_volume():
    """An earlier draft opened on 'the top of the ranking', which was turn three's."""
    assert "rank the covered areas by annual growth" in _turns()[3]["query"]
    assert "the top of the ranking" not in _turns()[3]["query"]
