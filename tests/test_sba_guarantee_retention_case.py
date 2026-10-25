"""Case-local gate for sba_guarantee_retention_versus_commercial_book.

The trap must keep separating: measuring a lender's exposure by the gross
approved amount rather than by the share the guarantee leaves with it has to
keep reporting a different count.  The screen's boundary must stay wide, size
and intensity must keep picking different lenders, and the fiscal-year period
must keep mattering.  Every assertion is written so that it can fail; where a
property is currently vacuous the test says so in its message.
"""

import json

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.data import load_expansion_table
from core.validation import numeric_equal
from evals.risk_screening import sba_guarantee_retention_versus_commercial_book as case


SPEC = json.loads(
    (PROJECT_ROOT / "evals/risk_screening/sba_guarantee_retention_versus_commercial_book.json").read_text()
)

GIVEAWAYS = (
    "unguaranteed", "not guaranteed", "net of the guarantee", "minus the guaranteed",
    "subtract the guaranteed", "guaranteed share is the government", "retained",
    "the remainder", "government bears", "only the portion",
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


@pytest.fixture(scope="module")
def screened():
    """The matched lender panel, rebuilt independently of the module's helpers."""
    loans = load_expansion_table("sba_7a")
    approved_on = pd.to_datetime(loans.ApprovalDate, errors="coerce")
    inside = approved_on.between(case.FISCAL_YEAR_START, case.FISCAL_YEAR_END)
    frame = pd.DataFrame({
        "gross": pd.to_numeric(loans.GrossApproval, errors="coerce"),
        "guaranteed": pd.to_numeric(loans.SBAGuaranteedApproval, errors="coerce"),
        "certificate": pd.to_numeric(loans.BankFDICNumber, errors="coerce"),
    }).loc[inside]
    del loans
    frame["own"] = frame.gross - frame.guaranteed

    banks = load_expansion_table("fdic_bankfind")
    reported_on = pd.to_datetime(banks.REPDTE, errors="coerce")
    books = pd.DataFrame({
        "certificate": pd.to_numeric(banks.CERT, errors="coerce"),
        "commercial": pd.to_numeric(banks.LNCI, errors="coerce") * case.THOUSANDS,
    }).loc[reported_on.eq(case.CALL_REPORT_DATE)]
    del banks

    identified = frame.loc[frame.certificate.notna()]
    grouped = identified.groupby(identified.certificate.astype(int))
    panel = pd.DataFrame({"gross": grouped.gross.sum(), "own": grouped.own.sum()})
    panel["commercial"] = books.set_index(books.certificate.astype(int)).commercial
    panel = panel.loc[panel.commercial.gt(0)]
    return frame, panel


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["fiscal_year_loan_count"] == 78078
    assert truth["gross_approval_usd_billions"] == 37.287816
    assert truth["guaranteed_approval_usd_billions"] == 27.503144
    assert truth["unidentified_lender_loan_count"] == 9562
    assert truth["matched_bank_lender_count"] == 1130
    assert truth["own_exposure_exceeds_commercial_book_lender_count"] == 6
    assert truth["largest_own_exposure_certificate"] == "58665"
    assert truth["largest_own_exposure_usd_billions"] == 0.757135
    assert truth["highest_exposure_intensity_certificate"] == "27417"
    assert truth["highest_exposure_intensity_ratio"] == 3.546408
    assert truth["highest_exposure_intensity_commercial_book_usd_billions"] == 0.009675
    assert truth["largest_own_exposure_intensity_ratio"] == 0.164963
    assert truth["three_quarter_guarantee_loan_count"] == 34472
    assert truth["one_half_guarantee_loan_count"] == 24375
    assert truth["top_ten_lender_own_exposure_share_pct"] == 38.812902
    assert truth["unidentified_lender_own_exposure_usd_billions"] == 1.065450


def test_the_gross_reading_still_misses(screened):
    """The trap, on the screened count and on the leaderboard it reorders."""
    _, panel = screened
    truth = _truth()
    by_own = int(panel.own.gt(panel.commercial).sum())
    by_gross = int(panel.gross.gt(panel.commercial).sum())
    assert by_own == truth["own_exposure_exceeds_commercial_book_lender_count"]
    assert by_gross == 45
    assert not numeric_equal(by_gross, by_own, decimals=0)
    assert by_gross - by_own == 39

    # The leader is the same either way; the reordering sits below the very top.
    assert str(panel.own.idxmax()) == truth["largest_own_exposure_certificate"]
    assert str(panel.gross.idxmax()) == truth["largest_own_exposure_certificate"], (
        "the two readings now name different leaders, so the docstring's account of the "
        "trap is stale and the third turn must be rechecked"
    )
    by_own = list(panel.own.sort_values(ascending=False).head(10).index)
    by_gross = list(panel.gross.sort_values(ascending=False).head(10).index)
    assert by_own[:5] == by_gross[:5]
    assert len(set(by_own) & set(by_gross)) == 7
    full_gross = list(panel.gross.sort_values(ascending=False).index)
    assert by_own.index(18409) + 1 == 7 and full_gross.index(18409) + 1 == 13


def test_the_trap_moves_six_of_the_sixteen_outputs(screened):
    """Pinned against the gross reading, output by output."""
    _, panel = screened
    truth = _truth()
    gross_intensity = panel.gross / panel.commercial
    moved = {
        "own_exposure_exceeds_commercial_book_lender_count": int(panel.gross.gt(panel.commercial).sum()),
        "largest_own_exposure_usd_billions": round(float(panel.gross.max()) / 1e9, 6),
        "highest_exposure_intensity_ratio": round(float(gross_intensity.max()), 6),
        "largest_own_exposure_intensity_ratio": round(float(gross_intensity[panel.own.idxmax()]), 6),
    }
    assert moved == {
        "own_exposure_exceeds_commercial_book_lender_count": 45,
        "largest_own_exposure_usd_billions": 2.852669,
        "highest_exposure_intensity_ratio": 14.562687,
        "largest_own_exposure_intensity_ratio": 0.621534,
    }
    for name, naive in moved.items():
        places = dict(zip((v.name for v in case.variables), case.DECIMALS))[name]
        assert not numeric_equal(naive, truth[name], decimals=places or 0), name

    # The two certificates are the outputs the trap leaves alone.
    assert str(gross_intensity.idxmax()) == truth["highest_exposure_intensity_certificate"]


def test_the_screen_boundary_is_not_a_close_call(screened):
    _, panel = screened
    intensity = (panel.own / panel.commercial).sort_values(ascending=False)
    assert round(float(intensity.iloc[5]), 6) == 1.102126
    assert round(float(intensity.iloc[6]), 6) == 0.866684
    assert float(intensity.iloc[5]) - 1.0 > 0.1, "the sixth lender has drifted onto the bar"
    assert 1.0 - float(intensity.iloc[6]) > 0.1, "the seventh lender has drifted onto the bar"


def test_size_and_intensity_pick_different_lenders():
    truth = _truth()
    assert truth["largest_own_exposure_certificate"] != truth["highest_exposure_intensity_certificate"], (
        "one lender now leads on both, so the third turn's required claim is unsupported"
    )
    assert truth["highest_exposure_intensity_ratio"] > 1.0 > truth["largest_own_exposure_intensity_ratio"]
    assert truth["highest_exposure_intensity_commercial_book_usd_billions"] < 0.05


def test_the_fiscal_year_is_load_bearing(screened):
    """A calendar year over the same file reports a different population."""
    frame, _ = screened
    loans = load_expansion_table("sba_7a")
    approved_on = pd.to_datetime(loans.ApprovalDate, errors="coerce")
    assert approved_on.max() == case.FISCAL_YEAR_END, (
        "the file no longer ends on the fiscal year end, so the period argument must be rechecked"
    )
    calendar = int(approved_on.dt.year.eq(2025).sum())
    assert calendar != len(frame)
    assert calendar == 58083
    assert "fiscal year that ended on 2025-09-30" in _turns()[0]["query"]


def test_the_positive_commercial_book_guard_currently_removes_nothing(screened):
    """The query no longer claims the guard filters; the module keeps it anyway.

    If it ever stops being vacuous the query must say what happens to the lenders
    it drops, because the match would then be silently losing them.
    """
    frame, panel = screened
    identified = frame.loc[frame.certificate.notna()]
    distinct = identified.certificate.astype(int).nunique()
    assert distinct == 1130
    assert len(panel) == distinct, (
        "some fiscal-year certificate no longer matches a bank with a positive commercial "
        "book, so the second turn must state what happens to the unmatched lenders"
    )
    assert "positive commercial and industrial loan book" not in _turns()[1]["query"]

    # The guard is capable of biting: banks with no commercial book do exist.
    banks = load_expansion_table("fdic_bankfind")
    reported_on = pd.to_datetime(banks.REPDTE, errors="coerce")
    at_date = reported_on.eq(case.CALL_REPORT_DATE)
    commercial = pd.to_numeric(banks.LNCI, errors="coerce")
    assert int((at_date & commercial.fillna(0).le(0)).sum()) == 217, (
        "no bank reports a zero commercial book any more, so the guard can never bite"
    )


def test_the_unidentified_certificates_are_genuinely_absent(screened):
    """Not a sentinel value: the column is null on every one of those rows."""
    frame, _ = screened
    missing = frame.certificate.isna()
    loans = load_expansion_table("sba_7a")
    approved_on = pd.to_datetime(loans.ApprovalDate, errors="coerce")
    raw = loans.BankFDICNumber[approved_on.between(case.FISCAL_YEAR_START, case.FISCAL_YEAR_END)]
    assert int(raw.isna().sum()) == int(missing.sum()) == 9562, (
        "some certificate is now unparseable rather than absent, so the fourth turn's "
        "population is no longer the non-bank lenders it claims to be"
    )


def test_the_guarantee_fractions_are_exact(screened):
    """Integer arithmetic, so the two counts do not depend on a float comparison."""
    frame, _ = screened
    truth = _truth()
    assert int(frame.guaranteed.mul(4).eq(frame.gross.mul(3)).sum()) == truth["three_quarter_guarantee_loan_count"]
    assert int(frame.guaranteed.mul(2).eq(frame.gross).sum()) == truth["one_half_guarantee_loan_count"]
    share = frame.guaranteed / frame.gross
    for decimals in (4, 6, 8, 10):
        assert int(share.round(decimals).eq(0.75).sum()) == truth["three_quarter_guarantee_loan_count"]
        assert int(share.round(decimals).eq(0.50).sum()) == truth["one_half_guarantee_loan_count"]
    assert int(frame.guaranteed.gt(frame.gross).sum()) == 0
    assert int(frame.guaranteed.eq(frame.gross).sum()) == 0, (
        "a fully guaranteed approval would carry no lender risk at all; the case does not handle that"
    )


def test_the_unidentified_lenders_are_material(screened):
    frame, _ = screened
    truth = _truth()
    missing = frame.certificate.isna()
    assert int(missing.sum()) == truth["unidentified_lender_loan_count"]
    assert round(float(frame.own[missing].sum()) / 1e9, 6) == truth["unidentified_lender_own_exposure_usd_billions"]
    share = 100 * float(frame.own[missing].sum()) / float(frame.own.sum())
    assert share > 10, "the non-bank share has fallen away, so the fourth turn's boundary claim is thin"


def test_the_certificate_comparison_accepts_leading_zeros():
    truth = _truth()
    outputs = {name: truth[name] for name in case.TURN_2_NAMES}
    padded = dict(outputs)
    padded["largest_own_exposure_certificate"] = "0" + truth["largest_own_exposure_certificate"]
    assert case._validate_subset(padded, case.TURN_2_NAMES).success
    wrong = dict(outputs)
    wrong["largest_own_exposure_certificate"] = "99999"
    assert not case._validate_subset(wrong, case.TURN_2_NAMES).success


@pytest.mark.parametrize("removed", ["sba_7a", "fdic_bankfind"])
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


def test_the_query_withholds_the_distinction():
    assert SPEC["case_type"] == "trap"
    trapped = " ".join(turn["query"] for turn in _turns()[:2]).lower()
    for giveaway in GIVEAWAYS:
        assert giveaway not in trapped, giveaway


def test_no_variable_description_states_the_avoidance():
    trapped = set(_turns()[0]["stores"]) | set(_turns()[1]["stores"])
    for variable in case.variables:
        if variable.name not in trapped:
            continue
        lowered = variable.description.lower()
        for giveaway in GIVEAWAYS:
            assert giveaway not in lowered, (variable.name, giveaway)


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
            broken[name] = "__WRONG__" if isinstance(broken[name], str) else float(broken[name]) + 1.0
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
    assert f"{truth['fiscal_year_loan_count']:,}" in facts
    assert f"{truth['unidentified_lender_loan_count']:,}" in facts
    assert f"{truth['matched_bank_lender_count']:,}" in facts
    assert f"{truth['gross_approval_usd_billions']:.6f}" in facts
    assert f"{truth['highest_exposure_intensity_ratio']:.6f}" in facts
    assert "45 lenders rather than 6" in facts
    assert set(SPEC["judge_reference"]) == {"source", "facts", "known_overclaims"}
    assert SPEC["judge_reference"]["known_overclaims"]


def test_the_judge_reference_quotes_the_six_intensities_exactly(screened):
    """These six sit only in the judge reference, so nothing else would catch a slip.

    An earlier draft carried three of them transcribed from a four-decimal print
    and padded with zeros, which would have let a judge mark a correct answer wrong.
    """
    _, panel = screened
    intensity = (panel.own / panel.commercial).sort_values(ascending=False)
    quoted = [3.546408, 2.501637, 2.024593, 1.932816, 1.327082, 1.102126]
    assert [round(float(value), 6) for value in intensity.head(6)] == quoted
    assert round(float(intensity.iloc[6]), 6) == 0.866684

    facts = " ".join(SPEC["judge_reference"]["facts"])
    for value in quoted + [0.866684]:
        assert f"{value:.6f}" in facts, value
