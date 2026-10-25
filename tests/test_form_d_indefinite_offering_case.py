"""Case-local gate for form_d_indefinite_offering_and_placement_cost.

The trap has to keep separating: a subscription share built from every notice's
sales must stay far from the one built only from the notices that supply a
denominator.  The two stated conventions -- originals only, and a positive stated
offering -- must each keep changing an answer, and the source identity and the
three-way partition must keep holding exactly.  Every assertion is written so it
can fail.
"""

import json

import pandas as pd
import pytest

from config import PROJECT_ROOT
from core.data import load_expansion_table, load_filings
from core.validation import numeric_equal
from evals.calculation_reconciliation import form_d_indefinite_offering_and_placement_cost as case


SPEC = json.loads(
    (PROJECT_ROOT
     / "evals/calculation_reconciliation/form_d_indefinite_offering_and_placement_cost.json"
     ).read_text()
)

GIVEAWAYS = (
    "same notices in the numerator", "both sides of the ratio", "matching denominator",
    "restrict the numerator", "excluding the notices that state none",
    "only the stated offerings' sales", "drop the unstated",
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


@pytest.fixture(scope="module")
def notices():
    return case._original_notices()


@pytest.fixture(scope="module")
def every_filing():
    raw = load_expansion_table("sec_form_d")
    filed_on = pd.to_datetime(raw.FILING_DATE, errors="coerce")
    return pd.DataFrame({
        "sold": pd.to_numeric(raw.TOTALAMOUNTSOLD, errors="coerce"),
        "amendment": raw.ISAMENDMENT.astype(str).str.strip().str.lower(),
    }).loc[filed_on.dt.year.eq(case.FILING_YEAR)]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["original_filing_count"] == 34717
    assert truth["indefinite_offering_filing_count"] == 10152
    assert truth["stated_offering_usd_billions"] == 832.467540
    assert truth["indefinite_offering_amount_sold_usd_billions"] == 364.657063
    assert truth["subscription_share_pct"] == 38.008355
    assert truth["fully_subscribed_filing_count"] == 12704
    assert truth["partially_subscribed_filing_count"] == 8130
    assert truth["no_sales_yet_filing_count"] == 3722
    assert truth["intermediary_paying_filing_count"] == 3086
    assert truth["sales_commission_usd_billions"] == 5.602919
    assert truth["finders_fee_usd_billions"] == 0.264248
    assert truth["commission_rate_pct"] == 2.1341
    assert truth["public_issuer_filing_count"] == 784
    assert truth["public_subscription_share_pct"] == 79.907441
    assert truth["public_commission_rate_pct"] == 4.1985
    assert truth["private_commission_rate_pct"] == 1.7619


def test_carrying_every_sale_into_the_numerator_still_misses(notices):
    """The trap, measured."""
    truth = _truth()
    stated = notices.offered.gt(0)
    denominator = float(notices.offered[stated].sum())
    naive = round(100.0 * float(notices.sold.sum()) / denominator, 6)
    assert naive == 81.812713
    assert not numeric_equal(naive, truth["subscription_share_pct"], decimals=6)
    assert round(naive - truth["subscription_share_pct"], 6) == 43.804358

    # The unstated notices outsell the stated ones, which is why it matters so much.
    assert truth["indefinite_offering_amount_sold_usd_billions"] > round(
        float(notices.sold[stated].sum()) / 1e9, 6
    )


def test_the_originals_convention_is_load_bearing(every_filing):
    """Amendments restate the cumulative figure, so summing everything explodes."""
    truth = _truth()
    originals = every_filing.amendment.eq(case.ORIGINAL_TOKEN)
    assert int(originals.sum()) == truth["original_filing_count"]
    assert int((~originals).sum()) == 21802
    every = round(float(every_filing.sold.sum()) / 1e9, 6)
    just_originals = round(float(every_filing.sold[originals].sum()) / 1e9, 6)
    assert every == 11557.813107
    assert just_originals == 681.064281
    assert every / just_originals > 16, "amendments no longer restate, so the convention is idle"
    assert "setting amendments aside" in _turns()[0]["query"]


def test_the_positive_offering_rule_is_load_bearing_and_not_vacuous(notices):
    """Nine notices state a zero offering and would be counted twice over."""
    truth = _truth()
    zero = notices.offered.eq(0)
    assert int(zero.sum()) == 9, (
        "no notice states a zero offering any more, so the positivity rule has gone idle "
        "and the query should stop claiming it does work"
    )
    loose = notices.offered.notna() & notices.remaining.eq(0)
    assert int(loose.sum()) == 12713
    assert not numeric_equal(int(loose.sum()), truth["fully_subscribed_filing_count"], decimals=0)
    # A zero offering is simultaneously fully sold and not begun, which is why it goes.
    degenerate = notices.loc[zero.fillna(False)]
    assert len(degenerate) == 9
    assert bool(degenerate.sold.eq(0).all()) and bool(degenerate.remaining.eq(0).all())
    assert "positive one" in _turns()[0]["query"]


def test_the_source_identity_holds_exactly(notices):
    stated = notices.offered.gt(0)
    residual = (notices.offered - notices.sold - notices.remaining)[stated]
    assert float(residual.abs().max()) == 0.0
    assert int(notices.sold[stated].gt(notices.offered[stated]).sum()) == 0
    assert int(notices.remaining[stated].lt(0).sum()) == 0


def test_the_three_outcomes_partition_the_stated_offerings(notices):
    truth = _truth()
    stated = int(notices.offered.gt(0).sum())
    assert stated == 24556
    total = (truth["fully_subscribed_filing_count"]
             + truth["partially_subscribed_filing_count"]
             + truth["no_sales_yet_filing_count"])
    assert total == stated, (total, stated)


def test_most_of_this_market_pays_no_intermediary(notices):
    """The third turn's required claim, pinned to the data."""
    truth = _truth()
    paying = notices.commission.gt(0) | notices.finders.gt(0)
    assert int(paying.sum()) == truth["intermediary_paying_filing_count"]
    assert 100 * float(paying.mean()) < 10, (
        "intermediated placements are now the majority, so the required claim is false"
    )
    assert int((notices.commission.gt(0) & ~notices.finders.gt(0)).sum()) == 2649
    assert int((notices.finders.gt(0) & ~notices.commission.gt(0)).sum()) == 214
    assert int((notices.commission.gt(0) & notices.finders.gt(0)).sum()) == 223


def test_the_public_issuers_pay_about_twice_as_much(notices):
    """The fourth turn's required claim."""
    truth = _truth()
    assert truth["public_commission_rate_pct"] > 2 * truth["private_commission_rate_pct"] * 0.9
    assert truth["public_commission_rate_pct"] / truth["private_commission_rate_pct"] > 2.0, (
        "the gap has closed below a factor of two, so the required claim overstates it"
    )
    assert truth["public_subscription_share_pct"] > truth["subscription_share_pct"]
    reporting = notices.cik.isin(case._periodic_report_issuers())
    assert int(reporting.sum()) == truth["public_issuer_filing_count"]
    assert 100 * float(reporting.mean()) < 5


def test_the_periodic_report_file_holds_only_periodic_reports():
    """The query defines a reporting issuer by presence in that file."""
    forms = set(load_filings().form.astype(str).str.upper().unique())
    assert forms <= {"10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A", "40-F/A"}, forms


@pytest.mark.parametrize("removed", ["sec_form_d"])
def test_the_notice_source_is_load_bearing(monkeypatch, removed):
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


def test_the_filing_index_is_load_bearing(monkeypatch):
    def without():
        raise FileNotFoundError("removed source")

    case.ground_truth.cache_clear()
    try:
        with monkeypatch.context() as ctx:
            ctx.setattr(case, "load_filings", without)
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
            broken[name] = float(broken[name]) + 1.0
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
    assert f"{truth['original_filing_count']:,}" in facts
    assert f"{truth['indefinite_offering_filing_count']:,}" in facts
    assert f"{truth['subscription_share_pct']:.6f}" in facts
    assert f"{truth['stated_offering_usd_billions']:.6f}" in facts
    assert "81.812713" in facts and "43.804358" in facts
    assert set(SPEC["judge_reference"]) == {"source", "facts", "known_overclaims"}
    assert SPEC["judge_reference"]["known_overclaims"]


def test_the_commission_rates_are_pinned_where_the_method_reproduces_them(notices):
    """At six decimals the private rate sits 0.0085 of a unit from a boundary."""
    stated = notices.offered.gt(0)
    commissioned = notices.commission.gt(0) & notices.sold.gt(0)
    reporting = notices.cik.isin(case._periodic_report_issuers())
    private = 100.0 * float(notices.commission[~reporting & commissioned].sum()) \
        / float(notices.sold[~reporting & commissioned].sum())
    at_six = abs(private * 1e6 - round(private * 1e6))
    at_four = abs(private * 1e4 - round(private * 1e4))
    assert at_six > 0.45, (
        "the private rate has moved away from the six-decimal boundary; the coarser "
        "pinning could be revisited, but nothing is broken by leaving it"
    )
    assert at_four < 0.45
    places = dict(zip((v.name for v in case.variables), case.DECIMALS))
    assert places["private_commission_rate_pct"] == 4
    assert places["public_commission_rate_pct"] == 4
    assert places["commission_rate_pct"] == 4
    # The shares stay at six because they reproduce there.
    share = 100.0 * float(notices.sold[stated].sum()) / float(notices.offered[stated].sum())
    assert abs(share * 1e6 - round(share * 1e6)) < 0.45
    assert places["subscription_share_pct"] == 6


def test_a_commission_without_sales_is_excluded_from_the_rate(notices):
    """742 notices report a commission and no sales; the query says 'against sales'."""
    commission = notices.commission.gt(0)
    assert int(commission.sum()) == 2872
    assert int((commission & notices.sold.eq(0)).sum()) == 742
    assert int((commission & notices.sold.gt(0)).sum()) == 2130
    assert "both a commission and a sale" in _turns()[2]["query"]
    assert "on that same basis" in _turns()[3]["query"]

    # The loose reading is materially different, which is why the query pins it.
    loose = round(100.0 * float(notices.commission[commission].sum())
                  / float(notices.sold[commission & notices.sold.gt(0)].sum()), 4)
    assert loose == 6.2882
    assert not numeric_equal(loose, _truth()["commission_rate_pct"], decimals=4)
