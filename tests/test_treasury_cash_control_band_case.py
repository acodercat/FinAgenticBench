"""Case-local gate for treasury_cash_buffer_control_band_calibration.

The case is a calibration, so what has to keep holding is that every input the
closed form consumes is uniquely determined by the data, that the five readings
the query pins really do move the answer past the tolerance, and that the two
findings the case rests on survive a refresh: the model-implied band sits far
below the balance the account carried, and the flows violate the serial
independence the policy is derived from.
"""

import json

import pandas as pd

from config import PROJECT_ROOT
from core.validation import numeric_equal
from evals.scenario_analysis import treasury_cash_buffer_control_band_calibration as case
from scripts.audit_case_design import has_closed_text_representation


SPEC = json.loads(
    (PROJECT_ROOT / "evals/scenario_analysis/treasury_cash_buffer_control_band_calibration.json").read_text()
)


def _truth():
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _turns():
    return SPEC["conversations"][0]["turns"]


def test_anchor_values_are_stable():
    truth = _truth()
    assert truth["business_day_count"] == 501
    assert truth["final_closing_balance_usd_millions"] == 872853
    assert truth["holding_cost_rate_pct"] == 3.64
    assert truth["largest_outflow_calendar_day"] == 1
    assert truth["minimum_closing_balance_usd_millions"] == 260807
    assert truth["minimum_closing_balance_date"] == "2025-06-12"
    assert truth["maximum_closing_balance_usd_millions"] == 1000632
    assert truth["days_single_flow_exceeds_band_count"] == 38
    assert round(truth["net_flow_standard_deviation_usd_millions"], 4) == 37597.6367
    assert round(truth["control_band_return_point_usd_millions"], 3) == 21887.416
    assert round(truth["implied_transfer_cost_usd_millions"], 2) == 1239.12


def test_the_five_pinned_readings_all_clear_the_tolerance():
    """Each clause the query spends words on has to be load-bearing."""
    truth = _truth()
    account = case._account()
    net_flow = account.net_flow
    pinned = truth["control_band_return_point_usd_millions"]
    variance = float(net_flow.std(ddof=1)) ** 2
    rate = truth["holding_cost_rate_pct"]
    daily = rate / 100.0 / case.DAY_COUNT_BASIS

    annual = case._return_point(variance, rate / 100.0)
    three_sixty_five = case._return_point(variance, rate / 100.0 / 365)
    level = case._return_point(float(account[case.CLOSING_LINE].std(ddof=1)) ** 2, daily)
    population = case._return_point(float(net_flow.std(ddof=0)) ** 2, daily)

    assert round(annual, 4) == 3076.7603
    assert round(three_sixty_five, 4) == 21988.2811
    assert round(level, 4) == 61205.6266
    assert round(population, 4) == 21872.8438
    # The calibration turn pins the degrees of freedom; the diagnostic turn repeats it.
    assert "one degree of freedom" in _turns()[0]["query"]
    assert "one-degree-of-freedom variance" in _turns()[2]["query"]
    for alternative in (annual, three_sixty_five, level, population):
        assert not numeric_equal(alternative, pinned, decimals=3), alternative

    # And the rate itself: the other overnight rate the file carries is a different number.
    from core.data import load_expansion_table

    rates = load_expansion_table("nyfed_reference_rates")
    overnight = rates.loc[rates["Rate Type"].astype(str).eq("SOFR")].copy()
    del rates
    overnight["observed_on"] = pd.to_datetime(overnight["Effective Date"])
    overnight["value"] = pd.to_numeric(overnight["Rate (%)"], errors="coerce")
    other = float(
        overnight.loc[overnight.observed_on.eq(account.index.max()), "value"].dropna().iloc[0]
    )
    assert other != rate
    assert not numeric_equal(
        case._return_point(variance, other / 100.0 / case.DAY_COUNT_BASIS), pinned, decimals=3
    )


def test_the_model_turn_states_no_formula():
    """P3, and the contract the other two cases in this capability keep.

    The agent has to supply Miller-Orr itself. The query may name the model, its
    objective and its constraints, and must pin every input the data decides, but
    handing over the closed form would turn the turn into arithmetic.
    """
    giveaways = ("cube root", "cube", "three times the transfer cost", "divided by four",
                 "four times that daily rate", "raised to", "exponent", "power of",
                 "three times the return point", "thrice")
    # The variable descriptions are a prompt surface too, so they are scanned as well.
    surfaces = [turn["query"] for turn in _turns()]
    surfaces += [variable.description for variable in case.variables]
    for surface in surfaces:
        lowered = surface.lower()
        for giveaway in giveaways:
            assert giveaway not in lowered, (giveaway, surface)
    model_turn = _turns()[1]["query"].lower()
    # Named model, objective and the parameters the data decides are all present.
    assert "miller-orr" in model_turn
    assert "band minimises expected transfer cost plus expected forgone interest" in model_turn
    for pinned in ("lower control limit at zero", "one million dollars",
                   "carried standard deviation of that flow", "not from any other overnight rate",
                   "three-hundred-sixty-day year"):
        assert pinned in model_turn, pinned


def test_the_band_sits_far_below_the_balance_the_account_carried():
    """The case's first finding. If this closed the case would measure nothing."""
    truth = _truth()
    upper_limit = 3.0 * truth["control_band_return_point_usd_millions"]
    assert truth["mean_closing_balance_usd_millions"] > 10 * upper_limit
    assert truth["minimum_closing_balance_usd_millions"] > upper_limit
    # The inversion is the same band read backwards, so it has to reproduce the mean.
    rebuilt = 3.0 * case._return_point(
        float(case._account().net_flow.std(ddof=1)) ** 2,
        truth["holding_cost_rate_pct"] / 100.0 / case.DAY_COUNT_BASIS,
        transfer_cost=truth["implied_transfer_cost_usd_millions"],
    )
    assert abs(rebuilt - truth["mean_closing_balance_usd_millions"]) < 1e-6


def test_the_final_turn_depends_on_the_band_and_the_count_is_not_degenerate():
    """T4 consumes T2's band, so the dependency is computational and not only thematic."""
    truth = _truth()
    count = truth["days_single_flow_exceeds_band_count"]
    total = truth["business_day_count"]
    assert 0 < count < total, "the count no longer separates any days"
    band = 3.0 * (truth["control_band_return_point_usd_millions"] - case.LOWER_LIMIT)
    net_flow = case._account().net_flow
    assert int(net_flow.abs().gt(band).sum()) == count
    # No day sits within a tolerance of the edge, so the strict comparison cannot flip.
    nearest = (net_flow.abs() - band).abs().min()
    assert nearest > 1.0, nearest
    assert "whole distance from the lower control limit to the upper one" in _turns()[3]["query"]


def test_the_maintained_assumption_still_fails():
    """The case's second finding: the flows are neither independent nor unseasonal."""
    truth = _truth()
    assert truth["short_horizon_variance_ratio"] < 0.9
    assert truth["long_horizon_variance_ratio"] < 0.9
    net_flow = case._account().net_flow
    by_day = net_flow.groupby(net_flow.index.day).mean()
    ranked = by_day.sort_values()
    assert float(ranked.iloc[0]) != float(ranked.iloc[1]), "the worst calendar day is tied"
    assert ranked.iloc[0] / ranked.iloc[1] > 3.0, "the seasonal day no longer stands out"
    # A seasonal mean this large against the daily dispersion is the point.
    assert abs(truth["largest_outflow_calendar_day_mean_usd_millions"]) > (
        truth["net_flow_standard_deviation_usd_millions"]
    )
    assert "breaking a tie by the earlier day" in _turns()[2]["query"]


def test_the_unconditional_drift_is_negligible():
    """The query does not pin a drift correction, so the drift has to stay small."""
    truth = _truth()
    assert abs(truth["mean_daily_net_flow_usd_millions"]) < (
        0.01 * truth["net_flow_standard_deviation_usd_millions"]
    )


def test_the_window_sits_inside_one_publication_era_and_outside_the_debt_limit_trough():
    from core.data import load_expansion_table

    cash = load_expansion_table("treasury_dts_operating_cash")
    frame = cash.loc[cash.account_type.astype(str).eq(case.CLOSING_LINE)].copy()
    del cash
    frame["observed_on"] = pd.to_datetime(frame.record_date)
    frame["amount"] = pd.to_numeric(frame.today_amount_million_usd, errors="coerce")
    era_start = frame.observed_on.min()
    assert era_start < pd.Timestamp(case.WINDOW_START), "the window now starts before the era does"
    earlier = frame.loc[frame.observed_on < pd.Timestamp(case.WINDOW_START), "amount"]
    assert float(earlier.min()) < _truth()["minimum_closing_balance_usd_millions"], (
        "the excluded stretch no longer contains a lower trough than the window"
    )


def test_the_account_grain_and_sign_guards_are_live():
    account = case._account()
    assert len(account) == _truth()["business_day_count"]
    assert account.index.is_unique and account.index.is_monotonic_increasing
    assert (account[case.WITHDRAWALS_LINE] > 0).all()
    # The four lines still balance to the file's published precision.
    residual = (
        account[case.CLOSING_LINE] - account[case.OPENING_LINE] - account.net_flow
    ).abs()
    assert residual.max() <= 1


def test_the_two_balance_extremes_are_unique():
    """Both are reported as levels with a date, so neither may tie."""
    closing = case._account()[case.CLOSING_LINE]
    assert int(closing.eq(closing.min()).sum()) == 1
    assert int(closing.eq(closing.max()).sum()) == 1


def test_no_business_day_in_the_window_is_dropped_for_a_missing_line():
    """The drop rule the query states is currently vacuous; assert that it is."""
    from core.data import load_expansion_table

    cash = load_expansion_table("treasury_dts_operating_cash")
    lines = (case.CLOSING_LINE, case.OPENING_LINE, case.DEPOSITS_LINE, case.WITHDRAWALS_LINE)
    frame = cash.loc[cash.account_type.astype(str).isin(lines)].copy()
    del cash
    frame["observed_on"] = pd.to_datetime(frame.record_date)
    inside = frame.loc[
        frame.observed_on.between(pd.Timestamp(case.WINDOW_START), pd.Timestamp(case.WINDOW_END))
    ]
    published = inside.groupby("observed_on").account_type.nunique()
    assert int(published.eq(len(lines)).sum()) == len(published), "a day now publishes only some lines"
    assert len(published) == _truth()["business_day_count"]


def test_the_rate_source_is_load_bearing():
    """Without the second source the band has no holding cost and turn two collapses."""
    account = case._account()
    assert "rate" not in {str(column).lower() for column in account.columns}
    assert isinstance(case._holding_rate_pct(), float)


def test_every_text_output_declares_a_closed_representation():
    truth = _truth()
    by_name = {variable.name: variable for variable in case.variables}
    text_outputs = [name for name, value in truth.items() if isinstance(value, str)]
    assert text_outputs
    for name in text_outputs:
        assert has_closed_text_representation(by_name[name].description), name


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
            "__WRONG__" if isinstance(broken[first], str) else float(broken[first]) + 1_000_000.0
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
    """The whole dict is serialised into the RQ judge prompt, so the guardrails matter."""
    reference = SPEC["judge_reference"]
    assert set(reference) == {"source", "facts", "known_overclaims"}
    assert reference["source"].strip() and reference["facts"]
    overclaims = reference["known_overclaims"]
    assert len(overclaims) >= 3 and all(str(o).strip() for o in overclaims)
    joined = " ".join(overclaims).lower()
    # The two readings the case is most exposed to have to be named.
    assert "ought to operate" in joined
    assert "significance test" in joined


def test_judge_reference_quotes_recomputed_figures():
    truth = _truth()
    facts = " ".join(SPEC["judge_reference"]["facts"])
    assert f"{truth['control_band_return_point_usd_millions']:,.3f}" in facts
    assert f"{truth['implied_transfer_cost_usd_millions']:,.2f}" in facts
    assert f"{truth['mean_closing_balance_usd_millions']:,.4f}" in facts
    assert truth["minimum_closing_balance_date"] in facts
    assert "3,076.7603" in facts
    assert f"{truth['days_single_flow_exceeds_band_count']} of the 501 days" in facts
