"""Case-local checks for the N-MFP stock/flow timing question."""

from core.data import load_expansion_table
from evals.temporal_path import money_fund_liquidity_stock_flow_path as case


def _reported():
    values = case.analysis()
    return {name: round(value, case.PRECISION[name]) if isinstance(value, float)
            else value for name, value in values.items()}


def test_event_is_unique_and_validator_accepts_display_values():
    values = _reported()
    assert values["liquidity_path_accession"] == "0002071691-25-002432"
    assert values["liquidity_path_category"] == "Other Tax Exempt"
    assert values["liquidity_path_peak_date"] == "2025-09-11"
    assert values["liquidity_path_prior_date"] == "2025-09-10"
    assert values["liquidity_path_next_date"] == "2025-09-12"
    assert values["liquidity_path_peak_daily_pct"] == 13.08
    assert values["liquidity_path_peak_weekly_pct"] == 100.0
    assert case.validate(values).success


def test_flow_is_not_the_change_in_daily_liquid_assets():
    values = case.analysis()
    selected_net_redemption = (values["liquidity_path_peak_redemptions_m"]
                               - values["liquidity_path_peak_subscriptions_m"])
    observed_daily_decline = (values["liquidity_path_prior_daily_m"]
                              - values["liquidity_path_peak_daily_m"])
    assert selected_net_redemption > 200
    assert observed_daily_decline < 30
    assert values["liquidity_path_peak_weekly_m"] > values["liquidity_path_peak_daily_m"]
    wrong = {**_reported(), "liquidity_path_peak_daily_m":
             round(values["liquidity_path_prior_daily_m"] - selected_net_redemption, 4)}
    assert not case.validate(wrong).success


def test_filing_keys_and_runner_up_margin():
    accession = case.analysis()["liquidity_path_accession"]
    flows = load_expansion_table("sec_nmfp_flows")
    liquidity = load_expansion_table("sec_nmfp_liquidity")
    flows = flows[flows.accession.eq(accession)]
    liquidity = liquidity[liquidity.accession.eq(accession)]
    assert len(flows) == len(liquidity) == 21
    assert not flows.duplicated(["class_id", "flow_date"]).any()
    assert not liquidity.liquidity_date.duplicated().any()
    flow = flows.set_index("flow_date").sort_index()
    daily = liquidity.set_index("liquidity_date").daily_liquid_assets_usd.sort_index()
    ratio = ((flow.gross_redemptions_usd - flow.gross_subscriptions_usd)
             / daily.shift(1)).dropna().sort_values(ascending=False)
    assert ratio.index[0] == "2025-09-11"
    assert ratio.index[1] == "2025-09-04"
    assert ratio.iloc[0] - ratio.iloc[1] > 0.14
