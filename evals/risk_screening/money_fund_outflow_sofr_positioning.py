"""Trace a money-fund liquidity drawdown into flows and SOFR positioning.

Among large original 2025 N-MFP3 reports, Columbia Short-Term Cash Fund has the
largest decline in reported weekly-liquid-assets fraction between consecutive
positive observations: 70.73% to 61.03% on 2025-01-30 to 2025-01-31.  Its
same-day net shareholder outflow is much smaller than the dollar decline in the
reported weekly bucket, so the two measures do not form an accounting identity.
The final turn decomposes the weekly-bucket change into daily-liquid assets and
the reported weekly-minus-daily increment rather than repeating an outflow.

The pinned-reading comparison measures form scope, liquidity population, ranking metric,
date ordering, flow aggregation, CFTC scope/window and liquidity decomposition.
The query pins these conventions, so this is a hard baseline, not a trap.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "eligible_nmfp_accession_count", "eligible_liquidity_step_count",
    "liquidity_drawdown_leader_accession", "liquidity_drawdown_leader_series_name",
    "liquidity_drawdown_leader_report_date", "liquidity_drawdown_leader_net_assets_usd_billions",
    "liquidity_drawdown_start_date", "liquidity_drawdown_end_date",
    "start_weekly_liquid_assets_usd_billions", "end_weekly_liquid_assets_usd_billions",
    "start_weekly_liquid_assets_pct", "end_weekly_liquid_assets_pct",
    "liquidity_drawdown_leader_margin_pp",
]
TURN_2_NAMES = [
    "event_flow_share_class_count", "event_gross_subscriptions_usd_billions",
    "event_gross_redemptions_usd_billions", "event_net_outflow_to_net_assets_pct",
    "net_outflow_to_weekly_liquidity_decline_pct",
]
TURN_3_NAMES = [
    "prior_cftc_report_date", "next_cftc_report_date",
    "prior_sofr3m_lev_money_net_contracts", "next_sofr3m_lev_money_net_contracts",
    "sofr3m_lev_money_net_change_contracts", "prior_sofr3m_open_interest_contracts",
    "next_sofr3m_open_interest_contracts", "sofr3m_open_interest_change_contracts",
]
TURN_4_NAMES = [
    "start_daily_liquid_assets_usd_billions", "end_daily_liquid_assets_usd_billions",
    "weekly_liquidity_decomposition_residual_usd_millions",
]

variables = [
    _v("eligible_nmfp_accession_count", "Store the eligible original N-MFP3 accession count as an integer."),
    _v("eligible_liquidity_step_count", "Store the eligible positive consecutive-liquidity-step count as an integer."),
    _v("liquidity_drawdown_leader_accession", "Store the selected accession as a string."),
    _v("liquidity_drawdown_leader_series_name", "Store the selected series name exactly as reported."),
    _v("liquidity_drawdown_leader_report_date", "Store the selected report date as an ISO YYYY-MM-DD string."),
    _v("liquidity_drawdown_leader_net_assets_usd_billions", "Store selected report-date net assets in USD billions, rounded to 6 decimals."),
    _v("liquidity_drawdown_start_date", "Store the starting liquidity date as an ISO YYYY-MM-DD string."),
    _v("liquidity_drawdown_end_date", "Store the ending liquidity date as an ISO YYYY-MM-DD string."),
    _v("start_weekly_liquid_assets_usd_billions", "Store starting weekly liquid assets in USD billions, rounded to 6 decimals."),
    _v("end_weekly_liquid_assets_usd_billions", "Store ending weekly liquid assets in USD billions, rounded to 6 decimals."),
    _v("start_weekly_liquid_assets_pct", "Store the starting reported weekly-liquid-assets fraction in percent, rounded to 4 decimals."),
    _v("end_weekly_liquid_assets_pct", "Store the ending reported weekly-liquid-assets fraction in percent, rounded to 4 decimals."),
    _v("liquidity_drawdown_leader_margin_pp", "Store the selected decline magnitude minus the runner-up decline magnitude in percentage points, rounded to 4 decimals."),
    _v("event_flow_share_class_count", "Store the selected accession-date flow-row count as an integer."),
    _v("event_gross_subscriptions_usd_billions", "Store aggregate gross subscriptions in USD billions, rounded to 6 decimals."),
    _v("event_gross_redemptions_usd_billions", "Store aggregate gross redemptions in USD billions, rounded to 6 decimals."),
    _v("event_net_outflow_to_net_assets_pct", "Store event net outflow divided by report-date net assets in percent, rounded to 4 decimals."),
    _v("net_outflow_to_weekly_liquidity_decline_pct", "Store event net outflow divided by the weekly-liquid-assets decline in percent, rounded to 4 decimals."),
    _v("prior_cftc_report_date", "Store the prior CFTC report date as an ISO YYYY-MM-DD string."),
    _v("next_cftc_report_date", "Store the next CFTC report date as an ISO YYYY-MM-DD string."),
    _v("prior_sofr3m_lev_money_net_contracts", "Store prior leveraged-money long minus short contracts as an integer."),
    _v("next_sofr3m_lev_money_net_contracts", "Store next leveraged-money long minus short contracts as an integer."),
    _v("sofr3m_lev_money_net_change_contracts", "Store next minus prior leveraged-money net contracts as an integer."),
    _v("prior_sofr3m_open_interest_contracts", "Store prior open interest in contracts as an integer."),
    _v("next_sofr3m_open_interest_contracts", "Store next open interest in contracts as an integer."),
    _v("sofr3m_open_interest_change_contracts", "Store next minus prior open interest in contracts as an integer."),
    _v("start_daily_liquid_assets_usd_billions", "Store starting daily liquid assets in USD billions, rounded to 6 decimals."),
    _v("end_daily_liquid_assets_usd_billions", "Store ending daily liquid assets in USD billions, rounded to 6 decimals."),
    _v("weekly_liquidity_decomposition_residual_usd_millions", "Store decomposed change minus reported weekly-liquid-assets change in USD millions, rounded to 3 decimals."),
]


@lru_cache(maxsize=1)
def _analysis():
    filings = load_expansion_table("sec_nmfp").copy()
    filings["report_date"] = pd.to_datetime(filings.report_date)
    eligible = filings.loc[
        filings.form.eq("N-MFP3")
        & filings.report_date.dt.year.eq(2025)
        & filings.net_assets_usd.ge(10e9)
    ].copy()

    liquidity = load_expansion_table("sec_nmfp_liquidity").copy()
    liquidity["liquidity_date"] = pd.to_datetime(liquidity.liquidity_date)
    path = eligible[["accession", "series_name", "report_date", "net_assets_usd"]].merge(
        liquidity, on="accession", validate="one_to_many"
    )
    path = path.loc[
        path.liquidity_date.dt.to_period("M").eq(path.report_date.dt.to_period("M"))
    ].sort_values(["accession", "liquidity_date"]).copy()
    shift_columns = [
        "liquidity_date", "weekly_liquid_assets_usd", "weekly_liquid_assets_fraction",
        "daily_liquid_assets_usd", "daily_liquid_assets_fraction",
    ]
    for column in shift_columns:
        path[f"prior_{column}"] = path.groupby("accession")[column].shift()
    path["weekly_fraction_change"] = (
        path.weekly_liquid_assets_fraction - path.prior_weekly_liquid_assets_fraction
    )
    steps = path.dropna(subset=["weekly_fraction_change"]).loc[
        path.weekly_liquid_assets_usd.gt(0)
        & path.prior_weekly_liquid_assets_usd.gt(0)
        & path.weekly_liquid_assets_fraction.gt(0)
        & path.prior_weekly_liquid_assets_fraction.gt(0)
    ].copy()
    ranking = steps.sort_values(
        ["weekly_fraction_change", "liquidity_date", "series_name", "accession"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    leader = ranking.iloc[0]

    flows = load_expansion_table("sec_nmfp_flows").copy()
    flows["flow_date"] = pd.to_datetime(flows.flow_date)
    event_flows = flows.loc[
        flows.accession.eq(leader.accession) & flows.flow_date.eq(leader.liquidity_date)
    ]
    if event_flows.empty:
        raise ValueError("selected liquidity step has no exact-date flow rows")

    cftc = load_expansion_table("cftc_cot").copy()
    cftc["date"] = pd.to_datetime(cftc["Report_Date_as_YYYY-MM-DD"])
    cftc = cftc.loc[
        cftc.report_scope.eq("futures_only")
        & cftc.CFTC_Contract_Market_Code.astype("string").eq("134741")
    ].sort_values("date")
    prior = cftc.loc[cftc.date.lt(leader.liquidity_date)].iloc[-1]
    following = cftc.loc[cftc.date.gt(leader.liquidity_date)].iloc[0]
    return eligible, ranking, leader, event_flows, prior, following


@lru_cache(maxsize=1)
def ground_truth():
    eligible, ranking, leader, event_flows, prior, following = _analysis()
    subscriptions = float(event_flows.gross_subscriptions_usd.sum())
    redemptions = float(event_flows.gross_redemptions_usd.sum())
    net_outflow = redemptions - subscriptions
    weekly_change = float(leader.weekly_liquid_assets_usd - leader.prior_weekly_liquid_assets_usd)
    weekly_decline = -weekly_change
    prior_net = int(prior.Lev_Money_Positions_Long_All - prior.Lev_Money_Positions_Short_All)
    following_net = int(following.Lev_Money_Positions_Long_All - following.Lev_Money_Positions_Short_All)
    start_non_daily = float(leader.prior_weekly_liquid_assets_usd - leader.prior_daily_liquid_assets_usd)
    end_non_daily = float(leader.weekly_liquid_assets_usd - leader.daily_liquid_assets_usd)
    daily_change = float(leader.daily_liquid_assets_usd - leader.prior_daily_liquid_assets_usd)
    non_daily_change = end_non_daily - start_non_daily
    decomposed = daily_change + non_daily_change

    return (
        len(eligible), len(ranking), str(leader.accession), str(leader.series_name),
        leader.report_date.strftime("%Y-%m-%d"), float(leader.net_assets_usd) / 1e9,
        leader.prior_liquidity_date.strftime("%Y-%m-%d"), leader.liquidity_date.strftime("%Y-%m-%d"),
        float(leader.prior_weekly_liquid_assets_usd) / 1e9,
        float(leader.weekly_liquid_assets_usd) / 1e9,
        float(leader.prior_weekly_liquid_assets_fraction) * 100,
        float(leader.weekly_liquid_assets_fraction) * 100,
        (abs(float(leader.weekly_fraction_change)) - abs(float(ranking.iloc[1].weekly_fraction_change))) * 100,
        len(event_flows), subscriptions / 1e9, redemptions / 1e9,
        net_outflow / float(leader.net_assets_usd) * 100,
        net_outflow / weekly_decline * 100,
        prior.date.strftime("%Y-%m-%d"), following.date.strftime("%Y-%m-%d"),
        prior_net, following_net, following_net - prior_net,
        int(prior.Open_Interest_All), int(following.Open_Interest_All),
        int(following.Open_Interest_All - prior.Open_Interest_All),
        float(leader.prior_daily_liquid_assets_usd) / 1e9,
        float(leader.daily_liquid_assets_usd) / 1e9,
        (decomposed - weekly_change) / 1e6,
    )


DECIMALS = [
    0, 0, None, None, None, 6, None, None, 6, 6, 4, 4, 4,
    0, 6, 6, 4, 4,
    None, None, 0, 0, 0, 0, 0, 0,
    6, 6, 3,
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_nmfp_liquidity_drawdown": turn_validator(validate_turn_1),
    "validate_event_flow_bridge": turn_validator(validate_turn_2),
    "validate_cftc_bracket": turn_validator(validate_turn_3),
    "validate_liquidity_decomposition": turn_validator(validate_turn_4),
}
