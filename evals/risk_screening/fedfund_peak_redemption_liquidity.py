"""Align a fund-series redemption event with daily broad-market stress.

FEDFUND accession 0001752724-25-098429 has its largest April series-level net
redemption on 2025-04-09. Fifteen class rows aggregate to 5.619207154 billion
USD redemptions, 2.867097246 billion subscriptions and 2.752109908 billion net
redemption. Same-date daily liquid assets are 136.0 billion USD, for 2.0236%.

After normalizing the N-MFP DD-MMM-YYYY date to the OFR ISO date, the event-day
FSI is 1.934. Volatility contributes +2.579 while Credit, Equity valuation,
Safe assets and Funding sum to -0.645. Across 22 April observations, the event
date ranks third; the April maximum is 2.743 on 2025-04-07, leaving an
event-minus-peak gap of -0.809.

Using the largest class gives 2.681589 billion USD net redemption; weekly rather
than daily liquid assets gives a different ratio; substituting the April FSI
maximum for the same-date FSI gives 2.743 instead of 1.934. The query pins the
series grain, daily-liquidity denominator, same-calendar-date alignment and OFR
monthly ranking, so these are regression alternatives and the integrated case is
a hard baseline. Numeric validation uses 0.6 x 10^-N rounding-boundary tolerance;
dates, counts and ranks are exact.
"""

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


ACCESSION = "0001752724-25-098429"
variables = [
    Variable("peak_net_redemption_date", None, "Store the peak net-redemption date as a YYYY-MM-DD string."),
    Variable("share_class_count", None, "Store the contributing share-class count as an integer."),
    Variable("gross_redemptions_usd_billions", None, "Store aggregated gross redemptions in USD billions, rounded to 6 decimals."),
    Variable("gross_subscriptions_usd_billions", None, "Store aggregated gross subscriptions in USD billions, rounded to 6 decimals."),
    Variable("same_day_daily_liquid_assets_usd_billions", None, "Store same-date daily liquid assets in USD billions, rounded to 6 decimals."),
    Variable("net_redemption_to_daily_liquid_assets_pct", None, "Store net redemption divided by daily liquid assets in percent, rounded to 4 decimals."),
    Variable("event_day_ofr_fsi", None, "Store the same-calendar-date OFR FSI, rounded to 3 decimals."),
    Variable("event_day_volatility_contribution", None, "Store the event-day OFR Volatility category contribution, rounded to 3 decimals."),
    Variable("event_day_other_market_categories_sum", None, "Store the event-day sum of Credit, Equity valuation, Safe assets and Funding contributions, rounded to 3 decimals."),
    Variable("april_ofr_observation_count", None, "Store the April 2025 OFR observation count as an integer."),
    Variable("april_peak_ofr_fsi_date", None, "Store the April 2025 maximum-FSI date as a YYYY-MM-DD string."),
    Variable("april_peak_ofr_fsi", None, "Store the April 2025 maximum OFR FSI, rounded to 3 decimals."),
    Variable("event_day_fsi_descending_rank", None, "Store the event day's descending FSI rank within April as an integer."),
]


def _display_date(value):
    # Reported in ISO, which is what the suite states elsewhere and, unlike
    # DD-MMM-YYYY, fixes the casing the validator compares.
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _analysis():
    flows = load_expansion_table("sec_nmfp_flows").loc[
        lambda frame: frame.accession.eq(ACCESSION)
    ].copy()
    flows["event_date"] = pd.to_datetime(
        flows.flow_date, format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    flows = flows.loc[flows.event_date.between("2025-04-01", "2025-04-30")]
    daily = flows.groupby("event_date").agg(
        gross_redemptions_usd=("gross_redemptions_usd", "sum"),
        gross_subscriptions_usd=("gross_subscriptions_usd", "sum"),
        classes=("class_id", "nunique"),
    )
    daily["net_redemption_usd"] = (
        daily.gross_redemptions_usd - daily.gross_subscriptions_usd
    )
    daily = daily.reset_index()
    event = daily.sort_values(
        ["net_redemption_usd", "event_date"], ascending=[False, True]
    ).iloc[0]

    liquidity = load_expansion_table("sec_nmfp_liquidity").loc[
        lambda frame: frame.accession.eq(ACCESSION)
    ].copy()
    liquidity["event_date"] = pd.to_datetime(
        liquidity.liquidity_date, format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    matched_liquidity = liquidity.loc[liquidity.event_date.eq(event.event_date)]
    if len(matched_liquidity) != 1:
        raise ValueError("expected one matching liquidity observation")

    fsi = load_expansion_table("ofr_market_stress").loc[
        lambda frame: frame.date.between("2025-04-01", "2025-04-30")
    ].copy()
    if fsi.date.duplicated().any() or fsi.ofr_fsi.isna().any():
        raise ValueError("invalid April OFR FSI population")
    ranked = fsi.sort_values(
        ["ofr_fsi", "date"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["descending_rank"] = ranked.index + 1
    event_fsi = ranked.loc[ranked.date.eq(event.event_date)]
    if len(event_fsi) != 1:
        raise ValueError("expected one same-date OFR FSI observation")
    event_fsi = event_fsi.iloc[0]
    peak_fsi = ranked.iloc[0]
    other_categories = (
        event_fsi.credit
        + event_fsi.equity_valuation
        + event_fsi.safe_assets
        + event_fsi.funding
    )
    if abs(event_fsi.ofr_fsi - event_fsi.volatility - other_categories) > 1e-9:
        raise ValueError("OFR market-category contributions do not reconcile to FSI")
    return event, matched_liquidity.iloc[0], event_fsi, peak_fsi, len(ranked), other_categories


def ground_truth():
    event, liquid, event_fsi, peak_fsi, observation_count, other_categories = _analysis()
    liquid_value = liquid.daily_liquid_assets_usd
    return (
        _display_date(event.event_date),
        int(event.classes),
        event.gross_redemptions_usd / 1e9,
        event.gross_subscriptions_usd / 1e9,
        liquid_value / 1e9,
        event.net_redemption_usd / liquid_value * 100,
        event_fsi.ofr_fsi,
        event_fsi.volatility,
        other_categories,
        observation_count,
        _display_date(peak_fsi.date),
        peak_fsi.ofr_fsi,
        int(event_fsi.descending_rank),
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs,
        variables,
        ground_truth(),
        [
    None, 0, 6, 6, 6, 4, 3, 3, 3, 0, None, 3, 0,
],
    )



TURN_1_NAMES = [
    "peak_net_redemption_date",
    "share_class_count",
    "gross_redemptions_usd_billions",
    "gross_subscriptions_usd_billions",
    "same_day_daily_liquid_assets_usd_billions",
    "net_redemption_to_daily_liquid_assets_pct",
]
TURN_2_NAMES = [
    "event_day_ofr_fsi",
    "event_day_volatility_contribution",
    "event_day_other_market_categories_sum",
]
TURN_3_NAMES = [
    "april_ofr_observation_count",
    "april_peak_ofr_fsi_date",
    "april_peak_ofr_fsi",
    "event_day_fsi_descending_rank",
]


def _turn_subset(names):
    """Score only `names`, filling the other outputs with their own answers."""
    def run(outputs):
        merged = dict(zip((variable.name for variable in variables), ground_truth()))
        merged.update({name: outputs.get(name) for name in names})
        return validate(merged)

    return run


# The turn contract: `validators` maps the name a turn's `validator`
# field cites to the callable that judges it.
validators = {
    "validate_fund_event": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_event_day_stress": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_month_ranking": turn_validator(_turn_subset(TURN_3_NAMES)),
}
