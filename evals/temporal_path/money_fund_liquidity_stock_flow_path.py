"""A tax-exempt money fund's daily flow and liquid-asset stock path.

The selected original September 2025 N-MFP3 accession has one share class and
one liquidity row on each of 21 reported dates. Ranking positive net
redemptions against the *previous reported date's* daily liquid assets picks
September 11 (52.7528%), ahead of September 4 (38.7308%). The September 11
net redemption is USD 247.15775846 million; the reported daily liquid asset
stock falls by only USD 27.04202153 million from September 10. Treating net
redemptions as the change in daily liquid assets is therefore a measured wrong
reconciliation, not a valid cash-flow identity. Weekly liquid assets include
daily liquid assets and cannot be added to them. This is a baseline because the
queries explicitly ask for the distinct reported stocks and flows.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


SERIES_ID = "S000001480"
REPORT_DATE = "2025-09-30"
CUTOFF = "2025-12-31"
MILLION = Decimal(1_000_000)

OUTPUTS = (
    ("liquidity_path_accession", None, "the selected original filing accession as an exact 10-2-6 digit SEC string"),
    ("liquidity_path_category", None, "the exact 2025 source-verbatim fund category token: Government, Other Tax Exempt, Prime, or Single State"),
    ("liquidity_path_net_assets_m", 4, "the September series net assets in USD millions"),
    ("liquidity_path_peak_date", None, "the selected net-redemption date as an ISO YYYY-MM-DD string"),
    ("liquidity_path_prior_date", None, "the preceding reported liquidity date as an ISO YYYY-MM-DD string"),
    ("liquidity_path_peak_subscriptions_m", 4, "selected-date gross subscriptions in USD millions"),
    ("liquidity_path_peak_redemptions_m", 4, "selected-date gross redemptions in USD millions"),
    ("liquidity_path_prior_daily_m", 4, "preceding-date daily liquid assets in USD millions"),
    ("liquidity_path_peak_daily_m", 4, "selected-date daily liquid assets in USD millions"),
    ("liquidity_path_peak_weekly_m", 4, "selected-date weekly liquid assets in USD millions"),
    ("liquidity_path_peak_daily_pct", 2, "selected-date reported daily liquid asset fraction in percent"),
    ("liquidity_path_peak_weekly_pct", 2, "selected-date reported weekly liquid asset fraction in percent"),
    ("liquidity_path_next_date", None, "next reported liquidity date as an ISO YYYY-MM-DD string"),
    ("liquidity_path_next_net_redemption_m", 4, "next-date gross redemptions less subscriptions in USD millions"),
    ("liquidity_path_next_daily_m", 4, "next-date daily liquid assets in USD millions"),
    ("liquidity_path_next_weekly_m", 4, "next-date weekly liquid assets in USD millions"),
)

variables = [
    Variable(name, None, "Store " + description +
             (f", rounded to {precision} decimals from unrounded source values."
              if precision is not None and precision > 0 else "."))
    for name, precision, description in OUTPUTS
]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 3), (3, 8), (8, 12), (12, 16)))


def _decimal(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite N-MFP value")
    return result


def _filing():
    series = load_expansion_table("sec_nmfp")
    rows = series.loc[
        series.series_id.eq(SERIES_ID) & series.report_date.eq(REPORT_DATE) &
        series.form.eq("N-MFP3") & series.filing_date.le(CUTOFF)
    ]
    if len(rows) != 1:
        raise ValueError("expected one original September filing")
    return rows.iloc[0]


def analysis() -> dict[str, str | float]:
    filing = _filing()
    accession = str(filing.accession)
    flows = load_expansion_table("sec_nmfp_flows")
    flows = flows.loc[flows.accession.eq(accession)].copy()
    liquidity = load_expansion_table("sec_nmfp_liquidity")
    liquidity = liquidity.loc[liquidity.accession.eq(accession)].copy()
    if flows.empty or liquidity.empty or flows.class_id.isna().any():
        raise ValueError("missing filing flow or liquidity rows")
    if flows.duplicated(["class_id", "flow_date"]).any():
        raise ValueError("duplicate physical class-day flow")
    if liquidity.liquidity_date.duplicated().any():
        raise ValueError("duplicate liquidity date")
    flow_by_day = flows.groupby("flow_date", sort=True)[
        ["gross_subscriptions_usd", "gross_redemptions_usd"]].sum()
    liquid_by_day = liquidity.set_index("liquidity_date").sort_index()
    days = list(liquid_by_day.index)
    if set(flow_by_day.index) != set(days) or len(days) < 3:
        raise ValueError("unmatched flow/liquidity dates")
    ranked = []
    for i in range(1, len(days)):
        date, prior = days[i], days[i - 1]
        gross = flow_by_day.loc[date]
        redemption = _decimal(gross.gross_redemptions_usd) - _decimal(gross.gross_subscriptions_usd)
        prior_daily = _decimal(liquid_by_day.loc[prior].daily_liquid_assets_usd)
        if redemption > 0 and prior_daily > 0:
            ranked.append((redemption / prior_daily, date, prior))
    if not ranked:
        raise ValueError("no positive net-redemption date with prior daily liquidity")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    _, date, prior = ranked[0]
    index = days.index(date)
    if index + 1 >= len(days):
        raise ValueError("selected date has no next reported liquidity day")
    next_date = days[index + 1]
    selected_flow, next_flow = flow_by_day.loc[date], flow_by_day.loc[next_date]
    selected_liquid, prior_liquid, next_liquid = (
        liquid_by_day.loc[date], liquid_by_day.loc[prior], liquid_by_day.loc[next_date]
    )
    def millions(value):
        return float(_decimal(value) / MILLION)
    return {
        "liquidity_path_accession": accession,
        "liquidity_path_category": str(filing.fund_category),
        "liquidity_path_net_assets_m": millions(filing.net_assets_usd),
        "liquidity_path_peak_date": str(date),
        "liquidity_path_prior_date": str(prior),
        "liquidity_path_peak_subscriptions_m": millions(selected_flow.gross_subscriptions_usd),
        "liquidity_path_peak_redemptions_m": millions(selected_flow.gross_redemptions_usd),
        "liquidity_path_prior_daily_m": millions(prior_liquid.daily_liquid_assets_usd),
        "liquidity_path_peak_daily_m": millions(selected_liquid.daily_liquid_assets_usd),
        "liquidity_path_peak_weekly_m": millions(selected_liquid.weekly_liquid_assets_usd),
        "liquidity_path_peak_daily_pct": float(_decimal(selected_liquid.daily_liquid_assets_fraction) * 100),
        "liquidity_path_peak_weekly_pct": float(_decimal(selected_liquid.weekly_liquid_assets_fraction) * 100),
        "liquidity_path_next_date": str(next_date),
        "liquidity_path_next_net_redemption_m": millions(
            _decimal(next_flow.gross_redemptions_usd) - _decimal(next_flow.gross_subscriptions_usd)),
        "liquidity_path_next_daily_m": millions(next_liquid.daily_liquid_assets_usd),
        "liquidity_path_next_weekly_m": millions(next_liquid.weekly_liquid_assets_usd),
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _check(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names],
                                    [truth[n] for n in names],
                                    [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 5)}
