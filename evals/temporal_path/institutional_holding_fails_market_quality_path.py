"""Trace the largest institutional holding into fails-to-deliver and market quality.

At the 2025-09-30 report period NVIDIA (CUSIP 67066G104) is the largest
plain-share institutional holding in the 13F data set: 5,234 original 13F-HR
filers report 2,950.433 billion USD over 15.83 billion shares, 202.307 billion
USD ahead of Microsoft. The CUSIP maps to symbol NVDA in the fails-to-deliver
files, which show fails on 50 of the 62 settlement dates in 2025 Q4, 2.78
million shares in total and a 761,202-share peak on 2025-12-18 — 0.0176
percent of the institutional share count. In MIDAS, NVDA's odd-lot volume
share over the same window is 31.1167 percent (hidden share 20.7796 percent on
SEC's eligible-volume denominator) against a 46.798 percent median
across the 17 of the twenty largest holdings that map to a MIDAS stock symbol
(IVV and SPY are ETFs; Berkshire is BRKB in the fails files and BRK.B in
MIDAS), ranking fourteenth. Against 2024-09-30 the plain-share value rose
14.35 percent while shares held rose 2.86 percent.

Rejected alternatives (each pinned by the query): ranking by value_usd_total,
which adds put/call and non-share rows; counting 13F-HR/A amendments; using
the aggregate table's issuer name rather than the CUSIP as the link key (the
same issuer appears under several classes); dividing fails by the period's
shares outstanding rather than the institutional share count; restricting
MIDAS to one exchange or to non-stock security types; ranking odd-lot shares
ascending; translating BRKB to BRK.B, which the query forbids; dividing hidden
volume by total trade volume instead of the SEC's hidden-eligible volume
(19.6305 instead of 20.7796 percent). Every ratio is a ratio of sums over the
window, not a mean of daily ratios. Dollar figures depend on the governed
table's per-accession unit inference (value_reporting_convention), which the
query relies on rather than re-deriving. The 2024-09-30 comparison in turn 4 stays inside the post-split,
dollar-unit periods, so no unit or split adjustment is required — the query
says so, and a thousands-of-dollars rescale would be wrong by 1,000x.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, codes and labels match exactly.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


PERIOD = "2025-09-30"
PRIOR_PERIOD = "2024-09-30"
WINDOW = ("2025-10-01", "2025-12-31")
COHORT_SIZE = 20


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "period_cusip_count", "leader_cusip", "leader_issuer_name", "leader_filer_count",
    "leader_share_value_usd_billions", "leader_share_amount", "leader_margin_usd_billions",
]
TURN_2_NAMES = [
    "leader_symbol", "window_settlement_date_count", "leader_fail_date_count", "leader_total_fails",
    "leader_peak_fails", "leader_peak_fail_settlement_date", "leader_fails_to_institutional_shares_pct",
]
TURN_3_NAMES = [
    "leader_midas_trading_day_count", "leader_odd_lot_volume_share_pct", "leader_hidden_volume_share_pct",
    "cohort_symbol_count", "cohort_median_odd_lot_volume_share_pct", "cohort_median_hidden_volume_share_pct",
    "leader_odd_lot_share_rank",
]
TURN_4_NAMES = [
    "prior_period_share_value_usd_billions", "share_value_change_pct", "prior_period_filer_count",
    "filer_count_change", "share_amount_change_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the count of distinct CUSIPs with a 2025-09-30 holdings row as an integer."),
    _v(TURN_1_NAMES[1], "Store the leading CUSIP as a string."),
    _v(TURN_1_NAMES[2], "Store the leading CUSIP's issuer name as text; casing and repeated whitespace are not significant."),
    _v(TURN_1_NAMES[3], "Store the leading CUSIP's distinct filer count as an integer."),
    _v(TURN_1_NAMES[4], "Store the leading CUSIP's plain-share institutional value in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[5], "Store the leading CUSIP's plain-share institutional share amount as an integer."),
    _v(TURN_1_NAMES[6], "Store the leader minus runner-up plain-share value in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[0], "Store the fails-to-deliver symbol mapped to the leading CUSIP as text."),
    _v(TURN_2_NAMES[1], "Store the count of distinct settlement dates in the 2025-10-01 to 2025-12-31 fails file window, across all securities, as an integer."),
    _v(TURN_2_NAMES[2], "Store the count of settlement dates on which the leading CUSIP has a fails row as an integer."),
    _v(TURN_2_NAMES[3], "Store the leading CUSIP's total failed shares over the window as an integer."),
    _v(TURN_2_NAMES[4], "Store the leading CUSIP's largest single-settlement-date fails quantity as an integer."),
    _v(TURN_2_NAMES[5], "Store the settlement date of that peak as a YYYY-MM-DD string (earliest date on a tie)."),
    _v(TURN_2_NAMES[6], "Store total failed shares divided by the turn-1 institutional share amount in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the count of MIDAS trading days for the symbol in the window as an integer."),
    _v(TURN_3_NAMES[1], "Store the symbol's odd-lot volume divided by trade volume over the window in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the symbol's hidden volume divided by the trade volume eligible for hidden orders over the window in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3], "Store the count of top-20 CUSIPs that map to a MIDAS stock symbol as an integer."),
    _v(TURN_3_NAMES[4], "Store the median odd-lot volume share across the mapped cohort in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store the median hidden volume share across the mapped cohort in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[6], "Store the leader's rank within the cohort by odd-lot volume share descending (1 = highest) as an integer."),
    _v(TURN_4_NAMES[0], "Store the leading CUSIP's plain-share institutional value at 2024-09-30 in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[1], "Store the 2024-09-30 to 2025-09-30 change in plain-share value in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the leading CUSIP's distinct filer count at 2024-09-30 as an integer."),
    _v(TURN_4_NAMES[3], "Store the 2025-09-30 filer count minus the 2024-09-30 filer count as an integer."),
    _v(TURN_4_NAMES[4], "Store the change in plain-share amount from 2024-09-30 to 2025-09-30 in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, None, None, 0, 6, 0, 6,
    None, 0, 0, 0, 0, None, 4,
    0, 4, 4, 0, 4, 4, 0,
    6, 4, 0, 0, 4,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def ground_truth():
    holdings = load_expansion_table("sec_13f_holdings").copy()
    holdings["period_of_report"] = holdings.period_of_report.astype(str)
    holdings["cusip"] = holdings.cusip.astype(str)
    period = holdings.loc[holdings.period_of_report.eq(PERIOD)].sort_values(
        ["share_value_usd_total", "cusip"], ascending=[False, True]).reset_index(drop=True)
    leader, runner = period.iloc[0], period.iloc[1]
    cohort = period.head(COHORT_SIZE)

    fails = load_expansion_table("sec_ftd").copy()
    fails["settlement_date"] = fails.settlement_date.astype(str)
    window = fails.loc[fails.settlement_date.between(*WINDOW)].copy()
    window["cusip"] = window.cusip.astype(str)
    window["symbol"] = window.symbol.astype("string")
    leader_fails = window.loc[window.cusip.eq(leader.cusip)].sort_values(["settlement_date"])
    symbols = leader_fails.symbol.dropna().unique()
    if len(symbols) != 1:
        raise ValueError("leader CUSIP does not map to exactly one fails-to-deliver symbol")
    symbol = str(symbols[0])
    peak = leader_fails.sort_values(["quantity_fails", "settlement_date"], ascending=[False, True]).iloc[0]

    midas = load_expansion_table("sec_midas_security_exchange").copy()
    midas["date"] = midas.date.astype(str)
    midas = midas.loc[midas.date.between(*WINDOW) & midas.security_type.astype(str).eq("Stock")].copy()
    midas["ticker"] = midas.ticker.astype(str)
    by_symbol = midas.groupby("ticker").agg(
        days=("date", "nunique"), odd=("odd_lot_volume_thousands", "sum"),
        hidden=("hidden_volume_thousands", "sum"), volume=("trade_volume_thousands", "sum"),
        hidden_base=("trade_volume_for_hidden_thousands", "sum"),
    )
    by_symbol["odd_share"] = by_symbol.odd / by_symbol.volume * 100
    # SEC's hidden-volume rate divides by the trade volume eligible to be hidden.
    by_symbol["hidden_share"] = by_symbol.hidden / by_symbol.hidden_base * 100
    cusip_symbol = window.dropna(subset=["symbol"]).sort_values("settlement_date").groupby("cusip").symbol.first()
    cohort_symbols = cohort.cusip.map(cusip_symbol).dropna().astype(str)
    cohort_stats = by_symbol.loc[by_symbol.index.isin(cohort_symbols)]
    leader_stats = by_symbol.loc[symbol]
    leader_rank = int((cohort_stats.odd_share > leader_stats.odd_share).sum()) + 1

    prior = holdings.loc[holdings.period_of_report.eq(PRIOR_PERIOD) & holdings.cusip.eq(leader.cusip)]
    if len(prior) != 1:
        raise ValueError("prior-period row for the leader is not unique")
    prior = prior.iloc[0]

    return (
        int(period.cusip.nunique()),
        str(leader.cusip),
        str(leader.issuer_name),
        int(leader.filer_count),
        float(leader.share_value_usd_total) / 1e9,
        int(leader.share_amount_total),
        float(leader.share_value_usd_total - runner.share_value_usd_total) / 1e9,
        symbol,
        int(window.settlement_date.nunique()),
        int(leader_fails.settlement_date.nunique()),
        int(leader_fails.quantity_fails.sum()),
        int(peak.quantity_fails),
        str(peak.settlement_date),
        float(leader_fails.quantity_fails.sum()) / float(leader.share_amount_total) * 100,
        int(leader_stats.days),
        float(leader_stats.odd_share),
        float(leader_stats.hidden_share),
        len(cohort_stats),
        float(cohort_stats.odd_share.median()),
        float(cohort_stats.hidden_share.median()),
        leader_rank,
        float(prior.share_value_usd_total) / 1e9,
        (float(leader.share_value_usd_total) / float(prior.share_value_usd_total) - 1) * 100,
        int(prior.filer_count),
        int(leader.filer_count) - int(prior.filer_count),
        (float(leader.share_amount_total) / float(prior.share_amount_total) - 1) * 100,
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in ("leader_issuer_name", "leader_symbol"):
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
    return validate_ordered_outputs(normalized, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_institutional_leader": turn_validator(validate_turn_1),
    "validate_fails_linkage": turn_validator(validate_turn_2),
    "validate_market_quality_context": turn_validator(validate_turn_3),
    "validate_prior_period_change": turn_validator(validate_turn_4),
}
