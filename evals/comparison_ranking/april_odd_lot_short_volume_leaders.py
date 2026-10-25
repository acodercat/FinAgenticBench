"""Compare April MIDAS odd-lot scale with FINRA short-volume share.

Across 6,516 tickers present on every one of the 21 common April 2025 trading
dates, the top five by aggregate MIDAS odd-lot volume are NVDA, TSLA, SPY, PLTR
and AAPL.  NVDA leads absolute odd-lot volume at 0.466144230 billion shares, but
PLTR ranks fourth on that measure and has the cohort's highest FINRA short-volume
share at 59.6526%.  Ranking by MIDAS trade volume or equal-weighted row ratios,
using raw ticker-string equality instead of the pinned dot-to-slash share-class
bridge, or adding short-exempt volume to the FINRA numerator, produces a
different population or value.

The query pins full common-date coverage, ratio-of-sums aggregation, absolute
odd-lot-volume cohort selection, short-only FINRA numerator and tie breaks, so
the integrated case is a hard baseline.  Numeric validation uses 0.6 x 10^-N
rounding-boundary tolerance; counts, ranks, ticker labels and the ordered cohort
are exact.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator


variables = [
    Variable("common_trading_date_count", None, "Store the common April trading-date count as an integer."),
    Variable("full_coverage_common_ticker_count", None, "Store the full-common-date-coverage ticker count as an integer."),
    Variable("top_five_midas_odd_lot_volume_tickers", None, "Store the top five tickers by aggregate MIDAS odd-lot volume as an ordered Python list."),
    Variable("largest_midas_odd_lot_volume_ticker", None, "Store the ticker with the largest aggregate MIDAS odd-lot volume."),
    Variable("largest_ticker_midas_trade_volume_billion_shares", None, "Store that ticker's aggregate MIDAS trade volume in billion shares, rounded to 6 decimals."),
    Variable("largest_ticker_midas_odd_lot_volume_billion_shares", None, "Store that ticker's aggregate MIDAS odd-lot volume in billion shares, rounded to 6 decimals."),
    Variable("largest_ticker_midas_odd_lot_share_pct", None, "Store that ticker's aggregate MIDAS odd-lot share in percent, rounded to 4 decimals."),
    Variable("top_five_highest_finra_short_share_ticker", None, "Store the top-five cohort ticker with the highest FINRA short-volume share."),
    Variable("finra_short_share_leader_midas_odd_lot_rank", None, "Store that ticker's MIDAS odd-lot-volume rank within the five-ticker cohort as an integer."),
    Variable("finra_short_share_leader_midas_odd_lot_volume_billion_shares", None, "Store that ticker's MIDAS odd-lot volume in billion shares, rounded to 4 decimals."),
    Variable("finra_short_share_leader_midas_odd_lot_share_pct", None, "Store that ticker's aggregate MIDAS odd-lot share in percent, rounded to 4 decimals."),
    Variable("finra_short_share_leader_short_volume_billion_shares", None, "Store that ticker's FINRA short volume in billion shares, rounded to 6 decimals."),
    Variable("finra_short_share_leader_short_exempt_volume_billion_shares", None, "Store that ticker's FINRA short-exempt volume in billion shares, rounded to 4 decimals."),
    Variable("finra_short_share_leader_total_volume_billion_shares", None, "Store that ticker's FINRA total volume in billion shares, rounded to 6 decimals."),
    Variable("finra_short_share_leader_short_volume_share_pct", None, "Store that ticker's FINRA short volume divided by FINRA total volume in percent, rounded to 4 decimals."),
]


def _analysis():
    midas = load_expansion_table("sec_midas_security_exchange").dropna(
        subset=["trade_volume_thousands", "odd_lot_volume_thousands"]
    ).copy()
    finra = load_expansion_table("finra_short_volume").copy()
    midas = midas.loc[midas.date.astype(str).between("2025-04-01", "2025-04-30")]
    finra = finra.loc[finra.date.astype(str).between("2025-04-01", "2025-04-30")]
    common_dates = sorted(set(midas.date) & set(finra.date))
    if len(common_dates) != 21:
        raise ValueError(f"expected 21 common April dates, found {len(common_dates)}")
    midas = midas.loc[midas.date.isin(common_dates)]
    finra = finra.loc[finra.date.isin(common_dates)]

    midas_tickers = midas.groupby("ticker", as_index=False).agg(
        midas_trade_thousands=("trade_volume_thousands", "sum"),
        midas_odd_thousands=("odd_lot_volume_thousands", "sum"),
        midas_days=("date", "nunique"),
    )
    midas_tickers["midas_odd_share_pct"] = (
        midas_tickers.midas_odd_thousands
        / midas_tickers.midas_trade_thousands
        * 100
    )
    finra_tickers = finra.groupby("symbol", as_index=False).agg(
        finra_short=("short_volume_shares", "sum"),
        finra_short_exempt=("short_exempt_volume_shares", "sum"),
        finra_total=("total_volume_shares", "sum"),
        finra_days=("date", "nunique"),
    )
    finra_tickers["finra_short_share_pct"] = (
        finra_tickers.finra_short / finra_tickers.finra_total * 100
    )
    midas_tickers["finra_symbol_key"] = midas_tickers.ticker.str.replace(
        ".", "/", regex=False
    )
    if midas_tickers.finra_symbol_key.duplicated().any():
        raise ValueError("MIDAS dot-to-slash ticker bridge is not one-to-one")
    common = midas_tickers.merge(
        finra_tickers,
        left_on="finra_symbol_key",
        right_on="symbol",
        validate="one_to_one",
    )
    full = common.loc[
        common.midas_days.eq(len(common_dates))
        & common.finra_days.eq(len(common_dates))
    ].copy()
    cohort = full.sort_values(
        ["midas_odd_thousands", "ticker"], ascending=[False, True]
    ).head(5).reset_index(drop=True)
    cohort["midas_odd_rank"] = cohort.index + 1
    short_leader = cohort.sort_values(
        ["finra_short_share_pct", "ticker"], ascending=[False, True]
    ).iloc[0]
    return common_dates, full, cohort, short_leader


@lru_cache(maxsize=1)
def ground_truth():
    common_dates, full, cohort, short_leader = _analysis()
    odd_leader = cohort.iloc[0]
    return (
        len(common_dates),
        len(full),
        cohort.ticker.tolist(),
        odd_leader.ticker,
        odd_leader.midas_trade_thousands / 1e6,
        odd_leader.midas_odd_thousands / 1e6,
        odd_leader.midas_odd_share_pct,
        short_leader.ticker,
        int(short_leader.midas_odd_rank),
        short_leader.midas_odd_thousands / 1e6,
        short_leader.midas_odd_share_pct,
        short_leader.finra_short / 1e9,
        short_leader.finra_short_exempt / 1e9,
        short_leader.finra_total / 1e9,
        short_leader.finra_short_share_pct,
    )


def validate(outputs):
    return validate_ordered_outputs(
        outputs,
        variables,
        ground_truth(),
        [
    0, 0, None, None, 6, 6, 4, None, 0, 4, 4, 6, 4, 6, 4,
],
    )



TURN_1_NAMES = [
    "common_trading_date_count",
    "full_coverage_common_ticker_count",
]
TURN_2_NAMES = [
    "top_five_midas_odd_lot_volume_tickers",
    "largest_midas_odd_lot_volume_ticker",
    "largest_ticker_midas_trade_volume_billion_shares",
    "largest_ticker_midas_odd_lot_volume_billion_shares",
    "largest_ticker_midas_odd_lot_share_pct",
]
TURN_3_NAMES = [
    "top_five_highest_finra_short_share_ticker",
    "finra_short_share_leader_midas_odd_lot_rank",
    "finra_short_share_leader_midas_odd_lot_volume_billion_shares",
    "finra_short_share_leader_midas_odd_lot_share_pct",
    "finra_short_share_leader_short_volume_billion_shares",
    "finra_short_share_leader_short_exempt_volume_billion_shares",
    "finra_short_share_leader_total_volume_billion_shares",
    "finra_short_share_leader_short_volume_share_pct",
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
    "validate_common_population": turn_validator(_turn_subset(TURN_1_NAMES)),
    "validate_odd_lot_cohort": turn_validator(_turn_subset(TURN_2_NAMES)),
    "validate_short_share_leader": turn_validator(_turn_subset(TURN_3_NAMES)),
}
