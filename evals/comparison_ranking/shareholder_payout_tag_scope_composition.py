"""Measure payout composition across alternative SEC cash-flow concepts.

The case builds one profitable-filer population from full-year 2024 Company
Facts windows, then compares a common-stock-only dividend measure with a total
dividend measure. `PaymentsOfDividends` includes distributions to common and
preferred shareholders and noncontrolling interests, so the total-dividend rule
uses that concept when present and falls back to
`PaymentsOfDividendsCommonStock` only when the broader concept is absent. This is
a dataset reading convention, not an assertion that the concepts are identical.

The later turns carry that measure into company-level payout composition and
identify the largest repurchaser and dividend payer. SEC MIDAS daily stock
volume supplies independent market-liquidity context for those two issuers; it
does not explain their payout policy or establish a causal relation between
trading activity and distributions.

Measured rejected alternatives cover the common-stock-only
measure, loss-making filers, first-filed duplicates, unrestricted reporting
windows and a mismatched market-data window. The query pins each convention, so
this is a hard baseline rather than a hidden-method trap.

Boundaries: absence of one concept is not evidence that no dividend was paid;
fiscal years ending in different 2024 months cover different twelve-month
periods; a payout above annual net income is not an arithmetic inconsistency;
and cross-sectional medians do not describe every firm's channel choice.
Numeric validation uses the shared rounding tolerance; counts and identifiers
match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_companies, load_expansion_table, load_facts
from core.validation import turn_validator, validate_ordered_outputs


YEAR = "2024"
ANNUAL_FORM = "10-K"
WINDOW_DAYS = (350, 380)
BUYBACK_CONCEPT = "PaymentsForRepurchaseOfCommonStock"
NARROW_DIVIDEND_CONCEPT = "PaymentsOfDividendsCommonStock"
BROAD_DIVIDEND_CONCEPT = "PaymentsOfDividends"
INCOME_CONCEPT = "NetIncomeLoss"
MARKET_YEAR = "2024"
MARKET_SECURITY_TYPE = "Stock"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "profitable_filer_count", "buyback_total_usd_billions", "narrow_tag_filer_count",
    "broad_tag_filer_count", "both_tag_filer_count",
]
TURN_2_NAMES = [
    "narrow_dividend_total_usd_billions", "combined_dividend_total_usd_billions",
    "narrow_buyback_multiple", "combined_buyback_multiple", "misclassified_payer_count",
]
TURN_3_NAMES = [
    "median_payout_ratio_pct", "median_buyback_share_pct", "both_channel_count",
    "dividend_only_count", "buyback_only_count", "neither_count", "payout_above_income_count",
]
TURN_4_NAMES = [
    "largest_repurchaser_cik", "largest_repurchaser_buyback_usd_billions",
    "largest_repurchaser_dividend_usd_billions", "largest_repurchaser_mean_daily_volume_millions",
    "largest_dividend_payer_cik", "largest_dividend_payer_dividend_usd_billions",
    "largest_dividend_payer_buyback_usd_billions", "largest_dividend_payer_mean_daily_volume_millions",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many profitable filers the annual population holds as an integer."),
    _v(TURN_1_NAMES[1], "Store their combined repurchases in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[2], "Store how many of them report the narrow dividend tag as an integer."),
    _v(TURN_1_NAMES[3], "Store how many report the broader dividend tag as an integer."),
    _v(TURN_1_NAMES[4], "Store how many report both as an integer."),
    _v(TURN_2_NAMES[0], "Store dividends under the narrow tag alone in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[1], "Store dividends under the broader-first fallback rule in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[2], "Store repurchases divided by narrow-tag dividends, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store repurchases divided by broader-first dividends, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store how many filers pay dividends under the broader-first rule but report nothing under the narrow tag, as an integer."),
    _v(TURN_3_NAMES[0], "Store the median payout ratio in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the median repurchase share of total payout in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store how many filers both repurchase and pay dividends as an integer."),
    _v(TURN_3_NAMES[3], "Store how many pay dividends without repurchasing as an integer."),
    _v(TURN_3_NAMES[4], "Store how many repurchase without paying dividends as an integer."),
    _v(TURN_3_NAMES[5], "Store how many do neither as an integer."),
    _v(TURN_3_NAMES[6], "Store how many return more than their net income as an integer."),
    _v(TURN_4_NAMES[0], "Store the central index key of the largest repurchaser as text."),
    _v(TURN_4_NAMES[1], "Store its repurchases in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[2], "Store its dividends under the broader-first rule in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[3], "Store its mean 2024 daily stock trading volume in millions of shares, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the central index key of the largest dividend payer as text."),
    _v(TURN_4_NAMES[5], "Store its dividends in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[6], "Store its repurchases in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[7], "Store its mean 2024 daily stock trading volume in millions of shares, rounded to 4 decimals."),
]

DECIMALS = [
    0, 6, 0, 0, 0,
    6, 6, 4, 4, 0,
    4, 4, 0, 0, 0, 0, 0,
    None, 6, 6, 4, None, 6, 6, 4,
]


@lru_cache(maxsize=1)
def _annual_facts():
    facts = load_facts().copy()
    facts["amount"] = pd.to_numeric(facts.value, errors="coerce")
    starts = pd.to_datetime(facts.start_date, errors="coerce")
    ends = pd.to_datetime(facts.end_date, errors="coerce")
    facts["span"] = (ends - starts).dt.days
    return facts.loc[
        facts.form.astype(str).eq(ANNUAL_FORM)
        & facts.span.between(*WINDOW_DAYS)
        & facts.end_date.astype(str).str.startswith(YEAR)
    ]


def _latest_by_filer(concept):
    rows = _annual_facts()
    rows = rows.loc[rows.concept.astype(str).eq(concept)]
    rows = rows.sort_values(["cik", "filed_date"]).groupby("cik").tail(1)
    return rows.set_index(rows.cik.astype(str).str.lstrip("0")).amount


@lru_cache(maxsize=1)
def _population():
    frame = pd.DataFrame({
        "buyback": _latest_by_filer(BUYBACK_CONCEPT),
        "narrow": _latest_by_filer(NARROW_DIVIDEND_CONCEPT),
        "broad": _latest_by_filer(BROAD_DIVIDEND_CONCEPT),
        "income": _latest_by_filer(INCOME_CONCEPT),
    })
    frame = frame.loc[frame.income.notna() & frame.income.gt(0)].copy()
    frame["dividend"] = frame.broad.fillna(frame.narrow).fillna(0.0)
    frame["narrow_dividend"] = frame.narrow.fillna(0.0)
    frame["buyback"] = frame.buyback.fillna(0.0)
    total = frame.buyback + frame.dividend
    frame["payout_ratio"] = total / frame.income * 100
    frame["buyback_share"] = frame.buyback / total.where(total > 0) * 100
    return frame


@lru_cache(maxsize=1)
def _ticker_by_cik():
    companies = load_companies().copy()
    companies["normalized_cik"] = companies.cik.astype(str).str.lstrip("0")
    return companies.set_index("normalized_cik").ticker


@lru_cache(maxsize=1)
def _market_rows():
    market = load_expansion_table("sec_midas_security_exchange").copy()
    market["market_date"] = pd.to_datetime(market.date, errors="coerce")
    market["volume"] = pd.to_numeric(market.trade_volume_thousands, errors="coerce")
    return market.loc[
        market.security_type.astype(str).eq(MARKET_SECURITY_TYPE)
        & market.market_date.dt.year.eq(int(MARKET_YEAR))
        & market.volume.notna()
    ].copy()


def _mean_daily_volume(cik):
    normalized_cik = str(cik).lstrip("0")
    if normalized_cik not in _ticker_by_cik().index:
        raise ValueError(f"no company row for {cik}")
    ticker = str(_ticker_by_cik().loc[normalized_cik])
    rows = _market_rows().loc[_market_rows().ticker.astype(str).eq(ticker)].copy()
    if rows.empty:
        raise ValueError(f"no {MARKET_YEAR} MIDAS stock rows for {ticker}")
    return float(rows.volume.mean()) / 1000


@lru_cache(maxsize=1)
def ground_truth():
    frame = _population()
    facts = _annual_facts()
    narrow_filers = set(facts.loc[facts.concept.astype(str).eq(NARROW_DIVIDEND_CONCEPT)].cik.astype(str).str.lstrip("0"))
    broad_filers = set(facts.loc[facts.concept.astype(str).eq(BROAD_DIVIDEND_CONCEPT)].cik.astype(str).str.lstrip("0"))
    population = set(frame.index)

    buyback_total = float(frame.buyback.sum())
    narrow_total = float(frame.narrow_dividend.sum())
    combined_total = float(frame.dividend.sum())

    by_buyback = frame.sort_values(["buyback"], ascending=False)
    by_dividend = frame.sort_values(["dividend"], ascending=False)
    top_buyer, top_payer = by_buyback.index[0], by_dividend.index[0]
    buyer_volume = _mean_daily_volume(top_buyer)
    payer_volume = _mean_daily_volume(top_payer)

    return (
        int(len(frame)),
        buyback_total / 1e9,
        int(len(narrow_filers & population)),
        int(len(broad_filers & population)),
        int(len(narrow_filers & broad_filers & population)),
        narrow_total / 1e9,
        combined_total / 1e9,
        buyback_total / narrow_total,
        buyback_total / combined_total,
        int((frame.dividend.gt(0) & frame.narrow_dividend.eq(0)).sum()),
        float(frame.payout_ratio.median()),
        float(frame.buyback_share.median()),
        int((frame.dividend.gt(0) & frame.buyback.gt(0)).sum()),
        int((frame.dividend.gt(0) & frame.buyback.eq(0)).sum()),
        int((frame.buyback.gt(0) & frame.dividend.eq(0)).sum()),
        int((frame.buyback.eq(0) & frame.dividend.eq(0)).sum()),
        int(frame.payout_ratio.gt(100).sum()),
        str(top_buyer),
        float(by_buyback.buyback.iloc[0]) / 1e9,
        float(by_buyback.dividend.iloc[0]) / 1e9,
        buyer_volume,
        str(top_payer),
        float(by_dividend.dividend.iloc[0]) / 1e9,
        float(by_dividend.buyback.iloc[0]) / 1e9,
        payer_volume,
    )


NAME_OUTPUTS = ("largest_repurchaser_cik", "largest_dividend_payer_cik")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = str(normalized[name]).strip().lstrip("0")
            truth[name] = str(truth[name]).strip().lstrip("0")
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_population_and_tags": turn_validator(validate_turn_1),
    "validate_tag_scope_effect": turn_validator(validate_turn_2),
    "validate_payout_composition": turn_validator(validate_turn_3),
    "validate_named_extremes": turn_validator(validate_turn_4),
}
