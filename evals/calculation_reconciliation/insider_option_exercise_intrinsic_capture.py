"""Reconcile Form 4 derivative M rows with co-reported same-day sale prices.

SEC Form 4 code M means exercise or conversion of a derivative security exempt
under Rule 16b-3; it is not limited to employee stock options.  Code A means a
grant, award or other acquisition under Rule 16b-3(d).  A code and conversion
price do not supply grant-date fair value or identify all contractual economics.

The case uses an explicitly limited matching heuristic.  Each 2024 M row with a
positive reported conversion/exercise price and underlying-share count is joined
to the median nonderivative S price in the same accession and transaction date.
No transaction identifier links the derivative row to a particular sale, and
multiple M rows can share the same sale-price summary.  The resulting sale-price
minus conversion-price times underlying shares is therefore a signed co-reported
price-gap proxy.  It is not realized intrinsic value, sale proceeds, compensation
income, or proof that the converted shares were sold.

Expiration dates describe reported remaining term for a mixed derivative
population; they do not prove voluntary early exercise or identify forfeited time
value.  The final turn joins unique M transaction dates to Fama-French daily
market-excess returns and compares event-date and all-2024 medians.  This is
descriptive timing context only, not evidence that the market caused an M event
or that an issuer-specific security moved with the broad factor.

The `insider_option_exercise` pinned-reading comparison records sensitivity to pairing an
exercise with a sale on any date in the same filing rather than the same date, to
using the mean rather than the median co-reported sale price, to admitting
non-positive strikes, to ranking issuers by M row count rather than by the price-gap
proxy, to defining the event population as M rows rather than unique M dates, and to
admitting Form 5 alongside Form 4. The filing-type scope is the one worth naming: ten
2024 code-M rows arrive on Form 5, the annual late-report obligation, and admitting
them moves the M count from 8,503 to 8,513, the priced count from 4,158 to 4,168, the
count carrying an expiration date from 3,817 to 3,827, and the median reported price
from 41.4000 to 41.2500 USD. Inside the usable-term window the dated count moves from
3,806 to 3,807, which flips the median remaining term from 604.5000 to 604.0000 days
because the count changes parity. Form 5 is a
different filing obligation rather than a variant of Form 4, so the query names the
Form 4 population for both code counts. Every one of these choices is pinned by the
query, making them robustness comparisons rather than hidden answer paths.

An issuer is its CIK, the key the query breaks ranking ties on, not a CIK and
name pair. One matched CIK, 0001385157, files as "TE Connectivity Ltd." on ten
rows and as "TE Connectivity plc" on three after its change of domicile, so
counting name pairs reports 167 issuers where there are 166. The two leaders
each file under a single name, so their reported names are unaffected.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and issuer names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_insider_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


YEAR = "2024"
FORM_TYPE = "4"
M_CODE = "M"
SALE_CODE = "S"
MAX_DAYS_TO_EXPIRY = 4000
LONG_DATED_DAYS = 730
NEAR_EXPIRY_DAYS = 90
MARKET_COLUMN = "mkt_rf_pct"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "m_transaction_count", "m_with_conversion_price_count", "m_with_expiry_count",
    "median_reported_conversion_price_usd", "a_transaction_count",
]
TURN_2_NAMES = [
    "matched_m_transaction_count", "matched_median_conversion_price_usd",
    "matched_median_co_reported_sale_price_usd", "median_sale_to_conversion_price_multiple",
    "sale_below_conversion_price_count",
]
TURN_3_NAMES = [
    "total_signed_price_gap_proxy_usd_billions", "median_price_gap_proxy_per_m_row_usd",
    "leading_issuer_name", "leading_issuer_price_gap_proxy_usd_billions", "leading_issuer_m_row_count",
    "runner_up_issuer_name", "runner_up_issuer_m_row_count", "issuer_count",
]
TURN_4_NAMES = [
    "expiry_dated_m_transaction_count", "median_days_to_expiry",
    "long_dated_m_transaction_count", "near_expiry_m_transaction_count",
    "unique_m_transaction_date_count", "factor_matched_m_date_count",
    "median_event_date_market_excess_return_pct", "median_2024_market_excess_return_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the number of derivative-table M transactions in the year as an integer."),
    _v(TURN_1_NAMES[1], "Store how many M rows report a conversion or exercise price as an integer."),
    _v(TURN_1_NAMES[2], "Store how many M rows report an expiration date as an integer."),
    _v(TURN_1_NAMES[3], "Store the median nonmissing reported conversion or exercise price in USD, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store the number of derivative-table A transactions in the year as an integer."),
    _v(TURN_2_NAMES[0], "Store how many M rows match a same-accession same-date sale-price summary as an integer."),
    _v(TURN_2_NAMES[1], "Store the matched rows' median conversion or exercise price in USD, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the median co-reported sale price in USD, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the median co-reported sale price divided by conversion price, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store how many matched rows have sale price below conversion price as an integer."),
    _v(TURN_3_NAMES[0], "Store the total signed co-reported price-gap proxy in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[1], "Store the median signed price-gap proxy per matched M row in USD, rounded to 2 decimals."),
    _v(TURN_3_NAMES[2], "Store the leading issuer name verbatim from its SEC submission as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_3_NAMES[3], "Store that issuer's proxy total in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[4], "Store how many matched M rows it accounts for as an integer."),
    _v(TURN_3_NAMES[5], "Store the runner-up issuer name verbatim from its SEC submission as a string, preserving capitalization, punctuation and spacing."),
    _v(TURN_3_NAMES[6], "Store how many matched M rows that issuer accounts for as an integer."),
    _v(TURN_3_NAMES[7], "Store how many issuers have at least one matched M row as an integer."),
    _v(TURN_4_NAMES[0], "Store how many M rows carry a usable expiration date as an integer."),
    _v(TURN_4_NAMES[1], "Store the median days from M transaction date to expiration, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store how many dated M rows have more than 730 days remaining as an integer."),
    _v(TURN_4_NAMES[3], "Store how many dated M rows have at most 90 days remaining as an integer."),
    _v(TURN_4_NAMES[4], "Store the number of unique 2024 M transaction dates as an integer."),
    _v(TURN_4_NAMES[5], "Store how many unique M dates match a daily factor observation as an integer."),
    _v(TURN_4_NAMES[6], "Store the median daily market excess return on matched M dates in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[7], "Store the median daily market excess return across all 2024 factor dates in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 0, 4, 0,
    0, 4, 4, 4, 0,
    6, 2, None, 6, 0, None, 0, 0,
    0, 4, 0, 0, 0, 0, 4, 4,
]


@lru_cache(maxsize=1)
def _derivatives():
    frame = load_insider_table("deriv_trans").copy()
    for column in ("conv_exercise_price", "undlyng_sec_shares"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Form 5 is the annual late-report obligation, a different filing from Form 4,
    # and the table carries both. Ten 2024 code-M rows arrive on Form 5, so the
    # form restriction the question states has to be applied here.
    return frame.loc[
        frame.trans_date.astype(str).str.startswith(YEAR)
        & frame.trans_form_type.astype(str).eq(FORM_TYPE)
    ]


@lru_cache(maxsize=1)
def _paired():
    exercises = _derivatives()
    exercises = exercises.loc[exercises.trans_code.astype(str).eq(M_CODE)]
    submissions = load_insider_table("submission")[["accession_number", "issuercik", "issuername"]]
    exercises = exercises.merge(submissions, on="accession_number", how="left")

    sales = load_insider_table("nonderiv_trans")
    sales = sales.loc[
        sales.trans_code.astype(str).eq(SALE_CODE) & sales.trans_date.astype(str).str.startswith(YEAR)
    ][["accession_number", "trans_date", "trans_pricepershare"]].copy()
    sales["price"] = pd.to_numeric(sales.trans_pricepershare, errors="coerce")
    same_day = sales.groupby(["accession_number", "trans_date"], as_index=False).price.median()

    paired = exercises.merge(same_day, on=["accession_number", "trans_date"], how="inner")
    paired = paired.dropna(subset=["conv_exercise_price", "price", "undlyng_sec_shares"])
    paired = paired.loc[paired.conv_exercise_price.gt(0)].copy()
    paired["price_gap_proxy"] = (paired.price - paired.conv_exercise_price) * paired.undlyng_sec_shares
    paired["price_multiple"] = paired.price / paired.conv_exercise_price
    return paired


@lru_cache(maxsize=1)
def ground_truth():
    derivatives = _derivatives()
    exercises = derivatives.loc[derivatives.trans_code.astype(str).eq(M_CODE)]
    paired = _paired()

    # The issuer is its CIK, the tie-break key; a name is what one filing calls it.
    by_issuer = paired.groupby("issuercik", as_index=False).agg(
        issuername=("issuername", "first"),
        price_gap_proxy=("price_gap_proxy", "sum"), m_rows=("price_gap_proxy", "size"),
    ).sort_values(["price_gap_proxy", "issuercik"], ascending=[False, True]).reset_index(drop=True)
    leader, runner_up = by_issuer.iloc[0], by_issuer.iloc[1]

    dated = exercises.dropna(subset=["expiration_date"]).copy()
    dated["days"] = (
        pd.to_datetime(dated.expiration_date, errors="coerce")
        - pd.to_datetime(dated.trans_date, errors="coerce")
    ).dt.days
    dated = dated.loc[dated.days.notna() & dated.days.between(0, MAX_DAYS_TO_EXPIRY)]

    factors = load_ken_french_table("daily_factors").copy()
    factors["day"] = factors.date.astype(str)
    factors[MARKET_COLUMN] = pd.to_numeric(factors[MARKET_COLUMN], errors="coerce")
    year_rows = factors.loc[factors.day.str.startswith(YEAR)].copy()
    m_dates = set(exercises.trans_date.astype(str))
    event_rows = year_rows.loc[year_rows.day.isin(m_dates)].copy()

    return (
        int(len(exercises)),
        int(exercises.conv_exercise_price.notna().sum()),
        int(exercises.expiration_date.notna().sum()),
        float(exercises.conv_exercise_price.median()),
        int(derivatives.trans_code.astype(str).eq("A").sum()),
        int(len(paired)),
        float(paired.conv_exercise_price.median()),
        float(paired.price.median()),
        float(paired.price_multiple.median()),
        int(paired.price.lt(paired.conv_exercise_price).sum()),
        float(paired.price_gap_proxy.sum()) / 1e9,
        float(paired.price_gap_proxy.median()),
        str(leader.issuername),
        float(leader.price_gap_proxy) / 1e9,
        int(leader.m_rows),
        str(runner_up.issuername),
        int(runner_up.m_rows),
        int(len(by_issuer)),
        int(len(dated)),
        float(dated.days.median()),
        int(dated.days.gt(LONG_DATED_DAYS).sum()),
        int(dated.days.le(NEAR_EXPIRY_DAYS).sum()),
        int(len(m_dates)),
        int(len(event_rows)),
        float(event_rows[MARKET_COLUMN].median()),
        float(year_rows[MARKET_COLUMN].median()),
    )


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
    "validate_m_population": turn_validator(validate_turn_1),
    "validate_co_reported_sale_match": turn_validator(validate_turn_2),
    "validate_price_gap_proxy_concentration": turn_validator(validate_turn_3),
    "validate_remaining_term_and_event_dates": turn_validator(validate_turn_4),
}
