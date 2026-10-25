"""Compare Form 4 transaction codes and a filing-level Rule 10b5-1 proxy.

Every Form 4 transaction carries a one-letter code, and the codes do not describe
the same act. Across the 351,580 non-derivative transaction rows on Form 4 filings
from 2019 through 2025, sale rows (S) number 116,010 and purchase rows (P) only
9,660, while Rule 16b-3 awards or other acquisitions (A) reach 76,360, code F
deliveries or withholdings for an exercise price or tax liability 69,758, and
exempt exercises or conversions (M) 55,061. Under the SEC code table, P and S
include both open-market and private transactions. The other codes cannot be
relabelled as P or S merely from the acquired-or-disposed marker.

The distinction changes the answer by an order of magnitude. Read off the
acquired-or-disposed indicator, insiders show 201,813 dispositions against 149,767
acquisitions, a ratio of 1.3475. Restricted to the P/S codes the same population
gives 116,010 sales against 9,660 purchases,
a ratio of 12.0093, and priced at the reported per-share amounts it gives
801.1562 billion USD sold against 36.8450 billion bought, a ratio of 21.7440.
Withholding alone accounts for 34.5657 percent of all dispositions, and purchases
for 6.4500 percent of all acquisitions.

The filing-level Rule 10b5-1 checkbox is unavailable for most of the window. It is empty on all
28,124 Form 4 filings of 2022, carries a value on 16,664 of the 27,366 filings of
2023, and on every one of the 28,172 filings of 2024. Classifying every S row in an
affirmative filing as a proxy group puts 68.9147 percent of 2024 sale rows but
29.1652 percent of sale value in that group. The checkbox says that a transaction
in the filing was made under an intended plan; it does not identify which row. The
flag is written both as a digit and as a word, and 3,993 sale rows inherit the word
`true` from their filing rather than the digit 1.

The filing-level proxy can support a descriptive transaction-date comparison. Of
the 252 trading days of 2024, 45.2381 percent closed below the risk-free rate.
Code P rows fall on such days 55.0562 percent of the time and carry a mean market
excess return of -0.1818 percent on the transaction date; S rows in affirmative
filings sit at 0.0184 percent and S rows in non-affirmative filings at 0.1422
percent. These transaction-row-weighted associations do not identify row-level plan
status, causal timing, discretion, or returns earned.

The `insider_discretionary_selling` pinned-reading comparison records sensitivity to using
the acquired/disposed marker, including other forms, valuing other codes, widening
the plan-field window, reading only the digit encoding, and using filing rather than
transaction dates. The query fixes the reported convention, so these are robustness
comparisons rather than hidden answer paths.

Regulatory boundary: SEC transaction code F covers payment of an exercise price or
tax liability through delivery or withholding incident to receipt, exercise, or
vesting. P and S cover open-market or private transactions. The Rule 10b5-1 checkbox
is at filing grain and indicates that a transaction was made under an intended plan;
it does not tag each row.

Boundaries: a transaction row is not an insider and not a trade of a fixed size, so
counts here measure filing activity rather than participants or volume. P/S rows
are not assumed to be discretionary, and gifts, conversions and residual codes are
left out of both sides.
The market comparison keeps only transactions dated on a trading day, which
retains 801 of the 908 purchases, and a mean daily return over a group of
transactions is not a return any portfolio earned.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_insider_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


ANNUAL_FORM = "4"
WINDOW_YEARS = (2019, 2025)
FOCUS_YEAR = 2024
SALE_CODE = "S"
PURCHASE_CODE = "P"
AWARD, WITHHOLDING, EXERCISE = "A", "F", "M"
PLAN_TRUE = ("1", "true")
MARKET_COLUMN = "mkt_rf_pct"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "transaction_rows", "sale_rows", "award_rows", "code_f_rows", "code_m_rows",
    "purchase_rows", "code_f_share_of_dispositions_pct", "purchase_share_of_acquisitions_pct",
]
TURN_2_NAMES = [
    "disposition_rows", "acquisition_rows", "indicator_ratio", "coded_sale_purchase_count_ratio",
    "sale_value_usd_billions", "purchase_value_usd_billions", "coded_sale_purchase_value_ratio",
]
TURN_3_NAMES = [
    "flagged_filings_2022", "flagged_filings_2023", "flagged_filings_2024",
    "sale_rows_in_affirmative_plan_filings_pct", "sale_value_in_affirmative_plan_filings_pct",
    "word_affirmative_sale_rows",
]
TURN_4_NAMES = [
    "trading_days", "baseline_down_day_share_pct", "purchase_down_day_share_pct",
    "purchase_mean_market_return_pct", "sale_in_affirmative_plan_filing_mean_market_return_pct",
    "sale_in_nonaffirmative_filing_mean_market_return_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many qualifying transaction rows the window carries as an integer."),
    _v(TURN_1_NAMES[1], "Store how many carry transaction code S as an integer."),
    _v(TURN_1_NAMES[2], "Store how many carry the award code as an integer."),
    _v(TURN_1_NAMES[3], "Store how many carry Rule 16b-3 transaction code F as an integer."),
    _v(TURN_1_NAMES[4], "Store how many carry Rule 16b-3 transaction code M as an integer."),
    _v(TURN_1_NAMES[5], "Store how many carry transaction code P as an integer."),
    _v(TURN_1_NAMES[6], "Store withholding rows as a percent of all disposition rows, rounded to 4 decimals."),
    _v(TURN_1_NAMES[7], "Store purchase rows as a percent of all acquisition rows, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store how many rows the indicator marks as dispositions as an integer."),
    _v(TURN_2_NAMES[1], "Store how many it marks as acquisitions as an integer."),
    _v(TURN_2_NAMES[2], "Store dispositions divided by acquisitions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store code S rows divided by code P rows, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store the reported-price value of valid code S rows in USD billions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store the reported-price value of valid code P rows in USD billions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[6], "Store sale value divided by purchase value, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store how many 2022 filings carry a value in the plan flag as an integer."),
    _v(TURN_3_NAMES[1], "Store how many 2023 filings do as an integer."),
    _v(TURN_3_NAMES[2], "Store how many 2024 filings do as an integer."),
    _v(TURN_3_NAMES[3], "Store the percent of 2024 code S rows belonging to an affirmative-plan filing, rounded to 4 decimals."),
    _v(TURN_3_NAMES[4], "Store those rows' valid reported-price value as a percent of 2024 code S value, rounded to 4 decimals."),
    _v(TURN_3_NAMES[5], "Store how many 2024 code S rows inherit the affirmative word encoding from their filing as an integer."),
    _v(TURN_4_NAMES[0], "Store how many trading days the factor table carries for the focus year as an integer."),
    _v(TURN_4_NAMES[1], "Store the percent of them with a negative market excess return, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store the percent of purchase transactions falling on such a day, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store the mean market excess return on purchase transaction dates in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the same transaction-row-weighted mean for code S rows in affirmative-plan filings, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the same mean for code S rows in non-affirmative filings, rounded to 4 decimals."),
]

DECIMALS = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4]


@lru_cache(maxsize=1)
def _filings():
    submissions = load_insider_table("submission").copy()
    submissions["year"] = pd.to_datetime(submissions.filing_date, errors="coerce").dt.year
    return submissions.loc[submissions.document_type.astype(str).eq(ANNUAL_FORM)].copy()


@lru_cache(maxsize=1)
def _transactions():
    frame = load_insider_table("nonderiv_trans").copy()
    frame["shares"] = pd.to_numeric(frame.trans_shares, errors="coerce")
    frame["price"] = pd.to_numeric(frame.trans_pricepershare, errors="coerce")
    merged = frame.merge(
        _filings()[["accession_number", "year", "aff10b5one"]], on="accession_number", how="inner",
    )
    return merged.loc[merged.year.between(*WINDOW_YEARS)].copy()


@lru_cache(maxsize=1)
def _market_days():
    factors = load_ken_french_table("daily_factors").copy()
    factors["day"] = pd.to_datetime(factors.date, errors="coerce")
    factors["market"] = pd.to_numeric(factors[MARKET_COLUMN], errors="coerce")
    return factors.loc[factors.day.dt.year.eq(FOCUS_YEAR), ["day", "market"]].dropna()


@lru_cache(maxsize=1)
def ground_truth():
    rows = _transactions()
    code = rows.trans_code.astype(str)
    indicator = rows.trans_acquired_disp_cd.astype(str)
    dispositions = int(indicator.eq("D").sum())
    acquisitions = int(indicator.eq("A").sum())
    sales, purchases = int(code.eq(SALE_CODE).sum()), int(code.eq(PURCHASE_CODE).sum())
    withheld = int(code.eq(WITHHOLDING).sum())
    sale_value = float(rows.loc[code.eq(SALE_CODE)].eval("shares * price").sum())
    purchase_value = float(rows.loc[code.eq(PURCHASE_CODE)].eval("shares * price").sum())

    filings = _filings()
    flagged = {
        year: int(filings.loc[filings.year.eq(year), "aff10b5one"].notna().sum())
        for year in (2022, 2023, FOCUS_YEAR)
    }

    focus = rows.loc[rows.year.eq(FOCUS_YEAR)]
    focus_sales = focus.loc[focus.trans_code.astype(str).eq(SALE_CODE)].copy()
    flag = focus_sales.aff10b5one.astype(str).str.strip().str.lower()
    planned = flag.isin(PLAN_TRUE)
    planned_value = float(focus_sales.loc[planned].eval("shares * price").sum())
    total_value = float(focus_sales.eval("shares * price").sum())

    days = _market_days()
    dated = focus.assign(day=pd.to_datetime(focus.trans_date, errors="coerce")).merge(days, on="day", how="inner")
    dated_flag = dated.aff10b5one.astype(str).str.strip().str.lower().isin(PLAN_TRUE)
    dated_code = dated.trans_code.astype(str)
    bought = dated.loc[dated_code.eq(PURCHASE_CODE)]
    plan_sales = dated.loc[dated_code.eq(SALE_CODE) & dated_flag]
    open_sales = dated.loc[dated_code.eq(SALE_CODE) & ~dated_flag]

    return (
        int(len(rows)), sales, int(code.eq(AWARD).sum()), withheld, int(code.eq(EXERCISE).sum()), purchases,
        withheld / dispositions * 100, purchases / acquisitions * 100,
        dispositions, acquisitions, dispositions / acquisitions, sales / purchases,
        sale_value / 1e9, purchase_value / 1e9, sale_value / purchase_value,
        flagged[2022], flagged[2023], flagged[FOCUS_YEAR],
        float(planned.mean()) * 100, planned_value / total_value * 100, int(flag.eq("true").sum()),
        int(len(days)), float(days.market.lt(0).mean()) * 100,
        float(bought.market.lt(0).mean()) * 100, float(bought.market.mean()),
        float(plan_sales.market.mean()), float(open_sales.market.mean()),
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
    "validate_code_composition": turn_validator(validate_turn_1),
    "validate_measure_divergence": turn_validator(validate_turn_2),
    "validate_plan_flag_availability": turn_validator(validate_turn_3),
    "validate_market_timing": turn_validator(validate_turn_4),
}
