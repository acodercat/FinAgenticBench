"""Read cross-border credit to emerging markets: what splits, and what cannot be compared.

The global liquidity indicators report credit to non-bank borrowers in emerging
economies in three currencies of denomination, each split into international
debt securities and bank loans. The split is exact: at 2025-Q4 dollar credit of
4,345,181.8 million comprises 2,433,764.1 of securities and 1,911,417.7 of
loans, and the same identity closes to zero for the euro and yen series.

The comparison the table invites is the one it cannot support. Each series is
denominated in its own currency, so the yen figure of 10,178,921.8 million is
millions of yen, not of dollars: dividing it by the dollar figure gives 2.3426,
a number with no economic content, and the table carries no exchange rate with
which to convert. Any statement that yen credit to these borrowers exceeds
dollar credit is a units error rather than a finding.

Within each currency the composition is comparable and it differs sharply.
Securities carry 56.0106 percent of dollar credit against 38.5757 percent of
euro credit and 28.5272 percent of yen credit, so the dollar series has the
highest securities share of the three. The dollar path shows
when: the securities share ran 32.2782 percent at 2010-Q4 and 34.2435 at
2015-Q4, then 48.9648 at 2020-Q4 and 56.0106 at 2025-Q4, while the stock of
dollar credit rose from 1,991,574.7 million to 4,345,181.8. Over the last of
those spans the dollar's broad nominal effective exchange rate rose from 94.9100
to 103.0400, 8.5660 percent. The case reports the index as separate currency
context and does not use two endpoint readings to explain the funding mix.

Measured rejected alternatives include ranking
unconverted published levels, using the wrong lender population, dividing by an
across-currency total, changing the requested periods, and substituting the real
effective exchange-rate series. The query pins each choice, so these are baseline
instruction-following probes rather than a live hidden-method trap. This is a hard
baseline case.

Knowledge used by the case: a currency of denomination is also the measurement
unit of the published level, while a within-currency component share is invariant
to that unit. The source identifies the instrument mix but not the ultimate holder
of each debt security, its tradability, or the cause of a change in composition.

Boundaries: the loan series is reported by banks while the securities series is
reported for all lending sectors, so the two components come from different
reporting populations even though they add to the published total. The effective
exchange rate index begins in 2019 in this artifact, which is why the exchange
rate context covers only the most recent span.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; period labels match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


LATEST_PERIOD = "2025-Q4"
PATH_PERIODS = ("2010-Q4", "2015-Q4", "2020-Q4", "2025-Q4")
CURRENCIES = ("USD", "EUR", "JPY")
CREDIT_INSTRUMENT = "B"
SECURITIES_INSTRUMENT = "D"
LOAN_INSTRUMENT = "G"
ALL_LENDERS = "A"
BANK_LENDERS = "B"
EER_MONTHS = ("2020-12", "2025-12")
EER_AREA = "US"
EER_TYPE = "N"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "period_count", "earliest_period", "dollar_credit_millions", "dollar_securities_millions",
    "dollar_loans_millions", "dollar_identity_residual_millions", "euro_identity_residual_millions",
    "yen_identity_residual_millions",
]
TURN_2_NAMES = [
    "euro_credit_millions", "yen_credit_millions", "yen_to_dollar_published_ratio",
    "euro_to_dollar_published_ratio", "distinct_unit_count",
]
TURN_3_NAMES = [
    "dollar_securities_share_pct", "euro_securities_share_pct", "yen_securities_share_pct",
    "highest_securities_share_currency",
]
TURN_4_NAMES = [
    "dollar_share_2010_pct", "dollar_share_2015_pct", "dollar_share_2020_pct",
    "dollar_credit_2010_millions", "dollar_credit_growth_pct",
    "dollar_index_2020", "dollar_index_2025", "dollar_index_change_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store how many distinct periods the table reports as an integer."),
    _v(TURN_1_NAMES[1], "Store the earliest period label as a YYYY-Qn string."),
    _v(TURN_1_NAMES[2], "Store dollar-denominated credit at the latest period in millions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[3], "Store its debt-securities component in millions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[4], "Store its bank-loan component in millions, rounded to 4 decimals."),
    _v(TURN_1_NAMES[5], "Store securities plus loans minus credit for the dollar series, rounded to 4 decimals."),
    _v(TURN_1_NAMES[6], "Store the same residual for the euro series, rounded to 4 decimals."),
    _v(TURN_1_NAMES[7], "Store the same residual for the yen series, rounded to 4 decimals."),
    _v(TURN_2_NAMES[0], "Store euro-denominated credit at the latest period in millions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[1], "Store yen-denominated credit at the latest period in millions, rounded to 4 decimals."),
    _v(TURN_2_NAMES[2], "Store the yen figure divided by the dollar figure as published, rounded to 4 decimals."),
    _v(TURN_2_NAMES[3], "Store the euro figure divided by the dollar figure as published, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store how many distinct units of measure the three series carry as an integer."),
    _v(TURN_3_NAMES[0], "Store the securities share of dollar credit in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[1], "Store the securities share of euro credit in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[2], "Store the securities share of yen credit in percent, rounded to 4 decimals."),
    _v(TURN_3_NAMES[3],
       "Store the currency with the highest securities share as its three-letter ISO code: "
       "one of USD, EUR, JPY. The table states each currency as a code and a name in one "
       "field, and only the code is wanted."),
    _v(TURN_4_NAMES[0], "Store the dollar securities share at the first named period in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[1], "Store it at the second named period, rounded to 4 decimals."),
    _v(TURN_4_NAMES[2], "Store it at the third named period, rounded to 4 decimals."),
    _v(TURN_4_NAMES[3], "Store dollar credit at the first named period in millions, rounded to 4 decimals."),
    _v(TURN_4_NAMES[4], "Store the growth in dollar credit from the first to the last named period in percent, rounded to 4 decimals."),
    _v(TURN_4_NAMES[5], "Store the dollar's nominal broad effective exchange rate at the earlier named month, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store it at the later named month, rounded to 4 decimals."),
    _v(TURN_4_NAMES[7], "Store the change between those two readings in percent, rounded to 4 decimals."),
]

DECIMALS = [
    0, None, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 0,
    4, 4, 4, None,
    4, 4, 4, 4, 4, 4, 4, 4,
]


@lru_cache(maxsize=1)
def _liquidity():
    frame = load_expansion_table("bis_global_liquidity").copy()
    frame.columns = [column.split(":")[0] for column in frame.columns]
    frame["value"] = pd.to_numeric(frame.OBS_VALUE, errors="coerce")
    frame["currency"] = frame.CURR_DENOM.astype(str).str.slice(0, 3)
    frame["instrument"] = frame.L_INSTR.astype(str).str.slice(0, 1)
    frame["lender"] = frame.LENDERS_SECTOR.astype(str).str.slice(0, 1)
    frame["period"] = frame.TIME_PERIOD.astype(str)
    frame["unit"] = frame.UNIT_MEASURE.astype(str).str.slice(0, 3)
    return frame


def _cell(period, currency, lender, instrument):
    frame = _liquidity()
    rows = frame.loc[
        frame.period.eq(period) & frame.currency.eq(currency)
        & frame.lender.eq(lender) & frame.instrument.eq(instrument), "value",
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {period} {currency} {lender}{instrument}, found {len(rows)}")
    return float(rows.iloc[0])


def _credit(period, currency):
    return _cell(period, currency, ALL_LENDERS, CREDIT_INSTRUMENT)


def _securities(period, currency):
    return _cell(period, currency, ALL_LENDERS, SECURITIES_INSTRUMENT)


def _loans(period, currency):
    return _cell(period, currency, BANK_LENDERS, LOAN_INSTRUMENT)


@lru_cache(maxsize=1)
def ground_truth():
    frame = _liquidity()
    credit = {code: _credit(LATEST_PERIOD, code) for code in CURRENCIES}
    securities = {code: _securities(LATEST_PERIOD, code) for code in CURRENCIES}
    loans = {code: _loans(LATEST_PERIOD, code) for code in CURRENCIES}
    shares = {code: securities[code] / credit[code] * 100 for code in CURRENCIES}
    highest = max(sorted(CURRENCIES), key=lambda code: shares[code])
    lowest = min(sorted(CURRENCIES), key=lambda code: shares[code])

    rates = load_expansion_table("bis_effective_exchange_rates").copy()
    rates.columns = [column.split(":")[0] for column in rates.columns]
    rates["value"] = pd.to_numeric(rates.OBS_VALUE, errors="coerce")
    rates = rates.loc[
        rates.REF_AREA.astype(str).str.startswith(EER_AREA)
        & rates.EER_TYPE.astype(str).str.startswith(EER_TYPE)
    ]

    def index_at(month):
        cell = rates.loc[rates.TIME_PERIOD.astype(str).eq(month), "value"]
        if cell.empty:
            raise ValueError(f"no effective exchange rate for {month}")
        return float(cell.iloc[0])

    earlier, later = (index_at(month) for month in EER_MONTHS)

    return (
        int(frame.period.nunique()),
        sorted(frame.period.unique())[0],
        credit["USD"],
        securities["USD"],
        loans["USD"],
        securities["USD"] + loans["USD"] - credit["USD"],
        securities["EUR"] + loans["EUR"] - credit["EUR"],
        securities["JPY"] + loans["JPY"] - credit["JPY"],
        credit["EUR"],
        credit["JPY"],
        credit["JPY"] / credit["USD"],
        credit["EUR"] / credit["USD"],
        int(frame.loc[frame.period.eq(LATEST_PERIOD)].unit.nunique()),
        shares["USD"],
        shares["EUR"],
        shares["JPY"],
        highest,
        _securities(PATH_PERIODS[0], "USD") / _credit(PATH_PERIODS[0], "USD") * 100,
        _securities(PATH_PERIODS[1], "USD") / _credit(PATH_PERIODS[1], "USD") * 100,
        _securities(PATH_PERIODS[2], "USD") / _credit(PATH_PERIODS[2], "USD") * 100,
        _credit(PATH_PERIODS[0], "USD"),
        (_credit(PATH_PERIODS[3], "USD") / _credit(PATH_PERIODS[0], "USD") - 1) * 100,
        earlier,
        later,
        (later / earlier - 1) * 100,
    )


NAME_OUTPUTS = ("earliest_period", "highest_securities_share_currency")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = str(normalized[name]).strip().upper()
            truth[name] = str(truth[name]).strip().upper()
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
    "validate_instrument_identity": turn_validator(validate_turn_1),
    "validate_denomination_boundary": turn_validator(validate_turn_2),
    "validate_composition_by_currency": turn_validator(validate_turn_3),
    "validate_shift_and_dollar_context": turn_validator(validate_turn_4),
}
