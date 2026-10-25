"""The Treasury's daily cash identity, its residual, and what debt issuance explains.

The Daily Treasury Statement reports the operating cash account under three
successive naming conventions in this window: a Federal Reserve Account line
through 2021-09-30, a single Treasury General Account line through 2022-04-15,
and from 2022-04-18 the four-line structure of opening balance, deposits,
withdrawals and closing balance. A screen written against any one vocabulary
silently drops the other eras, so the population has to be established before the
arithmetic starts.

Across the 928 dates carrying all four modern lines, opening plus deposits minus
withdrawals equals the closing balance exactly on 620 of them. On the remaining
308 the residual is one million dollars, positive on 156 dates and negative on
152, and it is never larger. That is the signature of four lines each rounded to
the nearest million, not of an account that fails to balance: the correct reading
is that the statement balances to its published precision, and a response that
reports a broken identity has over-read a rounding artifact.

The largest single-day movement in the window is 262,780 million on 2022-04-18,
from an opening balance of 578,473 to a close of 841,253. Marketable securities
issued that day net to 125,408 million, less than half the movement, and
nonmarketable issues net to 12,039 million, most of that traffic being government
account series that recycle inside the government rather than raising cash from
the public. Debt operations do not account for the day; the deposits line carries
receipts the debt tables never touch.

Rejected alternative conventions, each measured against the pinned reading:

- Treating the withdrawals line as a signed negative number and adding it, rather
  than as the positive magnitude the statement prints, turns the residual into
  twice the withdrawals and leaves zero exact dates out of 928.
- Ranking days by the closing balance itself rather than by the movement from
  opening to closing selects 2025-10-30 at a 1,000,632 million balance, which is a
  level rather than a flow.
- Counting the swing on the deposits-minus-withdrawals line instead of on the two
  balances gives 262,779 million for the same day, one million below the balance
  movement, because that day is one of the 308 whose lines round apart.
"""

from __future__ import annotations

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OPENING = "Treasury General Account (TGA) Opening Balance"
DEPOSITS = "Total TGA Deposits (Table II)"
WITHDRAWALS = "Total TGA Withdrawals (Table II) (-)"
CLOSING = "Treasury General Account (TGA) Closing Balance"
FOUR_LINE = (OPENING, DEPOSITS, WITHDRAWALS, CLOSING)

TURN_1_NAMES = [
    "account_type_label_count", "federal_reserve_account_last_date",
    "single_line_tga_last_date", "four_line_first_date", "four_line_date_count",
]
TURN_2_NAMES = [
    "identity_exact_date_count", "identity_maximum_absolute_residual",
    "identity_positive_residual_date_count",
]
TURN_3_NAMES = [
    "largest_balance_movement_date", "largest_balance_movement_amount",
    "largest_movement_opening_balance", "largest_movement_closing_balance",
]
TURN_4_NAMES = [
    "movement_day_marketable_net_issues", "movement_day_nonmarketable_net_issues",
    "marketable_net_issues_cover_movement",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v(TURN_1_NAMES[0], "Store the number of distinct account-type labels the operating cash statement uses as an integer."),
    _v(TURN_1_NAMES[1], "Store the last date carrying the Federal Reserve Account label as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[2], "Store the last date carrying the single Treasury General Account label as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[3], "Store the first date carrying the opening-balance label as an ISO YYYY-MM-DD string."),
    _v(TURN_1_NAMES[4], "Store the number of dates carrying all four of the modern labels as an integer."),
    _v(TURN_2_NAMES[0], "Store how many of those dates satisfy the identity exactly as an integer."),
    _v(TURN_2_NAMES[1], "Store the largest absolute residual across those dates in USD millions as an integer."),
    _v(TURN_2_NAMES[2], "Store how many of those dates carry a positive residual as an integer."),
    _v(TURN_3_NAMES[0], "Store the selected date as an ISO YYYY-MM-DD string."),
    _v(TURN_3_NAMES[1], "Store the signed closing-minus-opening movement on that date in USD millions as an integer."),
    _v(TURN_3_NAMES[2], "Store that date's opening balance in USD millions as an integer."),
    _v(TURN_3_NAMES[3], "Store that date's closing balance in USD millions as an integer."),
    _v(TURN_4_NAMES[0], "Store marketable issues minus marketable redemptions on that date in USD millions as an integer."),
    _v(TURN_4_NAMES[1], "Store nonmarketable issues minus nonmarketable redemptions on that date in USD millions as an integer."),
    _v(TURN_4_NAMES[2], "Store whether marketable net issues reach the balance movement as a boolean."),
]

DECIMALS = [0, None, None, None, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, None]


@lru_cache(maxsize=1)
def _cash():
    rows = load_expansion_table("treasury_dts_operating_cash")
    return rows.assign(
        date=rows.record_date.astype(str), account=rows.account_type.astype(str)
    )


@lru_cache(maxsize=1)
def _four_line_panel():
    """One row per date carrying all four modern lines, in USD millions."""
    rows = _cash()
    wide = rows.loc[rows.account.isin(FOUR_LINE)].pivot_table(
        index="date", columns="account", values="today_amount_million_usd", aggfunc="first"
    )
    return wide.dropna()


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    rows = _cash()
    panel = _four_line_panel()
    residual = (panel[OPENING] + panel[DEPOSITS] - panel[WITHDRAWALS] - panel[CLOSING]).astype(int)
    movement = (panel[CLOSING] - panel[OPENING]).astype(int)
    largest = movement.abs().max()
    tied = sorted(movement.loc[movement.abs().eq(largest)].index)
    if len(tied) != 1:
        raise ValueError(f"expected one largest movement date, found {len(tied)}")
    day = tied[0]

    debt = load_expansion_table("treasury_dts_public_debt")
    debt = debt.loc[debt.record_date.astype(str).eq(day)]

    def net(market: str) -> int:
        rows_for = debt.loc[debt.security_market.astype(str).eq(market)]
        issues = rows_for.loc[rows_for.transaction_type.astype(str).eq("Issues")]
        redemptions = rows_for.loc[rows_for.transaction_type.astype(str).eq("Redemptions")]
        return int(issues.transaction_today_amount_million_usd.sum()
                   - redemptions.transaction_today_amount_million_usd.sum())

    marketable, nonmarketable = net("Marketable"), net("Nonmarketable")
    return (
        int(rows.account.nunique()),
        str(rows.loc[rows.account.eq("Federal Reserve Account"), "date"].max()),
        str(rows.loc[rows.account.eq("Treasury General Account (TGA)"), "date"].max()),
        str(rows.loc[rows.account.eq(OPENING), "date"].min()),
        int(len(panel)),
        int((residual == 0).sum()), int(residual.abs().max()), int((residual > 0).sum()),
        day, int(movement.loc[day]), int(panel.loc[day, OPENING]), int(panel.loc[day, CLOSING]),
        marketable, nonmarketable, bool(marketable >= int(movement.loc[day])),
    )


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    places = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names]
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_naming_eras": turn_validator(validate_turn_1),
    "validate_identity_residual": turn_validator(validate_turn_2),
    "validate_largest_movement": turn_validator(validate_turn_3),
    "validate_debt_contribution": turn_validator(validate_turn_4),
}
