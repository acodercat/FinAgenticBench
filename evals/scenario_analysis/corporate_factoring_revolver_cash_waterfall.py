"""Hypothetical issuer receivable sale and six-month revolver waterfall.

SEC notes fix opening balances and already-sold factoring principal. Monthly
collections, spending, sale prices and limits are explicit scenario terms, not
reported Mondelez contracts or forecasts. Existing sold receivables are
serviced and remitted, never available as owned operating receipts.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "corporate_factoring_revolver_cash_waterfall"
CIK = "1103982"
ACCESSION = "0001103982-25-000030"
PERIOD = "2024-12-31"
SPENDING_MN = np.array([2300., 1800., 500., 200., 100., 100.])
COLLECTION_SHARES = np.array([.20, .30, .30, .20, 0., 0.])
LOAN_LIMIT_MN = 600.
FACTOR_OUTSTANDING_LIMIT_MN = 665.
LOT_MN = 200.
SPREAD = .02


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("issuer_cik", "Store the exact ten-K issuer CIK as a string token, without added leading zeroes."),
    _v("original_accession", "Store the exact original ten-K accession as a string token."),
    _v("owned_receivables_mn", "Store opening on-book owned receivables in USD millions, rounded to 2 decimals."),
    _v("usable_cash_mn", "Store source-disclosed unrestricted opening cash from its own unrounded filing fact, USD millions, rounded to 2 decimals."),
    _v("cash_including_restricted_mn", "Store source-disclosed cash including restricted cash from its own unrounded filing fact, USD millions, rounded to 2 decimals."),
    _v("current_restricted_cash_mn", "Store source-disclosed current restricted cash from its own unrounded filing fact, USD millions, rounded to 2 decimals."),
    _v("already_sold_face_mn", "Store the disclosed previously sold outstanding factoring principal separately from owned receivables, USD millions, rounded to 2 decimals."),
    _v("unfunded_peak_debt_mn", "Store peak revolver demand without factoring before enforcing the USD 600m credit cap, USD millions, rounded to 4 decimals."),
    _v("unfunded_interest_mn", "Store six-month revolver interest without factoring and before enforcing the cap, USD millions, rounded to 4 decimals."),
    _v("first_no_factor_breach_month", "Store the first month in which the no-factor path would need more than the credit cap, as a YYYY-MM string."),
    _v("day_zero_feb_lot", "Store exactly SELL or KEEP for the USD 200m February eligible lot."),
    _v("day_zero_mar_lot", "Store exactly SELL or KEEP for the USD 200m March eligible lot."),
    _v("day_zero_april_face_mn", "Store day-zero sale face drawn from the April owned batch in USD millions, rounded to 4 decimals."),
    _v("day_zero_total_financing_cost_mn", "Store full six-month factoring discounts, activation fee and revolver interest for the selected day-zero plan, USD millions, rounded to 4 decimals."),
    _v("end_feb_revolver_mn", "Store executed end-February debt under the selected day-zero plan, USD millions, rounded to 4 decimals."),
    _v("stressed_march_credit_demand_mn", "Store March credit demand under the delay if the original credit cap were diagnostically relaxed; it is not an authorized draw, USD millions, rounded to 4 decimals."),
    _v("stressed_first_breach_month", "Store the first original-limit breach after the March collection delay as a YYYY-MM string."),
    _v("amend_required_limit_mn", "Store the total credit limit required by the March amendment after paying its USD 15m cash fee, USD millions, rounded to 4 decimals."),
    _v("amend_total_financing_cost_mn", "Store six-month factoring discounts, fees and revolver interest for the amendment branch, USD millions, rounded to 4 decimals."),
    _v("additional_april_factor_face_mn", "Store extra March sale face from still-owned April receivables needed for the factor cure, USD millions, rounded to 4 decimals."),
    _v("factor_total_financing_cost_mn", "Store six-month factoring discounts, fees and revolver interest for the factor-cure branch, USD millions, rounded to 4 decimals."),
    _v("factor_remaining_headroom_mn", "Store factoring-program outstanding-face headroom immediately after the March additional sale and before March servicing settlements, USD millions, rounded to 4 decimals."),
    _v("chosen_cure", "Store exactly FACTOR or AMEND for the lower-cost feasible March cure."),
    _v("chosen_final_cash_mn", "Store June-end unrestricted cash after the chosen cure and debt repayment, USD millions, rounded to 4 decimals."),
]
TURN_NAMES = [
    ["issuer_cik", "original_accession", "owned_receivables_mn", "usable_cash_mn", "cash_including_restricted_mn", "current_restricted_cash_mn", "already_sold_face_mn"],
    ["unfunded_peak_debt_mn", "unfunded_interest_mn", "first_no_factor_breach_month"],
    ["day_zero_feb_lot", "day_zero_mar_lot", "day_zero_april_face_mn", "day_zero_total_financing_cost_mn", "end_feb_revolver_mn"],
    ["stressed_march_credit_demand_mn", "stressed_first_breach_month"],
    ["amend_required_limit_mn", "amend_total_financing_cost_mn", "additional_april_factor_face_mn", "factor_total_financing_cost_mn", "factor_remaining_headroom_mn", "chosen_cure", "chosen_final_cash_mn"],
]
DECIMALS = [None, None, 2, 2, 2, 2, 2, 4, 4, None, None, None, 4, 4, 4, 4, None, 4, 4, 4, 4, 4, None, 4]


@lru_cache(maxsize=1)
def _state():
    notes = load_expansion_table("sec_notes")
    subset = notes.loc[
        notes.cik.astype(str).eq(CIK)
        & notes.adsh.eq(ACCESSION)
        & notes.ddate.eq(PERIOD)
        & notes.qtrs.eq(0)
        & notes.uom.eq("USD")
        & notes.dimh.eq("0x00000000")
        & notes.coreg.isna()
    ]
    tags = (
        "AccountsReceivableNetCurrent",
        "AccountsReceivableSoldUnderFactoringProgramPrincipalAmountOutstanding",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "RestrictedCashCurrent",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
    )
    facts = {}
    for tag in tags:
        rows = subset.loc[subset.tag.eq(tag), "value"].dropna().unique()
        if len(rows) != 1:
            raise ValueError(f"Original ten-K note fact {tag} is not unique")
        facts[tag] = float(rows[0]) / 1e6
    receivables = facts[tags[0]]
    already_sold = facts[tags[1]]
    total_cash = facts[tags[2]]
    restricted = facts[tags[3]]
    cash = facts[tags[4]]
    if (abs(total_cash - restricted - cash) > 1e-8 or receivables <= 0
            or already_sold <= 0 or cash <= 0):
        raise ValueError("SEC cash or receivable opening reconciliation changed")

    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR")].sort_values("Effective Date")
    if rates["Effective Date"].duplicated().any():
        raise ValueError("SOFR effective date duplicated")
    months = [f"2025-{m:02d}" for m in range(1, 7)]
    annual = np.array([
        float(rates.loc[rates["Effective Date"].lt(month + "-01"), "Rate (%)"].iloc[-1]) / 100 + SPREAD
        for month in months
    ])
    days = np.array([pd.Period(month, freq="M").days_in_month for month in months])
    if np.any(~np.isfinite(annual)) or not np.all(days == [31, 28, 31, 30, 31, 30]):
        raise ValueError("SOFR fixing or day count changed")
    return receivables, cash, already_sold, annual, days, total_cash, restricted


def _path(state, feb_lot=0., mar_lot=0., april_sale=0., delay=.0, extra_sale=0., amend_fee=0.):
    receivables, cash, already_sold, annual, days = state[:5]
    owned = receivables * COLLECTION_SHARES
    if (feb_lot not in (0., LOT_MN) or mar_lot not in (0., LOT_MN)
            or april_sale < 0 or extra_sale < 0 or april_sale + extra_sale > owned[3] + 1e-7):
        raise ValueError("Unpermitted receivable cohort sale")
    sold = np.array([0., feb_lot, mar_lot, april_sale, 0., 0.])
    collection = owned - sold
    delay_face = delay * (owned[2] - mar_lot)
    collection[2] -= delay_face
    collection[4] += delay_face
    collection[3] -= extra_sale
    cash += .98*feb_lot + .97*mar_lot + .96*april_sale
    initial_fee = 10. if feb_lot + mar_lot + april_sale > 1e-9 else 0.
    cash -= initial_fee
    debt = 0.
    interest = 0.
    ledger = []
    for k in range(6):
        if k == 2:
            cash += .96*extra_sale - (2. if extra_sale > 1e-9 else 0.) - amend_fee
        charge = debt * annual[k] * days[k] / 360
        interest += charge
        cash += collection[k] - SPENDING_MN[k] - charge
        if cash < 0:
            debt -= cash
            cash = 0.
        else:
            repay = min(cash, debt)
            cash -= repay
            debt -= repay
        ledger.append((cash, debt, charge))
    cost = (.02*feb_lot + .03*mar_lot + .04*(april_sale + extra_sale)
            + initial_fee + (2. if extra_sale > 1e-9 else 0.) + amend_fee + interest)
    return {
        "cash": cash, "debt": debt, "interest": interest, "cost": cost,
        "peak": max(row[1] for row in ledger), "ledger": ledger,
    }


def _breaches(balance):
    # The executed plan drives end-February debt to exactly the limit, which the
    # query's "may not exceed" does not count as a breach; the slack absorbs the
    # float noise of a ninety-day accrual rather than widening the rule.
    return balance > LOAN_LIMIT_MN + 1e-7


def _day_zero_best(state):
    receivables, _, sold_existing, _, _ = state[:5]
    options = []
    for feb_lot in (0., LOT_MN):
        for mar_lot in (0., LOT_MN):
            upper = min(receivables * .20, FACTOR_OUTSTANDING_LIMIT_MN - sold_existing - feb_lot - mar_lot)
            if upper < 0:
                continue
            if _path(state, feb_lot, mar_lot, upper)["peak"] > LOAN_LIMIT_MN + 1e-8:
                continue
            if _path(state, feb_lot, mar_lot, 0)["peak"] <= LOAN_LIMIT_MN:
                minimum = 0.
            else:
                minimum = float(brentq(lambda z: _path(state, feb_lot, mar_lot, z)["peak"] - LOAN_LIMIT_MN,
                                        0., upper, xtol=1e-11))
            candidate = _path(state, feb_lot, mar_lot, minimum)
            # Across each branch a further April sale saves less prior-debt
            # interest than its four-percent discount; check the endpoints.
            if minimum + 1e-6 < upper and _path(state, feb_lot, mar_lot, upper)["cost"] <= candidate["cost"]:
                raise ValueError("April sale is not cost-increasing on a branch")
            options.append((candidate["cost"], feb_lot, mar_lot, minimum, candidate))
    if len(options) != 4:
        raise ValueError("Discrete factoring branch support changed")
    return min(options, key=lambda row: (row[0], row[1], row[2], row[3]))


@lru_cache(maxsize=1)
def analysis():
    state = _state()
    receivables, cash, already_sold, _, _, total_cash, restricted = state
    unfunded = _path(state)
    _, feb, mar, april, initial = _day_zero_best(state)
    if initial["peak"] > LOAN_LIMIT_MN + 1e-6:
        raise ValueError("Day-zero plan breaches the original revolver")
    stressed = _path(state, feb, mar, april, delay=.75)
    amended = _path(state, feb, mar, april, delay=.75, amend_fee=15.)
    amended_limit = amended["peak"]
    # The extra March face must fit both the remaining April owned batch and the cap.
    available_face = min(
        receivables * .20 - april,
        FACTOR_OUTSTANDING_LIMIT_MN - already_sold - mar - april,
    )
    if _path(state, feb, mar, april, delay=.75, extra_sale=available_face)["peak"] > LOAN_LIMIT_MN + 1e-6:
        raise ValueError("No factoring cure fits the outstanding-face cap")
    extra = float(brentq(
        lambda face: _path(state, feb, mar, april, delay=.75, extra_sale=face)["ledger"][2][1] - LOAN_LIMIT_MN,
        1e-7, available_face, xtol=1e-11,
    ))
    factored = _path(state, feb, mar, april, delay=.75, extra_sale=extra)
    if factored["peak"] > LOAN_LIMIT_MN + 1e-7 or amended_limit <= LOAN_LIMIT_MN:
        raise ValueError("March financing cure state changed")
    winner = "FACTOR" if factored["cost"] < amended["cost"] else "AMEND"
    return {
        "issuer_cik": CIK,
        "original_accession": ACCESSION,
        "owned_receivables_mn": receivables,
        "usable_cash_mn": cash,
        "cash_including_restricted_mn": total_cash,
        "current_restricted_cash_mn": restricted,
        "already_sold_face_mn": already_sold,
        "unfunded_peak_debt_mn": unfunded["peak"],
        "unfunded_interest_mn": unfunded["interest"],
        "first_no_factor_breach_month": next(f"2025-{j:02d}" for j, row in enumerate(unfunded["ledger"], 1) if _breaches(row[1])),
        "day_zero_feb_lot": "SELL" if feb else "KEEP",
        "day_zero_mar_lot": "SELL" if mar else "KEEP",
        "day_zero_april_face_mn": april,
        "day_zero_total_financing_cost_mn": initial["cost"],
        "end_feb_revolver_mn": initial["ledger"][1][1],
        "stressed_march_credit_demand_mn": stressed["ledger"][2][1],
        "stressed_first_breach_month": next(f"2025-{j:02d}" for j, row in enumerate(stressed["ledger"], 1) if _breaches(row[1])),
        "amend_required_limit_mn": amended_limit,
        "amend_total_financing_cost_mn": amended["cost"],
        "additional_april_factor_face_mn": extra,
        "factor_total_financing_cost_mn": factored["cost"],
        "factor_remaining_headroom_mn": available_face - extra,
        "chosen_cure": winner,
        "chosen_final_cash_mn": factored["cash"] if winner == "FACTOR" else amended["cash"],
    }


@lru_cache(maxsize=1)
def ground_truth():
    state = analysis()
    return tuple(state[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = analysis()
    precision = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [precision[n] for n in names])


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate_subset(outputs, names))
    for i, names in enumerate(TURN_NAMES, start=1)
}
