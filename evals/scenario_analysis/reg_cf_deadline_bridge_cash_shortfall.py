"""Regulation Crowdfunding deadline, pre-closing cash and bridge funding.

The 2025 Form C/A amendment extends this offering's deadline, but the reported
most-recent-fiscal-year cash and debt are used only as hypothetical June 18
opening amounts. Operating cash use is stipulated, not inferred from net
income. Subscriptions remain unavailable until successful closing. The
disclosed per-share price gives an integer-share funding boundary; a 10% cash
intermediary fee is an explicit scenario value within the disclosed range.
The daily loan resets use SOFR available after each morning publication, not
the same day's effective-date observation before it has been published.

Measured alternatives: treating target subscriptions as pre-closing cash can
erase the required 121,424-dollar bridge; continuously sized fundraising
understates the needed gross by up to one 2.93-dollar share. Neither is an
accepted financial convention for this case, which is labelled baseline.
"""

from decimal import Decimal, ROUND_CEILING
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


def D(value):
    amount = Decimal(str(value))
    if not amount.is_finite():
        raise ValueError("non-finite amount")
    return amount


variables = [
    Variable("prior_deadline", None, "Store the prior deadline as an ISO YYYY-MM-DD string."),
    Variable("revised_deadline", None, "Store the revised deadline as an ISO YYYY-MM-DD string."),
    Variable("reported_cash_usd", None, "Store the disclosed most-recent-fiscal-year cash and equivalents in USD as a finite whole-dollar number."),
    Variable("reported_short_debt_usd", None, "Store the disclosed most-recent-fiscal-year short-term debt in USD as a finite whole-dollar number."),
    Variable("target_offering_usd", None, "Store the disclosed target offering amount in USD as a finite number, rounded to 2 decimals."),
    Variable("maximum_offering_usd", None, "Store the disclosed maximum offering amount in USD as a finite number, rounded to 2 decimals."),
    Variable("prior_close_cash_usd", None, "Store cash after the prior-date hypothetical debt repayment in USD as a finite number, rounded to 4 decimals from unrounded inputs."),
    Variable("preclose_shortfall_usd", None, "Store the revised-date pre-closing operating cash shortfall without bridge funding in USD as a finite whole-dollar number."),
    Variable("first_negative_cash_date", None, "Store the first negative operating-cash date as an ISO YYYY-MM-DD string."),
    Variable("minimum_subscription_shares", None, "Store the minimum successful subscription as a finite integer number of common shares."),
    Variable("stressed_bridge_usd", None, "Store the revised minimum bridge principal in USD as a finite whole-dollar number."),
    Variable("stressed_subscription_shares", None, "Store the revised minimum successful subscription as a finite integer number of common shares."),
]
TURNS = [
    ["prior_deadline", "revised_deadline", "reported_cash_usd", "reported_short_debt_usd", "target_offering_usd", "maximum_offering_usd"],
    ["prior_close_cash_usd"],
    ["preclose_shortfall_usd", "first_negative_cash_date"],
    ["minimum_subscription_shares"],
    ["stressed_bridge_usd", "stressed_subscription_shares"],
]
DECIMAL_BY_NAME = {v.name: (None if v.name in {"prior_deadline", "revised_deadline", "first_negative_cash_date"} else 4 if v.name == "prior_close_cash_usd" else 2 if v.name in {"target_offering_usd", "maximum_offering_usd"} else 0) for v in variables}
DECIMALS = [DECIMAL_BY_NAME[v.name] for v in variables]
PRICE = D("2.93")


def _filing_state():
    filings = load_expansion_table("sec_form_c")
    selected = {}
    for accession in ("0001665160-25-000665", "0001665160-25-001224"):
        rows = filings.loc[filings["ACCESSION_NUMBER"].astype(str).eq(accession) & filings["CIK"].astype(str).eq("0001765489") & filings["SUBMISSION_TYPE"].eq("C/A")]
        if len(rows) != 1:
            raise ValueError("ambiguous Form C/A accession")
        selected[accession] = rows.iloc[0]
    old, new = selected.values()
    if str(old["DEADLINEDATE"]) != "2025-07-14" or str(new["DEADLINEDATE"]) != "2025-10-16":
        raise ValueError("offering deadline lineage changed")
    if str(new["FILING_DATE"]) != "2025-06-18" or D(new["OFFERINGAMOUNT"]) != D(old["OFFERINGAMOUNT"]):
        raise ValueError("offering identity changed")
    cash = D(new["CASHEQUIMOSTRECENTFISCALYEAR"])
    debt = D(new["SHORTTERMDEBTMRECENTFISCALYEAR"])
    target = D(new["OFFERINGAMOUNT"])
    maximum = D(new["MAXIMUMOFFERINGAMOUNT"])
    if target / PRICE != D(2559) or maximum / PRICE != D(1475025):
        raise ValueError("offering share-quantity boundary changed")
    return cash, debt, target, maximum


def _factor(start, end, spread):
    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]].dropna()
    if rates["Effective Date"].duplicated().any():
        raise ValueError("duplicate effective SOFR date")
    fixings = rates.set_index(pd.to_datetime(rates["Effective Date"]))["Rate (%)"].sort_index()
    if len(fixings) < 2:
        raise ValueError("insufficient SOFR publication history")
    # An effective-date fixing is published on the next SOFR publication day.
    # The next effective date in this frozen daily series identifies that day.
    published = pd.Series(fixings.to_numpy()[:-1], index=fixings.index[1:])
    dates = pd.date_range(start, end, freq="D")
    daily = published.reindex(dates, method="ffill")
    if daily.isna().any():
        raise ValueError("missing SOFR fixing")
    return sum((D(rate) / 100 + D(spread)) / 360 for rate in daily)


def _shares_needed(cash, debt, burn, bridge, debt_factor, bridge_factor):
    # Cash after burn and minimum bridge is zero immediately before closing.
    if cash + bridge != burn:
        raise ValueError("bridge does not preserve zero pre-close cash")
    payable = debt * (1 + debt_factor) + bridge * (1 + bridge_factor) + D(50000)
    shares = (payable / (PRICE * D("0.90"))).to_integral_value(rounding=ROUND_CEILING)
    if shares <= 0 or (shares - 1) * PRICE * D("0.90") >= payable:
        raise ValueError("incorrect minimum whole-share boundary")
    return int(shares)


def analysis():
    cash, debt, target, maximum = _filing_state()
    old_days = pd.date_range("2025-06-18", "2025-07-14", freq="D")
    new_days = pd.date_range("2025-06-18", "2025-10-16", freq="D")
    shock_days = pd.date_range("2025-09-01", "2025-10-16", freq="D")
    if (len(old_days), len(new_days), len(shock_days)) != (27, 121, 46):
        raise ValueError("scenario calendar changed")
    prior_cash = cash - D(3000) * len(old_days) - debt * (1 + _factor("2025-06-18", "2025-07-14", "0.03"))
    normal_burn = D(3000) * len(new_days)
    shortfall = normal_burn - cash
    if shortfall <= 0:
        raise ValueError("no pre-closing cash deficit")
    running_cash = cash
    first_negative = None
    for date in new_days:
        running_cash -= D(3000)
        if running_cash < 0:
            first_negative = date.strftime("%Y-%m-%d")
            break
    if first_negative is None or first_negative <= "2025-08-31":
        raise ValueError("bridge draw date is no longer feasible")
    old_debt_factor = _factor("2025-06-18", "2025-10-16", "0.03")
    bridge_factor = _factor("2025-08-31", "2025-10-16", "0.05")
    shares = _shares_needed(cash, debt, normal_burn, shortfall, old_debt_factor, bridge_factor)
    shocked_burn = normal_burn + D(450) * len(shock_days)
    stressed_bridge = shocked_burn - cash
    stressed_shares = _shares_needed(cash, debt, shocked_burn, stressed_bridge, old_debt_factor, bridge_factor)
    if not (target <= D(shares) * PRICE <= maximum and target <= D(stressed_shares) * PRICE <= maximum):
        raise ValueError("subscription outside disclosed offering range")
    return {
        "prior_deadline": "2025-07-14",
        "revised_deadline": "2025-10-16",
        "reported_cash_usd": float(cash),
        "reported_short_debt_usd": float(debt),
        "target_offering_usd": float(target),
        "maximum_offering_usd": float(maximum),
        "prior_close_cash_usd": float(prior_cash),
        "preclose_shortfall_usd": float(shortfall),
        "first_negative_cash_date": first_negative,
        "minimum_subscription_shares": shares,
        "stressed_bridge_usd": float(stressed_bridge),
        "stressed_subscription_shares": stressed_shares,
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _validate(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[name] for name in names], [truth[name] for name in names], [DECIMAL_BY_NAME[name] for name in names])


def validate(outputs):
    return _validate(outputs, [v.name for v in variables])


validators = {f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate(outputs, names)) for i, names in enumerate(TURNS, 1)}
