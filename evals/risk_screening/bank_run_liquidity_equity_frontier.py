"""Hypothetical bank security disposal, repo funding and book-capital frontier.

FDIC balances anchor only the opening balance sheet. Asset availability, sale
prices, repo collateral, reserve and equity/assets threshold are stipulated
internal scenario terms, not reported liquidity or regulatory capital. Paying a
deposit withdrawal directly out of equity, counting repo as new equity, or
selling pledged securities changes the stage-one frontier. The canonical
first-period maximum runoff is USD 2,783.9042m. Omitting the capital floor
raises it to USD 2,796.1151m; treating repo as equity raises it to USD
2,794.7223m; allowing pledged securities to be sold raises it to USD
2,966.6755m. These are different financial contracts outside PV tolerance.
Resetting prior losses or repo interest when spreads change corrupts later
states.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import linprog

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


REPORT_DATE = "2024-12-31"
TARGET_CERT = "622"
SALE_PRICES = np.array([.97, .94, .80])
TRANCHE_SHARES = np.array([.25, .25, .50])
ADVANCE = .80
REPO_LINE_MN = 1000.0
RESERVE_SHARE = .02
EQUITY_FLOOR = .085


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("selected_certificate", "Store the selected FDIC certificate as its exact string identifier."),
    _v("opening_cash_mn", "Store reported cash in USD millions, rounded to 3 decimals from the source's thousand-USD unit."),
    _v("opening_securities_mn", "Store reported securities in USD millions, rounded to 3 decimals from the source's thousand-USD unit."),
    _v("opening_equity_assets_pct", "Store opening book equity divided by assets in percent, rounded to 4 decimals."),
    _v("first_sale_book_mn", "Store book value of securities sold in stage one in USD millions, rounded to 4 decimals."),
    _v("first_repo_principal_mn", "Store stage-one repo principal in USD millions, rounded to 4 decimals."),
    _v("first_ending_equity_assets_pct", "Store stage-one ending book equity/assets in percent, rounded to 4 decimals."),
    _v("frontier_runoff_mn", "Store maximum first-period withdrawal with no new equity in USD millions, rounded to 3 decimals."),
    _v("frontier_runoff_pct", "Store maximum first-period withdrawal as percent of opening deposits, rounded to 3 decimals; compute from the unrounded maximum."),
    _v("frontier_bindings", "Store exactly one token from CASH_ONLY, CASH_COLLATERAL, CASH_CAPITAL, CASH_COLLATERAL_CAPITAL."),
    _v("second_new_equity_mn", "Store the minimum new equity for stage two in USD millions, rounded to 4 decimals."),
    _v("second_sale_book_mn", "Store additional stage-two book securities sold in USD millions, rounded to 4 decimals."),
    _v("second_ending_cash_mn", "Store stage-two ending cash in USD millions, rounded to 4 decimals, calculated from unrounded events."),
    _v("third_new_equity_mn", "Store the minimum new equity for stage three in USD millions, rounded to 4 decimals."),
    _v("third_ending_equity_assets_pct", "Store stage-three ending book equity/assets in percent, rounded to 4 decimals."),
]

TURN_NAMES = [
    ["selected_certificate", "opening_cash_mn", "opening_securities_mn", "opening_equity_assets_pct"],
    ["first_sale_book_mn", "first_repo_principal_mn", "first_ending_equity_assets_pct"],
    ["frontier_runoff_mn", "frontier_runoff_pct", "frontier_bindings"],
    ["second_new_equity_mn", "second_sale_book_mn", "second_ending_cash_mn"],
    ["third_new_equity_mn", "third_ending_equity_assets_pct"],
]
DECIMALS = [None, 3, 3, 4, 4, 4, 4, 3, 3, None, 4, 4, 4, 4, 4]


def _sofr(day):
    rates = load_expansion_table("nyfed_reference_rates")
    row = rates.loc[rates["Rate Type"].eq("SOFR") & rates["Effective Date"].eq(day)]
    if len(row) != 1:
        raise ValueError(f"SOFR fixing is not unique: {day}")
    return float(row.iloc[0]["Rate (%)"]) / 100


def _opening():
    banks = load_expansion_table("fdic_bankfind")
    day = banks.loc[banks.REPDTE.eq(REPORT_DATE)].copy()
    subset = day.loc[
        day.ASSET.between(5_000_000, 10_000_000)
        & day.DEP.gt(0) & day.EQ.gt(0) & day.CHBAL.gt(0)
        & day.SC.div(day.ASSET).gt(.30)
        & day.CHBAL.add(day.SC).le(day.ASSET)
    ].copy()
    subset["cash_dep"] = subset.CHBAL / subset.DEP
    row = subset.sort_values(["cash_dep", "CERT"]).iloc[0]
    if str(row.CERT) != TARGET_CERT:
        raise ValueError("FDIC screening leader changed")
    return {
        "cert": str(row.CERT),
        "asset": float(row.ASSET) / 1000,
        "cash": float(row.CHBAL) / 1000,
        "security": float(row.SC) / 1000,
        "deposit": float(row.DEP) / 1000,
        "equity": float(row.EQ) / 1000,
    }


def _first_lp(book, fixed_runoff=None):
    a, c, s, d, e = (book[n] for n in ("asset", "cash", "security", "deposit", "equity"))
    k = (_sofr("2024-12-31") + .015) * 90 / 360
    losses = 1 - SALE_PRICES
    widths = TRANCHE_SHARES * s
    fixed = fixed_runoff is not None
    objective = np.r_[losses, k] if fixed else np.array([0., 0., 0., 0., -1.])
    bounds = list(zip(np.zeros(3), widths)) + [(0, REPO_LINE_MN)]
    if not fixed:
        bounds.append((0, d))
    cash_row = np.r_[-SALE_PRICES, -(1-k)] if fixed else np.r_[-SALE_PRICES, -(1-k), 1-RESERVE_SHARE]
    collateral_row = np.r_[np.repeat(ADVANCE, 3), 1.] if fixed else np.r_[np.repeat(ADVANCE, 3), 1., 0.]
    capital_row = np.r_[(1-EQUITY_FLOOR)*losses, EQUITY_FLOOR+(1-EQUITY_FLOOR)*k]
    if not fixed:
        capital_row = np.r_[capital_row, -EQUITY_FLOOR]
    cash_rhs = c - RESERVE_SHARE*d
    capital_rhs = e - EQUITY_FLOOR*a
    if fixed:
        cash_rhs -= (1-RESERVE_SHARE)*fixed_runoff
        capital_rhs += EQUITY_FLOOR*fixed_runoff
    result = linprog(
        objective,
        A_ub=np.vstack([cash_row, collateral_row, capital_row]),
        b_ub=[cash_rhs, ADVANCE*s, capital_rhs],
        bounds=bounds, method="highs",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"First-period scenario infeasible: {result.message}")
    return result, k


def _post_first(book, result, k, runoff):
    x = result.x[:3]
    q = float(result.x[3])
    sale_book = float(x.sum())
    sale_cash = float(SALE_PRICES @ x)
    loss = float((1-SALE_PRICES) @ x)
    interest = k*q
    state = {
        "asset": book["asset"] - runoff + q - loss - interest,
        "cash": book["cash"] + sale_cash + q - runoff - interest,
        "security": book["security"] - sale_book,
        "deposit": book["deposit"] - runoff,
        "equity": book["equity"] - loss - interest,
        "repo": q,
        "other_liability": book["asset"] - book["deposit"] - book["equity"],
        "unpledged": book["security"] - sale_book - q/ADVANCE,
        "third_tranche_used": float(x[2]),
        "opening_security": book["security"],
        "sale_book": sale_book,
    }
    _check(state)
    return state


def _check(state):
    actual = state["asset"]
    balance = state["deposit"] + state["repo"] + state["other_liability"] + state["equity"]
    if abs(actual - balance) > 1e-6:
        raise ValueError("Balance sheet does not reconcile")
    if state["cash"] + 1e-6 < RESERVE_SHARE*state["deposit"]:
        raise ValueError("Cash reserve not met")
    if state["equity"]/state["asset"] + 1e-10 < EQUITY_FLOOR:
        raise ValueError("Book equity floor not met")
    if state["unpledged"] < -1e-6:
        raise ValueError("Pledged securities sold")


def _next_stage(previous, withdrawal, k):
    # All sellable security left after the first stage lies in the third
    # cumulative-price tranche. The existing repo cannot be prepaid within
    # this period and the USD 1bn line is already filled.
    max_sale = max(0.0, previous["unpledged"])
    max_sale = min(max_sale, TRANCHE_SHARES[2]*previous["opening_security"] - previous["third_tranche_used"])
    interest = previous["repo"] * k
    c, d, e, a = (previous[n] for n in ("cash", "deposit", "equity", "asset"))
    # Variables: additional book securities sold, fresh external equity.
    result = linprog(
        [0., 1.],
        A_ub=[[-.80, -1.], [(1-EQUITY_FLOOR)*.20, -(1-EQUITY_FLOOR)]],
        b_ub=[c - RESERVE_SHARE*d - (1-RESERVE_SHARE)*withdrawal - interest,
              e - EQUITY_FLOOR*a + EQUITY_FLOOR*withdrawal - (1-EQUITY_FLOOR)*interest],
        bounds=[(0, max_sale), (0, None)], method="highs",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Later-stage scenario infeasible: {result.message}")
    sale, injection = map(float, result.x)
    loss = .20*sale
    state = previous.copy()
    state["asset"] = a - withdrawal - loss - interest + injection
    state["cash"] = c + .80*sale + injection - withdrawal - interest
    state["security"] -= sale
    state["deposit"] -= withdrawal
    state["equity"] += injection - loss - interest
    state["unpledged"] -= sale
    state["third_tranche_used"] += sale
    state["additional_sale"] = sale
    state["new_equity"] = injection
    _check(state)
    return state


def analysis():
    opening = _opening()
    runoff1 = .35 * opening["deposit"]
    first_result, k1 = _first_lp(opening, runoff1)
    first = _post_first(opening, first_result, k1, runoff1)
    if abs(first["repo"]-REPO_LINE_MN) > 1e-7:
        raise ValueError("Frozen first-stage repo regime changed")
    frontier_result, _ = _first_lp(opening)
    max_runoff = float(frontier_result.x[4])
    frontier = _post_first(opening, frontier_result, k1, max_runoff)
    # Cash, collateral and capital all bind at the independent frontier.
    if not (abs(frontier["cash"]-RESERVE_SHARE*frontier["deposit"]) < 1e-5
            and abs(frontier["unpledged"]) < 1e-5
            and abs(frontier["equity"]/frontier["asset"]-EQUITY_FLOOR) < 1e-8):
        raise ValueError("Frozen first-period frontier regimes changed")
    k2 = (_sofr("2025-03-31") + .035) * 91 / 360
    second = _next_stage(first, .04*opening["deposit"], k2)
    k3 = (_sofr("2025-06-30") + .015) * 92 / 360
    third = _next_stage(second, .01*opening["deposit"], k3)
    return {
        "selected_certificate": opening["cert"],
        "opening_cash_mn": opening["cash"],
        "opening_securities_mn": opening["security"],
        "opening_equity_assets_pct": 100*opening["equity"]/opening["asset"],
        "first_sale_book_mn": first["sale_book"],
        "first_repo_principal_mn": first["repo"],
        "first_ending_equity_assets_pct": 100*first["equity"]/first["asset"],
        "frontier_runoff_mn": max_runoff,
        "frontier_runoff_pct": 100*max_runoff/opening["deposit"],
        "frontier_bindings": "CASH_COLLATERAL_CAPITAL",
        "second_new_equity_mn": second["new_equity"],
        "second_sale_book_mn": second["additional_sale"],
        "second_ending_cash_mn": second["cash"],
        "third_new_equity_mn": third["new_equity"],
        "third_ending_equity_assets_pct": 100*third["equity"]/third["asset"],
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names],
        [precision[n] for n in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(
        lambda outputs, names=names: _validate_subset(outputs, names)
    )
    for i, names in enumerate(TURN_NAMES, start=1)
}
