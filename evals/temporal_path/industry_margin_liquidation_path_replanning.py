"""Hypothetical margin account on Ken French industry total-return paths.

The selected industries are return-index proxies, not claims of tradable ETF
prices. All credit, margin and trading terms are scenario assumptions.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from cave_agent import Variable
from core.data import load_expansion_table, load_ken_french_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "industry_margin_liquidation_path_replanning"
INDUSTRIES = {
    "nodur_pct": "NoDur", "durbl_pct": "Durbl", "manuf_pct": "Manuf",
    "enrgy_pct": "Enrgy", "hitec_pct": "HiTec", "telcm_pct": "Telcm",
    "shops_pct": "Shops", "hlth_pct": "Hlth", "utils_pct": "Utils",
    "other_pct": "Other",
}
START = "2022-01-01"
END = "2023-12-29"
OPENING_EQUITY_MN = 10.
SPREAD = .02
SALE_FEE_RATE = .002
RESTORE_RATIO = .4


def _v(name, description):
    return Variable(name, None, description)


TOKEN_CONTRACT = "Store the exact source-canonical industry token: one of NoDur, Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils or Other."
variables = [
    _v("industry_1", "Store the highest selected 2021 sample-volatility industry. " + TOKEN_CONTRACT),
    _v("industry_2", "Store the second selected 2021 sample-volatility industry. " + TOKEN_CONTRACT),
    _v("industry_3", "Store the third selected 2021 sample-volatility industry. " + TOKEN_CONTRACT),
    _v("initial_loan_mn", "Store opening financed debt in USD millions, rounded to 4 decimals."),
    _v("unliquidated_low_margin_pct", "Store the lowest trading-day equity-to-assets ratio in the no-sale diagnostic, in percent, rounded to 4 decimals."),
    _v("unliquidated_low_margin_day", "Store the no-sale diagnostic's first lowest-ratio trading date as an ISO YYYY-MM-DD string."),
    _v("first_liquidation_day", "Store the first compulsory-sale trading date as an ISO YYYY-MM-DD string."),
    _v("cumulative_liquidation_mn", "Store cumulative gross market value of compulsory sales on the original maintenance path, USD millions, rounded to 4 decimals."),
    _v("liquidation_terminal_equity_mn", "Store terminal account equity after original-rule compulsory sales, USD millions, rounded to 4 decimals."),
    _v("minimum_equity_injection_mn", "Store the single first-call outside cash repayment needed to avoid that day's sale, USD millions, rounded to 4 decimals."),
    _v("post_injection_sale_count", "Store the exact integer count of later compulsory-sale trading events on the injection branch."),
    _v("injected_adjusted_terminal_equity_mn", "Store terminal equity after the injection branch less terminal value of outside cash at unspread SOFR, USD millions, rounded to 4 decimals."),
    _v("tightened_first_sale_day", "Store the first compulsory-sale date after the later maintenance tightening on the injected path as an ISO YYYY-MM-DD string."),
    _v("tightened_cumulative_sale_mn", "Store cumulative gross compulsory sale value after the later maintenance tightening, USD millions, rounded to 4 decimals."),
    _v("tightened_adjusted_terminal_equity_mn", "Store the tightened injected path's terminal equity less terminal value of outside cash at unspread SOFR, USD millions, rounded to 4 decimals."),
    _v("no_sale_leverage_count", "Store the exact integer count of eligible zero-compulsory-sale starting-leverage grid points."),
    _v("best_no_sale_leverage", "Store the terminal-equity-maximizing eligible initial assets/equity leverage, rounded to 2 decimals."),
    _v("best_no_sale_terminal_equity_mn", "Store terminal equity at the selected eligible leverage with no outside cash or forced sale, USD millions, rounded to 4 decimals."),
]
TURN_NAMES = [
    ["industry_1", "industry_2", "industry_3", "initial_loan_mn", "unliquidated_low_margin_pct", "unliquidated_low_margin_day"],
    ["first_liquidation_day", "cumulative_liquidation_mn", "liquidation_terminal_equity_mn"],
    ["minimum_equity_injection_mn", "post_injection_sale_count", "injected_adjusted_terminal_equity_mn"],
    ["tightened_first_sale_day", "tightened_cumulative_sale_mn", "tightened_adjusted_terminal_equity_mn"],
    ["no_sale_leverage_count", "best_no_sale_leverage", "best_no_sale_terminal_equity_mn"],
]
DECIMALS = [None, None, None, 4, 4, None, None, 4, 4, 4, 0, 4, None, 4, 4, 0, 2, 4]


@lru_cache(maxsize=1)
def _inputs():
    frame = load_ken_french_table("daily_industry_returns")
    train = frame.loc[frame.date.between("2021-01-01", "2021-12-31")]
    replay = frame.loc[frame.date.between(START, END)]
    if len(train) != 252 or len(replay) != 501 or train[list(INDUSTRIES)].isna().any().any():
        raise ValueError("Ken French training or replay support changed")
    ranked = sorted(
        INDUSTRIES,
        key=lambda col: (-float(train[col].to_numpy(float).std(ddof=1)), INDUSTRIES[col]),
    )
    columns = tuple(ranked[:3])
    trading_dates = pd.to_datetime(replay.date)
    calendar = pd.date_range(START, END)
    positions = np.searchsorted(calendar.to_numpy(), trading_dates.to_numpy())
    if not np.array_equal(calendar.to_numpy()[positions], trading_dates.to_numpy()):
        raise ValueError("Trading day outside calendar replay")
    published = replay[list(columns)].to_numpy(float)
    # Ken French encodes a missing observation as -99.99, which is finite and
    # would otherwise be replayed as a -99.99% daily return.
    if not np.isfinite(published).all() or (published <= -99.99).any():
        raise ValueError("Missing industry return")
    returns = np.zeros((len(calendar), 3))
    returns[positions] = published / 100

    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]].copy()
    rates["Effective Date"] = pd.to_datetime(rates["Effective Date"])
    rates = rates.sort_values("Effective Date")
    if rates["Effective Date"].duplicated().any():
        raise ValueError("Duplicate SOFR effective date")
    fixed = pd.merge_asof(
        pd.DataFrame({"day": calendar}), rates,
        left_on="day", right_on="Effective Date", direction="backward",
    )
    sofr = fixed["Rate (%)"].to_numpy(float) / 100
    if not np.isfinite(sofr).all():
        raise ValueError("Missing calendar-day SOFR")
    trading_mask = np.zeros(len(calendar), dtype=bool)
    trading_mask[positions] = True
    return tuple(INDUSTRIES[col] for col in columns), calendar.to_numpy(dtype="datetime64[D]"), returns, sofr, trading_mask


def _path(*, leverage=2.5, policy="hold", tightened=False):
    tokens, dates, returns, sofr, trading = _inputs()
    price = np.ones(3)
    units = np.full(3, OPENING_EQUITY_MN * leverage / 3)
    debt = OPENING_EQUITY_MN * (leverage - 1)
    low = (float("inf"), None)
    first_sale = None
    sales = []
    injection = 0.
    injection_day = None
    terminal_state = None
    for i, day in enumerate(dates):
        if trading[i]:
            price *= 1 + returns[i]
        debt *= 1 + (sofr[i] + SPREAD) / 360
        if not trading[i]:
            continue
        assets = float(price @ units)
        equity = assets - debt
        if assets <= 0 or equity <= 0:
            return {"failed": True, "failure_date": str(day)}
        margin = equity / assets
        if margin < low[0] - 1e-14:
            low = (margin, str(day))
        maintenance = .35 if tightened and day >= np.datetime64("2023-03-01") else .30
        if margin < maintenance - 1e-12:
            if policy == "inject" and injection_day is None:
                injection = max(0., debt - (1 - maintenance) * assets)
                debt -= injection
                injection_day = i
                margin = (assets - debt) / assets
            if policy != "hold" and margin < maintenance - 1e-12:
                sale = (debt - (1 - RESTORE_RATIO) * assets) / (RESTORE_RATIO - SALE_FEE_RATE)
                if sale < 0 or sale > assets + 1e-8 or sale * (1 - SALE_FEE_RATE) > debt + 1e-8:
                    return {"failed": True, "failure_date": str(day)}
                units *= 1 - sale / assets
                debt -= (1 - SALE_FEE_RATE) * sale
                sales.append((str(day), float(sale)))
                if first_sale is None:
                    first_sale = str(day)
                post_assets = float(price @ units)
                if abs((post_assets - debt) / post_assets - RESTORE_RATIO) > 1e-8:
                    raise ValueError("Margin restoration invariant failed")
        terminal_state = (float(price @ units), float(debt))
    if terminal_state is None:
        raise ValueError("No trading observations")
    terminal_equity = terminal_state[0] - terminal_state[1]
    opportunity = (
        injection * float(np.prod(1 + sofr[injection_day + 1:] / 360))
        if injection_day is not None else 0.
    )
    return {
        "failed": False, "tokens": tokens,
        "initial_debt": OPENING_EQUITY_MN * (leverage - 1),
        "low_pct": 100 * low[0], "low_day": low[1],
        "first_sale": first_sale, "sales": sales,
        "sale_sum": sum(value for _, value in sales),
        "injection": injection,
        "terminal_equity": terminal_equity,
        "adjusted_terminal_equity": terminal_equity - opportunity,
        "terminal_assets": terminal_state[0], "terminal_debt": terminal_state[1],
    }


@lru_cache(maxsize=1)
def analysis():
    diagnostic = _path()
    liquidated = _path(policy="liquidate")
    injected = _path(policy="inject")
    tightened = _path(policy="inject", tightened=True)
    if any(z["failed"] for z in (diagnostic, liquidated, injected, tightened)):
        raise ValueError("Chosen margin path insolvent")
    eligible = []
    for tick in range(61):
        leverage = 1 + tick * .05
        path = _path(leverage=leverage, policy="liquidate", tightened=True)
        if not path["failed"] and not path["sales"]:
            eligible.append((leverage, path))
    if not eligible:
        raise ValueError("No zero-sale leverage candidate")
    best_leverage, best = max(
        eligible, key=lambda row: (row[1]["terminal_equity"], -row[0]),
    )
    return {
        "industry_1": diagnostic["tokens"][0],
        "industry_2": diagnostic["tokens"][1],
        "industry_3": diagnostic["tokens"][2],
        "initial_loan_mn": diagnostic["initial_debt"],
        "unliquidated_low_margin_pct": diagnostic["low_pct"],
        "unliquidated_low_margin_day": diagnostic["low_day"],
        "first_liquidation_day": liquidated["first_sale"],
        "cumulative_liquidation_mn": liquidated["sale_sum"],
        "liquidation_terminal_equity_mn": liquidated["terminal_equity"],
        "minimum_equity_injection_mn": injected["injection"],
        "post_injection_sale_count": len(injected["sales"]),
        "injected_adjusted_terminal_equity_mn": injected["adjusted_terminal_equity"],
        "tightened_first_sale_day": tightened["first_sale"],
        "tightened_cumulative_sale_mn": tightened["sale_sum"],
        "tightened_adjusted_terminal_equity_mn": tightened["adjusted_terminal_equity"],
        "no_sale_leverage_count": len(eligible),
        "best_no_sale_leverage": best_leverage,
        "best_no_sale_terminal_equity_mn": best["terminal_equity"],
    }


@lru_cache(maxsize=1)
def ground_truth():
    state = analysis()
    return tuple(state[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = analysis()
    precision = dict(zip((v.name for v in variables), DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[n] for n in names], [truth[n] for n in names],
        [precision[n] for n in names],
    )


def validate(outputs):
    return _validate_subset(outputs, [v.name for v in variables])


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _validate_subset(outputs, names))
    for i, names in enumerate(TURN_NAMES, start=1)
}
