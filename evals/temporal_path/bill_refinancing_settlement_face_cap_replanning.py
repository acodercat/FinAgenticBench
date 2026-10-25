"""Retrospective issuer bill refinancing with settlement-order and face-state constraints.

MSPD/auction/SOFR values are observed source inputs. The issuer, its lot size,
credit facilities and subsequent policy changes are hypothetical.
"""

from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd

from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_ID = "bill_refinancing_settlement_face_cap_replanning"
ISSUE_DATES = ("2025-01-07", "2025-01-28", "2025-02-18", "2025-03-04")
START = "2025-01-01"
END = "2025-03-31"
OPENING_CASH_MN = .3
LOT_FACE_MN = .1
LOT_MAX = 20
BASE_LIMIT_MN = 2.7
EMERGENCY_FEE_MN = .01


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("legacy_due_mn", "Store the scaled 2025-Q1 old-bill principal due in USD millions, rounded to 6 decimals."),
    _v("last_legacy_maturity_date", "Store the latest selected old-bill maturity date as an ISO YYYY-MM-DD string."),
    _v("first_issue_pre_cash_gap_mn", "Store base bridge principal needed on the first candidate issue date before its new issue proceeds, USD millions, rounded to 6 decimals."),
    _v("jan_7_lots", "Store the chosen integer number of USD0.1m face lots issued on 2025-01-07."),
    _v("jan_28_lots", "Store the chosen integer number of USD0.1m face lots issued on 2025-01-28."),
    _v("feb_18_lots", "Store the chosen integer number of USD0.1m face lots issued on 2025-02-18."),
    _v("mar_4_lots", "Store the chosen integer number of USD0.1m face lots issued on 2025-03-04."),
    _v("initial_total_cost_mn", "Store minimum original-plan note discount plus bridge interest through 2025-03-31, USD millions, rounded to 6 decimals."),
    _v("initial_peak_preissue_bridge_mn", "Store the highest original-plan base bridge principal immediately before any candidate issue settlement, USD millions, rounded to 6 decimals."),
    _v("post_jan_28_cash_mn", "Store unrestricted cash after all 2025-01-28 ordered settlements and cash sweep, USD millions, rounded to 6 decimals."),
    _v("post_jan_28_base_bridge_mn", "Store base bridge principal after all 2025-01-28 ordered settlements and cash sweep, USD millions, rounded to 6 decimals."),
    _v("tightened_feb_18_lots", "Store the chosen integer 2025-02-18 lot count under the tightened live-note face cap."),
    _v("tightened_mar_4_lots", "Store the proposed integer 2025-03-04 lot count under the tightened cap; it has not been executed at 2025-02-27."),
    _v("first_emergency_draw_date", "Store the first emergency-bridge draw date as an ISO YYYY-MM-DD string."),
    _v("tightened_peak_emergency_mn", "Store the highest emergency-bridge principal on the tightened full-window proposed path through 2025-03-31, including post-2025-02-27 stages, USD millions, rounded to 6 decimals."),
    _v("tightened_total_cost_mn", "Store full-window note discount, both bridge interest charges and emergency activation fee for the tightened plan, USD millions, rounded to 6 decimals."),
    _v("restored_mar_4_lots", "Store the chosen integer 2025-03-04 lot count after future face capacity is restored."),
    _v("restored_total_cost_mn", "Store full-window note discount, both bridge interest charges and the retained emergency fee under the restored continuation, USD millions, rounded to 6 decimals."),
    _v("restored_end_base_bridge_mn", "Store base bridge principal at 2025-03-31 after the restored continuation, USD millions, rounded to 6 decimals."),
]
TURN_NAMES = [
    ["legacy_due_mn", "last_legacy_maturity_date", "first_issue_pre_cash_gap_mn"],
    ["jan_7_lots", "jan_28_lots", "feb_18_lots", "mar_4_lots", "initial_total_cost_mn", "initial_peak_preissue_bridge_mn"],
    ["post_jan_28_cash_mn", "post_jan_28_base_bridge_mn"],
    ["tightened_feb_18_lots", "tightened_mar_4_lots", "first_emergency_draw_date", "tightened_peak_emergency_mn", "tightened_total_cost_mn"],
    ["restored_mar_4_lots", "restored_total_cost_mn", "restored_end_base_bridge_mn"],
]
DECIMALS = [6, None, 6, None, None, None, None, 6, 6, 6, 6, None, None, None, 6, 6, None, 6, 6]


@lru_cache(maxsize=1)
def _inputs():
    debt = load_expansion_table("treasury_marketable_securities")
    debt = debt.loc[
        debt.record_date.eq("2024-12-31")
        & debt.security_class.eq("Bills Maturity Value")
        & debt.maturity_date.between(START, END)
    ].copy()
    ident = "security_identifier_or_total_label"
    nonempty = debt.loc[debt.outstanding_million_usd.notna()].copy()
    if (len(debt) != 59 or debt[ident].nunique() != 25 or len(nonempty) != 25
            or not nonempty[ident].is_unique
            or (nonempty.outstanding_million_usd <= 0).any()):
        raise ValueError("MSPD detail/outstanding support changed")
    due = (nonempty.groupby("maturity_date").outstanding_million_usd.sum() / 1e6).to_dict()

    auctions = load_expansion_table("treasury_auctions")
    auctions = auctions.loc[
        auctions.security_type.eq("Bill")
        & auctions.security_term.eq("4-Week")
        & auctions.issue_date.isin(ISSUE_DATES)
    ].sort_values("issue_date")
    if tuple(auctions.issue_date) != ISSUE_DATES or not auctions.issue_date.is_unique:
        raise ValueError("Candidate issue calendar changed")
    maturity = tuple(auctions.maturity_date)
    prices = tuple(auctions.price_per100.astype(float) / 100)
    if any(not (0 < p < 1) for p in prices):
        raise ValueError("Bill discount price changed")

    rates = load_expansion_table("nyfed_reference_rates")
    rates = rates.loc[rates["Rate Type"].eq("SOFR"), ["Effective Date", "Rate (%)"]].copy()
    rates["Effective Date"] = pd.to_datetime(rates["Effective Date"])
    rates = rates.sort_values("Effective Date")
    if rates["Effective Date"].duplicated().any():
        raise ValueError("Duplicate SOFR effective date")
    daily = pd.DataFrame({"day": pd.date_range(START, END)})
    daily = pd.merge_asof(daily, rates, left_on="day", right_on="Effective Date", direction="backward")
    if daily["Rate (%)"].isna().any():
        raise ValueError("Missing retrospective SOFR fixing")
    events = tuple(sorted(set(due) | set(ISSUE_DATES) | {d for d in maturity if d <= END} | {END}))
    factors = []
    previous = "2024-12-31"
    for day in events:
        mask = daily.day.between(pd.Timestamp(previous) + pd.Timedelta(days=1), pd.Timestamp(day))
        fixing = daily.loc[mask, "Rate (%)"].to_numpy(float) / 100
        factors.append((float(np.sum(fixing + .03) / 360), float(np.sum(fixing + .06) / 360)))
        previous = day
    return due, maturity, prices, events, tuple(factors)


def _simulate(lots, *, face_cap=2.5, emergency=False, restore=False):
    due, maturity, prices, events, factors = _inputs()
    if len(lots) != 4 or any(not isinstance(n, int) or not 0 <= n <= LOT_MAX for n in lots):
        raise ValueError("Invalid note lot tuple")
    cash = OPENING_CASH_MN
    base = extra = live_face = interest = fee = 0.
    peak_preissue = peak_extra = 0.
    first_extra_date = None
    rows = {}

    for i, day in enumerate(events):
        def clear(delta):
            nonlocal cash, base, extra, fee, peak_extra, first_extra_date
            cash += delta
            if cash < -1e-12:
                shortage = -cash
                cash = 0.
                draw = min(shortage, BASE_LIMIT_MN - base)
                base += draw
                shortage -= draw
                if shortage > 1e-10:
                    if not emergency:
                        return False
                    extra += shortage
                    if first_extra_date is None:
                        first_extra_date = day
                        fee += EMERGENCY_FEE_MN
                        if not clear(-EMERGENCY_FEE_MN):
                            return False
            elif cash > 0:
                repay = min(cash, extra)
                extra -= repay
                cash -= repay
                repay = min(cash, base)
                base -= repay
                cash -= repay
            peak_extra = max(peak_extra, extra)
            return True

        charge = base * factors[i][0] + extra * factors[i][1]
        interest += charge
        if not clear(-charge) or not clear(-due.get(day, 0.)):
            return None
        for j, maturity_day in enumerate(maturity):
            if maturity_day == day:
                if not clear(-LOT_FACE_MN * lots[j]):
                    return None
                live_face -= LOT_FACE_MN * lots[j]
        preissue = base + extra
        if day in ISSUE_DATES:
            peak_preissue = max(peak_preissue, preissue)
        for j, issue_day in enumerate(ISSUE_DATES):
            if issue_day == day:
                live_face += LOT_FACE_MN * lots[j]
                current_cap = 2.5 if restore and day >= "2025-03-01" else face_cap
                if live_face > current_cap + 1e-9:
                    return None
                if not clear(LOT_FACE_MN * lots[j] * prices[j]):
                    return None
        rows[day] = (cash, base, extra, live_face, preissue)

    discount = sum(LOT_FACE_MN * n * (1 - p) for n, p in zip(lots, prices))
    cost = discount + interest + fee
    if abs(cash - base - extra - live_face - (OPENING_CASH_MN - sum(due.values()) - cost)) > 1e-8:
        raise ValueError("Issuer cash/debt conservation failed")
    return {
        "lots": lots, "cost": cost, "cash": cash, "base": base, "extra": extra,
        "live_face": live_face, "peak_preissue": peak_preissue,
        "peak_extra": peak_extra, "first_extra_date": first_extra_date,
        "fee": fee, "rows": rows,
    }


def _best(*, prefix=(), face_cap=2.5, emergency=False, restore=False):
    candidates = []
    for future in product(range(LOT_MAX + 1), repeat=4 - len(prefix)):
        lots = tuple(prefix) + future
        result = _simulate(lots, face_cap=face_cap, emergency=emergency, restore=restore)
        if result is not None:
            candidates.append(result)
    if not candidates:
        raise ValueError("No feasible bill refinancing plan")
    return min(candidates, key=lambda z: (z["cost"], z["lots"]))


@lru_cache(maxsize=1)
def analysis():
    due, _, _, _, _ = _inputs()
    initial = _best()
    tightened = _best(prefix=initial["lots"][:2], face_cap=2.3, emergency=True)
    restored = _best(prefix=tightened["lots"][:3], face_cap=2.3, emergency=True, restore=True)
    if (initial["rows"]["2025-01-28"][:2] != tightened["rows"]["2025-01-28"][:2]
            or tightened["rows"]["2025-02-27"] != restored["rows"]["2025-02-27"]):
        raise ValueError("Executed-state branch replay changed")
    return {
        "legacy_due_mn": sum(due.values()),
        "last_legacy_maturity_date": max(due),
        "first_issue_pre_cash_gap_mn": initial["rows"][ISSUE_DATES[0]][4],
        "jan_7_lots": initial["lots"][0],
        "jan_28_lots": initial["lots"][1],
        "feb_18_lots": initial["lots"][2],
        "mar_4_lots": initial["lots"][3],
        "initial_total_cost_mn": initial["cost"],
        "initial_peak_preissue_bridge_mn": initial["peak_preissue"],
        "post_jan_28_cash_mn": initial["rows"][ISSUE_DATES[1]][0],
        "post_jan_28_base_bridge_mn": initial["rows"][ISSUE_DATES[1]][1],
        "tightened_feb_18_lots": tightened["lots"][2],
        "tightened_mar_4_lots": tightened["lots"][3],
        "first_emergency_draw_date": tightened["first_extra_date"],
        "tightened_peak_emergency_mn": tightened["peak_extra"],
        "tightened_total_cost_mn": tightened["cost"],
        "restored_mar_4_lots": restored["lots"][3],
        "restored_total_cost_mn": restored["cost"],
        "restored_end_base_bridge_mn": restored["base"],
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
