"""Trace the largest monthly debt increase through marketable stock and bill flows.

Each MSPD month end is aligned to the latest Debt-to-the-Penny observation on or
before it.  Across 24 snapshots, 2025-06-30 to 2025-07-31 has the largest total-
debt increase: about $705.518bn, split between public and intragovernmental debt.
MSPD marketable stock increases about $312.677bn and reconciles exactly across
five published class totals; Bills drive the increase.  Gross bill auction flows
issued in the interval are much larger than the net bill-stock change because
maturities and other stock mechanics intervene.

The measured interactions cover exact versus prior-observation debt alignment,
total versus public-debt ranking, total-row versus detail aggregation, issue versus
auction-date flows, offering versus accepted weighting, and 3-month versus 6-month
short-tenor context.  The query pins every convention, so these are regression probes for
a hard baseline.  The case never treats auction flow, marketable stock and total
public debt as one accounting identity.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import validate_ordered_outputs, turn_validator

# The governed MSPD table starts 2019-01-31; the case is pinned to the monthly
# snapshots the query names.
SNAPSHOT_WINDOW = ("2024-01-31", "2025-12-31")

CLASS_TOTALS = {
    "Bills Maturity Value": "Total Treasury Bills",
    "Notes": "Total Treasury Notes",
    "Bonds": "Total Treasury Bonds",
    "Inflation-Protected Securities": "Total Treasury TIPS",
    "Floating Rate Notes": "Total Treasury Floating Rate Notes",
}
CLASS_DISPLAY = {
    "Bills Maturity Value": "Bills",
    "Notes": "Notes",
    "Bonds": "Bonds",
    "Inflation-Protected Securities": "TIPS",
    "Floating Rate Notes": "Floating Rate Notes",
}
AUCTION_SECURITY_TYPE = {
    "Bills Maturity Value": "Bill",
    "Notes": "Note",
    "Bonds": "Bond",
    "Inflation-Protected Securities": "TIPS",
    "Floating Rate Notes": "FRN",
}


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "aligned_mspd_snapshot_count", "selected_interval_start_mspd_date",
    "selected_interval_end_mspd_date", "selected_interval_start_debt_observation_date",
    "selected_interval_end_debt_observation_date", "selected_total_debt_change_usd_billions",
    "selected_public_debt_change_usd_billions", "selected_intragovernmental_change_usd_billions",
    "selected_total_debt_change_margin_usd_billions",
]
TURN_2_NAMES = [
    "start_total_marketable_outstanding_usd_billions",
    "end_total_marketable_outstanding_usd_billions", "bills_outstanding_change_usd_billions",
    "notes_outstanding_change_usd_billions", "bonds_outstanding_change_usd_billions",
    "tips_outstanding_change_usd_billions", "frn_outstanding_change_usd_billions",
    "largest_marketable_change_class", "class_change_reconciliation_residual_usd_millions",
    "marketable_change_share_of_public_debt_change_pct",
]
TURN_3_NAMES = [
    "selected_class_issue_event_count", "selected_class_issue_distinct_cusip_count",
    "selected_class_issue_offering_usd_billions", "selected_class_issue_total_accepted_usd_billions",
    "largest_selected_class_term_by_offering", "largest_selected_class_term_issue_count",
    "largest_selected_class_term_offering_usd_billions", "largest_selected_class_term_offering_margin_usd_billions",
    "selected_class_offering_to_stock_change_ratio",
]
TURN_4_NAMES = [
    "curve_start_observation_date", "curve_end_observation_date",
    "curve_start_three_month_yield_pct", "curve_end_three_month_yield_pct",
    "curve_start_one_year_yield_pct", "curve_end_one_year_yield_pct",
    "selected_class_accepted_weighted_high_investment_rate_pct",
]

variables = [
    _v("aligned_mspd_snapshot_count", "Store the aligned MSPD snapshot count as an integer."),
    _v("selected_interval_start_mspd_date", "Store the selected interval's starting MSPD date as an ISO YYYY-MM-DD string."),
    _v("selected_interval_end_mspd_date", "Store the selected interval's ending MSPD date as an ISO YYYY-MM-DD string."),
    _v("selected_interval_start_debt_observation_date", "Store the starting aligned Debt-to-the-Penny observation date as an ISO YYYY-MM-DD string."),
    _v("selected_interval_end_debt_observation_date", "Store the ending aligned Debt-to-the-Penny observation date as an ISO YYYY-MM-DD string."),
    _v("selected_total_debt_change_usd_billions", "Store the selected interval's total public-debt-outstanding change in USD billions, rounded to 6 decimals. Compute it from its own unrounded endpoints, not from the two component changes reported here."),
    _v("selected_public_debt_change_usd_billions", "Store the selected interval's debt-held-by-public change in USD billions, rounded to 6 decimals. Compute it from its own unrounded endpoints, not from the total and the other component reported here."),
    _v("selected_intragovernmental_change_usd_billions", "Store the selected interval's intragovernmental-holdings change in USD billions, rounded to 6 decimals. Compute it from its own unrounded endpoints, not from the total and the other component reported here."),
    _v("selected_total_debt_change_margin_usd_billions", "Store selected total-debt change minus the runner-up interval change in USD billions, rounded to 6 decimals."),
    _v("start_total_marketable_outstanding_usd_billions", "Store starting total marketable outstanding in USD billions, rounded to 6 decimals."),
    _v("end_total_marketable_outstanding_usd_billions", "Store ending total marketable outstanding in USD billions, rounded to 6 decimals."),
    _v("bills_outstanding_change_usd_billions", "Store the Treasury Bills class-total outstanding change in USD billions, rounded to 6 decimals."),
    _v("notes_outstanding_change_usd_billions", "Store the Treasury Notes class-total outstanding change in USD billions, rounded to 6 decimals."),
    _v("bonds_outstanding_change_usd_billions", "Store the Treasury Bonds class-total outstanding change in USD billions, rounded to 6 decimals."),
    _v("tips_outstanding_change_usd_billions", "Store the Treasury TIPS class-total outstanding change in USD billions, rounded to 6 decimals."),
    _v("frn_outstanding_change_usd_billions", "Store the Treasury Floating Rate Notes class-total outstanding change in USD billions, rounded to 6 decimals."),
    _v("largest_marketable_change_class", "Store the marketable class with the greatest outstanding increase as Bills | Notes | Bonds | TIPS | Floating Rate Notes."),
    _v("class_change_reconciliation_residual_usd_millions", "Store total marketable change minus the sum of the five class-total changes in USD millions, rounded to 6 decimals."),
    _v("marketable_change_share_of_public_debt_change_pct", "Store total marketable change divided by debt-held-by-public change in percent, rounded to 4 decimals."),
    _v("selected_class_issue_event_count", "Store the selected class's issued auction-record count as an integer."),
    _v("selected_class_issue_distinct_cusip_count", "Store the selected class's distinct issued CUSIP count as an integer."),
    _v("selected_class_issue_offering_usd_billions", "Store the selected class's aggregate offering amount in USD billions, rounded to 6 decimals."),
    _v("selected_class_issue_total_accepted_usd_billions", "Store the selected class's aggregate total accepted amount in USD billions, rounded to 6 decimals."),
    _v("largest_selected_class_term_by_offering", "Store the selected class's security term with the greatest aggregate offering amount as a string."),
    _v("largest_selected_class_term_issue_count", "Store that security term's issue-event count as an integer."),
    _v("largest_selected_class_term_offering_usd_billions", "Store that security term's aggregate offering in USD billions, rounded to 6 decimals."),
    _v("largest_selected_class_term_offering_margin_usd_billions", "Store the leading term's offering minus the runner-up term's offering in USD billions, rounded to 6 decimals."),
    _v("selected_class_offering_to_stock_change_ratio", "Store selected-class offering divided by its outstanding-stock increase as a ratio, rounded to 4 decimals."),
    _v("curve_start_observation_date", "Store the starting Treasury curve observation date as an ISO YYYY-MM-DD string."),
    _v("curve_end_observation_date", "Store the ending Treasury curve observation date as an ISO YYYY-MM-DD string."),
    _v("curve_start_three_month_yield_pct", "Store the starting 3-month Treasury par yield in percent, rounded to 2 decimals."),
    _v("curve_end_three_month_yield_pct", "Store the ending 3-month Treasury par yield in percent, rounded to 2 decimals."),
    _v("curve_start_one_year_yield_pct", "Store the starting 1-year Treasury par yield in percent, rounded to 2 decimals."),
    _v("curve_end_one_year_yield_pct", "Store the ending 1-year Treasury par yield in percent, rounded to 2 decimals."),
    _v("selected_class_accepted_weighted_high_investment_rate_pct", "Store the selected class's total-accepted-weighted high investment rate in percent, rounded to 4 decimals."),
]


def _aligned_debt_snapshots(marketable, debt):
    debt = debt.copy()
    debt["record_date"] = pd.to_datetime(debt.record_date)
    rows = []
    for snapshot_date in sorted(pd.to_datetime(marketable.record_date).unique()):
        eligible = debt.loc[debt.record_date.le(snapshot_date)].sort_values("record_date")
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        rows.append({
            "mspd_date": pd.Timestamp(snapshot_date), "debt_date": pd.Timestamp(row.record_date),
            "public": float(row.debt_held_by_public_usd) / 1e9,
            "intra": float(row.intragovernmental_holdings_usd) / 1e9,
            "total": float(row.total_public_debt_outstanding_usd) / 1e9,
        })
    aligned = pd.DataFrame(rows).sort_values("mspd_date").reset_index(drop=True)
    for column in ["public", "intra", "total"]:
        aligned[f"{column}_change"] = aligned[column].diff()
    return aligned


@lru_cache(maxsize=1)
def ground_truth():
    marketable = load_expansion_table("treasury_marketable_securities").copy()
    marketable["record_date"] = pd.to_datetime(marketable.record_date)
    marketable = marketable.loc[marketable.record_date.between(*SNAPSHOT_WINDOW)]
    debt = load_expansion_table("treasury_debt_to_penny")
    aligned = _aligned_debt_snapshots(marketable, debt)
    ranked = aligned.dropna(subset=["total_change"]).sort_values(
        ["total_change", "mspd_date"], ascending=[False, True]
    ).reset_index(drop=True)
    selected, runner = ranked.iloc[0], ranked.iloc[1]
    selected_index = aligned.index[aligned.mspd_date.eq(selected.mspd_date)][0]
    if selected_index == 0:
        raise ValueError("selected interval has no prior MSPD snapshot")
    start = aligned.iloc[selected_index - 1]

    def total_value(snapshot_date, security_class, label=None):
        rows = marketable.loc[
            marketable.record_date.eq(snapshot_date) & marketable.security_class.eq(security_class)
        ]
        if label is not None:
            rows = rows.loc[rows.security_identifier_or_total_label.eq(label)]
        if len(rows) != 1:
            raise ValueError(f"unexpected MSPD total row for {security_class}")
        return float(rows.iloc[0].outstanding_million_usd) / 1000

    start_total = total_value(start.mspd_date, "Total Marketable")
    end_total = total_value(selected.mspd_date, "Total Marketable")
    marketable_change = end_total - start_total
    class_changes = {}
    for security_class, label in CLASS_TOTALS.items():
        class_changes[security_class] = (
            total_value(selected.mspd_date, security_class, label)
            - total_value(start.mspd_date, security_class, label)
        )
    class_order = sorted(class_changes, key=lambda key: (-class_changes[key], CLASS_DISPLAY[key]))
    driver = class_order[0]
    class_residual = marketable_change - sum(class_changes.values())

    auctions = load_expansion_table("treasury_auctions").copy()
    auctions["issue_date"] = pd.to_datetime(auctions.issue_date)
    selected_issues = auctions.loc[
        auctions.security_type.eq(AUCTION_SECURITY_TYPE[driver])
        & auctions.issue_date.gt(start.mspd_date)
        & auctions.issue_date.le(selected.mspd_date)
    ].copy()
    for column in ["offering_amt", "total_accepted", "high_investment_rate"]:
        selected_issues[column] = pd.to_numeric(selected_issues[column], errors="coerce")
    selected_issues = selected_issues.dropna(subset=["offering_amt", "total_accepted", "high_investment_rate"])
    term_totals = selected_issues.groupby("security_term", as_index=False).agg(
        issue_count=("cusip", "size"), offering=("offering_amt", "sum")
    ).sort_values(["offering", "security_term"], ascending=[False, True]).reset_index(drop=True)
    term_leader, term_runner = term_totals.iloc[0], term_totals.iloc[1]
    weighted_rate = float(np.average(
        selected_issues.high_investment_rate, weights=selected_issues.total_accepted
    ))

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    curve_start = curve.loc[curve.Date.le(start.mspd_date)].sort_values("Date").iloc[-1]
    curve_end = curve.loc[curve.Date.le(selected.mspd_date)].sort_values("Date").iloc[-1]
    start_3m, end_3m = float(curve_start["3 Mo"]), float(curve_end["3 Mo"])
    start_1y, end_1y = float(curve_start["1 Yr"]), float(curve_end["1 Yr"])
    start_slope, end_slope = start_3m - start_1y, end_3m - end_1y

    return (
        len(aligned),
        start.mspd_date.strftime("%Y-%m-%d"),
        selected.mspd_date.strftime("%Y-%m-%d"),
        start.debt_date.strftime("%Y-%m-%d"),
        selected.debt_date.strftime("%Y-%m-%d"),
        float(selected.total_change),
        float(selected.public_change),
        float(selected.intra_change),
        float(selected.total_change - runner.total_change),
        start_total,
        end_total,
        class_changes["Bills Maturity Value"],
        class_changes["Notes"],
        class_changes["Bonds"],
        class_changes["Inflation-Protected Securities"],
        class_changes["Floating Rate Notes"],
        CLASS_DISPLAY[driver],
        class_residual * 1000,
        marketable_change / float(selected.public_change) * 100,
        len(selected_issues),
        selected_issues.cusip.nunique(),
        float(selected_issues.offering_amt.sum()) / 1e9,
        float(selected_issues.total_accepted.sum()) / 1e9,
        str(term_leader.security_term),
        int(term_leader.issue_count),
        float(term_leader.offering) / 1e9,
        float(term_leader.offering - term_runner.offering) / 1e9,
        float(selected_issues.offering_amt.sum()) / 1e9 / class_changes[driver],
        curve_start.Date.strftime("%Y-%m-%d"),
        curve_end.Date.strftime("%Y-%m-%d"),
        start_3m,
        end_3m,
        start_1y,
        end_1y,
        weighted_rate,
    )


DECIMALS = [
    0, None, None, None, None, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, None, 6, 4,
    0, 0, 6, 6, None, 0, 6, 6, 4,
    None, None, 2, 2, 2, 2, 4,
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [decimals[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_debt_interval": turn_validator(validate_turn_1),
    "validate_marketable_stock": turn_validator(validate_turn_2),
    "validate_selected_class_auction_flow": turn_validator(validate_turn_3),
    "validate_short_curve_context": turn_validator(validate_turn_4),
}
