"""Trace a Regulation Crowdfunding amendment into a Regulation D market window.

The case selects the largest first-original-to-latest-amendment increase in a
2025 Form C file-number lineage, inspects the latest issuer financial snapshot,
then uses the dynamically selected amendment date to define a thirty-day initial
Form D filing window and locate the crowdfunding maximum within that distribution.
The exemption populations remain distinct; the cross-source comparison is scale
context, not an accounting reconciliation.  This is a hard baseline because all
lineage, finite-amount, window, percentile and tie conventions are explicit.

Fragility: the lineage leader is 0.0225 million USD ahead of the runner-up, on
Form C data that receives late amendments.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


TURN_1_NAMES = [
    "eligible_reg_cf_lineage_count", "largest_increase_file_number",
    "largest_increase_issuer_name", "original_form_c_accession", "original_form_c_filing_date",
    "original_maximum_offering_usd_millions", "latest_form_c_a_accession",
    "latest_form_c_a_filing_date", "latest_maximum_offering_usd_millions",
    "maximum_increase_leader_margin_usd_millions",
]
TURN_2_NAMES = [
    "latest_amendment_target_offering_usd_millions", "latest_amendment_deadline_date",
    "latest_amendment_current_employees", "latest_amendment_assets_usd_millions",
    "latest_amendment_cash_usd_millions", "latest_amendment_debt_usd_millions",
    "latest_amendment_revenue_usd_millions", "latest_amendment_net_income_usd_millions",
    "latest_maximum_offering_to_assets_pct", "latest_maximum_offering_to_cash_pct",
    "latest_debt_to_assets_pct",
]
TURN_3_NAMES = [
    "form_d_window_start_date", "initial_form_d_filing_count",
    "indefinite_initial_form_d_filing_count", "finite_positive_initial_form_d_count",
    "finite_initial_form_d_offering_usd_billions", "finite_initial_form_d_sold_usd_billions",
    "finite_initial_form_d_sold_to_offering_pct", "finite_initial_form_d_median_offering_usd_millions",
    "finite_initial_form_d_q75_offering_usd_millions", "form_d_offering_leader_industry",
    "form_d_offering_leader_industry_count", "form_d_offering_leader_industry_usd_billions",
]
TURN_4_NAMES = [
    "form_d_offerings_at_or_below_selected_maximum_count",
    "selected_maximum_form_d_inclusive_percentile_pct",
    "selected_maximum_to_form_d_median_ratio",
]


def _v(name: str, description: str) -> Variable:
    return Variable(name, None, description)


variables = [
    _v("eligible_reg_cf_lineage_count", "Store the eligible Form C file-number lineage count as an integer."),
    _v("largest_increase_file_number", "Store the selected SEC file number exactly as reported."),
    _v("largest_increase_issuer_name", "Store the selected issuer name exactly as reported in the original Form C."),
    _v("original_form_c_accession", "Store the selected lineage's original Form C accession as a string."),
    _v("original_form_c_filing_date", "Store the original Form C filing date as an ISO YYYY-MM-DD string."),
    _v("original_maximum_offering_usd_millions", "Store the original maximum offering in USD millions, rounded to 6 decimals."),
    _v("latest_form_c_a_accession", "Store the selected lineage's latest Form C/A accession as a string."),
    _v("latest_form_c_a_filing_date", "Store the latest Form C/A filing date as an ISO YYYY-MM-DD string."),
    _v("latest_maximum_offering_usd_millions", "Store the latest maximum offering in USD millions, rounded to 6 decimals."),
    _v("maximum_increase_leader_margin_usd_millions", "Store the increase margin over the runner-up in USD millions, rounded to 6 decimals."),
    _v("latest_amendment_target_offering_usd_millions", "Store the latest amendment's target offering in USD millions, rounded to 6 decimals."),
    _v("latest_amendment_deadline_date", "Store the latest amendment's offering deadline as an ISO YYYY-MM-DD string."),
    _v("latest_amendment_current_employees", "Store the latest amendment's current employee count as an integer."),
    _v("latest_amendment_assets_usd_millions", "Store the latest amendment's most-recent-fiscal-year assets in USD millions, rounded to 6 decimals."),
    _v("latest_amendment_cash_usd_millions", "Store the latest amendment's most-recent-fiscal-year cash and equivalents in USD millions, rounded to 6 decimals."),
    _v("latest_amendment_debt_usd_millions", "Store latest short-term plus long-term debt in USD millions, rounded to 6 decimals."),
    _v("latest_amendment_revenue_usd_millions", "Store the latest amendment's most-recent-fiscal-year revenue in USD millions, rounded to 6 decimals."),
    _v("latest_amendment_net_income_usd_millions", "Store the latest amendment's most-recent-fiscal-year net income in USD millions, rounded to 6 decimals."),
    _v("latest_maximum_offering_to_assets_pct", "Store latest maximum offering divided by assets in percent, rounded to 4 decimals."),
    _v("latest_maximum_offering_to_cash_pct", "Store latest maximum offering divided by cash in percent, rounded to 4 decimals."),
    _v("latest_debt_to_assets_pct", "Store latest debt divided by assets in percent, rounded to 4 decimals."),
    _v("form_d_window_start_date", "Store the first date of the selected thirty-calendar-day Form D window as an ISO YYYY-MM-DD string."),
    _v("initial_form_d_filing_count", "Store the initial Form D filing count in the selected window as an integer."),
    _v("indefinite_initial_form_d_filing_count", "Store the initial Form D count whose total offering is reported as Indefinite as an integer."),
    _v("finite_positive_initial_form_d_count", "Store the finite-positive initial Form D offering count as an integer."),
    _v("finite_initial_form_d_offering_usd_billions", "Store aggregate finite-positive initial Form D offering amount in USD billions, rounded to 6 decimals."),
    _v("finite_initial_form_d_sold_usd_billions", "Store aggregate reported amount sold for that finite-positive cohort in USD billions, rounded to 6 decimals."),
    _v("finite_initial_form_d_sold_to_offering_pct", "Store aggregate sold divided by aggregate offering amount in percent, rounded to 4 decimals."),
    _v("finite_initial_form_d_median_offering_usd_millions", "Store the finite-positive cohort median offering amount in USD millions, rounded to 6 decimals."),
    _v("finite_initial_form_d_q75_offering_usd_millions", "Store the finite-positive cohort 75th-percentile offering amount in USD millions, rounded to 6 decimals."),
    _v("form_d_offering_leader_industry", "Store the industry group with the largest aggregate finite offering amount exactly as reported."),
    _v("form_d_offering_leader_industry_count", "Store the leader industry's finite-positive filing count as an integer."),
    _v("form_d_offering_leader_industry_usd_billions", "Store the leader industry's aggregate finite offering amount in USD billions, rounded to 6 decimals."),
    _v("form_d_offerings_at_or_below_selected_maximum_count", "Store the finite Form D offering count at or below the selected crowdfunding maximum as an integer."),
    _v("selected_maximum_form_d_inclusive_percentile_pct", "Store the selected maximum's inclusive empirical percentile in the finite Form D distribution in percent, rounded to 4 decimals."),
    _v("selected_maximum_to_form_d_median_ratio", "Store selected crowdfunding maximum divided by the Form D median as a ratio, rounded to 4 decimals."),
]


def _lineages() -> pd.DataFrame:
    filings = load_expansion_table("sec_form_c").loc[lambda frame: frame.FILING_DATE.astype(str).str.startswith("2025")]
    records = []
    for file_number, group in filings.groupby("FILE_NUMBER"):
        originals = group.loc[group.SUBMISSION_TYPE.eq("C")].sort_values(
            ["FILING_DATE", "ACCESSION_NUMBER"]
        )
        amendments = group.loc[group.SUBMISSION_TYPE.eq("C/A")].sort_values(
            ["FILING_DATE", "ACCESSION_NUMBER"]
        )
        if originals.empty or amendments.empty:
            continue
        original = originals.iloc[0]
        latest = amendments.iloc[-1]
        if pd.isna(original.MAXIMUMOFFERINGAMOUNT) or pd.isna(
            latest.MAXIMUMOFFERINGAMOUNT
        ):
            continue
        records.append({
            "file_number": file_number,
            "issuer": original.NAMEOFISSUER,
            "original_accession": original.ACCESSION_NUMBER,
            "original_date": original.FILING_DATE,
            "original_maximum": float(original.MAXIMUMOFFERINGAMOUNT),
            "latest_accession": latest.ACCESSION_NUMBER,
            "latest_date": latest.FILING_DATE,
            "latest_maximum": float(latest.MAXIMUMOFFERINGAMOUNT),
            "increase": float(
                latest.MAXIMUMOFFERINGAMOUNT - original.MAXIMUMOFFERINGAMOUNT
            ),
        })
    return pd.DataFrame(records).sort_values(
        ["increase", "latest_date", "file_number"], ascending=[False, True, True]
    ).reset_index(drop=True)


@lru_cache(maxsize=1)
def ground_truth() -> tuple:
    lineages = _lineages()
    leader = lineages.iloc[0]
    form_c = load_expansion_table("sec_form_c").loc[lambda frame: frame.FILING_DATE.astype(str).str.startswith("2025")]
    latest = form_c.loc[
        form_c.ACCESSION_NUMBER.eq(leader.latest_accession)
    ].iloc[0]
    debt = float(
        latest.SHORTTERMDEBTMRECENTFISCALYEAR
        + latest.LONGTERMDEBTRECENTFISCALYEAR
    )
    assets = float(latest.TOTALASSETMOSTRECENTFISCALYEAR)
    cash = float(latest.CASHEQUIMOSTRECENTFISCALYEAR)

    window_end = pd.Timestamp(leader.latest_date)
    window_start = window_end - pd.Timedelta(days=29)
    form_d = load_expansion_table("sec_form_d")
    initial = form_d.loc[
        ~form_d.ISAMENDMENT.fillna(False)
        & form_d.FILING_DATE.between(
            window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        )
    ].copy()
    initial["finite_offering"] = pd.to_numeric(
        initial.TOTALOFFERINGAMOUNT, errors="coerce"
    )
    finite = initial.loc[initial.finite_offering.gt(0)].copy()
    industries = finite.groupby("INDUSTRYGROUPTYPE").agg(
        count=("ACCESSIONNUMBER", "size"),
        offering=("finite_offering", "sum"),
    ).sort_values(["offering", "INDUSTRYGROUPTYPE"], ascending=[False, True])
    industry_name = str(industries.index[0])
    maximum = float(leader.latest_maximum)
    at_or_below = int(finite.finite_offering.le(maximum).sum())
    median = float(finite.finite_offering.median())

    return (
        len(lineages),
        str(leader.file_number),
        str(leader.issuer),
        str(leader.original_accession),
        str(leader.original_date),
        float(leader.original_maximum) / 1e6,
        str(leader.latest_accession),
        str(leader.latest_date),
        maximum / 1e6,
        float(leader.increase - lineages.iloc[1].increase) / 1e6,
        float(latest.OFFERINGAMOUNT) / 1e6,
        str(latest.DEADLINEDATE),
        int(latest.CURRENTEMPLOYEES),
        assets / 1e6,
        cash / 1e6,
        debt / 1e6,
        float(latest.REVENUEMOSTRECENTFISCALYEAR) / 1e6,
        float(latest.NETINCOMEMOSTRECENTFISCALYEAR) / 1e6,
        maximum / assets * 100,
        maximum / cash * 100,
        debt / assets * 100,
        window_start.strftime("%Y-%m-%d"),
        len(initial),
        int(initial.TOTALOFFERINGAMOUNT.eq("Indefinite").sum()),
        len(finite),
        float(finite.finite_offering.sum()) / 1e9,
        float(finite.TOTALAMOUNTSOLD.sum()) / 1e9,
        float(finite.TOTALAMOUNTSOLD.sum() / finite.finite_offering.sum() * 100),
        median / 1e6,
        float(finite.finite_offering.quantile(0.75)) / 1e6,
        industry_name,
        int(industries.iloc[0]["count"]),
        float(industries.iloc[0]["offering"]) / 1e9,
        at_or_below,
        at_or_below / len(finite) * 100,
        maximum / median,
    )


DECIMALS = [
    0, None, None, None, None, 6, None, None, 6, 6,
    6, None, 0, 6, 6, 6, 6, 6, 4, 4, 4,
    None, 0, 0, 0, 6, 6, 4, 6, 6, None, 0, 6,
    0, 4, 4,
]


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(
        outputs, [by_name[name] for name in names],
        [truth[name] for name in names], [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])


validators = {
    "validate_reg_cf_lineage": turn_validator(validate_turn_1),
    "validate_latest_amendment_financials": turn_validator(validate_turn_2),
    "validate_form_d_window": turn_validator(validate_turn_3),
    "validate_cross_exemption_scale": turn_validator(validate_turn_4),
}
