"""Reconcile Apple's disclosed commitment ladders and run a curve sensitivity.

This is a hard baseline with explicit fact-grain audit outputs.  The measured alternatives retain both a physical-row sum and a distinct-value sum as measured wrong
paths, plus the dimensioned-component sum as a documented numerical equivalent.
Those alternatives test the XBRL surface without misclassifying a prompt that
asks the analyst to interpret its row-grain evidence as a diagnostic trap.

Gross debt is the face amount, which the query defines, not the net carrying
amount. The filing reports both: the face amount on four rows, two of them
undimensioned and both 91.281 billion, and the net carrying amount on a single
undimensioned row of 90.678 billion. The 0.603 billion between them is the
unamortized discount and issuance costs (0.309) and the hedge-accounting
adjustments. Reading the net figure reports counts of 1, 1 and 1, a gross of
90.678 and a difference of 603.0 million where the face amount reconciles to 0.0.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


DEBT_CONCEPTS = [
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",
]
PURCHASE_CONCEPTS = [
    "UnrecordedUnconditionalPurchaseObligationBalanceOnFirstAnniversary",
    "UnrecordedUnconditionalPurchaseObligationBalanceOnSecondAnniversary",
    "UnrecordedUnconditionalPurchaseObligationBalanceOnThirdAnniversary",
    "UnrecordedUnconditionalPurchaseObligationBalanceOnFourthAnniversary",
    "UnrecordedUnconditionalPurchaseObligationBalanceOnFifthAnniversary",
    "UnrecordedUnconditionalPurchaseObligationDueAfterFiveYears",
]
TIMES = [1, 2, 3, 4, 5, 10]

T1 = [
    "debt_year_1_usd_billions", "debt_year_2_usd_billions",
    "debt_year_3_usd_billions", "debt_year_4_usd_billions",
    "debt_year_5_usd_billions", "debt_after_year_5_usd_billions",
    "debt_maturity_sum_usd_billions", "gross_debt_fact_physical_row_count",
    "gross_debt_undimensioned_physical_row_count",
    "gross_debt_distinct_undimensioned_value_count",
    "gross_long_term_debt_usd_billions", "maturity_sum_minus_gross_debt_usd_millions",
]
T2 = [
    "purchase_year_1_usd_billions", "purchase_year_2_usd_billions",
    "purchase_year_3_usd_billions", "purchase_year_4_usd_billions",
    "purchase_year_5_usd_billions", "purchase_after_year_5_usd_billions",
    "purchase_obligation_sum_usd_billions", "combined_year_1_commitments_usd_billions",
    "combined_undiscounted_commitments_usd_billions",
]
T3 = [
    "curve_observation_date", "five_year_par_yield_pct",
    "base_debt_present_value_usd_billions", "base_purchase_present_value_usd_billions",
    "base_combined_present_value_usd_billions",
]
T4 = [
    "shocked_flat_discount_rate_pct", "shocked_debt_present_value_usd_billions",
    "shocked_purchase_present_value_usd_billions", "shocked_combined_present_value_usd_billions",
    "combined_pv_change_pct",
]


def _v(name, description):
    return Variable(name, None, description)


variables = [
    *[_v(name, "Store the disclosed debt-maturity amount in USD billions, rounded to 3 decimals.") for name in T1[:6]],
    _v(T1[6], "Store the sum of the six debt-maturity buckets in USD billions, rounded to 3 decimals."),
    _v(T1[7], "Store the physical matching fact-row count as an integer."),
    _v(T1[8], "Store the matching undimensioned physical-row count as an integer."),
    _v(T1[9], "Store the distinct undimensioned numeric-value count as an integer."),
    _v(T1[10], "Store consolidated gross long-term debt in USD billions, rounded to 3 decimals."),
    _v(T1[11], "Store maturity sum minus consolidated gross debt in USD millions, rounded to 1 decimal."),
    *[_v(name, "Store the disclosed purchase-obligation amount in USD billions, rounded to 3 decimals.") for name in T2[:6]],
    _v(T2[6], "Store the six-bucket purchase-obligation sum in USD billions, rounded to 3 decimals."),
    _v(T2[7], "Store combined first-year commitments in USD billions, rounded to 3 decimals."),
    _v(T2[8], "Store combined undiscounted commitments in USD billions, rounded to 3 decimals."),
    _v(T3[0], "Store the curve observation date as an ISO YYYY-MM-DD string."),
    _v(T3[1], "Store the source-reported five-year par yield in percent, rounded to 2 decimals."),
    *[_v(name, "Store the mechanical present value in USD billions, rounded to 3 decimals. "
               "Compute each from the unrounded discounted cash flows, not from the other two "
               "present values reported here.") for name in T3[2:5]],
    _v(T4[0], "Store the shocked flat discount rate in percent, rounded to 2 decimals."),
    *[_v(name, "Store the shocked mechanical present value in USD billions, rounded to 3 decimals. "
               "Compute each from the unrounded discounted cash flows, not from the other two "
               "present values reported here.") for name in T4[1:4]],
    _v(T4[4], "Store combined shocked-minus-base present value divided by base combined present value in percent, rounded to 4 decimals."),
]


APPLE_CIK = "320193"
UNDIMENSIONED = "0x00000000"


def _apple_fy2025_10k(notes):
    """Apple's fiscal-2025 Form 10-K facts from the universe-wide FS&Notes table."""
    return notes.loc[
        notes.cik.astype(str).str.lstrip("0").eq(APPLE_CIK)
        & notes.form.astype(str).eq("10-K")
        & notes.ddate.astype(str).eq("2025-09-30")
    ]


def _one_value(notes, concept):
    rows = notes.loc[
        notes.tag.astype(str).eq(concept)
        & notes.qtrs.astype(str).eq("0")
        & notes.uom.astype(str).eq("USD")
        & notes.dimh.astype(str).eq(UNDIMENSIONED)
    ]
    values = rows.value.drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"expected one distinct undimensioned value for {concept}, found {len(values)}")
    return float(values.iloc[0]) / 1e9


@lru_cache(maxsize=1)
def ground_truth():
    notes = _apple_fy2025_10k(load_expansion_table("sec_notes"))
    debt = [_one_value(notes, concept) for concept in DEBT_CONCEPTS]
    purchase = [_one_value(notes, concept) for concept in PURCHASE_CONCEPTS]
    carrying = notes.loc[
        notes.tag.astype(str).eq("DebtInstrumentCarryingAmount")
        & notes.qtrs.astype(str).eq("0")
        & notes.uom.astype(str).eq("USD")
    ]
    undimensioned = carrying.loc[carrying.dimh.astype(str).eq(UNDIMENSIONED)]
    distinct = undimensioned.value.drop_duplicates()
    if len(distinct) != 1:
        raise ValueError("consolidated gross-debt fact is not unique after exact-value collapse")
    gross = float(distinct.iloc[0]) / 1e9

    curve = load_expansion_table("treasury_yield_curve").copy()
    curve["Date"] = pd.to_datetime(curve.Date)
    curve_row = curve.loc[curve.Date.le("2025-09-30")].sort_values("Date").iloc[-1]
    base_rate = float(curve_row["5 Yr"]) / 100

    def pv(values, rate):
        return sum(value / (1 + rate) ** time for value, time in zip(values, TIMES, strict=True))

    debt_total, purchase_total = sum(debt), sum(purchase)
    debt_pv, purchase_pv = pv(debt, base_rate), pv(purchase, base_rate)
    shocked_rate = base_rate + 0.01
    shocked_debt, shocked_purchase = pv(debt, shocked_rate), pv(purchase, shocked_rate)
    base_combined = debt_pv + purchase_pv
    shocked_combined = shocked_debt + shocked_purchase
    return (
        *debt, debt_total, len(carrying), len(undimensioned), len(distinct), gross,
        (debt_total - gross) * 1000,
        *purchase, purchase_total, debt[0] + purchase[0], debt_total + purchase_total,
        curve_row.Date.strftime("%Y-%m-%d"), float(curve_row["5 Yr"]),
        debt_pv, purchase_pv, base_combined,
        shocked_rate * 100, shocked_debt, shocked_purchase, shocked_combined,
        (shocked_combined / base_combined - 1) * 100,
    )


DECIMALS = [
    *([3] * 7), 0, 0, 0, 3, 1,
    *([3] * 9), None, 2, 3, 3, 3,
    2, 3, 3, 3, 4,
]


def _validate(outputs, names):
    truth = dict(zip((v.name for v in variables), ground_truth()))
    places = dict(zip((v.name for v in variables), DECIMALS))
    by_name = {v.name: v for v in variables}
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [places[n] for n in names])


def validate_turn_1(outputs): return _validate(outputs, T1)
def validate_turn_2(outputs): return _validate(outputs, T2)
def validate_turn_3(outputs): return _validate(outputs, T3)
def validate_turn_4(outputs): return _validate(outputs, T4)
def validate(outputs): return _validate(outputs, [v.name for v in variables])


validators = {
    "validate_debt_ladder": turn_validator(validate_turn_1),
    "validate_purchase_ladder": turn_validator(validate_turn_2),
    "validate_base_discount": turn_validator(validate_turn_3),
    "validate_parallel_shock": turn_validator(validate_turn_4),
}
