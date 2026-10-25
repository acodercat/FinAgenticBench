"""Gross HMDA originations and Z.1 household mortgage transactions.

HMDA's restricted 2024 cohort has 3,887,656 reported originations totaling
1426.415070 USD bn. Ordinary refinancing contributes 160.540365 bn, cash-out
refinancing 128.749150 bn, and purchase 1097.545825 bn. The 208,420 residual
rows, totaling 39.579730 bn, are neither of those refinancing classes nor
purchase. Treating all non-refinance originations as purchase changes the
purchase amount from 1097.545825 to 1137.125555 bn.

The first supplied Z.1 vintage with all 2024 quarters is 2025-03-13 and sums
to 344.006 bn; the latest supplied vintage is 2025-09-11 and sums to
367.886 bn. FA is a seasonally adjusted annual rate; summing its four
quarterly entries instead of the FU unadjusted transactions changes the object.
FU appears in both fu101 and fu218 and is counted only once. Neither HMDA gross
origination nor the restricted purchase gross is a same-population net financing
flow or the change in outstanding debt. This is a baseline, as the queries pin
the FU and HMDA populations rather than leave a live diagnostic trap.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("hmda_cohort_rows", 0, "the 2024 HMDA cohort's reported-row count as an integer"),
    ("hmda_gross_bn", 4, "the cohort's gross loan amount in USD billions"),
    ("ordinary_refinance_rows", 0, "the ordinary-refinance reported-row count as an integer"),
    ("ordinary_refinance_bn", 4, "the ordinary-refinance gross amount in USD billions"),
    ("cashout_refinance_rows", 0, "the cash-out-refinance reported-row count as an integer"),
    ("cashout_refinance_bn", 4, "the cash-out-refinance gross amount in USD billions"),
    ("latest_z1_release", None, "the latest supplied Z.1 release date as an ISO YYYY-MM-DD string"),
    ("latest_z1_transactions_bn", 4, "the annual sum of unadjusted Z.1 transactions in USD billions"),
    ("first_full_z1_release", None, "the first supplied Z.1 release covering all four quarters as an ISO YYYY-MM-DD string"),
    ("first_full_z1_transactions_bn", 4, "the annual sum of unadjusted Z.1 transactions in USD billions"),
    ("purchase_gross_bn", 4, "the carried cohort's home-purchase gross amount in USD billions"),
)
variables = [
    Variable(name, None, f"Store {description}, rounded to {precision} decimals from unrounded source rows." if precision else f"Store {description}.")
    for name, precision, description in OUTPUTS
]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = (
    tuple(name for name, *_ in OUTPUTS[:2]),
    tuple(name for name, *_ in OUTPUTS[2:6]),
    tuple(name for name, *_ in OUTPUTS[6:8]),
    tuple(name for name, *_ in OUTPUTS[8:]),
)


@lru_cache(maxsize=1)
def _cohort():
    data = load_expansion_table("hmda")
    cohort = data.loc[
        data.activity_year.eq("2024")
        & data.action_taken.eq("1")
        & data.lien_status.eq("1")
        & data.occupancy_type.eq("1")
        & data.business_or_commercial_purpose.eq("2")
        & data.derived_dwelling_category.eq("Single Family (1-4 Units):Site-Built")
    ].copy()
    if cohort.empty or cohort.loan_amount.isna().any() or (cohort.loan_amount <= 0).any():
        raise ValueError("HMDA cohort is empty or has invalid loan amounts")
    return cohort


@lru_cache(maxsize=1)
def _z1():
    data = load_expansion_table("fed_z1_vintages")
    dates = {"2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"}
    selected = data.loc[
        data["series"].eq("FU153165105.Q")
        & data["table"].eq("fu101")
        & data["frequency"].eq("Q")
        & data["date"].isin(dates)
    ].copy()
    if selected.duplicated(["vintage", "date"]).any():
        raise ValueError("duplicate Z.1 vintage-quarter observation")
    complete = selected.groupby("vintage").filter(lambda g: set(g["date"]) == dates and len(g) == 4)
    if complete.empty or complete.value.isna().any():
        raise ValueError("no complete 2024 Z.1 vintage")
    amounts = complete.groupby("vintage")["value"].sum().sort_index()
    return amounts


@lru_cache(maxsize=1)
def analysis():
    cohort = _cohort()
    ordinary = cohort.loc[cohort.loan_purpose.eq("31")]
    cashout = cohort.loc[cohort.loan_purpose.eq("32")]
    purchase = cohort.loc[cohort.loan_purpose.eq("1")]
    z1 = _z1()
    bn = 1_000_000_000
    return {
        "hmda_cohort_rows": len(cohort),
        "hmda_gross_bn": float(cohort.loan_amount.sum() / bn),
        "ordinary_refinance_rows": len(ordinary),
        "ordinary_refinance_bn": float(ordinary.loan_amount.sum() / bn),
        "cashout_refinance_rows": len(cashout),
        "cashout_refinance_bn": float(cashout.loan_amount.sum() / bn),
        "latest_z1_release": str(z1.index[-1]),
        "latest_z1_transactions_bn": float(z1.iloc[-1] / 1000),
        "first_full_z1_release": str(z1.index[0]),
        "first_full_z1_transactions_bn": float(z1.iloc[0] / 1000),
        "purchase_gross_bn": float(purchase.loan_amount.sum() / bn),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _check(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [PRECISION[n] for n in names])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _check(outputs, names))
    for i, names in enumerate(TURNS, 1)
}
