"""Synthetic BDC cash, PIK, debt and borrowing-base ledgers.

The 2024 annual investment schedule supplies only opening positions and PIK
fractions. The credit facility and distribution policy are stated hypothetical
terms. Missing PIK is excluded, not read as zero. The later 2025 comparative
quarters are not used as revisions to the selected annual source version.

Wrong conventions: cashing PIK overstates quarter-one cash by its accrual;
advancing against fresh PIK overstates borrowing capacity by 65% of its accrual;
reversing the paid distribution after a mark shock erases the actual debt cure.
"""

from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


CASE_CIK = "1843162"
REPORT_DATE = "2024-12-31"
FILING_DATE = "2025-03-31"
MATURITY_FLOOR = "2025-06-30"
DEBT_TYPES = {
    "TermLoanMember", "DelayedDrawTermLoanMember", "SeniorSecuredNoteMember",
    "IncrementalTermLoanMember", "SeriesASeniorNoteMember",
}
ADVANCE = 0.65
INITIAL_DEBT_TO_MARK = 0.645
LOAN_SPREAD = 0.05
FINANCE_SPREAD = 0.025
PAYOUT_RATIO = 0.80
SHOCK = 0.15
Q2_EQUITY_BUDGET = 3_000_000.0


def _v(name, description):
    return Variable(name, None, description)


variables = [
    _v("loan_count", "Store the number of eligible original loan positions as an integer."),
    _v("opening_principal_mn", "Store original loan principal in USD millions, rounded to 4 decimals."),
    _v("opening_mark_mn", "Store original loan fair value in USD millions, rounded to 4 decimals."),
    _v("q1_cash_coupon_mn", "Store first-quarter cash loan interest in USD millions, rounded to 4 decimals."),
    _v("q1_capitalized_pik_mn", "Store first-quarter PIK added to loan principal in USD millions, rounded to 4 decimals."),
    _v("q1_financing_interest_mn", "Store first-quarter interest paid on the opening borrowing in USD millions, rounded to 4 decimals."),
    _v("q1_distribution_mn", "Store the first-quarter cash distribution in USD millions, rounded to 3 decimals; carry unrounded cash through later turns."),
    _v("q1_new_borrowing_mn", "Store the additional borrowing drawn at first-quarter end in USD millions, rounded to 4 decimals."),
    _v("q1_external_equity_mn", "Store the minimum first-quarter external equity cash injection in USD millions, rounded to 3 decimals; carry unrounded cash through later turns."),
    _v("shocked_borrowing_base_mn", "Store the borrowing base after the original-mark shock in USD millions, rounded to 3 decimals."),
    _v("shock_cure_equity_mn", "Store immediate additional equity needed to cure the borrowing-base excess, in USD millions rounded to 3 decimals; carry unrounded debt through later turns."),
    _v("q2_early_debt_repayment_mn", "Store the optimal April 1 equity-funded debt repayment in USD millions, rounded to 4 decimals."),
    _v("q2_max_distribution_mn", "Store the maximum second-quarter cash distribution in USD millions, rounded to 3 decimals."),
    _v("q2_ending_principal_mn", "Store loan principal after second-quarter PIK capitalization in USD millions, rounded to 4 decimals."),
]

TURN_NAMES = [
    ["loan_count", "opening_principal_mn", "opening_mark_mn"],
    ["q1_cash_coupon_mn", "q1_capitalized_pik_mn", "q1_financing_interest_mn"],
    ["q1_distribution_mn", "q1_new_borrowing_mn", "q1_external_equity_mn"],
    ["shocked_borrowing_base_mn", "shock_cure_equity_mn"],
    ["q2_early_debt_repayment_mn", "q2_max_distribution_mn", "q2_ending_principal_mn"],
]
DECIMALS = [0, 4, 4, 4, 4, 4, 3, 4, 3, 3, 3, 4, 3, 4]


def _book():
    table = load_expansion_table("sec_bdc")
    rows = table.loc[
        table.cik.eq(CASE_CIK) & table.report_date.eq(REPORT_DATE)
        & table.form.eq("10-K") & table.filed_date.eq(FILING_DATE)
    ].copy()
    if rows.accession.nunique() != 1 or rows.investment_id.duplicated().any():
        raise ValueError("The annual source version is not unique")
    rows = rows.dropna(subset=[
        "principal_amount_usd", "fair_value_usd", "pik_interest_rate",
        "total_interest_rate", "maturity_date", "investment_type",
    ])
    kind = rows.investment_type.astype(str).str.rsplit("#", n=1).str[-1]
    rows = rows.loc[
        rows.principal_amount_usd.gt(0) & rows.fair_value_usd.gt(0)
        & rows.pik_interest_rate.ge(0) & rows.maturity_date.gt(MATURITY_FLOOR)
        & kind.isin(DEBT_TYPES)
    ].copy()
    if len(rows) != 27 or (rows.pik_interest_rate > 0).sum() != 10:
        raise ValueError("Frozen loan population changed")
    return rows


def _sofr(day):
    table = load_expansion_table("nyfed_reference_rates")
    match = table.loc[table["Rate Type"].eq("SOFR") & table["Effective Date"].eq(day)]
    if len(match) != 1:
        raise ValueError("SOFR fixing is not unique")
    return float(match.iloc[0]["Rate (%)"]) / 100


def analysis():
    rows = _book()
    opening_p = float(rows.principal_amount_usd.sum())
    opening_mark = float(rows.fair_value_usd.sum())
    pik_mark = float(rows.loc[rows.pik_interest_rate.gt(0), "fair_value_usd"].sum())
    opening_debt = INITIAL_DEBT_TO_MARK * opening_mark
    q1_sofr = _sofr("2024-12-31")
    q1_total = opening_p * (q1_sofr + LOAN_SPREAD) * 90 / 360
    q1_pik = float((rows.principal_amount_usd * rows.pik_interest_rate).sum()) * 90 / 360
    q1_cash = q1_total - q1_pik
    q1_finance = opening_debt * (q1_sofr + FINANCE_SPREAD) * 90 / 360
    distribution = PAYOUT_RATIO * q1_total
    deficit = max(0.0, distribution + q1_finance - q1_cash)
    old_base = ADVANCE * opening_mark
    new_borrowing = min(deficit, max(0.0, old_base - opening_debt))
    equity_q1 = deficit - new_borrowing
    debt_after_payout = opening_debt + new_borrowing
    shocked_base = ADVANCE * (opening_mark - SHOCK * pik_mark)
    shock_cure = max(0.0, debt_after_payout - shocked_base)
    debt_after_cure = debt_after_payout - shock_cure

    # All new budget used at quarter start to save financing interest. The
    # capacity it frees may be borrowed again only after quarter-two interest.
    early_repayment = min(Q2_EQUITY_BUDGET, debt_after_cure)
    q2_debt_for_interest = debt_after_cure - early_repayment
    q2_sofr = _sofr("2025-03-31")
    q2_principal = rows.principal_amount_usd.astype(float) * (
        1 + rows.pik_interest_rate.astype(float) * 90 / 360
    )
    q2_total = float(q2_principal.sum()) * (q2_sofr + LOAN_SPREAD) * 91 / 360
    q2_pik = float((q2_principal * rows.pik_interest_rate.astype(float)).sum()) * 91 / 360
    q2_cash = q2_total - q2_pik
    q2_finance = q2_debt_for_interest * (q2_sofr + FINANCE_SPREAD) * 91 / 360
    q2_redraw = shocked_base - q2_debt_for_interest
    q2_max_distribution = q2_cash - q2_finance + q2_redraw
    if q2_redraw < 0 or q2_max_distribution < 0:
        raise ValueError("Hypothetical capital policy infeasible")
    mn = lambda amount: amount / 1_000_000
    return {
        "loan_count": len(rows), "opening_principal_mn": mn(opening_p),
        "opening_mark_mn": mn(opening_mark), "q1_cash_coupon_mn": mn(q1_cash),
        "q1_capitalized_pik_mn": mn(q1_pik),
        "q1_financing_interest_mn": mn(q1_finance),
        "q1_distribution_mn": mn(distribution),
        "q1_new_borrowing_mn": mn(new_borrowing),
        "q1_external_equity_mn": mn(equity_q1),
        "shocked_borrowing_base_mn": mn(shocked_base),
        "shock_cure_equity_mn": mn(shock_cure),
        "q2_early_debt_repayment_mn": mn(early_repayment),
        "q2_max_distribution_mn": mn(q2_max_distribution),
        "q2_ending_principal_mn": mn(float(q2_principal.sum()) + q2_pik),
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
