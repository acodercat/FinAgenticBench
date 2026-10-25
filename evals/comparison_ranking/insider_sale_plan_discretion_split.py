"""Split exact-Form-4 sale rows by a filing-level Rule 10b5-1 proxy, then re-rank.

Over calendar 2024 the exact-Form-4 population records 20,945 qualifying S/D
transaction rows across 370 issuer CIKs worth 107.697155 billion USD,
led by Amazon.com at 13.729432 billion, 0.560549 billion ahead of Walmart. Two
filing-level groups change what that leadership means. S/D rows in filings with an
affirmative Rule 10b5-1 checkbox account for 14,444 rows but 31.422153 billion
USD, 29.1764 percent of value. When that proxy group is removed, Amazon.com has no
remaining sale-row value and leaves the ranking, which Walmart then leads at
13.152360 billion, 2.261679 billion ahead of Bank of America. Against the
purchase side, 850 P/A rows total 4.970704 billion USD and the sale-to-purchase
reported-value ratio is 21.6664. The table describes the asymmetry but not its cause.

The third turn requires recomputation over the narrowed proxy population; carrying
the turn-1 leader forward without reranking gives the wrong issuer.

The `insider_plan_discretion` pinned-reading comparison records sensitivity to treating every
disposition code as a sale, changing the affirmative-only flag reading, splitting
on counts instead of value, reranking on shares, and choosing a different outstanding-
share fact. The query fixes the reported convention, so these are robustness
comparisons rather than hidden answer paths.

The source writes the filing-level checkbox as 1, 0, true, false or missing. In the
exact-Form-4 qualifying-sale population both encodings are live. An affirmative
filing says at least one transaction was made under an intended plan; it does not
identify which row, so the split cannot be called transaction-level plan status or
discretionary activity.

One qualifying row reports a price of zero. It counts as reporting a price, so the
first turn has 20,945 rows; dropping it as unpriced gives 20,944 and moves no value.

The late-filing count is zero, and that is a fact about the window rather than a
missing field: 207 of the 54,849 exact-Form-4 rows carry a timeliness mark and every
one of them is E for early, so counting marked rows instead of late rows reports 207.
The query asks for the rows reported as late.

Entity and measurement boundaries: rankings aggregate by issuer CIK and use the
largest-value name variant only for display. S/P include open-market or private
transactions. The final percentage compares annual sale-row volume with a
point-in-time share count as of 2025-03-12 and is not ownership sold or unique-share
turnover.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts, identifiers and names match exactly.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_facts, load_insider_table
from core.validation import turn_validator, validate_ordered_outputs


TRANSACTION_YEAR = "2024"
FORM_TYPE = "4"
SHARES_CUTOFF_DATE = "2025-03-31"
SHARES_CONCEPT = "EntityCommonStockSharesOutstanding"
PLAN_FLAGS = ("1", "true")


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = [
    "sale_transaction_count", "sale_issuer_count", "sale_value_usd_billions",
    "gross_leader_issuer", "gross_leader_value_usd_billions", "gross_leader_margin_usd_billions",
]
TURN_2_NAMES = [
    "affirmative_plan_filing_sale_count", "affirmative_plan_filing_sale_value_usd_billions",
    "nonaffirmative_filing_sale_count", "nonaffirmative_filing_sale_value_usd_billions",
    "affirmative_plan_filing_share_of_value_pct",
    "affirmative_plan_filing_share_of_transactions_pct",
]
TURN_3_NAMES = [
    "nonaffirmative_filing_issuer_count", "nonaffirmative_filing_leader_issuer",
    "nonaffirmative_filing_leader_value_usd_billions",
    "nonaffirmative_filing_leader_margin_usd_billions",
    "gross_leader_nonaffirmative_filing_value_usd_billions",
    "gross_leader_nonaffirmative_filing_rank",
]
TURN_4_NAMES = [
    "purchase_transaction_count", "purchase_issuer_count", "purchase_value_usd_billions",
    "purchase_leader_issuer", "purchase_leader_value_usd_billions", "sale_to_purchase_value_ratio",
    "late_reported_transaction_count", "nonaffirmative_filing_leader_sale_shares",
    "nonaffirmative_filing_leader_shares_outstanding",
    "nonaffirmative_filing_leader_shares_outstanding_end_date",
    "nonaffirmative_filing_leader_sale_shares_to_outstanding_pct",
]

variables = [
    _v(TURN_1_NAMES[0], "Store the number of qualifying sale transactions as an integer."),
    _v(TURN_1_NAMES[1], "Store the number of issuers with at least one qualifying sale as an integer."),
    _v(TURN_1_NAMES[2], "Store the total qualifying sale value in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[3], "Store the leading issuer's name as the filings record it, as text."),
    _v(TURN_1_NAMES[4], "Store that issuer's sale value in USD billions, rounded to 6 decimals."),
    _v(TURN_1_NAMES[5], "Store the leader's value minus the runner-up's, in USD billions rounded to 6 decimals."),
    _v(TURN_2_NAMES[0], "Store the number of qualifying sale rows in affirmative-plan filings as an integer."),
    _v(TURN_2_NAMES[1], "Store their reported-price value in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[2], "Store the number of qualifying sale rows in non-affirmative filings as an integer."),
    _v(TURN_2_NAMES[3], "Store their reported-price value in USD billions, rounded to 6 decimals."),
    _v(TURN_2_NAMES[4], "Store affirmative-filing sale-row value as a percent of total sale-row value, rounded to 4 decimals."),
    _v(TURN_2_NAMES[5], "Store affirmative-filing sale rows as a percent of all qualifying sale rows, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store the number of issuer CIKs with any non-affirmative-filing sale-row value as an integer."),
    _v(TURN_3_NAMES[1], "Store the leading issuer in that filing-level proxy ranking, as text."),
    _v(TURN_3_NAMES[2], "Store that issuer's proxy sale-row value in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[3], "Store its value minus the runner-up's, in USD billions rounded to 6 decimals."),
    _v(TURN_3_NAMES[4], "Store the turn-1 leader's non-affirmative-filing sale-row value in USD billions, rounded to 6 decimals."),
    _v(TURN_3_NAMES[5], "Store the turn-1 leader's proxy rank, or 0 if it has no such value, as an integer."),
    _v(TURN_4_NAMES[0], "Store the number of qualifying purchase transactions as an integer."),
    _v(TURN_4_NAMES[1], "Store the number of issuers with at least one qualifying purchase as an integer."),
    _v(TURN_4_NAMES[2], "Store the total purchase value in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[3], "Store the leading issuer by purchase value, as text."),
    _v(TURN_4_NAMES[4], "Store that issuer's purchase value in USD billions, rounded to 6 decimals."),
    _v(TURN_4_NAMES[5], "Store total sale value divided by total purchase value, rounded to 4 decimals."),
    _v(TURN_4_NAMES[6], "Store how many transactions in the window are reported as late, as an integer."),
    _v(TURN_4_NAMES[7], "Store the proxy leader's shares summed across its qualifying non-affirmative-filing sale rows as an integer."),
    _v(TURN_4_NAMES[8], "Store that issuer's shares outstanding from the selected count as an integer."),
    _v(TURN_4_NAMES[9], "Store the as-of date of that outstanding-share count as an ISO YYYY-MM-DD string."),
    _v(TURN_4_NAMES[10], "Store sale-row shares as a percent of shares outstanding, rounded to 4 decimals."),
]

DECIMALS = [
    0, 0, 6, None, 6, 6,
    0, 6, 0, 6, 4, 4,
    0, None, 6, 6, 6, 0,
    0, 0, 6, None, 6, 4, 0, 0, 0, None, 4,
]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def _window():
    transactions = load_insider_table("nonderiv_trans").copy()
    submissions = load_insider_table("submission").copy()
    submissions = submissions.loc[submissions.document_type.astype(str).eq(FORM_TYPE),
        ["accession_number", "issuercik", "issuername", "aff10b5one"]
    ]
    transactions["trans_shares"] = pd.to_numeric(transactions.trans_shares, errors="coerce")
    transactions["trans_pricepershare"] = pd.to_numeric(transactions.trans_pricepershare, errors="coerce")
    joined = transactions.merge(submissions, on="accession_number", how="inner")
    window = joined.loc[joined.trans_date.astype(str).str.startswith(TRANSACTION_YEAR)].copy()
    window["value"] = window.trans_shares * window.trans_pricepershare
    window["affirmative_plan_filing"] = (
        window.aff10b5one.astype("string").str.strip().str.lower().isin(PLAN_FLAGS)
    )
    return window


def _ranked(frame):
    by_name = frame.groupby(["issuercik", "issuername"], as_index=False).agg(
        name_value=("value", "sum"),
    ).sort_values(["issuercik", "name_value", "issuername"], ascending=[True, False, True])
    display = by_name.drop_duplicates("issuercik")[["issuercik", "issuername"]]
    ranked = frame.groupby("issuercik", as_index=False).agg(
        value=("value", "sum"), shares=("trans_shares", "sum"),
    ).merge(display, on="issuercik", how="left")
    return ranked.sort_values(["value", "issuercik"], ascending=[False, True]).reset_index(drop=True)


@lru_cache(maxsize=1)
def ground_truth():
    window = _window()
    sales = window.loc[
        window.trans_code.astype(str).eq("S")
        & window.trans_acquired_disp_cd.astype(str).eq("D")
        & window.value.notna()
    ]
    gross = _ranked(sales)
    gross_leader = gross.iloc[0]

    affirmative = sales.loc[sales.affirmative_plan_filing]
    nonaffirmative = sales.loc[~sales.affirmative_plan_filing]
    nonaffirmative_rank = _ranked(nonaffirmative)
    nonaffirmative_leader = nonaffirmative_rank.iloc[0]
    carried = nonaffirmative_rank.loc[nonaffirmative_rank.issuercik.eq(gross_leader.issuercik)]
    carried_value = float(carried.value.iloc[0]) if len(carried) else 0.0
    carried_rank = int(carried.index[0]) + 1 if len(carried) else 0

    purchases = window.loc[
        window.trans_code.astype(str).eq("P")
        & window.trans_acquired_disp_cd.astype(str).eq("A")
        & window.value.notna()
    ]
    purchase_rank = _ranked(purchases)
    purchase_leader = purchase_rank.iloc[0]

    facts = load_facts()
    counts = facts.loc[
        facts.cik.astype(str).str.lstrip("0").eq(str(nonaffirmative_leader.issuercik).lstrip("0"))
        & facts.concept.astype(str).eq(SHARES_CONCEPT)
        & facts.unit.astype(str).eq("shares")
        & facts.filed_date.astype(str).le(SHARES_CUTOFF_DATE)
    ].sort_values(["filed_date", "end_date"])
    if counts.empty:
        raise ValueError("no share count is available for the non-affirmative-filing leader")
    share_count = counts.iloc[-1]
    shares_outstanding = float(share_count.value)
    shares_sold = float(nonaffirmative_leader.shares)

    return (
        int(len(sales)),
        int(sales.issuercik.nunique()),
        float(sales.value.sum()) / 1e9,
        str(gross_leader.issuername),
        float(gross_leader.value) / 1e9,
        float(gross_leader.value - gross.iloc[1].value) / 1e9,
        int(len(affirmative)),
        float(affirmative.value.sum()) / 1e9,
        int(len(nonaffirmative)),
        float(nonaffirmative.value.sum()) / 1e9,
        float(affirmative.value.sum() / sales.value.sum() * 100),
        float(len(affirmative) / len(sales) * 100),
        int(len(nonaffirmative_rank)),
        str(nonaffirmative_leader.issuername),
        float(nonaffirmative_leader.value) / 1e9,
        float(nonaffirmative_leader.value - nonaffirmative_rank.iloc[1].value) / 1e9,
        carried_value / 1e9,
        carried_rank,
        int(len(purchases)),
        int(purchases.issuercik.nunique()),
        float(purchases.value.sum()) / 1e9,
        str(purchase_leader.issuername),
        float(purchase_leader.value) / 1e9,
        float(sales.value.sum() / purchases.value.sum()),
        int(window.trans_timeliness.astype(str).eq("L").sum()),
        int(shares_sold),
        int(shares_outstanding),
        str(share_count.end_date),
        shares_sold / shares_outstanding * 100,
    )


NAME_OUTPUTS = ("gross_leader_issuer", "nonaffirmative_filing_leader_issuer", "purchase_leader_issuer")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_gross_sale_leadership": turn_validator(validate_turn_1),
    "validate_plan_decomposition": turn_validator(validate_turn_2),
    "validate_nonaffirmative_reranking": turn_validator(validate_turn_3),
    "validate_purchase_asymmetry": turn_validator(validate_turn_4),
}
