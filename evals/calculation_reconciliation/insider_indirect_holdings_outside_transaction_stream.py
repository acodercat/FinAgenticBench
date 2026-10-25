"""Beneficial ownership disclosed in Form 4 holding rows rather than transactions.

A Form 4 Table I reports two kinds of row. A transaction row carries a trade and
the position left after it; a holding row carries a position with no trade at
all, which is how an insider discloses stock held through trusts, partnerships
and retirement plans. An analyst who reads only the transaction stream never
sees those positions, and the omission is not random: in this window holding
rows are overwhelmingly indirect while transaction rows are overwhelmingly
direct.

Three conventions this case fixes, each because the alternative is defensible
but wrong on this data:

1. ``shrs_ownd_folwng_trans`` on a transaction row is a running balance, not a
   delta. Within one filing 71.2% of transaction rows in this window share an
   ownership line with another row, so summing that column over transaction rows double counts.
   This case therefore never aggregates transaction balances; it counts rows.
2. A standing indirect position can be re-disclosed on later Form 4 filings. Summing
   holding rows across those filings can count it repeatedly: Walmart's Walton family
   trust line alone reports 3,002,673,393 shares on each of many filings and
   sums to 138.9 billion shares across 2024, against 8,033,386,215 shares
   outstanding on its 2024-12-04 cover page. Taking one filing per reporting owner is what makes a
   cross-sectional total meaningful.
3. A holding row and a transaction row inside the same filing can share
   ``(security_title, direct_indirect_ownership)`` while describing different
   vehicles, so the two row sets cannot be combined into a filing-level total
   without footnote text this runtime does not carry. The measured quantity is
   deliberately the holding-row side alone.

The window starts after NVIDIA's June 2024 ten-for-one split so that a single
issuer's share counts are on one basis; a window spanning the split would mix
pre- and post-split counts and make the outstanding-share ratio meaningless.
``security_title`` is not a grouping key: the selected issuer's own filings
carry both "Common Stock" and "Common".

The two indirect shares count rows, which the query states because weighting
them by the shares each row carries is equally defensible and gives a different
answer: 88.6274% of the holding-row shares and 6.7025% of the transaction-row
shares sit in indirect form, against the 83.6923% and 15.6661% of rows
reported. The row count is what the turn's argument needs, since the point is
how often each row type carries an indirect position rather than how much stock
does, and a row count is not dominated by a handful of very large trusts. The
contributing-owner count is not exposed the same way: none of the selected
issuer's eighteen retained holding rows carries zero shares, so counting owners
with any row and counting owners with a positive one both give six.
"""

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_facts, load_insider_table
from core.validation import turn_validator, validate_ordered_outputs


WINDOW_START = "2024-10-01"
WINDOW_END = "2024-12-31"
SHARES_CONCEPT = "EntityCommonStockSharesOutstanding"

OUTPUTS = (
    ("eligible_filing_count", 0, "the eligible Form 4 filing count as an integer"),
    ("filings_with_holding_rows", 0, "the count of those filings carrying at least one holding row as an integer"),
    ("holding_row_count", 0, "the total holding-row count as an integer"),
    ("holding_row_indirect_pct", 4, "the indirect share of those holding rows in percent"),
    ("transaction_row_indirect_pct", 4, "the indirect share of the same filings' transaction rows in percent"),
    ("issuer_owner_pair_count", 0, "the count of distinct issuer and reporting-owner pairs as an integer"),
    ("latest_filing_holding_row_count", 0, "the holding-row count on the retained latest filings as an integer"),
    ("selected_issuer_cik", None, "the selected issuer's exact CIK as a source-verbatim string"),
    ("selected_holding_shares", 2, "the selected issuer's retained holding-row share total"),
    ("selected_indirect_holding_shares", 2, "the indirect portion of that total"),
    ("selected_owner_count", 0, "the number of reporting owners contributing to it as an integer"),
    ("largest_holding_row_shares", 2, "the largest single retained holding row for that issuer"),
    ("selected_holdings_to_shares_outstanding_pct", 4, "that holding-row total against shares outstanding, in percent"),
)
variables = [
    Variable(name, None, f"Store {text}" + (f", rounded to {p} decimals from unrounded source rows." if p is not None else "."))
    for name, p, text in OUTPUTS
]
PRECISION = {name: p for name, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:5]),
    tuple(n for n, _, _ in OUTPUTS[5:7]),
    tuple(n for n, _, _ in OUTPUTS[7:11]),
    tuple(n for n, _, _ in OUTPUTS[11:]),
)


def _dec(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("non-finite share amount")
    return number


@lru_cache(maxsize=1)
def _population():
    """Single-owner original Form 4 filings whose period falls in the window."""
    submissions = load_insider_table("submission")
    owners = load_insider_table("reportingowner")
    period = submissions.period_of_report.astype(str)
    eligible = submissions.loc[
        submissions.document_type.eq("4") & period.between(WINDOW_START, WINDOW_END)
    ].copy()
    # 0.8% of filings name more than one reporting owner; a holding row on those
    # cannot be attributed to a single owner, so they are excluded rather than
    # split. This drops 92 filings from the window, not zero.
    counts = owners.groupby("accession_number").size()
    sole = set(counts[counts.eq(1)].index)
    eligible = eligible.loc[eligible.accession_number.isin(sole)]
    if eligible.accession_number.duplicated().any():
        raise ValueError("submission accessions are not unique in the window")
    named = eligible.merge(
        owners.loc[owners.accession_number.isin(sole), ["accession_number", "rptownercik"]],
        on="accession_number", validate="one_to_one",
    )
    if len(named) != len(eligible):
        raise ValueError("reporting-owner join changed the filing count")
    return named


def _rows(table_name, accessions):
    rows = load_insider_table(table_name)
    return rows.loc[rows.accession_number.isin(accessions)]


@lru_cache(maxsize=1)
def analysis():
    filings = _population()
    accessions = set(filings.accession_number)
    holdings = _rows("nonderiv_holding", accessions)
    transactions = _rows("nonderiv_trans", accessions)
    if holdings.empty or transactions.empty:
        raise ValueError("window carries no Form 4 table-one rows")
    if holdings.shrs_ownd_folwng_trans.isna().any():
        raise ValueError("a holding row reports no share amount")
    for frame in (holdings, transactions):
        if not frame.direct_indirect_ownership.isin({"D", "I"}).all():
            raise ValueError("ownership form outside the reported D/I vocabulary")

    # One filing per issuer and reporting owner. A standing indirect position is
    # repeated on later Form 4 filings, so without this the same trust can be
    # counted on multiple filings.
    ordered = filings.sort_values(
        ["issuercik", "rptownercik", "period_of_report", "filing_date", "accession_number"]
    )
    latest = ordered.drop_duplicates(["issuercik", "rptownercik"], keep="last")
    retained = holdings.loc[holdings.accession_number.isin(set(latest.accession_number))].merge(
        latest[["accession_number", "issuercik", "rptownercik"]],
        on="accession_number", validate="many_to_one",
    )

    by_issuer = retained.groupby("issuercik").shrs_ownd_folwng_trans.sum()
    ranked = by_issuer.sort_index().sort_values(ascending=False, kind="stable")
    selected_cik = str(ranked.index[0])
    if ranked.iloc[0] <= ranked.iloc[1]:
        raise ValueError("selected issuer is not a strict leader")
    chosen = retained.loc[retained.issuercik.eq(selected_cik)]

    total = sum((_dec(x) for x in chosen.shrs_ownd_folwng_trans), Decimal(0))
    indirect = sum(
        (_dec(x) for x in chosen.loc[chosen.direct_indirect_ownership.eq("I"), "shrs_ownd_folwng_trans"]),
        Decimal(0),
    )
    largest = chosen.shrs_ownd_folwng_trans.sort_values(ascending=False)
    if largest.iloc[0] <= largest.iloc[1]:
        raise ValueError("largest holding row is not a strict maximum")

    facts = load_facts()
    cover = facts.loc[
        facts.cik.astype(str).str.lstrip("0").eq(selected_cik.lstrip("0"))
        & facts.concept.eq(SHARES_CONCEPT)
        & facts.unit.eq("shares")
        & facts.end_date.le(WINDOW_END)
    ].sort_values(["end_date", "filed_date", "accession"])
    if cover.empty:
        raise ValueError("no cover-page share count on or before the window end")
    outstanding = _dec(cover.iloc[-1].value)
    if outstanding <= 0:
        raise ValueError("nonpositive shares outstanding")

    return {
        "eligible_filing_count": len(filings),
        "filings_with_holding_rows": int(holdings.accession_number.nunique()),
        "holding_row_count": len(holdings),
        "holding_row_indirect_pct": float(100 * holdings.direct_indirect_ownership.eq("I").mean()),
        "transaction_row_indirect_pct": float(100 * transactions.direct_indirect_ownership.eq("I").mean()),
        "issuer_owner_pair_count": len(latest),
        "latest_filing_holding_row_count": len(retained),
        "selected_issuer_cik": selected_cik,
        "selected_holding_shares": float(total),
        "selected_indirect_holding_shares": float(indirect),
        "selected_owner_count": int(chosen.rptownercik.nunique()),
        "largest_holding_row_shares": float(largest.iloc[0]),
        # Divides a 2-decimal total by a share count near 2.4e10, so the rounded
        # operand moves this by about 2e-11 against a 6e-5 tolerance.
        "selected_holdings_to_shares_outstanding_pct": float(total / outstanding * 100),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[variable.name] for variable in variables)


def _check(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(
        outputs,
        [by_name[n] for n in names],
        [truth[n] for n in names],
        [PRECISION[n] for n in names],
    )


def validate_turn_1(outputs):
    return _check(outputs, TURNS[0])


def validate_turn_2(outputs):
    return _check(outputs, TURNS[1])


def validate_turn_3(outputs):
    return _check(outputs, TURNS[2])


def validate_turn_4(outputs):
    return _check(outputs, TURNS[3])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
    for i in range(1, 5)
}
