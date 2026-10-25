"""13F option-underlier disclosures versus issuer Form 4 derivative transactions.

The governed 13F-HR aggregation is by period and underlying CUSIP. Its option
values describe the underlying security under Form 13F instructions; they are
not option premiums, position delta, or a directional net. The aggregate is
selected by submission type in the frozen builder. One 2025Q2 13F-HR filing
also has an inconsistent amendment flag and contributes 52 Oracle common
shares, but no Oracle option rows; the case describes the governed 13F-HR
aggregate and does not claim amendment-chain reconstruction.

Baseline alternatives measured from the governed data: treating put-minus-call
as net bearish exposure yields +USD1.274053032bn for 2024Q4 and
-USD1.754376701bn for 2025Q2, but neither is delta-adjusted portfolio risk.
Form 4 code M appears once in 2024Q4 and twenty times in 2025Q2; only ten of
the latter rows are titled Stock Option. Those issuer-insider transactions are
neither the 13F reporting population nor evidence of a causal link to it.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table, load_insider_table
from core.validation import turn_validator, validate_ordered_outputs


ISSUER_CIK = "0001341439"
PERIODS = ("2024-12-31", "2025-06-30")
CUTOFF = "2025-12-31"

OUTPUTS = (
    ("option_underlier_cusip", None, "selected Oracle common-stock CUSIP as an exact nine-character source string"),
    ("option_underlier_q4_accessions", 0, "number of distinct 13F-HR information-table accessions aggregated for the CUSIP, as an integer"),
    ("option_underlier_q4_common_shares", 0, "2024 Q4 reported non-option common shares, as an integer"),
    ("option_underlier_q4_common_value_bn", 6, "2024 Q4 reported non-option common-share value in USD billions"),
    ("option_underlier_q4_put_bn", 6, "2024 Q4 Form 13F put-designated reported value in USD billions"),
    ("option_underlier_q4_call_bn", 6, "2024 Q4 Form 13F call-designated reported value in USD billions"),
    ("option_underlier_q2_put_bn", 6, "2025 Q2 Form 13F put-designated reported value in USD billions"),
    ("option_underlier_q2_call_bn", 6, "2025 Q2 Form 13F call-designated reported value in USD billions"),
    ("option_underlier_q4_form4_m_rows", 0, "2024 Q4 Oracle Form 4 derivative transaction rows with code M, as an integer"),
    ("option_underlier_q2_form4_m_rows", 0, "2025 Q2 Oracle Form 4 derivative transaction rows with code M, as an integer"),
    ("option_underlier_q2_form4_stock_option_m_rows", 0, "2025 Q2 Oracle Form 4 code-M derivative transaction rows explicitly titled Stock Option, as an integer"),
)

variables = [
    Variable(name, None, f"Store {description}" +
             (f", rounded to {precision} decimals from unrounded inputs." if precision is not None else "."))
    for name, precision, description in OUTPUTS
]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 4), (4, 6), (6, 8), (8, 11)))


def _decimal(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite source value")
    return result


def _holdings_rows():
    frame = load_expansion_table("sec_13f_holdings")
    candidates = frame.loc[
        frame.period_of_report.eq(PERIODS[0]) &
        frame.issuer_name.eq("ORACLE CORP") &
        frame.title_of_class.eq("COM") &
        frame.share_value_usd_total.gt(0)
    ].sort_values(["share_value_usd_total", "cusip"], ascending=[False, True])
    if len(candidates) < 2 or candidates.iloc[0].share_value_usd_total <= \
            candidates.iloc[1].share_value_usd_total:
        raise ValueError("unexpected Oracle CUSIP ranking")
    selected_cusip = str(candidates.iloc[0].cusip)
    selected = frame.loc[frame.cusip.eq(selected_cusip) &
                         frame.period_of_report.isin(PERIODS)]
    if len(selected) != 2 or selected.period_of_report.nunique() != 2:
        raise ValueError("unexpected Microsoft CUSIP-period population")
    rows = {str(row.period_of_report): row for row in selected.itertuples(index=False)}
    for row in rows.values():
        if not str(row.issuer_name).upper().startswith("ORACLE"):
            raise ValueError("CUSIP issuer mismatch")
        if any(_decimal(getattr(row, col)) <= 0 for col in
               ("share_amount_total", "share_value_usd_total", "put_value_usd_total",
                "call_value_usd_total")):
            raise ValueError("nonpositive reported value")
    return selected_cusip, rows


def _form4_code_m_counts() -> tuple[int, int, int]:
    submissions = load_insider_table("submission")
    filings = submissions.loc[
        submissions.issuercik.eq(ISSUER_CIK) &
        submissions.document_type.eq("4") &
        submissions.filing_date.le(CUTOFF), ["accession_number"]]
    if filings.accession_number.duplicated().any():
        raise ValueError("duplicate Oracle Form 4 accession")
    transactions = load_insider_table("deriv_trans")
    selected = transactions.loc[
        transactions.accession_number.isin(filings.accession_number) &
        transactions.trans_code.eq("M")]
    q4 = selected.loc[selected.trans_date.between("2024-10-01", PERIODS[0])]
    q2 = selected.loc[selected.trans_date.between("2025-04-01", PERIODS[1])]
    return len(q4), len(q2), int(q2.security_title.eq("Stock Option").sum())


def analysis() -> dict[str, int | str | float]:
    selected_cusip, rows = _holdings_rows()
    q4, q2 = (rows[p] for p in PERIODS)
    q4_m, q2_m, q2_option_m = _form4_code_m_counts()
    billion = Decimal(1_000_000_000)
    return {
        "option_underlier_cusip": selected_cusip,
        "option_underlier_q4_accessions": int(q4.filer_count),
        "option_underlier_q4_common_shares": int(_decimal(q4.share_amount_total)),
        "option_underlier_q4_common_value_bn": float(_decimal(q4.share_value_usd_total) / billion),
        "option_underlier_q4_put_bn": float(_decimal(q4.put_value_usd_total) / billion),
        "option_underlier_q4_call_bn": float(_decimal(q4.call_value_usd_total) / billion),
        "option_underlier_q2_put_bn": float(_decimal(q2.put_value_usd_total) / billion),
        "option_underlier_q2_call_bn": float(_decimal(q2.call_value_usd_total) / billion),
        "option_underlier_q4_form4_m_rows": q4_m,
        "option_underlier_q2_form4_m_rows": q2_m,
        "option_underlier_q2_form4_stock_option_m_rows": q2_option_m,
    }


@lru_cache(maxsize=1)
def ground_truth():
    result = analysis()
    return tuple(result[v.name] for v in variables)


def _check(outputs, names):
    by = {v.name: v for v in variables}
    truth = dict(zip(by, ground_truth()))
    return validate_ordered_outputs(outputs, [by[n] for n in names],
                                    [truth[n] for n in names],
                                    [PRECISION[n] for n in names])


def validate_turn_1(outputs): return _check(outputs, TURNS[0])
def validate_turn_2(outputs): return _check(outputs, TURNS[1])
def validate_turn_3(outputs): return _check(outputs, TURNS[2])
def validate_turn_4(outputs): return _check(outputs, TURNS[3])
def validate(outputs): return _check(outputs, tuple(v.name for v in variables))


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"])
              for i in range(1, 5)}
