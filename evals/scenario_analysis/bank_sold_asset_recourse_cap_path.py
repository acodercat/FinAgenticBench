"""Sold-asset principal, remaining contractual recourse and hypothetical paid loss.

Baseline convention checks: using sold face instead of remaining recourse for
the residential 10% pool produces 991.3 rather than 113 thousand USD. Pooling
the two caps at the 10% stress produces 2,480,826.3 rather than 2,479,948
thousand USD; pooling happens to agree at the 40% stress. The stated
counterfactual caps each column separately. These are not observed losses.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("rcs_bank_count", 0, "the matched eligible-bank count as an integer"),
    ("rcs_selected_rssd", None, "the selected bank's exact RSSD ID as a string"),
    ("rcs_exposure_cash_pct", 4, "the selected combined maximum recourse exposure divided by FDIC cash, in percent"),
    ("rcs_residential_face_m", 3, "reported residential sold unpaid principal in USD millions"),
    ("rcs_other_face_m", 3, "reported all-other sold unpaid principal in USD millions"),
    ("rcs_residential_cap_m", 3, "reported residential remaining maximum contractual exposure in USD millions"),
    ("rcs_other_cap_m", 3, "reported all-other remaining maximum contractual exposure in USD millions"),
    ("rcs_ten_pct_payment_m", 4, "the first hypothetical total recourse payment in USD millions"),
    ("rcs_ten_pct_cash_m", 4, "book cash after the first hypothetical payment in USD millions"),
    ("rcs_ten_pct_equity_m", 4, "book equity after the first hypothetical payment in USD millions"),
    ("rcs_forty_pct_payment_m", 4, "the revised hypothetical total recourse payment in USD millions"),
    ("rcs_forty_pct_cash_m", 4, "book cash after the revised hypothetical payment, recomputed from opening cash, in USD millions"),
    ("rcs_forty_pct_equity_m", 4, "book equity after the revised hypothetical payment, recomputed from opening equity, in USD millions"),
)
variables = [Variable(name, None, f"Store {description}" +
    (f", rounded to {precision} decimals from unrounded inputs." if precision is not None else "."))
    for name, precision, description in OUTPUTS]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 3), (3, 7), (7, 10), (10, 13)))


def d(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite bank amount")
    return result


def analysis() -> dict[str, int | str | float]:
    rcs = load_expansion_table("ffiec_call_reports_recourse")
    fdic = load_expansion_table("fdic_bankfind")
    fdic = fdic.loc[fdic.REPDTE.eq("2025-09-30")]
    joined = rcs.merge(fdic[["CERT", "CHBAL", "EQ"]],
        left_on="fdic_certificate", right_on="CERT", validate="one_to_one")
    columns = ["residential_sold_principal_thousand_usd",
               "other_sold_principal_thousand_usd",
               "residential_max_recourse_thousand_usd",
               "other_max_recourse_thousand_usd", "CHBAL", "EQ"]
    good = joined.filing_type.eq("031") & joined[columns].notna().all(axis=1)
    good &= joined[columns].gt(0).all(axis=1)
    sample = joined.loc[good].copy()
    sample["ratio"] = (sample[columns[2]] + sample[columns[3]]) / sample.CHBAL
    sample = sample.sort_values(["ratio", "bank_rssd_id"], ascending=[False, True])
    if len(sample) < 2:
        raise ValueError("insufficient comparable banks")
    row = sample.iloc[0]
    residential_face, other_face = d(row[columns[0]]), d(row[columns[1]])
    residential_cap, other_cap = d(row[columns[2]]), d(row[columns[3]])
    cash, equity = d(row.CHBAL), d(row.EQ)
    first = min(residential_face * Decimal(".10"), residential_cap) + min(other_face * Decimal(".10"), other_cap)
    revised = min(residential_face * Decimal(".10"), residential_cap) + min(other_face * Decimal(".40"), other_cap)
    scale = Decimal(1000)
    return {
        "rcs_bank_count": len(sample),
        "rcs_selected_rssd": str(row.bank_rssd_id),
        "rcs_exposure_cash_pct": float((residential_cap + other_cap) / cash * 100),
        "rcs_residential_face_m": float(residential_face / scale),
        "rcs_other_face_m": float(other_face / scale),
        "rcs_residential_cap_m": float(residential_cap / scale),
        "rcs_other_cap_m": float(other_cap / scale),
        "rcs_ten_pct_payment_m": float(first / scale),
        "rcs_ten_pct_cash_m": float((cash - first) / scale),
        "rcs_ten_pct_equity_m": float((equity - first) / scale),
        "rcs_forty_pct_payment_m": float(revised / scale),
        "rcs_forty_pct_cash_m": float((cash - revised) / scale),
        "rcs_forty_pct_equity_m": float((equity - revised) / scale),
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
