"""RC-L unused commitments become loans only after stipulated draws.

The baseline excludes three measured wrong paths: treating unused face as
already-funded loans, treating a par draw or bridge principal as an equity loss,
and charging the hypothetical provision as a second cash outflow. No actual
draw probability, cancelability, CCF, or credit loss follows from RC-L face.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("eligible_bank_count", 0, "the eligible matched-bank count as an integer"),
    ("selected_rssd_id", None, "the selected bank's exact RSSD identifier as a string"),
    ("selected_commitment_to_equity_pct", 4, "the selected combined unused commitments divided by opening FDIC book equity, in percent"),
    ("runner_up_commitment_to_equity_pct", 4, "the runner-up ratio in percent"),
    ("unused_card_m", 3, "the selected unused credit-card lines in USD millions"),
    ("unused_ci_m", 3, "the selected unused C&I commitments in USD millions"),
    ("opening_net_loans_leases_m", 3, "the selected opening FDIC net loans and leases in USD millions"),
    ("bridge_principal_m", 4, "the par bridge principal needed after using opening cash in USD millions"),
    ("hypothetical_provision_m", 4, "the stipulated credit-loss provision on new funded loans in USD millions"),
    ("ending_book_equity_m", 4, "book equity after the stipulated provision in USD millions"),
)
variables = [Variable(n, None, f"Store {d}" + (f", rounded to {p} decimals from unrounded inputs." if p is not None else ".")) for n, p, d in OUTPUTS]
PRECISION = {n: p for n, p, _ in OUTPUTS}
TURNS = (
    tuple(n for n, _, _ in OUTPUTS[:4]),
    tuple(n for n, _, _ in OUTPUTS[4:7]),
    tuple(n for n, _, _ in OUTPUTS[7:8]),
    tuple(n for n, _, _ in OUTPUTS[8:]),
)
M = Decimal(1000)  # Source values are thousand USD.


def dec(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite amount")
    return result


def analysis():
    commitments = load_expansion_table("ffiec_call_reports_commitments")
    fdic = load_expansion_table("fdic_bankfind")
    fdic = fdic.loc[fdic.REPDTE.eq("2025-09-30")]
    joined = commitments.merge(
        fdic[["CERT", "ASSET", "CHBAL", "EQ", "LNLSNET"]],
        left_on="fdic_certificate", right_on="CERT", validate="one_to_one",
    )
    good = joined.filing_type.eq("031") & joined.ASSET.gt(10_000_000)
    positive = ["unused_credit_card_lines_thousand_usd", "unused_ci_commitments_thousand_usd", "CHBAL", "EQ", "LNLSNET"]
    good &= joined[positive].notna().all(axis=1) & joined[positive].gt(0).all(axis=1)
    sample = joined.loc[good].copy()
    sample["ratio"] = (
        sample.unused_credit_card_lines_thousand_usd + sample.unused_ci_commitments_thousand_usd
    ) / sample.EQ * 100
    sample = sample.sort_values(["ratio", "bank_rssd_id"], ascending=[False, True])
    if len(sample) < 2:
        raise ValueError("insufficient complete banks")
    row, second = sample.iloc[0], sample.iloc[1]
    card = dec(row.unused_credit_card_lines_thousand_usd)
    ci = dec(row.unused_ci_commitments_thousand_usd)
    cash = dec(row.CHBAL)
    equity = dec(row.EQ)
    card_draw = card * Decimal("0.15")
    ci_draw = ci * Decimal("0.30")
    funded = card_draw + ci_draw
    bridge = max(Decimal(0), funded - cash)
    if bridge <= 0:
        raise ValueError("selected draws do not exercise bridge branch")
    provision = card_draw * Decimal("0.04") + ci_draw * Decimal("0.02")
    return {
        "eligible_bank_count": len(sample),
        "selected_rssd_id": str(row.bank_rssd_id),
        "selected_commitment_to_equity_pct": float((card + ci) / equity * 100),
        "runner_up_commitment_to_equity_pct": float((dec(second.unused_credit_card_lines_thousand_usd) + dec(second.unused_ci_commitments_thousand_usd)) / dec(second.EQ) * 100),
        "unused_card_m": float(card / M),
        "unused_ci_m": float(ci / M),
        "opening_net_loans_leases_m": float(dec(row.LNLSNET) / M),
        "bridge_principal_m": float(bridge / M),
        "hypothetical_provision_m": float(provision / M),
        "ending_book_equity_m": float((equity - provision) / M),
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


validators = {f"validate_turn_{i}": turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1, 5)}
