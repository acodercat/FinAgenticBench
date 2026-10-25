"""Fed stressed pretax-income bridge and a bounded revenue sensitivity.

The 2025 severely adverse results contain three matched firms with positive
PPNR less loan provisions but negative reported pretax income. The simplified
subtotal is not an accounting identity for pretax income: the Fed table also
contains other revenue and loss components. The selected residual is a signed
unreported-in-runtime bridge, not a measured trading loss. A 25 bp opening-
assets PPNR increase is a hypothetical horizon-total sensitivity, not a Fed
projection or a CET1-ratio calculation. Fed figures are USD billions, whereas
the Y-9C asset field is USD thousands. The prompt pins these conventions, so
the wrong shortcut is regression coverage, not a live diagnostic trap.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("stress_bridge_matched_firms", 0, "number of matched named firms as an integer"),
    ("stress_bridge_reversal_firms", 0, "number of matched firms with a positive simplified PPNR-after-provision result but negative reported pretax income as an integer"),
    ("stress_bridge_rssd", None, "selected institution RSSD identifier as an exact digit string"),
    ("stress_bridge_assets_b", 4, "selected firm's 2024 year-end Y-9C consolidated assets in USD billions"),
    ("stress_bridge_ppnr_b", 4, "selected firm's cumulative projected pre-provision net revenue in USD billions"),
    ("stress_bridge_provision_b", 4, "selected firm's cumulative projected loan and lease loss provisions in USD billions"),
    ("stress_bridge_pretax_b", 4, "selected firm's reported projected pretax income in USD billions"),
    ("stress_bridge_residual_b", 4, "signed residual that reconciles PPNR after loan provisions to reported pretax income in USD billions, calculated from unrounded source amounts"),
    ("stress_bridge_restored_firms", 0, "number of carried firms with strictly positive pretax income under the hypothetical cumulative PPNR increase as an integer"),
    ("stress_bridge_selected_stressed_pretax_b", 4, "selected firm's pretax income under that hypothetical cumulative PPNR increase in USD billions"),
)
variables = [Variable(name, None, "Store " + description +
                      (f", rounded to {places} decimals." if places and places > 0 else "."))
             for name, places, description in OUTPUTS]
PRECISION = {name: places for name, places, _ in OUTPUTS}
TURNS = tuple(tuple(name for name, _, _ in OUTPUTS[a:b]) for a, b in
              ((0, 2), (2, 4), (4, 8), (8, 10)))


def dec(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite amount")
    return result


def analysis() -> dict[str, str | int | float]:
    stress = load_expansion_table("fed_stress_tests")
    y9c = load_expansion_table("ffiec_y9c")
    stress = stress.loc[
        stress.exercise_name.eq("2025 Stress Test")
        & stress.scenario_name.eq("Supervisory Severely Adverse")
        & stress.id_rssd.notna()
    ].copy()
    stress["rssd"] = pd.to_numeric(stress.id_rssd, errors="raise").astype("int64").astype(str)
    if stress.rssd.duplicated().any():
        raise ValueError("duplicate firm in stress exercise")
    y9c = y9c.loc[y9c.report_date.eq("2024-12-31")].copy()
    y9c["rssd"] = pd.to_numeric(y9c.reporter_rssd_id, errors="raise").astype("int64").astype(str)
    if y9c.rssd.duplicated().any():
        raise ValueError("duplicate Y-9C reporter/date")
    matched = stress.merge(y9c[["rssd", "consolidated_total_assets_thousand_usd"]],
                           on="rssd", how="inner", validate="one_to_one")
    rows = []
    for row in matched.itertuples(index=False):
        assets_b = dec(row.consolidated_total_assets_thousand_usd) / Decimal(1_000_000)
        if assets_b <= 0:
            continue
        ppnr = dec(row.revenue_preprovision_net_amt)
        provision = dec(row.provision_amt)
        pretax = dec(row.income_pretax_net_amt)
        subtotal = ppnr - provision
        residual = pretax - subtotal
        rows.append((row.rssd, assets_b, ppnr, provision, pretax, subtotal, residual))
    reversals = [r for r in rows if r[5] > 0 and r[4] < 0]
    if not reversals:
        raise ValueError("no sign-reversal cohort")
    selected = min(reversals, key=lambda r: (r[6] / r[1], int(r[0])))
    uplift = lambda r: Decimal("0.0025") * r[1]
    return {
        "stress_bridge_matched_firms": len(rows),
        "stress_bridge_reversal_firms": len(reversals),
        "stress_bridge_rssd": selected[0],
        "stress_bridge_assets_b": float(selected[1]),
        "stress_bridge_ppnr_b": float(selected[2]),
        "stress_bridge_provision_b": float(selected[3]),
        "stress_bridge_pretax_b": float(selected[4]),
        "stress_bridge_residual_b": float(selected[6]),
        "stress_bridge_restored_firms": sum(r[4] + uplift(r) > 0 for r in reversals),
        "stress_bridge_selected_stressed_pretax_b": float(selected[4] + uplift(selected)),
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
