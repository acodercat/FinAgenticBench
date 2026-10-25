"""Financial and source-native checks for RC-S recourse and N-PORT repo cases."""

import zipfile

import pandas as pd
import pytest

from core.data import load_expansion_table
from core.dataset_layers import RAW_DIR
from evals.scenario_analysis import bank_sold_asset_recourse_cap_path as bank
from evals.scenario_analysis import nport_reverse_repo_collateral_floor as fund


def _reported(module):
    data = module.analysis()
    return {name: round(value, module.PRECISION[name])
            if module.PRECISION[name] is not None else value
            for name, value in data.items()}


def test_rcs_pool_caps_and_fresh_scenario():
    result = bank.analysis()
    assert result["rcs_bank_count"] == 12
    assert result["rcs_selected_rssd"] == "280110"
    assert result["rcs_ten_pct_payment_m"] == pytest.approx(2479.948)
    assert result["rcs_forty_pct_payment_m"] == pytest.approx(8043.739)
    assert bank.validate(_reported(bank)).success
    wrong = _reported(bank)
    wrong["rcs_forty_pct_payment_m"] = round(result["rcs_forty_pct_payment_m"] + result["rcs_ten_pct_payment_m"], 4)
    assert not bank.validate(wrong).success
    wrong = _reported(bank)
    wrong["rcs_ten_pct_payment_m"] = round((9.913 + 24798.35) * .1, 4)
    assert not bank.validate(wrong).success


def test_nport_position_grain_and_contract_rate():
    rows = load_expansion_table("sec_nport_repo_collateral")
    assert rows["holding_id"].nunique() == 9
    assert not rows.duplicated(["accession", "holding_id", "collateral_id"]).any()
    result = fund.analysis()
    assert (result["repo_position_count"], result["reverse_position_count"]) == (1, 8)
    assert result["reverse_covered_count"] == 3
    assert result["reverse_selected_holding_id"] == "153742555"
    assert result["reverse_sofr_spread_bp"] == 4
    assert result["reverse_floor_topup_m"] == pytest.approx(1.4003848125)
    assert fund.validate(_reported(fund)).success
    wrong = _reported(fund)
    wrong["reverse_floor_topup_m"] = round(13.4457348125, 4)
    assert not fund.validate(wrong).success


@pytest.mark.raw_data
def test_source_native_values_match_governed_projections():
    bank_archive = RAW_DIR / "ffiec_call_reports" / "FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
    with zipfile.ZipFile(bank_archive) as z:
        with z.open("FFIEC CDR Call Schedule RCS 09302025.txt") as handle:
            native = pd.read_csv(handle, sep="\t", skiprows=[1], dtype="string",
                usecols=["IDRSSD", "RCFDB790", "RCFDB796", "RCFDB797", "RCFDB803"])
    source = native.loc[native.IDRSSD.eq("280110")].iloc[0]
    governed = load_expansion_table("ffiec_call_reports_recourse")
    row = governed.loc[governed.bank_rssd_id.eq("280110")].iloc[0]
    assert tuple(int(source[c]) for c in ["RCFDB790", "RCFDB796", "RCFDB797", "RCFDB803"]) == tuple(
        int(row[c]) for c in ["residential_sold_principal_thousand_usd",
                           "other_sold_principal_thousand_usd",
                           "residential_max_recourse_thousand_usd",
                           "other_max_recourse_thousand_usd"])
    fund_archive = RAW_DIR / "sec_nport" / "2025q3_nport.zip"
    with zipfile.ZipFile(fund_archive) as z:
        with z.open("REPURCHASE_COLLATERAL.tsv") as handle:
            native_collateral = pd.read_csv(handle, sep="\t", dtype="string",
                usecols=["HOLDING_ID", "COLLATERAL_AMOUNT", "PRINCIPAL_AMOUNT"])
    selected = native_collateral.loc[native_collateral.HOLDING_ID.eq("153742555")]
    assert len(selected) == 1
    governed = load_expansion_table("sec_nport_repo_collateral")
    position = governed.loc[governed.holding_id.eq("153742555")].iloc[0]
    assert float(selected.iloc[0].COLLATERAL_AMOUNT) == position.collateral_value_usd
    assert float(selected.iloc[0].PRINCIPAL_AMOUNT) == position.collateral_security_principal
