"""RC-E maturity/repricing case numerical and source-native checks."""

import zipfile

import pandas as pd
import pytest

from core.data import load_expansion_table
from core.dataset_layers import RAW_DIR
from evals.scenario_analysis import bank_time_deposit_reset_maturity_replan as case


def _reported():
    return {k: round(v, case.PRECISION[k]) if case.PRECISION[k] is not None else v
            for k, v in case.analysis().items()}


def test_reset_maturity_and_replanning():
    result = case.analysis()
    assert result["time_deposit_eligible_banks"] == 86
    assert result["time_deposit_selected_rssd"] == "673440"
    assert result["time_deposit_half_renewal_cash_m"] == pytest.approx(72.091)
    assert result["time_deposit_floor_min_renewal_pct"] == pytest.approx(52.579723809)
    assert case.validate(_reported()).success
    wrong = _reported()
    wrong["time_deposit_half_renewal_cash_m"] = round(
        613.021 - .5 * (709.935 + 804.584), 3)
    assert not case.validate(wrong).success
    wrong = _reported()
    wrong["time_deposit_floor_extra_expense_m"] = round(
        result["time_deposit_floor_extra_expense_m"] +
        result["time_deposit_half_renewal_extra_expense_m"], 3)
    assert not case.validate(wrong).success


@pytest.mark.raw_data
def test_selected_bank_rc_e_raw_projection():
    source = RAW_DIR / "ffiec_call_reports" / "FFIEC_CDR_Call_Bulk_All_Schedules_09302025.zip"
    with zipfile.ZipFile(source) as archive:
        with archive.open("FFIEC CDR Call Schedule RCE 09302025.txt") as handle:
            raw = pd.read_csv(handle, sep="\t", skiprows=[1], dtype="string",
                              usecols=["IDRSSD", "RCONHK07", "RCONHK08", "RCONHK11",
                                       "RCONHK12", "RCONHK13", "RCONK222"])
    row = raw.loc[raw.IDRSSD.eq("673440")].iloc[0]
    governed = load_expansion_table("ffiec_call_reports_time_deposits")
    projected = governed.loc[governed.bank_rssd_id.eq("673440")].iloc[0]
    native = ["RCONHK07", "RCONHK08", "RCONHK11", "RCONHK12", "RCONHK13", "RCONK222"]
    named = ["small_within_3m_maturity_or_reset_thousand_usd",
             "small_3_to_12m_maturity_or_reset_thousand_usd",
             "small_within_1y_maturity_thousand_usd",
             "large_within_3m_maturity_or_reset_thousand_usd",
             "large_3_to_12m_maturity_or_reset_thousand_usd",
             "large_within_1y_maturity_thousand_usd"]
    assert tuple(int(row[c]) for c in native) == tuple(int(projected[c]) for c in named)
