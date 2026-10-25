from __future__ import annotations

import json

from config import DATASETS_DIR
from core.data import load_expansion_table, runtime_variables


def test_phase4_governed_catalog_and_table_selection() -> None:
    catalog = json.loads(
        (DATASETS_DIR / "catalog.json").read_text(encoding="utf-8")
    )
    source_by_table = {
        "dol_form5500.plan_financials": "dol_form5500_financials",
        "dol_form5500.schedule_sb": "dol_form5500_schedule_sb",
        "ffiec_call_reports.bank_capital": "ffiec_call_reports_capital",
        "treasury_debt.debt_to_penny": "treasury_debt_to_penny",
        "treasury_debt.marketable_securities": "treasury_marketable_securities",
        "treasury_usfr.reconciliations": "treasury_usfr_reconciliations",
    }
    for table_id, source in source_by_table.items():
        assert len(load_expansion_table(source)) == catalog["tables"][table_id]["row_count"]
    assert [
        variable.name for variable in runtime_variables(["dol_form5500_financials"])
    ] == ["form5500_plan_financials_df"]
    assert [
        variable.name for variable in runtime_variables(["ffiec_call_reports_capital"])
    ] == ["ffiec_bank_capital_df"]

