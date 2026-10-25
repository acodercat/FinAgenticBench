"""13F amendment lineage and concentration-decision checks."""

import zipfile

import pandas as pd
import pytest

from core.data import load_expansion_table
from core.dataset_layers import RAW_DIR
from evals.temporal_path import thirteen_f_amendment_public_position_path as case


def _reported():
    return {k: round(v, case.PRECISION[k]) if case.PRECISION[k] is not None else v
            for k, v in case.analysis().items()}


def test_restated_plus_additive_public_table():
    result = case.analysis()
    assert result["thirteen_f_chain_count"] == 4
    assert result["thirteen_f_latest_restatement"] == "0001214659-25-004240"
    assert result["thirteen_f_added_accession"] == "0001214659-25-004241"
    assert result["thirteen_f_final_value_m"] == pytest.approx(263.917093)
    assert result["thirteen_f_first_cutoff_sale_shares"] == 564
    assert result["thirteen_f_later_cutoff_sale_shares"] == 0
    assert case.validate(_reported()).success
    wrong = _reported()
    wrong["thirteen_f_final_value_m"] = 1041.907
    assert not case.validate(wrong).success
    wrong = _reported()
    wrong["thirteen_f_final_value_m"] = 4.002
    assert not case.validate(wrong).success
    wrong = _reported()
    wrong["thirteen_f_later_cutoff_sale_shares"] = 564
    assert not case.validate(wrong).success


@pytest.mark.raw_data
def test_amendment_information_tables_match_source():
    governed = load_expansion_table("sec_13f_amendment_chain")
    for archive_name, accession in [
        ("01dec2024-28feb2025_form13f.zip", "0001214659-25-002354"),
        ("01dec2024-28feb2025_form13f.zip", "0001214659-25-002443"),
        ("01mar2025-31may2025_form13f.zip", "0001214659-25-004240"),
        ("01mar2025-31may2025_form13f.zip", "0001214659-25-004241"),
    ]:
        with zipfile.ZipFile(RAW_DIR / "sec_13f" / archive_name) as archive:
            with archive.open("INFOTABLE.tsv") as handle:
                raw = pd.read_csv(handle, sep="\t", dtype="string",
                                  usecols=["ACCESSION_NUMBER", "INFOTABLE_SK", "VALUE"])
        raw = raw.loc[raw.ACCESSION_NUMBER.eq(accession)]
        projection = governed.loc[governed.accession.eq(accession)]
        assert len(raw) == len(projection)
        assert set(raw.INFOTABLE_SK) == set(projection.source_row_key)
        assert int(pd.to_numeric(raw.VALUE).sum()) == int(projection.reported_value_usd.sum())
