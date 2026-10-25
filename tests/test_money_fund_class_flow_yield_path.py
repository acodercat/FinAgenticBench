"""Financial and source-native checks for the share-class flow/yield path."""

from decimal import Decimal
import csv
import io
import zipfile

import pytest

from core.data import load_expansion_table
from core.dataset_layers import RAW_DIR
from evals.temporal_path import money_fund_class_flow_yield_path as case


def _reported():
    return {name: round(value, case.PRECISION[name]) if case.PRECISION[name] is not None else value
            for name, value in case.analysis().items()}


def test_fixed_class_ids_and_cross_month_flow_reversal():
    actual = case.analysis()
    assert actual["class_flow_sep_matched_classes"] == 6
    assert actual["class_flow_inflow_id"] == "C000013784"
    assert actual["class_flow_outflow_id"] == "C000013788"
    assert actual["class_flow_inflow_net_yield_pct"] < actual["class_flow_outflow_net_yield_pct"]
    assert actual["class_flow_inflow_m"] > 0 > actual["class_flow_outflow_m"]
    assert actual["class_flow_oct_outflow_id_net_m"] > 0
    assert actual["class_flow_oct_total_m"] < actual["class_flow_sep_total_m"]


def test_same_day_keys_and_original_version_are_unique():
    series = load_expansion_table("sec_nmfp")
    flows = load_expansion_table("sec_nmfp_flows")
    yields = load_expansion_table("sec_nmfp_class_yields")
    for date in ("2025-09-30", "2025-10-31"):
        filing = series.loc[series.series_id.eq(case.SERIES_ID) & series.report_date.eq(date)]
        assert len(filing) == 1 and filing.iloc[0].form == "N-MFP3"
        accession = filing.iloc[0].accession
        f = flows.loc[flows.accession.eq(accession) & flows.flow_date.eq(date)]
        y = yields.loc[yields.accession.eq(accession) & yields.yield_date.eq(date)]
        assert len(f) == len(y) == 6
        assert not f.class_id.duplicated().any() and not y.class_id.duplicated().any()
        assert set(f.class_id) == set(y.class_id)
        assert y.seven_day_gross_yield.nunique() == 1


def test_strict_validator_rejects_gross_flow_and_unit_paths():
    reported = _reported()
    assert case.validate(reported).success
    for name, wrong_value in (
        ("class_flow_inflow_m", 2164.7783),  # gross subscriptions, not net flow
        ("class_flow_outflow_m", -1725.6992),  # gross redemptions, not net flow
        ("class_flow_sep_total_m", 1179.6854),  # only the two selected classes
        ("class_flow_oct_outflow_id_net_m", -17.5722),  # copied September sign
        ("class_flow_inflow_net_yield_pct", 0.0395),  # decimal rate, not percent
    ):
        wrong = dict(reported)
        wrong[name] = wrong_value
        assert not case.validate(wrong).success, name


def test_decimal_precision_boundaries_and_exact_ids():
    reported = _reported()
    for name in ("class_flow_sep_assets_m", "class_flow_inflow_m", "class_flow_oct_total_m"):
        for sign in (-1, 1):
            wrong = dict(reported)
            wrong[name] = case.analysis()[name] + sign * 0.00007
            assert not case.validate(wrong).success
    for name in ("class_flow_sep_accession", "class_flow_inflow_id"):
        wrong = dict(reported)
        wrong[name] = (reported[name].lstrip("0") if name.endswith("accession")
                       else "C" + reported[name][2:])
        assert not case.validate(wrong).success


@pytest.mark.raw_data
def test_native_sec_rows_independently_reproduce_path():
    """Use the SEC zip member fields directly, without the case's calculation path."""
    expected = {
        "2025-09-30": ("20250909-20251007_nmfp.zip", "0001410368-25-013991", Decimal("554849663.47")),
        "2025-10-31": ("20251008-20251107_nmfp.zip", "0001410368-25-021425", Decimal("6187798.38")),
    }
    for date, (archive, accession, net_total) in expected.items():
        with zipfile.ZipFile(RAW_DIR / "sec_nmfp" / archive) as bundle:
            def rows(member):
                with bundle.open(member) as stream:
                    return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"), delimiter="\t"))

            submission = [r for r in rows("NMFP_SUBMISSION.tsv") if r["ACCESSION_NUMBER"] == accession]
            assert len(submission) == 1 and submission[0]["SUBMISSIONTYPE"] == "N-MFP3"
            daily = [r for r in rows("NMFP_DLYSHAREHOLDERFLOWREPORT.tsv")
                     if r["ACCESSION_NUMBER"] == accession
                     and r["DAILYSHAREHOLDERFLOWDATE"] == ("30-SEP-2025" if date.endswith("09-30") else "31-OCT-2025")]
            net_yields = [r for r in rows("NMFP_SEVENDAYNETYIELD.tsv")
                          if r["ACCESSION_NUMBER"] == accession
                          and r["SEVENDAYNETYIELDDATE"] == ("30-SEP-2025" if date.endswith("09-30") else "31-OCT-2025")]
            gross = [r for r in rows("NMFP_SEVENDAYGROSSYIELD.tsv")
                     if r["ACCESSION_NUMBER"] == accession
                     and r["SEVENDAYGROSSYIELDDATE"] == ("30-SEP-2025" if date.endswith("09-30") else "31-OCT-2025")]
            assert len(daily) == len(net_yields) == 6 and len(gross) == 1
            assert {r["CLASSESID"] for r in daily} == {r["CLASSESID"] for r in net_yields}
            by_id = {r["CLASSESID"]: Decimal(r["SEVENDAYNETYIELDVALUE"].strip()) for r in net_yields}
            signed = {r["CLASSESID"]: Decimal(r["DAILYGROSSSUBSCRIPTIONS"].strip())
                      - Decimal(r["DAILYGROSSREDEMPTIONS"].strip()) for r in daily}
            assert sum(signed.values()) == net_total
            if date == "2025-09-30":
                assert max(signed, key=signed.get) == "C000013784"
                assert min(signed, key=signed.get) == "C000013788"
                assert signed["C000013784"] == Decimal("2088371001.84")
                assert signed["C000013788"] == Decimal("-908685565.65")
                assert by_id["C000013784"] == Decimal(".0395")
                assert by_id["C000013788"] == Decimal(".0405")
                assert Decimal(gross[0]["SEVENDAYGROSSYIELDVALUE"].strip()) == Decimal(".0419")
            else:
                assert signed["C000013784"] == Decimal("484128931.26")
                assert signed["C000013788"] == Decimal("17572180.27")
                assert by_id["C000013784"] == Decimal(".0394")
                assert by_id["C000013788"] == Decimal(".0404")
