"""Targeted financial and acceptance checks for the BDC commitment path."""

from core.data import load_expansion_table
from evals.temporal_path import bdc_commitment_mark_maturity_path as case


def _display():
    values = case.analysis()
    return {
        name: round(value, case.PRECISION[name])
        if case.PRECISION[name] is not None else value
        for name, value in values.items()
    }


def test_original_filing_and_funded_unfunded_cohorts():
    values = case.analysis()
    assert values["bdc_maturity_accession"] == "0001193125-25-271826"
    assert values["bdc_maturity_assets_m"] == 2142.302058
    assert values["bdc_maturity_cash_m"] == 47.135851
    assert values["bdc_maturity_schedule_fair_m"] == 2069.092639
    assert values["bdc_maturity_funded_count"] == 342
    assert values["bdc_maturity_negative_count"] == 55
    assert values["bdc_maturity_negative_fair_m"] == -2.31728
    assert case.validate(_display()).success


def test_missing_mark_and_negative_rows_do_not_become_funded_principal():
    schedule = load_expansion_table("sec_bdc")
    rows = schedule.loc[
        schedule.accession.eq("0001193125-25-271826")
        & schedule.report_date.eq("2025-09-30")
    ]
    assert len(rows) == 428
    assert not rows.investment_id.duplicated().any()
    assert ((rows.principal_amount_usd > 0) & rows.fair_value_usd.isna()).sum() == 1
    assert rows.loc[rows.fair_value_usd < 0, "principal_amount_usd"].eq(0).all()
    wrong = {**_display(), "bdc_maturity_funded_count": 343}
    assert not case.validate(wrong).success
    wrong = {**_display(), "bdc_maturity_schedule_fair_m": 2071.4099}
    assert not case.validate(wrong).success


def test_dynamic_maturity_boundary_and_negative_mark_subset():
    values = case.analysis()
    assert values["bdc_maturity_threshold_date"] == "2028-09-30"
    assert values["bdc_maturity_by_threshold_count"] == 67
    assert values["bdc_maturity_by_threshold_principal_m"] == 219.302955
    assert values["bdc_maturity_by_threshold_fair_m"] == 214.352424
    assert values["bdc_maturity_negative_by_threshold_count"] == 6
    assert values["bdc_maturity_negative_by_threshold_fair_m"] == -0.051437
    assert values["bdc_maturity_by_threshold_principal_m"] >= (
        values["bdc_maturity_funded_principal_m"] * 0.10)
    wrong = {**_display(), "bdc_maturity_threshold_date": "2027-09-30"}
    assert not case.validate(wrong).success
    wrong = {**_display(), "bdc_maturity_negative_by_threshold_fair_m": 0.0514}
    assert not case.validate(wrong).success


def test_display_boundary_and_wrong_unit_are_distinct():
    values = case.analysis()
    name = "bdc_maturity_by_threshold_principal_m"
    near = {**_display(), name: values[name] + 0.000059}
    outside = {**_display(), name: values[name] + 0.000061}
    wrong_unit = {**_display(), name: values[name] * 1_000_000}
    assert case.validate(near).success
    assert not case.validate(outside).success
    assert not case.validate(wrong_unit).success
