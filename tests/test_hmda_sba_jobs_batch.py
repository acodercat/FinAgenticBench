"""Data and scoring boundaries for the mortgage-flow and SBA-jobs batch."""

from evals.calculation_reconciliation import hmda_origination_vs_mortgage_transactions as mortgage
from evals.calculation_reconciliation import sba_reported_jobs_vs_state_employment as jobs


def _reported(case):
    result = case.analysis()
    return {
        v.name: round(result[v.name], case.PRECISION[v.name])
        if case.PRECISION[v.name] is not None else result[v.name]
        for v in case.variables
    }


def test_mortgage_origination_scope_and_vintages():
    result = mortgage.analysis()
    assert result["hmda_cohort_rows"] == 3_887_656
    assert result["latest_z1_release"] == "2025-09-11"
    assert result["first_full_z1_release"] == "2025-03-13"
    assert result["latest_z1_transactions_bn"] > result["first_full_z1_transactions_bn"]
    other_rows = result["hmda_cohort_rows"] - result["ordinary_refinance_rows"] - result["cashout_refinance_rows"]
    purchase_rows = int(mortgage._cohort().loan_purpose.eq("1").sum())
    assert other_rows - purchase_rows == 208_420
    assert result["purchase_gross_bn"] > result["latest_z1_transactions_bn"]


def test_mortgage_wrong_scope_and_unit_fail_strict_validator():
    reported = _reported(mortgage)
    assert mortgage.validate(reported).success
    wrong = dict(reported)
    wrong["purchase_gross_bn"] = round(
        mortgage.analysis()["hmda_gross_bn"]
        - mortgage.analysis()["ordinary_refinance_bn"]
        - mortgage.analysis()["cashout_refinance_bn"], 3
    )
    assert not mortgage.validate(wrong).success
    wrong = dict(reported)
    wrong["latest_z1_transactions_bn"] *= 1000
    assert not mortgage.validate(wrong).success
    wrong = dict(reported)
    wrong["hmda_gross_bn"] = round(mortgage.analysis()["hmda_gross_bn"], 3)
    assert not mortgage.validate(wrong).success


def test_mortgage_release_dates_require_exact_iso_tokens():
    reported = _reported(mortgage)
    assert mortgage.validate(reported).success
    for field in ("latest_z1_release", "first_full_z1_release"):
        for alternate in (
            reported[field].replace("-", "/"),
            reported[field].replace("2025-", ""),
            "September 11, 2025",
        ):
            wrong = dict(reported)
            wrong[field] = alternate
            assert not mortgage.validate(wrong).success


def test_sba_reported_jobs_are_not_state_employment_change():
    result = jobs.analysis()
    assert result["seven_a_rows"] == 7_330
    assert result["five_o_four_rows"] == 811
    assert result["qcew_2025_employment"] < result["qcew_2024_employment"]
    assert result["seven_a_jobs"] > 0 and result["five_o_four_jobs"] > 0
    assert jobs._cohort("sba_7a").duplicated().sum() == 3
    assert result["seven_a_zero_job_rows"] > 0
    assert result["five_o_four_zero_job_rows"] > 0


def test_sba_guarantee_and_zero_job_scope_fail_strict_validator():
    reported = _reported(jobs)
    assert jobs.validate(reported).success
    wrong = dict(reported)
    wrong["seven_a_guaranteed_m"] = wrong["seven_a_gross_m"]
    assert not jobs.validate(wrong).success
    wrong = dict(reported)
    wrong["five_o_four_zero_job_cdc_m"] = round(
        jobs.analysis()["five_o_four_zero_job_cdc_m"]
        + jobs.analysis()["five_o_four_zero_job_third_party_m"], 2
    )
    assert not jobs.validate(wrong).success
    wrong = dict(reported)
    wrong["seven_a_guaranteed_m"] = round(jobs.analysis()["seven_a_guaranteed_m"], 2)
    assert not jobs.validate(wrong).success
