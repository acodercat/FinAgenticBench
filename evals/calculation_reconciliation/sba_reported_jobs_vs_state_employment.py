"""SBA reported jobs supported are not measured net employment creation.

In the frozen 2026-06-30 snapshot, California FY2025 approvals first disbursed
by 2025-12-31 contain 7,330 7(a) rows and 811 504 rows. They report 72,245
and 10,293 jobs supported. The 7(a) rows have 3933.7961 USD million gross loan
approval and 2924.6249265 million guaranteed; 504 has 957.823 million
SBA/CDC loan approval and 1254.91718722 million third-party loans. Three
7(a) rows look identical in this privacy-reduced extract; none is deduplicated.

California QCEW private all-industry annual-average employment changes from
15,546,764 in calendar 2024 to 15,544,576 in calendar 2025. Subtracting the
two SBA reported-job totals from that change would conflate fiscal approval
cohorts, self-reported created/retained jobs and a statewide average stock.
847 7(a) rows with zero reported jobs carry 272.8045 million approved loans;
26 504 zero-job rows carry 22.756 million CDC and 33.340888 million third-party
approval. Excluding them lowers a descriptive financing-per-reported-job
quotient solely by changing the financing numerator and population. None of
these approved figures is cash subsidy cost, realized disbursement or causal
employment impact. The query pins its retrospective population, so this is a
baseline with measured wrong readings for regression, not a live trap.
"""

from functools import lru_cache

import pandas as pd
from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


OUTPUTS = (
    ("seven_a_rows", 0, "the eligible 7(a) reported-row count as an integer"),
    ("seven_a_gross_m", 4, "the 7(a) gross approval in USD millions"),
    ("seven_a_guaranteed_m", 4, "the 7(a) SBA-guaranteed approval in USD millions"),
    ("seven_a_jobs", 0, "the 7(a) reported jobs-supported total as an integer"),
    ("five_o_four_rows", 0, "the eligible 504 reported-row count as an integer"),
    ("five_o_four_cdc_m", 4, "the 504 SBA/CDC gross approval in USD millions"),
    ("five_o_four_third_party_m", 4, "the 504 third-party loan amount in USD millions"),
    ("five_o_four_jobs", 0, "the 504 reported jobs-supported total as an integer"),
    ("qcew_2024_employment", 0, "California's 2024 private all-industry annual-average employment as an integer"),
    ("qcew_2025_employment", 0, "California's 2025 private all-industry annual-average employment as an integer"),
    ("seven_a_zero_job_rows", 0, "the eligible 7(a) zero-reported-jobs row count as an integer"),
    ("seven_a_zero_job_gross_m", 4, "gross 7(a) approved loans on those rows in USD millions"),
    ("five_o_four_zero_job_rows", 0, "the eligible 504 zero-reported-jobs row count as an integer"),
    ("five_o_four_zero_job_cdc_m", 4, "SBA/CDC approved loans on those rows in USD millions"),
    ("five_o_four_zero_job_third_party_m", 4, "third-party approved loans on those rows in USD millions"),
)
variables = [
    Variable(name, None, f"Store {description}, rounded to {precision} decimals from unrounded source rows." if precision else f"Store {description}.")
    for name, precision, description in OUTPUTS
]
PRECISION = {name: precision for name, precision, _ in OUTPUTS}
TURNS = (
    tuple(name for name, *_ in OUTPUTS[:4]),
    tuple(name for name, *_ in OUTPUTS[4:8]),
    tuple(name for name, *_ in OUTPUTS[8:10]),
    tuple(name for name, *_ in OUTPUTS[10:]),
)


@lru_cache(maxsize=2)
def _cohort(source):
    data = load_expansion_table(source)
    cohort = data.loc[
        data.ApprovalFY.eq(2025)
        & data.ProjectState.eq("CA")
        & data.FirstDisbursementDate.notna()
        & data.FirstDisbursementDate.astype(str).le("2025-12-31")
    ].copy()
    fields = ["GrossApproval", "JobsSupported"]
    fields += ["SBAGuaranteedApproval"] if source == "sba_7a" else ["ThirdPartyDollars"]
    for field in fields:
        cohort[field] = pd.to_numeric(cohort[field], errors="coerce")
    if cohort.empty or cohort[fields].isna().any().any():
        raise ValueError(f"{source} cohort empty or has missing measures")
    if (cohort.GrossApproval <= 0).any() or (cohort.JobsSupported < 0).any():
        raise ValueError(f"{source} cohort has invalid financing or jobs")
    if source == "sba_7a":
        if (cohort.SBAGuaranteedApproval < 0).any() or (cohort.SBAGuaranteedApproval > cohort.GrossApproval).any():
            raise ValueError("invalid 7(a) guaranteed approval")
    elif (cohort.ThirdPartyDollars <= 0).any():
        raise ValueError("invalid 504 third-party loan")
    return cohort


@lru_cache(maxsize=1)
def _employment():
    data = load_expansion_table("bls_qcew")
    selected = data.loc[
        data.state_fips.eq("06")
        & data.aggregation_level.eq("state_by_ownership")
        & data.ownership_scope.eq("private")
        & data.industry_code.astype(str).str.zfill(2).eq("10")
        & data.year.isin([2024, 2025])
    ]
    if len(selected) != 2 or selected.year.nunique() != 2 or selected.annual_avg_employment.isna().any():
        raise ValueError("QCEW annual employment is not one row per year")
    return dict(zip(selected.year, selected.annual_avg_employment))


@lru_cache(maxsize=1)
def analysis():
    seven = _cohort("sba_7a")
    five = _cohort("sba_504")
    seven_zero = seven.loc[seven.JobsSupported.eq(0)]
    five_zero = five.loc[five.JobsSupported.eq(0)]
    employment = _employment()
    m = 1_000_000
    return {
        "seven_a_rows": len(seven),
        "seven_a_gross_m": float(seven.GrossApproval.sum() / m),
        "seven_a_guaranteed_m": float(seven.SBAGuaranteedApproval.sum() / m),
        "seven_a_jobs": int(seven.JobsSupported.sum()),
        "five_o_four_rows": len(five),
        "five_o_four_cdc_m": float(five.GrossApproval.sum() / m),
        "five_o_four_third_party_m": float(five.ThirdPartyDollars.sum() / m),
        "five_o_four_jobs": int(five.JobsSupported.sum()),
        "qcew_2024_employment": int(employment[2024]),
        "qcew_2025_employment": int(employment[2025]),
        "seven_a_zero_job_rows": len(seven_zero),
        "seven_a_zero_job_gross_m": float(seven_zero.GrossApproval.sum() / m),
        "five_o_four_zero_job_rows": len(five_zero),
        "five_o_four_zero_job_cdc_m": float(five_zero.GrossApproval.sum() / m),
        "five_o_four_zero_job_third_party_m": float(five_zero.ThirdPartyDollars.sum() / m),
    }


@lru_cache(maxsize=1)
def ground_truth():
    answers = analysis()
    return tuple(answers[v.name] for v in variables)


def _check(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names], [truth[n] for n in names], [PRECISION[n] for n in names])


def validate(outputs):
    return _check(outputs, tuple(v.name for v in variables))


validators = {
    f"validate_turn_{i}": turn_validator(lambda outputs, names=names: _check(outputs, names))
    for i, names in enumerate(TURNS, 1)
}
