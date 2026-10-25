"""Boundary checks for the SBA risk-transfer and ARCC PIK cases."""

from evals.scenario_analysis import arcc_pik_income_dividend_nav_stress as bdc
from evals.scenario_analysis import sba_504_7a_private_loss_syndication as sba


def _answers(case):
    return dict(zip((variable.name for variable in case.variables), case.ground_truth()))


def _reported(module):
    answers = _answers(module)
    return {
        variable.name: round(answers[variable.name], precision)
        if precision is not None else answers[variable.name]
        for variable, precision in zip(module.variables, module.DECIMALS)
    }


def test_sba_program_risk_and_whole_row_boundary():
    answers = _answers(sba)
    assert sba.validate(answers).success
    assert answers["leader_state"] == "CA"
    assert answers["leader_naics2"] == "72"
    assert answers["minimum_whole_504_sales"] == 8
    assert not sba.validate(dict(answers, leader_private_approval_m=980.587173)).success
    assert not sba.validate(dict(answers, minimum_whole_504_sales=7)).success


def test_bdc_accrual_and_valuation_boundaries():
    answers = _answers(bdc)
    assert bdc.validate(answers).success
    assert answers["q3_recognized_pik_interest_m"] == 59
    assert answers["q3_unaffiliated_pik_interest_m"] == 50
    assert answers["q3_after_tax_nii_m"] == 338
    assert answers["partial_cash_like_dividend_gap_m"] == 63
    assert round(answers["bridge_interest_m"], 4) == 0.3276
    assert not bdc.validate(dict(answers, q3_after_tax_nii_m=349)).success
    assert answers["stressed_equity_m"] < 14_000
    assert not bdc.validate(dict(answers, q3_recognized_pik_interest_m=165)).success
    assert not bdc.validate(dict(answers, q3_unaffiliated_pik_interest_m=59)).success
    assert not bdc.validate(dict(answers, partial_cash_like_dividend_gap_m=0)).success
    assert not bdc.validate(dict(answers, bridge_interest_m=0.0)).success


def test_strict_display_boundary_for_each_numeric_output():
    for module in (sba, bdc):
        answers = _answers(module)
        reported = _reported(module)
        for variable, precision in zip(module.variables, module.DECIMALS):
            if precision in (None, 0):
                continue
            outside = dict(reported)
            outside[variable.name] = answers[variable.name] + .7 * 10 ** (-precision)
            assert not module.validate(outside).success, variable.name


def test_derived_outputs_are_pinned_coarser_than_their_operands():
    """Both quantities are arithmetic functions of outputs reported in earlier turns,
    so they are pinned at least one decimal coarser than their operands."""
    sba_precision = dict(zip((v.name for v in sba.variables), sba.DECIMALS))
    assert sba_precision["seven_a_loss_share_pct"] < sba_precision["leader_504_private_share_pct"]
    assert sba_precision["modeled_private_loss_m"] < min(
        sba_precision["leader_private_approval_m"], sba_precision["leader_504_private_share_pct"]
    )
    bdc_precision = dict(zip((v.name for v in bdc.variables), bdc.DECIMALS))
    assert bdc_precision["partial_cash_like_dividend_gap_m"] < min(
        bdc_precision["q3_after_tax_nii_m"], bdc_precision["q3_recognized_pik_interest_m"]
    )
