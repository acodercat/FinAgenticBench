"""Oracle 13F option-underlier and Form 4 population boundary."""

from evals.temporal_path import thirteen_f_option_underlier_exposure_boundary as case
from core.data import load_expansion_table


def _reported():
    values = case.analysis()
    return {name: round(value, case.PRECISION[name]) if isinstance(value, float) else value
            for name, value in values.items()}


def test_underlier_ordering_and_form4_instrument_split():
    values = _reported()
    assert values["option_underlier_cusip"] == "68389X105"
    assert values["option_underlier_q4_put_bn"] > values["option_underlier_q4_call_bn"]
    assert values["option_underlier_q2_put_bn"] < values["option_underlier_q2_call_bn"]
    assert values["option_underlier_q4_form4_m_rows"] == 1
    assert values["option_underlier_q2_form4_m_rows"] == 20
    assert values["option_underlier_q2_form4_stock_option_m_rows"] == 10
    assert case.validate(values).success


def test_wrong_unit_and_all_m_as_options_are_rejected():
    correct = _reported()
    wrong_unit = {**correct, "option_underlier_q4_put_bn": 4327.612410}
    assert not case.validate(wrong_unit).success
    wrong_instrument = {**correct, "option_underlier_q2_form4_stock_option_m_rows": 20}
    assert not case.validate(wrong_instrument).success


def test_literal_cusip_ranking_resolves_source_case_variant():
    rows = load_expansion_table("sec_13f_holdings")
    q4 = rows.loc[
        rows.period_of_report.eq("2024-12-31") &
        rows.issuer_name.eq("ORACLE CORP") & rows.title_of_class.eq("COM")]
    assert set(q4.cusip) == {"068389X10", "68389X105"}
    ranked = q4.sort_values("share_value_usd_total", ascending=False)
    leader, runner_up = ranked.iloc[0], ranked.iloc[1]
    assert leader.cusip == case.analysis()["option_underlier_cusip"]
    assert leader.share_value_usd_total > 100_000 * runner_up.share_value_usd_total
