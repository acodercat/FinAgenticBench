"""Amortized-cost measurements and conservation of reinsurance capacity."""
import numpy as np

from evals.temporal_path import debt_modification_effective_interest_derecognition as d
from evals.temporal_path import flood_occurrence_window_reinstatement_treaty as t
from scripts.verdict_baseline import mutate


def test_debt_initial_and_modified_amortization_and_measurement_rates():
    result = d.analysis()
    face, *_ = d.inputs()
    rate = result['original_semiannual_eir_pct'] / 100
    original = np.full(10, .025 * face); original[-1] += face
    remaining, _ = d.accrue(result['net_issuance_mn'], rate, original)
    assert abs(remaining) < 1e-8
    assert abs(d.pv(original[4:], rate) - result['pre_modification_carrying_mn']) < 1e-8
    modified = np.full(8, .02 * face); modified[-1] += face
    modified_rate = result['A_fee_amortizing_semiannual_eir_pct'] / 100
    remaining, _ = d.accrue(result['A_adjusted_carrying_mn'], modified_rate, modified)
    assert abs(remaining) < 1e-8
    assert abs(result['A_adjusted_carrying_mn'] + .01 * face - d.pv(modified, rate)) < 1e-8
    assert abs(d.pv(modified[2:], modified_rate) - result['second_prechange_carrying_mn']) < 1e-8
    assert result['A_ten_percent_test_pct'] < 10 < result['second_ten_percent_test_pct']
    assert modified_rate > rate
    assert result['B_classification'] == 'DERECOGNIZE'
    reduced = np.full(6, .025 * .8 * face); reduced[-1] += .8 * face
    assert abs(result['B_postchange_carrying_mn'] - d.market_pv(reduced, 200)) < 1e-8
    assert abs(result['pre_modification_carrying_mn'] - result['B_postchange_carrying_mn'] - .01 * face - result['B_accounting_gain_mn']) < 1e-8
    assert result['B_accounting_gain_mn'] > result['A_modification_gain_mn']
    wrong = dict(result, B_postchange_carrying_mn=d.pv(reduced, rate) - .01 * face,
                 B_accounting_gain_mn=result['pre_modification_carrying_mn'] - d.pv(reduced, rate))
    assert not d.validate(wrong).success
    # A repeated-modification test is not an original-book cash flow reset.
    next_flows = modified[2:].copy(); next_flows[-1] -= .15 * face
    old_legal = d.pv(modified[2:], rate)
    assert abs(old_legal - result['second_prechange_carrying_mn']) > 5
    wrong_test = 100 * abs(d.pv(next_flows, rate) - result['second_prechange_carrying_mn']) / result['second_prechange_carrying_mn']
    assert abs(wrong_test - result['second_ten_percent_test_pct']) > .5


def test_treaty_source_union_and_gross_net_capacity_conservation():
    count, gross, dates, daily, fractions = t.inputs()
    assert count == 2638
    assert abs(sum(daily) - .4 * gross) < 1e-8
    assert len(fractions) == 69
    windows = np.convolve(daily, np.ones(3), 'valid')
    result = t.analysis()
    starts = [dates.get_loc(x) for x in result['optimal_window_starts'].split(',')]
    assert len(starts) <= 3 and all(b - a >= 3 for a, b in zip(starts, starts[1:]))
    net, recovery, premium, state = t.ledger(starts, windows)
    assert abs(net - recovery + premium) < 1e-10
    assert abs(18 - recovery - state[1]) < 1e-10
    # Restored capacity consumes reinstatement and restores the occurrence layer.
    restored = 10 - state[2]
    assert abs(10 - recovery + restored - state[0]) < 1e-10
    assert abs(net - result['maximum_net_recovery_mn']) < 1e-9
    actual = t.ledger(starts[:1], windows)
    assert abs(actual[3][0] - 10) < 1e-9
    assert abs(actual[3][1] - 8) < 1e-9
    assert abs(actual[3][2]) < 1e-9
    assert result['future_optimal_starts'] != result['endorsement_optimal_starts']
    assert result['endorsement_max_price_mn'] > 1
    assert result['endorsement_decision'] == 'BUY'
    # Empty occurrence selections change neither capacity nor required premium.
    assert t.ledger([], windows) == (0., 0., 0., (10., 18., 10.))


def test_treaty_time_premium_and_aggregate_are_distinct_contracts():
    _, _, dates, _, fractions = t.inputs()
    early = dates.get_loc('2020-08-25')
    later = dates.get_loc('2020-10-08')
    early_net, recovery, early_premium, state = t.settlement(20., early, (10., 18., 10.))
    later_net, later_recovery, later_premium, later_state = t.settlement(20., later, (10., 18., 10.))
    assert recovery == later_recovery == 10
    assert early_premium > later_premium > 0
    assert state == later_state == (10., 8., 0.)
    assert early_net < later_net
    assert abs(early_premium - 3 * fractions[early]) < 1e-12
    _, next_recovery, next_premium, terminal = t.settlement(20., later, state)
    assert next_recovery == 8 and next_premium == 0
    assert terminal == (2., 0., 0.)


def test_all_debt_and_treaty_storage_outputs_and_legal_representation():
    for module in [d, t]:
        truth = dict(zip((v.name for v in module.variables), module.ground_truth()))
        assert module.validate(truth).success
        precision = dict(zip(truth, module.DECIMALS))
        rounded = {k: round(value, precision[k]) if precision[k] is not None else value for k, value in truth.items()}
        assert module.validate(rounded).success
        for name, value in truth.items():
            bad = truth.copy(); bad[name] = mutate(value)
            assert not module.validate(bad).success
            if isinstance(value, str):
                for invalid in [value.lower(), ' ' + value, value + ' ', 'WRONG']:
                    if invalid != value:
                        bad[name] = invalid
                        assert not module.validate(bad).success
        for names in module.TURN_NAMES:
            assert module._subset({n: rounded[n] for n in names}, names).success
