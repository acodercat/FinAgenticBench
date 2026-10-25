"""Cash-flow timing, delivery envelope and redemption wealth conservation."""
import numpy as np
import pandas as pd

from evals.comparison_ranking import treasury_delivery_basket_ctd_switch as c
from evals.scenario_analysis import fund_redemption_swing_price_wealth_transfer as s
from scripts.verdict_baseline import mutate


def test_delivery_coupon_calendar_and_optimized_switch():
    ids, auctions, *_ = c.inputs()
    dates, _ = c.coupon_dates(auctions.loc[ids[2], 'maturity_date'], pd.Timestamp('2024-12-31'))
    assert pd.Timestamp('2025-02-28') in dates
    assert pd.Timestamp('2025-08-31') in dates
    result = c.analysis()
    crossing = result['first_switch_bp']
    assert 0 < crossing < 200
    assert abs(c.quotes(crossing)[0] - c.quotes(crossing)[1]) < 1e-9
    assert np.argmin(c.quotes(crossing - .001)) == 0
    assert np.argmin(c.quotes(crossing + .001)) == 1
    assert c.quotes(crossing)[2] > c.quotes(crossing)[0] + 10
    assert result['left_switch_dv01'] > 0
    assert result['right_switch_dv01'] > result['left_switch_dv01']
    # The cash-price ranking does not identify the optimal shifted delivery.
    assert np.argmin([c.prices(x, 200)[1] for x in ids]) == 0
    assert np.argmin(c.quotes(200)) == 1
    # Forgoing the coupon receipt produces a materially higher boundary.
    assert c.prices(ids[0])[2] / .73 > .7


def test_redemption_funding_wealth_and_consumed_cost_bands():
    nav, cash, request, *_ = s.inputs()
    units = nav / 100
    faces = (nav - cash) * np.array([.3, .4, .3])
    first = s.transaction(units, faces, cash, request, 100)
    naive = s.transaction(units, faces, cash, request, 100, fair=False)
    assert abs(first['post_nav'] - first['pre_nav']) < 1e-10
    assert naive['post_nav'] < first['post_nav']
    assert abs(cash + first['gross'] - first['cost'] - request * first['price'] - first['cash']) < 1e-9
    assert abs(request * (first['pre_nav'] - first['price']) - first['cost']) < 1e-9
    assert .05 * nav < first['previous'] < .15 * nav
    second = s.transaction(first['units'], first['faces'], first['cash'], .1 * units, 150, first['previous'])
    reset = s.transaction(first['units'], first['faces'], first['cash'], .1 * units, 150, 0)
    assert abs(second['post_nav'] - second['pre_nav']) < 1e-10
    assert reset['price'] > second['price'] + .8
    assert second['previous'] > .15 * nav
    assert abs(s.sale_cost(second['gross'], first['previous']) - second['cost']) < 1e-9


def test_swing_capacity_boundary_uses_actual_first_state():
    nav, cash, request, *_ = s.inputs()
    units = nav / 100
    faces = (nav - cash) * np.array([.3, .4, .3])
    first = s.transaction(units, faces, cash, request, 100)
    capacity = s.analysis()['maximum_next_request_units_mn']
    assert 0 < capacity < .95 * first['units']
    for change, direction in [(-1e-5, -1), (0, 0), (1e-5, 1)]:
        next_trade = s.transaction(first['units'], first['faces'], first['cash'], capacity + change, 150, first['previous'])
        discount = 1 - next_trade['price'] / next_trade['pre_nav']
        if direction == 0:
            assert abs(discount - .012) < 1e-12
        else:
            assert direction * (discount - .012) > 0
        assert next_trade['gross'] < sum(first['faces'] * s.marks(150))


def test_delivery_and_swing_all_outputs_and_closed_identifiers():
    for module in [c, s]:
        truth = dict(zip((v.name for v in module.variables), module.ground_truth()))
        assert module.validate(truth).success
        rounded = {k: round(value, 4) if isinstance(value, (float, int)) else value for k, value in truth.items()}
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
