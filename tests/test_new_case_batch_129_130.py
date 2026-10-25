"""Independent author checks for the first 129–138 candidate pair."""

from datetime import date
from math import exp, log, sqrt

import numpy as np
from scipy.integrate import quad

from core.data import load_expansion_table
from evals.calculation_reconciliation import bdc_pik_cash_distribution_borrowing_base as bdc
from evals.scenario_analysis import tips_reopening_deflation_floor_implied_growth as tips


def test_bdc_independent_event_ledger_and_strict_rounding():
    source = load_expansion_table("sec_bdc")
    source = source.loc[
        source.cik.eq("1843162") & source.report_date.eq("2024-12-31")
        & source.form.eq("10-K") & source.filed_date.eq("2025-03-31")
    ]
    permitted = {
        "TermLoanMember", "DelayedDrawTermLoanMember", "IncrementalTermLoanMember",
        "SeniorSecuredNoteMember", "SeriesASeniorNoteMember",
    }
    selected = []
    for row in source.itertuples(index=False):
        kind = str(row.investment_type).split("#")[-1]
        if kind not in permitted:
            continue
        if any(np.isnan(float(v)) for v in (
            row.principal_amount_usd, row.fair_value_usd,
            row.pik_interest_rate, row.total_interest_rate,
        )):
            continue
        if (row.principal_amount_usd <= 0 or row.fair_value_usd <= 0
                or row.pik_interest_rate < 0 or str(row.maturity_date) <= "2025-06-30"):
            continue
        selected.append({
            "p": float(row.principal_amount_usd),
            "mark": float(row.fair_value_usd),
            "pik": float(row.pik_interest_rate),
        })
    assert len(selected) == 27
    assert sum(item["pik"] > 0 for item in selected) == 10

    result = bdc.analysis()
    rate = load_expansion_table("nyfed_reference_rates")
    sofr = {}
    for day in ("2024-12-31", "2025-03-31"):
        match = rate[(rate["Effective Date"] == day) & (rate["Rate Type"] == "SOFR")]
        assert len(match) == 1
        sofr[day] = float(match.iloc[0]["Rate (%)"]) / 100
    gross = cash = pik = 0.0
    for item in selected:
        item["q1_pik"] = item["p"] * item["pik"] * 90 / 360
        gross += item["p"] * (sofr["2024-12-31"] + .05) * 90 / 360
        pik += item["q1_pik"]
        cash += item["p"] * (sofr["2024-12-31"] + .05 - item["pik"]) * 90 / 360
    assert abs(gross - cash - pik) < 1e-6
    mark = sum(item["mark"] for item in selected)
    opening_debt = .645 * mark
    q1_finance = opening_debt * (sofr["2024-12-31"] + .025) * 90 / 360
    paid_distribution = .8 * gross
    deficit = paid_distribution + q1_finance - cash
    draw = .65 * mark - opening_debt
    equity = deficit - draw
    assert equity > 0
    shocked_mark = sum(
        item["mark"] * (.85 if item["pik"] > 0 else 1)
        for item in selected
    )
    shocked_base = .65 * shocked_mark
    cure = opening_debt + draw - shocked_base
    assert cure > 0
    assert abs(result["q1_cash_coupon_mn"] - cash / 1e6) < 1e-9
    assert abs(result["q1_capitalized_pik_mn"] - pik / 1e6) < 1e-9
    assert abs(result["q1_new_borrowing_mn"] - draw / 1e6) < 1e-9
    assert abs(result["q1_external_equity_mn"] - equity / 1e6) < 1e-9
    assert abs(result["shock_cure_equity_mn"] - cure / 1e6) < 1e-9

    # Event ledger: each capitalized position rolls individually, and the new
    # equity reduces debt at April 1 rather than pretending it is investment gain.
    q2_debt = shocked_base - 3_000_000
    q2_finance = q2_debt * (sofr["2025-03-31"] + .025) * 91 / 360
    q2_cash = q2_pik = 0.0
    for item in selected:
        new_opening = item["p"] + item["q1_pik"]
        q2_cash += new_opening * (sofr["2025-03-31"] + .05 - item["pik"]) * 91 / 360
        q2_pik += new_opening * item["pik"] * 91 / 360
    max_distribution = q2_cash - q2_finance + 3_000_000
    assert abs(result["q2_max_distribution_mn"] - max_distribution / 1e6) < 1e-9
    assert abs(result["q2_ending_principal_mn"] -
               (sum(item["p"] + item["q1_pik"] for item in selected) + q2_pik) / 1e6) < 1e-9
    bdc_precision = dict(zip((v.name for v in bdc.variables), bdc.DECIMALS))
    assert bdc.validate({
        name: round(value, bdc_precision[name]) if bdc_precision[name] is not None else value
        for name, value in result.items()
    }).success
    reported_path = result.copy()
    reported_path["q1_distribution_mn"] = round(
        .8 * (round(result["q1_cash_coupon_mn"], 4)
              + round(result["q1_capitalized_pik_mn"], 4)), 3
    )
    reported_path["q1_external_equity_mn"] = round(
        reported_path["q1_distribution_mn"]
        + round(result["q1_financing_interest_mn"], 4)
        - round(result["q1_cash_coupon_mn"], 4)
        - round(result["q1_new_borrowing_mn"], 4), 3
    )
    assert bdc.validate(reported_path).success
    wrong = result.copy()
    wrong["q1_cash_coupon_mn"] += pik / 1e6
    assert not bdc.validate(wrong).success


def test_tips_independent_linear_solve_quadrature_and_bisection():
    auction = load_expansion_table("treasury_auctions")
    auction = auction[(auction.cusip == "91282CNB3") & (auction.auction_date == "2025-06-17")]
    assert len(auction) == 1
    a = auction.iloc[0]
    rates = load_expansion_table("treasury_yield_curve")
    rates = rates[rates.Date.astype(str) == "2025-06-30"]
    assert len(rates) == 1
    curve = rates.iloc[0]
    known_t = [0.5, 1, 2, 3, 5]
    known_y = [float(curve[x]) / 100 for x in ("6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr")]
    grid_t = np.arange(.5, 5.01, .5)
    interpolated = np.interp(grid_t, known_t, known_y)
    matrix = np.zeros((len(grid_t), len(grid_t)))
    for i, annual_y in enumerate(interpolated):
        matrix[i, :i] = annual_y / 2
        matrix[i, i] = 1 + annual_y / 2
    grid_d = np.linalg.solve(matrix, np.ones(len(grid_t)))
    assert np.all(grid_d > 0)
    dates = [date(y, m, 15) for y in range(2025, 2031) for m in (4, 10)
             if date(2025, 6, 30) < date(y, m, 15) <= date(2030, 4, 15)]
    t = np.array([(d - date(2025, 6, 30)).days / 365 for d in dates])
    d = np.exp(np.interp(t, np.r_[0, grid_t], np.log(np.r_[1, grid_d])))
    ratio = float(a.index_ratio_on_issue_date)
    invoice = float(a.adj_price) + float(a.adj_accrued_int_per1000) / 10
    coupon = float(a.int_rate) / 2

    def independent_price(mu, sigma):
        expected = ratio * np.exp(mu * t)
        base = coupon * np.dot(d, expected) + 100 * d[-1] * expected[-1]
        terminal_time = t[-1]
        def terminal_ratio(z):
            return ratio * exp((mu - .5 * sigma * sigma) * terminal_time
                               + sigma * sqrt(terminal_time) * z)
        density = lambda z: exp(-z * z / 2) / sqrt(2 * np.pi)
        floor_expectation = quad(
            lambda z: max(1 - terminal_ratio(z), 0) * density(z),
            -12, 12, epsabs=1e-11,
        )[0]
        floor_probability = quad(
            lambda z: (1 if terminal_ratio(z) < 1 else 0) * density(z),
            -12, 12, epsabs=1e-9, points=[
                (log(1 / ratio) - (mu - .5 * sigma * sigma) * terminal_time)
                / (sigma * sqrt(terminal_time))
            ],
        )[0]
        return base + 100 * d[-1] * floor_expectation, 100 * d[-1] * floor_expectation, floor_probability

    def independent_root(sigma):
        lo, hi = -.10, .15
        assert independent_price(lo, sigma)[0] < invoice < independent_price(hi, sigma)[0]
        for _ in range(55):
            mid = (lo + hi) / 2
            if independent_price(mid, sigma)[0] < invoice:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    result = tips.analysis()
    mu3, mu5 = independent_root(.03), independent_root(.05)
    v3 = independent_price(mu3, .03)
    v_fixed = independent_price(mu3, .05)
    v5 = independent_price(mu5, .05)
    assert abs(v3[0] - invoice) < 1e-9
    assert abs(v5[0] - invoice) < 1e-9
    assert abs(result["maturity_discount_factor"] - d[-1]) < 1e-12
    assert abs(result["fitted_growth_3pct"] - 100 * mu3) < 1e-8
    assert abs(result["refitted_growth_5pct"] - 100 * mu5) < 1e-8
    assert abs(result["floor_value_3pct_per_100"] - v3[1]) < 1e-8
    assert abs(result["floor_probability_3pct"] - 100 * v3[2]) < 1e-5
    assert abs(result["fixed_growth_price_5pct_per_100"] - v_fixed[0]) < 1e-8
    assert abs(result["refitted_floor_probability_5pct"] - 100 * v5[2]) < 1e-5
    assert v_fixed[0] > invoice
    assert mu5 < mu3
    assert abs((v_fixed[0] - v_fixed[1]) - (v3[0] - v3[1])) < 1e-10
    assert independent_price(.02, 1e-5)[1] < 1e-8
    assert independent_price(-.10, 1e-5)[1] > 30
    tips_precision = dict(zip((v.name for v in tips.variables), tips.DECIMALS))
    assert tips.validate({
        name: round(value, tips_precision[name]) if tips_precision[name] is not None else value
        for name, value in result.items()
    }).success
    reported_path = result.copy()
    reported_path["fixed_growth_price_5pct_per_100"] = round(
        round(result["dirty_invoice_per_100"], 6)
        + round(result["fixed_growth_floor_5pct_per_100"], 4)
        - round(result["floor_value_3pct_per_100"], 4), 3
    )
    assert tips.validate(reported_path).success
    wrong = result.copy()
    wrong["dirty_invoice_per_100"] *= ratio
    assert not tips.validate(wrong).success
