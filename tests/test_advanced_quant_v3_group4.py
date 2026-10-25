"""Independent model certificates, wrong paths and actual runtime binding."""

import json
from pathlib import Path
import numpy as np
import pytest
from evals.scenario_analysis import treasury_kyle_public_disclosure as ky
from evals.scenario_analysis import industry_variance_gamma_esscher_hedge as vg


@pytest.mark.parametrize("m", [ky, vg])
def test_strict_rounding_replay_and_turns(m):
    a = dict(zip((v.name for v in m.variables), m.ground_truth()))
    assert m.validate(a).success
    assert m.validate(
        {k: round(v, d) for (k, v), d in zip(a.items(), m.DECIMALS)}
    ).success
    m.ground_truth.cache_clear()
    assert tuple(a.values()) == m.ground_truth()
    spec = json.loads(Path(m.__file__).with_suffix(".json").read_text())
    names = []
    for i, t in enumerate(spec["conversations"][0]["turns"], 1):
        assert len(t["stores"]) >= 2
        assert getattr(m, f"validate_turn_{i}")({k: a[k] for k in t["stores"]}).success
        names += t["stores"]
    assert names == list(a) and not any("residual" in k for k in a)
    for (k, v), d in zip(a.items(), m.DECIMALS):
        assert not m.validate({n: w for n, w in a.items() if n != k}).success
        for x in [None, True, float("nan"), float("inf"), str(v), -v, v * 100, v / 100]:
            assert not m.validate(a | {k: x}).success
        if d == 0:
            assert not m.validate(a | {k: v + 0.1}).success
        else:
            eps = 0.6 * 10**-d
            assert m.validate(a | {k: v + 0.999 * eps}).success
            assert not m.validate(a | {k: v + 1.001 * eps}).success
    assert not m.validate(dict.fromkeys(a, 0)).success


def test_kyle_equilibrium_information_and_global_domain():
    a, e = ky.calculate()
    S = np.array(e["Sigma"])
    O = np.array(e["Omega"])
    L = np.array(e["impact"])
    C = np.array(e["posterior_covariance"])
    V = np.array(e["final_covariance"])
    B = np.array(e["informed_strategy"])
    for mat in [S, O, L, C, V]:
        assert np.linalg.eigvalsh(mat).min() > 0
    for k in [
        "riccati_relative",
        "post_riccati_relative",
        "camouflage_relative",
        "posterior_partition_relative",
    ]:
        assert e[k] < 1e-10
    assert abs(e["root_residual"]) < 1e-8 and np.linalg.eigvalsh(S - C).min() > 0
    assert np.linalg.norm(B @ C @ B.T - O) / np.linalg.norm(O) < 1e-10
    assert abs(np.trace(S @ np.linalg.inv(L)) / 4 - e["profit"]) < 1e-8
    p = [2, 0, 1]
    assert (
        np.max(abs(ky.impact(S[np.ix_(p, p)], O[np.ix_(p, p)]) - L[np.ix_(p, p)]))
        < 1e-15
    )
    with pytest.raises(AssertionError):
        ky.impact(-S, O)
    bad = ky.calculate(scope="futures_and_options_combined")[0]
    assert not ky.validate(bad).success


def test_vg_moments_martingale_projection_and_measure():
    a, e = vg.calculate()
    mu, theta, sigma, nu = e["physical_parameters"]
    s2 = sigma * sigma
    kk = np.array(
        [
            mu + theta,
            s2 + theta * theta * nu,
            3 * s2 * theta * nu + 2 * theta**3 * nu**2,
            3 * s2 * s2 * nu + 12 * s2 * theta * theta * nu**2 + 6 * theta**4 * nu**3,
        ]
    )
    assert np.allclose(kk, e["cumulants_per_year"], rtol=1e-10, atol=1e-14)
    assert abs(e["martingale_error"]) < 1e-8
    stock = e["stock_mean"]
    stock2 = e["stock_second_moment"]
    call = e["call_mean"]
    cross = e["stock_call_cross_moment"]
    w = e["hedge_shares"]
    cash = e["hedge_terminal_cash"]
    assert abs(call - w * stock - cash) < 1e-10
    assert abs(cross - w * stock2 - cash * stock) < 1e-8
    assert e["hedge_residual_variance"] > 0 and stock2 - stock * stock > 0
    assert e["mgf_domain"][0] < e["esscher_h"] < e["esscher_h"] + 2 < e["mgf_domain"][1]
    assert e["shortfall_thresholds"][0] < 102 < e["shortfall_thresholds"][1]
    assert 0 < a["one_dollar_shortfall_probability"] < 1
    for kw in [
        dict(mean_correction=True),
        dict(rate_type="EFFR"),
        dict(cash="force"),
        dict(days=10),
    ]:
        assert not vg.validate(vg.calculate(**kw)[0]).success
    with pytest.raises(AssertionError):
        vg.fit(np.zeros(20))


@pytest.mark.parametrize(
    "m,labels",
    [
        (ky, ["treasury_yield_curve", "cftc_cot"]),
        (vg, ["ken_french_10_industry_portfolios", "nyfed_reference_rates"]),
    ],
)
def test_actual_runtime_and_removal(m, labels, monkeypatch):
    from core.runtime_catalog import runtime_variables, select_runtime_tables

    j = json.loads(Path(m.__file__).with_suffix(".json").read_text())
    assert j["data_sources"] == labels
    frames = {v.name: v.value for v in runtime_variables(labels)}
    assert len(frames) == 2 and all(len(v) > 0 for v in frames.values())
    assert set(frames) == {t.name for t in select_runtime_tables(labels)}
    for label in labels:
        loader = (
            "load_ken_french_table"
            if label.startswith("ken_french")
            else "load_expansion_table"
        )
        target = "daily_industry_returns" if label.startswith("ken_french") else label
        original = getattr(m, loader)
        with monkeypatch.context() as patch:

            def missing(name):
                if name == target:
                    raise FileNotFoundError(target)
                return original(name)

            patch.setattr(m, loader, missing)
            m.inputs.cache_clear()
            with pytest.raises(FileNotFoundError):
                m.inputs()
        m.inputs.cache_clear()
