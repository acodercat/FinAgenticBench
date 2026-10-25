"""Stochastic control, stopping verification and strict financial-object acceptance."""

import json
from pathlib import Path
import numpy as np
import pytest
from evals.scenario_analysis import hmda_deaton_consumption_credit_welfare as hh
from evals.scenario_analysis import wti_dixit_operating_switching as dx


@pytest.mark.parametrize("module", [hh, dx])
def test_strict_contract_and_replay(module):
    a = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert module.validate(a).success
    assert module.validate(
        {k: round(v, d) for (k, v), d in zip(a.items(), module.DECIMALS)}
    ).success
    module.ground_truth.cache_clear()
    assert tuple(a.values()) == module.ground_truth()
    spec = json.loads(Path(module.__file__).with_suffix(".json").read_text())
    names = []
    for i, t in enumerate(spec["conversations"][0]["turns"], 1):
        assert len(t["stores"]) >= 2
        assert getattr(module, f"validate_turn_{i}")(
            {k: a[k] for k in t["stores"]}
        ).success
        names += t["stores"]
    assert names == list(a) and not any("residual" in k for k in a)
    for (k, v), d in zip(a.items(), module.DECIMALS):
        for bad in [
            None,
            True,
            str(v),
            float("nan"),
            float("inf"),
            -v,
            v * 100,
            v / 100,
        ]:
            assert not module.validate({**a, k: bad}).success
        assert not module.validate({n: w for n, w in a.items() if n != k}).success
        if d == 0:
            assert not module.validate({**a, k: v + 0.1}).success
        else:
            eps = 0.6 * 10**-d
            assert module.validate({**a, k: v + 0.999 * eps}).success
            assert not module.validate({**a, k: v + 1.001 * eps}).success
    assert not module.validate(dict.fromkeys(a, 0)).success


def test_source_removal(monkeypatch):
    for module, loader, sources in [
        (hh, "household_inputs", ["nyfed_reference_rates"]),
        (dx, "producer_inputs", ["eia_bulk_wti_daily", "treasury_yield_curve"]),
    ]:
        for source in sources:
            original = module.load_expansion_table
            with monkeypatch.context() as patch:

                def absent(name):
                    if name == source:
                        raise FileNotFoundError(source)
                    return original(name)

                patch.setattr(module, "load_expansion_table", absent)
                getattr(module, loader).cache_clear()
                with pytest.raises(FileNotFoundError):
                    getattr(module, loader)()
            getattr(module, loader).cache_clear()
    original = hh.load_catalog
    with monkeypatch.context() as patch:

        def missing_catalog():
            c = original()
            return {
                **c,
                "tables": {
                    k: v
                    for k, v in c["tables"].items()
                    if k != "hmda.loan_applications"
                },
            }

        patch.setattr(hh, "load_catalog", missing_catalog)
        hh.household_inputs.cache_clear()
        with pytest.raises(KeyError):
            hh.household_inputs()
    hh.household_inputs.cache_clear()


@pytest.mark.parametrize(
    "module,expected",
    [
        (hh, {"hmda_lar_df", "nyfed_reference_rates_df"}),
        (dx, {"eia_wti_daily_spot_prices_df", "treasury_yield_curve_df"}),
    ],
)
def test_actual_runtime_injection(module, expected):
    from core.runtime_catalog import runtime_variables

    spec = json.loads(Path(module.__file__).with_suffix(".json").read_text())
    injected = {v.name: v.value for v in runtime_variables(spec["data_sources"])}
    assert set(injected) == expected
    assert all(len(frame) > 0 for frame in injected.values())
    if module is dx:
        data = injected["eia_wti_daily_spot_prices_df"]
        selected = data[data.observation_date.between("2024-01-01", "2024-12-31")]
        assert selected.observation_date.is_unique
        assert len(selected) - 1 == module.ground_truth()[0]
    else:
        data = injected["hmda_lar_df"]
        selected = data[data.state_code.eq("TX")]
        mask = selected.activity_year.astype(str).eq("2024")
        for field in [
            "action_taken",
            "loan_type",
            "loan_purpose",
            "lien_status",
            "occupancy_type",
            "total_units",
        ]:
            mask &= selected[field].astype(str).eq("1")
        import pandas as pd

        income = pd.to_numeric(selected.loc[mask, "income"], errors="coerce")
        income = income[income.between(10, 500)]
        assert len(income) == module.ground_truth()[0]
        assert np.array_equal(
            np.quantile(income, [0.2, 0.5, 0.8], method="inverted_cdf"),
            module.ground_truth()[1:4],
        )


def test_active_registry_runtime_labels():
    from core.runtime_catalog import select_runtime_tables

    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "benchmarks.json").read_text())
    for path in registry["cases"].values():
        spec = json.loads((root / path).read_text())
        assert select_runtime_tables(spec["data_sources"])
