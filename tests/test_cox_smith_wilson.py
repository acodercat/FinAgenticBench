"""Strict PV boundaries and independent numerical objects for the new pair."""

import json
from pathlib import Path
import pytest
from evals.scenario_analysis import sba_cox_seasoned_amortizing_loan as cox
from evals.scenario_analysis import pension_smith_wilson_immunization as sw


@pytest.mark.parametrize("module", [cox, sw])
def test_strict_pv_rounding_and_turn_contract(module):
    truth = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert module.validate(truth).success
    assert module.validate(
        {k: round(v, d) for (k, v), d in zip(truth.items(), module.DECIMALS)}
    ).success
    module.ground_truth.cache_clear()
    assert tuple(truth.values()) == module.ground_truth()
    spec = json.loads(Path(module.__file__).with_suffix(".json").read_text())
    stores = []
    for i, turn in enumerate(spec["conversations"][0]["turns"], 1):
        assert len(turn["stores"]) >= 2
        stores.extend(turn["stores"])
        assert getattr(module, f"validate_turn_{i}")(
            {n: truth[n] for n in turn["stores"]}
        ).success
    assert stores == list(truth)
    assert not any("residual" in n for n in stores)
    for (key, value), decimals in zip(truth.items(), module.DECIMALS):
        assert not module.validate({k: v for k, v in truth.items() if k != key}).success
        for invalid in [
            None,
            True,
            str(value),
            float("nan"),
            float("inf"),
            float("-inf"),
        ]:
            assert not module.validate({**truth, key: invalid}).success
        if decimals == 0:
            assert not module.validate({**truth, key: value + 0.1}).success
        else:
            tolerance = 0.6 * 10**-decimals
            assert module.validate({**truth, key: value + 0.999 * tolerance}).success
            assert not module.validate(
                {**truth, key: value + 1.001 * tolerance}
            ).success
            for invalid in [-value, value * 100, value / 100, value + 0.001]:
                assert not module.validate({**truth, key: invalid}).success
    assert not module.validate(dict.fromkeys(truth, 0)).success


def test_source_removal(monkeypatch):
    for module, loader, names in [
        (cox, "cox_inputs", ["sba_7a", "nyfed_reference_rates"]),
        (sw, "sw_inputs", ["treasury_yield_curve", "dol_form5500_schedule_sb"]),
    ]:
        for source in names:
            original = module.load_expansion_table
            with monkeypatch.context() as patch:

                def missing(name):
                    if name == source:
                        raise FileNotFoundError(source)
                    return original(name)

                patch.setattr(module, "load_expansion_table", missing)
                getattr(module, loader).cache_clear()
                with pytest.raises(FileNotFoundError):
                    getattr(module, loader)()
            getattr(module, loader).cache_clear()
