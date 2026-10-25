"""Strict financial objects and numerical certificates for HJ / Gerber–Shiu."""

import json
from pathlib import Path
import pytest
from evals.scenario_analysis import industry_hansen_jagannathan_good_deal as hj
from evals.scenario_analysis import nfip_gerber_shiu_ruin_capital as gs


@pytest.mark.parametrize("module", [hj, gs])
def test_strict_pv_contract(module):
    t = dict(zip((v.name for v in module.variables), module.ground_truth()))
    assert module.validate(t).success
    assert module.validate(
        {k: round(v, d) for (k, v), d in zip(t.items(), module.DECIMALS)}
    ).success
    module.ground_truth.cache_clear()
    assert tuple(t.values()) == module.ground_truth()
    spec = json.loads(Path(module.__file__).with_suffix(".json").read_text())
    names = []
    for i, turn in enumerate(spec["conversations"][0]["turns"], 1):
        assert len(turn["stores"]) >= 2
        assert getattr(module, f"validate_turn_{i}")(
            {k: t[k] for k in turn["stores"]}
        ).success
        names += turn["stores"]
    assert names == list(t) and not any("residual" in k for k in t)
    for (k, v), d in zip(t.items(), module.DECIMALS):
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
            assert not module.validate({**t, k: bad}).success
        assert not module.validate({n: w for n, w in t.items() if n != k}).success
        if d == 0:
            assert not module.validate({**t, k: v + 0.1}).success
        else:
            eps = 0.6 * 10**-d
            assert module.validate({**t, k: v + 0.999 * eps}).success
            assert not module.validate({**t, k: v + 1.001 * eps}).success
    assert not module.validate(dict.fromkeys(t, 0)).success


def test_source_removal(monkeypatch):
    for module, loader, source_loader, sources in [
        (hj, "hj_inputs", "load_ken_french_table", ["daily_industry_returns"]),
        (hj, "hj_inputs", "load_expansion_table", ["nyfed_reference_rates"]),
        (gs, "ruin_inputs", "load_expansion_table", ["fema_nfip", "fema_disasters"]),
    ]:
        for source in sources:
            original = getattr(module, source_loader)
            with monkeypatch.context() as patch:

                def absent(name):
                    if name == source:
                        raise FileNotFoundError(source)
                    return original(name)

                patch.setattr(module, source_loader, absent)
                getattr(module, loader).cache_clear()
                with pytest.raises(FileNotFoundError):
                    getattr(module, loader)()
            getattr(module, loader).cache_clear()
