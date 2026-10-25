"""Independent quantitative certificates, financial wrong paths and runtime binding."""

import json
from pathlib import Path
import pytest
from evals.scenario_analysis import bank_arellano_bond_fieller_margin as ab
from evals.scenario_analysis import scf_borch_catastrophe_risk_exchange as br


@pytest.mark.parametrize("m", [ab, br])
def test_strict_rounding_replay_and_turns(m):
    a = dict(zip((v.name for v in m.variables), m.ground_truth()))
    assert m.validate(a).success
    rounded = {
        k: round(v, d) if d is not None else v
        for (k, v), d in zip(a.items(), m.DECIMALS)
    }
    assert m.validate(rounded).success
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
        bad = [None, True, float("nan"), float("inf")]
        if d is None:
            bad += [v.lower(), v + " ", "BOUNDED_INTERVAL"]
        else:
            bad += [str(v), -v, v * 100, v / 100]
        for x in bad:
            assert not m.validate({**a, k: x}).success
        if d == 0:
            assert not m.validate({**a, k: v + 0.1}).success
        elif d is not None:
            eps = 0.6 * 10 ** (-d)
            assert m.validate({**a, k: v + 0.999 * eps}).success
            assert not m.validate({**a, k: v + 1.001 * eps}).success
    assert not m.validate(dict.fromkeys(a, 0)).success


@pytest.mark.parametrize(
    "m,labels",
    [(ab, ["fdic_bankfind", "nyfed_reference_rates"]), (br, ["fed_scf", "fema_nfip"])],
)
def test_actual_runtime_and_source_removal(m, labels, monkeypatch):
    from core.runtime_catalog import runtime_variables, select_runtime_tables

    spec = json.loads(Path(m.__file__).with_suffix(".json").read_text())
    assert spec["data_sources"] == labels
    frames = {v.name: v.value for v in runtime_variables(labels)}
    assert len(frames) == 2 and all(len(d) > 0 for d in frames.values())
    assert set(frames) == {t.name for t in select_runtime_tables(labels)}
    for label in labels:
        original = m.load_expansion_table
        with monkeypatch.context() as patch:

            def missing(name):
                if name == label:
                    raise FileNotFoundError(label)
                return original(name)

            patch.setattr(m, "load_expansion_table", missing)
            m.inputs.cache_clear()
            with pytest.raises(FileNotFoundError):
                m.inputs()
        m.inputs.cache_clear()
