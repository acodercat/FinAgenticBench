import json
from pathlib import Path

import numpy as np

from evals.scenario_analysis import bis_var_generalized_connectedness as connectedness
from evals.scenario_analysis import nport_pca_keyrate_factor_hedge as hedge
from scripts.audit_case_design import has_closed_text_representation

ROOT = Path(__file__).resolve().parents[1]

def _truth(module): return dict(zip((v.name for v in module.variables), module.ground_truth()))

def test_group5_contract_and_turn_validators():
    for path, module in (("evals/scenario_analysis/bis_var_generalized_connectedness.json", connectedness), ("evals/scenario_analysis/nport_pca_keyrate_factor_hedge.json", hedge)):
        spec=json.loads((ROOT/path).read_text());turns=spec["conversations"][0]["turns"]
        assert len(turns)==4 and len(spec["data_sources"])>=2
        assert [name for turn in turns for name in turn["stores"]]==[v.name for v in module.variables]
        truth=_truth(module);assert module.validate(truth).success
        assert all(has_closed_text_representation(v.description) for v,value in zip(module.variables,module.ground_truth()) if isinstance(value,str))
        for i in range(1,5):
            names=getattr(module,f"TURN_{i}_NAMES");assert getattr(module,f"validate_turn_{i}")({n:truth[n] for n in names}).success

def test_connectedness_accounting_and_stability():
    truth=_truth(connectedness)
    assert truth["var_spectral_radius"]<1
    assert truth["gfevd_maximum_row_sum_residual"]<1e-12
    assert np.isclose(sum(truth[f"{n}_to_others_pct"]-truth[f"{n}_from_others_pct"] for n in connectedness.NODES),0)
    assert truth["maximum_ols_normal_moment"]<1e-10
    assert truth["net_transmitter"] in connectedness.NODES
    assert truth["largest_directed_edge_source"] in connectedness.NODES
    candidate={name:truth[name] for name in connectedness.TURN_3_NAMES};candidate["largest_directed_edge_source"]="OFR Financial Stress Index";assert not connectedness.validate_turn_3(candidate).success

def test_pca_orthogonality_hedge_and_realized_reduction():
    truth=_truth(hedge);state=hedge._state();vectors=state[3]
    assert np.max(np.abs(vectors.T@vectors-np.eye(5)))<1e-12
    assert truth["hedge_maximum_first_three_factor_residual"]<1e-8
    assert truth["absolute_pnl_reduction_pct"]>90
    # The residual is a pinned value at 7 decimals rather than a one-sided bound, so the
    # shared rounding-boundary tolerance is what admits a convergent solver's noise. That
    # tolerance is symmetric, so it cannot reject a value just above a threshold; what it
    # does reject is a residual large enough to mean the overlay did not solve.
    candidate={name:truth[name] for name in hedge.TURN_3_NAMES};candidate["hedge_maximum_first_three_factor_residual"]=1e-8;assert hedge.validate_turn_3(candidate).success
    candidate["hedge_maximum_first_three_factor_residual"]=1e-6;assert not hedge.validate_turn_3(candidate).success

def test_nport_registered_fund_name_is_source_verbatim():
 truth=_truth(hedge);candidate={name:truth[name] for name in hedge.TURN_2_NAMES};candidate["selected_fund_name"]="PIMCO 25+ Year Zero Coupon Treasury ETF";assert not hedge.validate_turn_2(candidate).success
 description=next(v.description for v in hedge.variables if v.name=="selected_fund_name");assert "verbatim" in description

def test_group5_deterministic_and_wrong_rejected():
    for module in (connectedness,hedge):
        first=module.ground_truth();module.ground_truth.cache_clear();assert first==module.ground_truth();truth=_truth(module);names=module.TURN_2_NAMES;candidate={n:truth[n] for n in names};numeric=next(n for n in names if isinstance(candidate[n],(int,float)) and not isinstance(candidate[n],bool));candidate[numeric]=float(candidate[numeric])+1;assert not module.validate_turn_2(candidate).success
