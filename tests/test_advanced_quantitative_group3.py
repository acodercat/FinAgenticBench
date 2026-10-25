import json
from pathlib import Path
import numpy as np
from evals.scenario_analysis import industry_cvar_ofr_optimization as cvar
from evals.temporal_path import treasury_rank1_vecm_positioning as vecm
from scripts.audit_case_design import has_closed_text_representation
ROOT=Path(__file__).resolve().parents[1]
def truth(m):return dict(zip((v.name for v in m.variables),m.ground_truth()))
def assert_declared_precision(m):
 for variable,decimals in zip(m.variables,m.DECIMALS):
  if isinstance(decimals,int) and decimals>0:assert f"rounded to {decimals} decimals" in variable.description
def test_group3_contract_and_validators():
 for path,m in (("evals/scenario_analysis/industry_cvar_ofr_optimization.json",cvar),("evals/temporal_path/treasury_rank1_vecm_positioning.json",vecm)):
  s=json.loads((ROOT/path).read_text());ts=s["conversations"][0]["turns"];assert len(ts)==4 and len(s["data_sources"])>=2
  assert [x for t in ts for x in t["stores"]]==[v.name for v in m.variables];d=truth(m);assert m.validate(d).success
  assert all(has_closed_text_representation(v.description) for v,value in zip(m.variables,m.ground_truth()) if isinstance(value,str))
  for i in range(1,5):
   ns=getattr(m,f"TURN_{i}_NAMES");assert getattr(m,f"validate_turn_{i}")({n:d[n] for n in ns}).success
def test_group3_output_descriptions_and_vecm_window_match_contract():
 for m in (cvar,vecm):assert_declared_precision(m)
 assert vecm._panel().date.ge("2021-01-01").all()
def test_cvar_constraints_and_tail_improvement():
 d=truth(cvar);*_,w,target,var,cv,count,res=cvar._state();assert np.isclose(w.sum(),1) and np.all(w>=0) and np.all(w<=.35+1e-12);assert res<1e-10 and cv>=var;assert d["optimized_training_cvar_loss_pct"]<d["equal_weight_training_cvar_loss_pct"]
def test_cvar_industry_outputs_require_exact_canonical_tokens():
 d=truth(cvar);candidate={n:d[n] for n in cvar.TURN_1_NAMES};candidate["selected_industry_1"]="Durable";assert not cvar.validate_turn_1(candidate).success
 descriptions={v.name:v.description for v in cvar.variables};assert all("canonical Ken French" in descriptions[n] for n in cvar.TURN_1_NAMES[2:7])
def test_vecm_normal_moment_and_test_identity():
 d=truth(vecm);assert d["ecm_maximum_normal_moment"]<1e-10
def test_group3_deterministic_and_wrong_rejected():
 for m in (cvar,vecm):
  a=m.ground_truth();m.ground_truth.cache_clear();assert a==m.ground_truth();d=truth(m);ns=m.TURN_2_NAMES;c={n:d[n] for n in ns};k=next(n for n in ns if isinstance(c[n],(int,float)) and not isinstance(c[n],bool));c[k]=float(c[k])+1;assert not m.validate_turn_2(c).success
