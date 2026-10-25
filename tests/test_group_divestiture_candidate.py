"""Task-local checks for bank_group_concentration_merger_divestiture."""
from types import SimpleNamespace

import numpy as np
import pytest

from evals.scenario_analysis import bank_group_concentration_merger_divestiture as case


def test_group_roots_distinguishes_diamonds_ambiguity_and_cyclic_ancestry():
    edges=[('1','2'),('1','3'),('2','4'),('3','4'),('5','6'),('7','6'),
           ('8','9'),('9','8'),('9','10'),('1','10'),('11','11')]
    result=case.group_roots(edges,{'4','6','10','11','12'})
    assert result=={'4':'1','12':'12'}


def test_full_branch_objective_boundary_ties_and_no_solution():
    groups={'1':60,'2':20,'3':20}
    # x=20 and x=60 exactly restore the original HHI; amount takes priority.
    options=[('9','b',60),('3','z',20),('2','a',20),('2','b',20),('1','small',19)]
    choice,feasible,_=case.transfer_options(groups,'1','2',options)
    assert choice[:3]==('2','a',20)
    assert choice[3]==case.hhi(groups.values())
    assert len(feasible)==4
    choice,feasible,_=case.transfer_options(groups,'1','2',[('1','small',19),('1','zero',0)])
    assert not feasible and choice[:3]==('NONE','NONE',0)
    assert choice[3]==case.hhi([80,20])
    with pytest.raises(ValueError):case.transfer_options(groups,'1','2',[('1','bad',81)])


def test_runtime_ground_truth_turn_scope_and_wrong_financial_objective():
    answers,evidence=case.analysis()
    assert case.validate(answers).success
    runtime=SimpleNamespace(retrieve=lambda name: answers[name])
    for names,validator in zip(case.TURN_NAMES,case.validators.values()):
        result=validator('prose is not a numerical source',runtime,SimpleNamespace(stores=names))
        assert result.success and set(result.variable_results)==set(names)
    minimum_hhi=min(evidence['options'],key=lambda x:x[3])
    assert minimum_hhi[1]!=answers['transfer_branch']
    wrong=dict(answers,transfer_bank=minimum_hhi[0],transfer_branch=minimum_hhi[1],
               transferred_deposits_thousand_usd=minimum_hhi[2],final_hhi=minimum_hhi[3])
    assert not case.validate_transfer(wrong).success


def test_missing_nonfinite_units_and_rounding_acceptance():
    answers,_=case.analysis()
    for value in [None, float('nan'),float('inf'),True,'4536.1711']:
        result=case.validate_transfer(dict(answers,final_hhi=value))
        assert not result.success
        if value is None:assert result.variables_not_set
    missing=dict(answers);del missing['transfer_branch']
    assert case.validate_transfer(missing).variables_not_set
    rounded={name:round(value,places) if places is not None else value
             for name,_,places in case.SPECS for value in [answers[name]]}
    assert case.validate(rounded).success
    assert case.validate_transfer(dict(answers,final_hhi=np.float64(answers['final_hhi']))).success
    assert not case.validate_transfer(dict(answers,transferred_deposits_thousand_usd=answers['transferred_deposits_thousand_usd']*1000)).success
    assert not case.validate_transfer(dict(answers,final_hhi=answers['final_hhi']+0.000061)).success
    assert case.validate_transfer(dict(answers,final_hhi=answers['final_hhi']+0.000059)).success


