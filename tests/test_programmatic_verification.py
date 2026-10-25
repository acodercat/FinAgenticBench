import importlib
import json

import pytest

from config import BENCHMARKS_JSON
from core.evaluator import load_specs, output_contract_fingerprint
from core.results import (
    RESULT_SCHEMA, atomic_write_json, channel_path, result_path,
)
from scripts.score_programmatic_verification import score_result_file


def _query(spec):
    return spec["conversations"][0]["turns"][0]["query"]


def _spec():
    return next(
        spec for spec in load_specs(BENCHMARKS_JSON)
        if spec["name"] == "walmart_fy2025_revenue"
    )


def _write_run(
    experiment, outputs, snapshot=None, *, later_outputs=None, **result_fields
):
    spec = _spec()
    frozen_spec = snapshot or spec
    path = result_path(experiment, spec["name"], "model", "run-1")
    later_outputs = later_outputs or {}
    turns = []
    for index, turn_spec in enumerate(
        frozen_spec["conversations"][0]["turns"], start=1
    ):
        turn_outputs = outputs if index == 1 else later_outputs.get(index, {})
        turn = {
            "turn": index,
            "query": turn_spec["query"],
            "stores": sorted(turn_outputs),
            "response": "answer",
            "outputs": turn_outputs,
            "stop_reason": "completed",
        }
        if index == 1:
            turn.update(result_fields)
        turns.append(turn)
    atomic_write_json(path, {
        "schema": RESULT_SCHEMA,
        "case": spec["name"],
        "model": {"model_id": "model"},
        "conversations": [{"id": "main", "turns": turns}],
    })
    return path


def test_post_hoc_pv_scores_stored_outputs_without_changing_run(
    tmp_path, monkeypatch
):
    case_module = importlib.import_module(_spec()["module"])
    monkeypatch.setattr(
        case_module,
        "ground_truth",
        lambda: (
            "accession",
            "2025-01-31",
            "2025-03-14T12:00:00Z",
            "2024-02-01",
            680.985,
        ),
    )
    path = _write_run(tmp_path, {
        "filing_accession": "accession",
        "fiscal_period_end": "2025-01-31",
        "acceptance_datetime": "2025-03-14T12:00:00Z",
    }, later_outputs={2: {
        "revenue_period_start": "2024-02-01",
        "revenue_usd_billions": 680.985,
    }})
    before = path.read_bytes()
    assert score_result_file(path, _spec()) == "scored"
    assert score_result_file(path, _spec()) == "already_scored"
    assert path.read_bytes() == before
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["programmatic_verification"]["success"] is True
    variables = turn["programmatic_verification"]["variable_verification"]
    assert variables["total"] == variables["set"] == variables["correct"] == 3
    assert variables["set_rate"] == variables["accuracy"] == 1.0
    assert set(variables["items"]) == {
        "filing_accession", "fiscal_period_end", "acceptance_datetime",
    }


def test_post_hoc_pv_refuses_a_changed_query(tmp_path):
    snapshot = {
        key: value for key, value in _spec().items() if not key.startswith("_")
    }
    snapshot["conversations"][0]["turns"][0]["query"] = "an older question"
    path = _write_run(tmp_path, {}, snapshot=snapshot)
    assert score_result_file(path, _spec()) == "partially_scored"
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["status"] == "query_changed"
    assert "programmatic_verification" not in payload["conversations"][0]["turns"][0]


def test_post_hoc_pv_refuses_a_changed_output_contract(tmp_path):
    """A precision or format edit is a changed question, one surface along.

    The stored answer was given under the description the agent actually read.
    If that description now asks for different decimals, a different unit or a
    different date format, scoring the old answer against today's validator
    reports the edit as the model's mistake.
    """
    path = _write_run(
        tmp_path,
        {
            "filing_accession": "0000104169-25-000037",
            "fiscal_period_end": "2025-01-31",
            "acceptance_datetime": "2025-03-14T12:00:00Z",
        },
        output_contract="a-contract-that-no-longer-matches",
    )
    assert score_result_file(path, _spec()) == "partially_scored"
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["status"] == "contract_changed"
    assert turn["stored"] == "a-contract-that-no-longer-matches"
    assert turn["current"] != turn["stored"]
    assert "programmatic_verification" not in turn


def test_post_hoc_pv_scores_a_run_written_before_contracts_were_fingerprinted(tmp_path):
    """No fingerprint on disk means an older run, not a changed contract."""
    path = _write_run(
        tmp_path,
        {
            "filing_accession": "0000104169-25-000037",
            "fiscal_period_end": "2025-01-31",
            "acceptance_datetime": "2025-03-14T12:00:00Z",
        },
    )
    score_result_file(path, _spec())
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["status"] != "contract_changed"


@pytest.mark.runtime_data   # the validator recomputes from the runtime tables
def test_a_matching_output_contract_is_scored(tmp_path):
    """The fingerprint the evaluator writes today must not refuse itself."""
    module = importlib.import_module(_spec()["module"])
    stores = _spec()["conversations"][0]["turns"][0]["stores"]
    path = _write_run(
        tmp_path,
        {
            "filing_accession": "0000104169-25-000037",
            "fiscal_period_end": "2025-01-31",
            "acceptance_datetime": "2025-03-14T12:00:00Z",
        },
        output_contract=output_contract_fingerprint(module, stores),
    )
    score_result_file(path, _spec())
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["status"] == "scored"


def test_an_output_that_only_reached_disk_as_text_is_refused_not_failed(tmp_path):
    """A storage limit must never be reported as a wrong answer."""
    path = _write_run(
        tmp_path,
        {
            "filing_accession": "<DataFrame object>",
            "fiscal_period_end": "2025-01-31",
            "acceptance_datetime": "2025-03-14T12:00:00Z",
        },
        outputs_unserializable=["filing_accession"],
    )
    assert score_result_file(path, _spec()) == "partially_scored"
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["unserializable"] == ["filing_accession"]
    assert "programmatic_verification" not in payload["conversations"][0]["turns"][0]


def test_a_crashing_validator_is_recorded_and_does_not_abort_the_batch(
    tmp_path, monkeypatch
):
    case_module = importlib.import_module(_spec()["module"])

    def explode():
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(case_module, "ground_truth", explode)
    path = _write_run(tmp_path, {
        "filing_accession": "accession",
        "fiscal_period_end": "2025-01-31",
        "acceptance_datetime": "2025-03-14T12:00:00Z",
    })
    assert score_result_file(path, _spec()) == "partially_scored"
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["error_type"] == "ZeroDivisionError"
    assert "programmatic_verification" not in payload["conversations"][0]["turns"][0]


def test_a_case_that_grew_an_output_marks_the_older_run_stale(tmp_path, monkeypatch):
    path = _write_run(tmp_path, {
        "filing_accession": "accession",
        "fiscal_period_end": "2025-01-31",
    })
    assert score_result_file(path, _spec()) == "partially_scored"
    payload = json.loads(channel_path(path, "programmatic_verification").read_text())
    turn = payload["conversations"][0]["turns"][0]
    assert turn["missing"] == ["acceptance_datetime"]
