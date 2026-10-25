import json

import pytest
import requests

from core.judging import (
    JudgeCallTracker, evaluate_panel, infer_tested_lab, rater_pool,
)
from core.llm import LLMClient, extract_json_object, with_retry
from core.models import ModelConfig, ModelRegistry


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"status {self.status_code}")
            error.response = self
            raise error


def config(name="gpt-test", **overrides):
    values = dict(
        name=name, api_model="provider-model", api_key="secret",
        base_url="https://example.test/v1", lab="openai",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_registry_infers_lab_and_never_exposes_api_key(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[qwen-test]\napi_model="q"\napi_key="secret"\n'
        'base_url="https://example.test/v1"\n'
    )
    registry = ModelRegistry.load(path)
    loaded = registry.get("qwen-test")
    assert loaded.lab == "alibaba"
    assert "api_key" not in loaded.public_fingerprint()
    assert "secret" not in repr(loaded)


def test_chat_client_uses_json_mode_and_parses_usage():
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({
            "choices": [{"message": {"content": '{"score": 4}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    completion = LLMClient(config(), request=request).complete("judge", json_mode=True)
    assert json.loads(completion.text) == {"score": 4}
    assert (completion.prompt_tokens, completion.completion_tokens) == (3, 2)
    assert calls[0][0].endswith("/chat/completions")
    assert calls[0][1]["json"]["response_format"] == {"type": "json_object"}
    assert "secret" not in repr(calls[0][1]["json"])
    value, usage = LLMClient(config(), request=request).complete_json_with_usage("judge")
    assert value == {"score": 4}
    assert (usage.prompt_tokens, usage.completion_tokens) == (3, 2)


def test_retry_only_retries_transient_failures():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise requests.ConnectionError("temporary")
        return "ok"

    assert with_retry(operation, sleeper=lambda _: None) == "ok"
    assert calls == 3
    with pytest.raises(ValueError):
        with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")), sleeper=lambda _: None)


def test_extract_json_accepts_fenced_prose_but_requires_an_object():
    assert extract_json_object("result: ```json\n{\"x\": 1}\n```") == {"x": 1}
    with pytest.raises(ValueError):
        extract_json_object("no object")


def test_panel_exclusion_and_health_tracking():
    judges = [
        config("gpt-one", lab="openai"),
        config("qwen-one", lab="alibaba"),
        config("deepseek-one", lab="deepseek"),
    ]
    assert [item.name for item in rater_pool(
        "openai", judges, exclude_same_lab=True
    )] == ["qwen-one", "deepseek-one"]
    tracker = JudgeCallTracker()

    def score(judge):
        if judge.name == "qwen-one":
            raise RuntimeError("down")
        return {"judge": judge.name}

    panel = evaluate_panel(judges[:2], score, tracker=tracker)
    assert [judge.name for judge, _ in panel.successes] == ["gpt-one"]
    assert panel.failures[0]["judge"] == "qwen-one"
    assert tracker.summary()["qwen-one"]["failures"] == 1
    tracker.record("gpt-one", True, {"prompt_tokens": 7, "completion_tokens": 3})
    assert tracker.summary()["gpt-one"]["total_tokens"] == 10


def test_a_panel_that_clears_quorum_is_scored_even_if_a_judge_is_lost():
    judges = [config("gpt-one", lab="openai"), config("qwen-one", lab="alibaba"),
              config("deepseek-one", lab="deepseek")]
    tracker = JudgeCallTracker()

    def score(judge):
        if judge.name == "deepseek-one":
            raise RuntimeError("down")
        return {"judge": judge.name}

    panel = evaluate_panel(judges, score, tracker=tracker)
    assert panel.responded == 2 and panel.quorum_met and not panel.complete
    assert panel.skip_reason() is None


def test_a_panel_below_quorum_is_skipped():
    tracker = JudgeCallTracker()

    def down(judge):
        raise RuntimeError("down")

    panel = evaluate_panel(
        [config("a", lab="openai"), config("b", lab="alibaba")], down,
        tracker=tracker,
    )
    assert not panel.quorum_met
    assert panel.skip_reason().startswith("judges_below_quorum: 0/2")


def test_explicit_single_judge_panel_is_supported_for_exploratory_scoring():
    judge = config("deepseek-only", lab="deepseek")
    tracker = JudgeCallTracker()
    panel = evaluate_panel(
        [judge], lambda selected: {"judge": selected.name},
        tracker=tracker, minimum=1,
    )
    assert [selected.name for selected, _ in panel.successes] == ["deepseek-only"]
    assert panel.failures == []
    assert panel.quorum_met and panel.complete


def test_saved_tested_lab_takes_precedence_over_mutable_registry_metadata():
    registry = ModelRegistry({"tested": config("tested", lab="openai")})
    saved = {"model": {"name": "tested", "model_id": "provider-model", "lab": "deepseek"}}
    assert infer_tested_lab(saved, registry) == "deepseek"
