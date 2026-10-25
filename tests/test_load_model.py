"""The tested agent's model carries every provider switch its config declares."""

from core.models import load_model

TOML = """
[thinking-off]
api_model = "m"
api_key = "k"
base_url = "http://localhost:1/v1"
extra_body = { enable_thinking = false }

[effort]
api_model = "m"
api_key = "k"
base_url = "http://localhost:1/v1"
max_tokens = 1000
temperature = 1.0
reasoning = true
reasoning_effort = "high"

[plain]
api_model = "m"
api_key = "k"
base_url = "http://localhost:1/v1"
"""


def test_extra_body_reaches_the_model(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    model, config = load_model(path, "thinking-off")
    assert config.extra_body == {"enable_thinking": False}
    assert model.kwargs["extra_body"] == {"enable_thinking": False}


def test_no_extra_body_is_sent_when_none_is_declared(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    model, _ = load_model(path, "plain")
    assert "extra_body" not in model.kwargs


def test_reasoning_model_gets_completion_cap_and_effort(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    model, _ = load_model(path, "effort")
    assert model.kwargs["max_completion_tokens"] == 1000
    assert "max_tokens" not in model.kwargs
    assert model.kwargs["extra_body"] == {"reasoning_effort": "high"}
    assert model.kwargs["temperature"] == 1.0


def test_plain_model_keeps_max_tokens(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(TOML)
    model, config = load_model(path, "plain")
    assert model.kwargs["max_tokens"] == config.max_tokens
    assert "max_completion_tokens" not in model.kwargs
