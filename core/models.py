from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import tomllib

import httpx
from cave_agent.models.base import Model

from core.responses_model import ResponsesModel
from core.stream_repair import FenceRepairingModel
from core.tool_gateway import ToolGateway


_LAB_PREFIXES = (
    (("claude-", "opus", "sonnet", "haiku"), "anthropic"),
    (("gpt-", "gpt", "o1-", "o3-", "o4-"), "openai"),
    (("gemini",), "google"),
    (("kimi",), "moonshot"),
    (("qwen",), "alibaba"),
    (("deepseek",), "deepseek"),
    (("glm",), "zhipu"),
    (("minimax",), "minimax"),
)


def infer_lab(name: str) -> str | None:
    lowered = name.lower()
    for prefixes, lab in _LAB_PREFIXES:
        if lowered.startswith(prefixes):
            return lab
    return None


# Wire protocols a models.toml entry may name. "google_genai" (the Gemini API
# through the google-genai SDK) and "anthropic" (the Messages API through the
# anthropic SDK) are spoken only by the tool-calling harness.
PROVIDERS = ("openai_compatible", "openai_responses", "google_genai", "anthropic")
TOOL_HARNESS_ONLY = ("google_genai", "anthropic")
# Seconds to open a connection: an unreachable endpoint fails fast and is
# retried, instead of holding a step for a whole read timeout per retry.
CONNECT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by tested agents and LLM judges."""

    name: str
    api_model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 8192
    timeout: float = 300.0
    lab: str | None = None
    provider: str = "openai_compatible"
    reasoning: bool = False
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    litellm_provider: str = "openai"
    input_cost_per_million_usd: float | None = None
    output_cost_per_million_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.api_model or not self.api_key:
            raise ValueError("model name, api_model and api_key must be non-empty")
        if self.max_tokens < 1 or self.timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if any(
            value is not None and value < 0
            for value in (
                self.input_cost_per_million_usd,
                self.output_cost_per_million_usd,
            )
        ):
            raise ValueError("model token prices cannot be negative")
        if self.provider not in PROVIDERS:
            raise ValueError(f"unsupported provider {self.provider!r}")

    def request_timeout(self) -> httpx.Timeout:
        """``timeout`` for each phase of a request, but :data:`CONNECT_TIMEOUT_S` at
        most to connect. On a stream the read timeout is the longest silence allowed."""
        return httpx.Timeout(self.timeout, connect=min(CONNECT_TIMEOUT_S, self.timeout))

    def chat_options(self) -> dict[str, Any]:
        """Sampling options for a Chat Completions request, as every tested agent sends them.

        A ``reasoning`` model takes its cap as ``max_completion_tokens``, which
        OpenAI reasoning models require in place of ``max_tokens``; LiteLLM
        renames the cap only for model ids it recognises. ``reasoning_effort``
        rides in ``extra_body`` with the provider switches -- whether a model
        thinks, for one -- so it reaches endpoints the client does not know.
        """
        body = dict(self.extra_body)
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        cap = "max_completion_tokens" if self.reasoning else "max_tokens"
        return {
            "temperature": self.temperature,
            cap: self.max_tokens,
            **({"extra_body": body} if body else {}),
        }

    def responses_options(self) -> dict[str, Any]:
        """Sampling options for a Responses API request, as every tested agent sends them."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            **({"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}),
            **({"extra_body": dict(self.extra_body)} if self.extra_body else {}),
        }

    def gemini_options(self) -> dict[str, Any]:
        """Generation options for a Gemini API request, as the tool-calling agent sends them.

        ``reasoning_effort`` names Gemini 3's ``thinking_level`` (low, medium or
        high; the models have no level without thinking).
        """
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            **({"thinking_config": {"thinking_level": self.reasoning_effort}}
               if self.reasoning_effort else {}),
        }

    def anthropic_options(self) -> dict[str, Any]:
        """Generation options for a Messages API request, as the tool-calling agent sends them.

        The anthropic SDK takes no ``temperature`` argument, so it rides in
        ``extra_body`` with the provider switches -- ``thinking``, for one -- and
        ``reasoning_effort``, which names ``output_config.effort``.
        """
        body = {**self.extra_body, "temperature": self.temperature}
        if self.reasoning_effort:
            body["output_config"] = {"effort": self.reasoning_effort}
        return {"max_tokens": self.max_tokens, "extra_body": body}

    def public_fingerprint(self) -> dict[str, Any]:
        """Return model metadata safe to persist (never includes the API key)."""
        return {
            "name": self.name,
            "model_id": self.api_model,
            "base_url": self.base_url,
            "lab": self.lab,
            "provider": self.provider,
            "reasoning": self.reasoning,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "litellm_provider": self.litellm_provider,
            "input_cost_per_million_usd": self.input_cost_per_million_usd,
            "output_cost_per_million_usd": self.output_cost_per_million_usd,
            "extra_body_fingerprint": hashlib.sha256(
                json.dumps(self.extra_body, sort_keys=True).encode()
            ).hexdigest()[:16],
        }


class ModelRegistry:
    def __init__(self, configs: dict[str, ModelConfig]):
        self._configs = dict(configs)

    @classmethod
    def load(cls, path: Path | str) -> "ModelRegistry":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist; create it from models.toml.example"
            )
        raw = tomllib.loads(path.read_text())
        configs = {}
        for name, values in raw.items():
            fields = dict(values)
            fields.setdefault("lab", infer_lab(name))
            try:
                configs[name] = ModelConfig(name=name, **fields)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid model {name!r} in {path}: {error}") from error
        return cls(configs)

    def get(self, name: str) -> ModelConfig:
        try:
            return self._configs[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model {name!r}; available: {sorted(self._configs)}"
            ) from error

    def select(self, names: Iterable[str]) -> list[ModelConfig]:
        return [self.get(name) for name in names]

    def find_by_api_model(self, api_model: str) -> ModelConfig | None:
        return next(
            (config for config in self._configs.values() if config.api_model == api_model),
            None,
        )

    @property
    def names(self) -> list[str]:
        return sorted(self._configs)


def load_model(path: Path, name: str) -> tuple[Model, ModelConfig]:
    """Build the agent's model wrapper from a models.toml entry.

    ``FenceRepairingModel`` is cave_agent's ``LiteLLMModel`` with its stream
    passed through :mod:`core.stream_repair`, which turns a provider's native
    tool envelope into the markdown fences cave-agent parses, and closes a fence
    the provider left open at a clean stop. Without it those turns run no code
    and set no variable while still reporting ``stop_reason == "completed"``,
    which is indistinguishable from a wrong answer -- ten such turns are on disk
    under ``experiments/``. It is a no-op on a well-formed stream.

    Only the tested agent is wrapped, and only this function builds it. Judges
    reach their provider through :class:`core.llm.LLMClient`, which posts to the
    endpoint directly and never constructs a ``LiteLLMModel``, so a repair that
    rewrites ``<...>`` runs cannot reach prose scoring.

    Request options come from :meth:`ModelConfig.chat_options`, so this agent
    and the tool-calling one (:func:`load_tool_gateway`) ask a model alike.
    ``provider = "openai_responses"`` selects :class:`core.responses_model.ResponsesModel`
    instead, for a model whose agent steps only the Responses API returns.
    """
    cfg = ModelRegistry.load(path).get(name)
    if cfg.provider in TOOL_HARNESS_ONLY:
        raise ValueError(f"{name}: only the tool harness speaks {cfg.provider} (--harness tool)")
    if cfg.provider == "openai_responses":
        return ResponsesModel(
            model_id=cfg.api_model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.request_timeout(),
            options=cfg.responses_options(),
        ), cfg
    model = FenceRepairingModel(
        model_id=cfg.api_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        custom_llm_provider=cfg.litellm_provider,
        # One float: LiteLLM's `acompletion` takes no per-phase timeout, so this
        # harness has no separate connect cap (`ModelConfig.request_timeout`).
        timeout=cfg.timeout,
        num_retries=3,
        **cfg.chat_options(),
    )
    return model, cfg


def load_tool_gateway(path: Path, name: str) -> tuple[ToolGateway, ModelConfig]:
    """The tool-calling harness's counterpart of :func:`load_model`.

    Reads the same ``models.toml`` entry: ``provider`` picks Chat Completions,
    the Responses API, the Gemini API or the Messages API, with the same
    request options.
    """
    cfg = ModelRegistry.load(path).get(name)
    return ToolGateway(cfg), cfg
