"""Provider-aware synchronous client used by FinAgenticBench's LLM judges."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable

import requests

from core.models import ModelConfig


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def retryable(error: Exception) -> bool:
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return True
    return isinstance(error, requests.HTTPError) and error.response is not None and (
        error.response.status_code == 429 or error.response.status_code >= 500
    )


def with_retry(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    delay: Callable[[int], float] = lambda attempt: 2 ** attempt + 0.5,
    sleeper: Callable[[float], None] = time.sleep,
):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts - 1 or not retryable(error):
                raise
            sleeper(delay(attempt))
    raise RuntimeError("retry loop exhausted without returning or raising")


def extract_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


class LLMClient:
    """One judge client with explicit retry and provider dispatch."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        request: Callable[..., requests.Response] = requests.post,
        attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self._request = request
        self._attempts = attempts
        self._sleeper = sleeper

    def complete(self, prompt: str, *, json_mode: bool = False) -> Completion:
        return with_retry(
            lambda: self._complete_once(prompt, json_mode=json_mode),
            attempts=self._attempts,
            sleeper=self._sleeper,
        )

    def complete_json(self, prompt: str) -> dict[str, Any]:
        value, _ = self.complete_json_with_usage(prompt)
        return value

    def complete_json_with_usage(
        self, prompt: str
    ) -> tuple[dict[str, Any], Completion]:
        last_error: Exception | None = None
        for _ in range(self._attempts):
            try:
                completion = self.complete(prompt, json_mode=True)
                return extract_json_object(completion.text), completion
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
        raise ValueError(
            f"{self.config.name}: malformed JSON after {self._attempts} attempts"
        ) from last_error

    def _complete_once(self, prompt: str, *, json_mode: bool) -> Completion:
        if not self.config.base_url:
            raise ValueError(f"{self.config.name}: base_url is required for judge calls")
        responses_api = self.config.provider == "openai_responses"
        endpoint = "/responses" if responses_api else "/chat/completions"
        url = self.config.base_url.rstrip("/") + endpoint
        body = (
            self._responses_body(prompt, json_mode)
            if responses_api else self._chat_body(prompt, json_mode)
        )
        response = self._post(url, body)
        if response.status_code == 400 and json_mode:
            fallback = dict(body)
            fallback.pop("response_format", None)
            fallback.pop("text", None)
            if fallback != body:
                response = self._post(url, fallback)
        response.raise_for_status()
        payload = response.json()
        if responses_api:
            usage = payload.get("usage", {})
            return Completion(
                payload.get("output_text") or self._responses_text(payload),
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
            )
        usage = payload.get("usage", {})
        return Completion(
            payload["choices"][0]["message"].get("content") or "",
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
        )

    def _post(self, url: str, body: dict[str, Any]):
        return self._request(
            url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json=body,
            timeout=self.config.timeout,
        )

    def _chat_body(self, prompt: str, json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.api_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.reasoning:
            body["max_completion_tokens"] = min(self.config.max_tokens, 4096)
            if self.config.reasoning_effort:
                body["reasoning_effort"] = self.config.reasoning_effort
        else:
            body["temperature"] = 0.1
            body["max_tokens"] = min(self.config.max_tokens, 4096)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.config.extra_body)
        return body

    def _responses_body(self, prompt: str, json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.api_model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": min(self.config.max_tokens, 4096),
        }
        reasoning = {}
        if self.config.reasoning_effort:
            reasoning["effort"] = self.config.reasoning_effort
        if self.config.reasoning_summary:
            reasoning["summary"] = self.config.reasoning_summary
        if reasoning:
            body["reasoning"] = reasoning
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        body.update(self.config.extra_body)
        return body

    @staticmethod
    def _responses_text(payload: dict[str, Any]) -> str:
        parts = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(content.get("text", ""))
        return "".join(parts)
