"""A run the model ended without an answer is told apart from an outage, with its reason."""

import pytest
from cave_agent.models.errors import (
    PromptTooLongError,
    ProviderBillingError,
    ProviderError,
    StreamStalledError,
)

from core.failure import classify


@pytest.mark.parametrize("error,expected", [
    (ProviderError("The model completed without returning an answer."), ("empty_answer", True)),
    (ProviderError("Model output was truncated and could not be recovered"),
     ("output_limit", True)),
    (PromptTooLongError("context_length_exceeded"), ("prompt_too_long", True)),
    (ProviderError("The provider filtered this response and returned no content."),
     ("content_filter", False)),
    (ProviderError("The provider filtered this response after partially streaming it."),
     ("content_filter", False)),
    (ProviderError("BadRequestError: Output data may contain inappropriate content."),
     ("content_filter", False)),
    (ProviderError("APIConnectionError: connection reset"), ("provider", False)),
    (ProviderError("The model stream ended without a terminal finish reason."),
     ("provider", False)),
    (StreamStalledError("no chunk for 600 s"), ("stream_stalled", False)),
    (ProviderBillingError("out of credit"), ("billing", False)),
    (None, ("unknown", False)),
])
def test_a_failure_is_the_models_only_when_the_reply_itself_failed(error, expected):
    failure = classify(error)
    assert (failure.kind, failure.model_fault) == expected

