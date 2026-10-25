"""The failure classes that decide whether work is worth retrying.

A provider outage, a dropped connection or a runtime that died is transient: the
case is unchanged and the next attempt may succeed, so the runner records the
failure and re-queues it. A missing module, an unknown validator or a validator
that raises is deterministic: every retry burns another paid model call on a
defect that will fail identically, so the run stops and says so. Between them, a
model that ended a run without an answer is asked again a few times and then
scored, and an endpoint out of credit stops the run until it is topped up.

Catching these as one class makes each look like the others.
"""

from __future__ import annotations


class FinAgenticBenchError(Exception):
    """Base class for benchmark failures FinAgenticBench classifies deliberately."""


class InfrastructureError(FinAgenticBenchError):
    """Transient: provider, network, runtime. Retryable and resumable."""


class UnansweredError(FinAgenticBenchError):
    """The model ended a run without a usable answer (:mod:`core.failure`).

    The model's own failure, not an outage: asked again only so many times,
    without easing the provider's load, and then the runner asks for the run
    to be recorded as it ended, and it is scored as a failure.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"the model gave no answer: {reason}"
                         + (f" ({detail})" if detail else ""))
        self.reason = reason


class BillingError(FinAgenticBenchError):
    """The endpoint is out of credit. Every retry fails alike until a human tops
    it up, so the run stops; it can be resumed afterwards."""


class BenchmarkSpecificationError(FinAgenticBenchError):
    """Deterministic defect in a case, its module or its validator. Never retried."""
