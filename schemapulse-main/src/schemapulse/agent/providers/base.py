"""The Gemma provider interface.

SchemaPulse does not know which model produced a candidate.  It knows that
*something* produced a :class:`~schemapulse.agent.schemas.CandidateSet`.  That
separation is why the reference model can be swapped for a local Ollama model,
or a fixture for an offline demo, without touching the engine or the experiments.

What every adapter must guarantee
---------------------------------
* ``is_live`` is True only if a real model was actually called.
* Latency is always reported, so a slow provider is visible rather than assumed.
* Raw text is retained for the report.  Transparency beats tidiness when you are
  demonstrating that a model was really used.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from schemapulse.agent.context import DiagnosticContext
from schemapulse.agent.prompts import render_prompt
from schemapulse.agent.schemas import CandidateSet, ExperimentFeedback


class GemmaResponse(BaseModel):
    """What an adapter returns to the engine."""

    model_config = ConfigDict(extra="forbid")

    candidates: CandidateSet
    raw_text: str = ""
    latency_ms: float = 0.0
    model_id: str = ""
    provider: str = ""
    is_live: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    attempts_made: int = 1
    repair_note: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@runtime_checkable
class GemmaProvider(Protocol):
    """Adapter contract for a Gemma-backed candidate generator."""

    name: str

    @property
    def is_live(self) -> bool:
        """True when a real model was contacted for this call."""
        ...

    @property
    def model_id(self) -> str:
        """Human-readable model identifier, recorded in every report."""
        ...

    def propose(
        self,
        *,
        context: DiagnosticContext,
        response_schema: dict[str, Any],
        attempt: int = 1,
        feedback: list[ExperimentFeedback] | None = None,
        max_candidates: int = 3,
    ) -> GemmaResponse:
        """Return a validated candidate set for the given diagnostic context."""
        ...

    def health(self) -> dict[str, Any]:
        """Cheap reachability probe used by ``schemapulse doctor``."""
        ...


class BaseProvider:
    """Shared plumbing: model id, timeout, retry budget."""

    name = "base"

    def __init__(
        self,
        model_id: str,
        *,
        timeout_s: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        self._model_id = model_id
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_live(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model_id, "reachable": self.is_live}

    def propose(
        self,
        *,
        context: DiagnosticContext,
        response_schema: dict[str, Any],
        attempt: int = 1,
        feedback: list[ExperimentFeedback] | None = None,
        max_candidates: int = 3,
    ) -> GemmaResponse:
        """Generate candidates. Subclasses must implement this.

        Declared here so ``BaseProvider`` structurally satisfies the
        :class:`GemmaProvider` protocol -- every concrete provider does
        implement it, but without this declaration the type checker cannot see
        that and rejects every ``ResolvedProvider`` construction.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement propose()")

    @staticmethod
    def render(
        context: DiagnosticContext,
        feedback: list[ExperimentFeedback] | None,
        max_candidates: int = 3,
    ) -> str:
        """Render the user prompt. Identical for every provider, by design."""
        return render_prompt(context, feedback=feedback, max_candidates=max_candidates)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model_id, "is_live": self.is_live}


class Stopwatch:
    """Context manager returning elapsed milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def default_response_schema() -> dict[str, Any]:
    """JSON Schema handed to the model as the structured-output contract.

    Derived from the Pydantic models so the two can never drift apart: if
    :class:`~schemapulse.agent.schemas.CandidateSet` changes, the schema the
    model sees changes with it.
    """
    return CandidateSet.model_json_schema()
