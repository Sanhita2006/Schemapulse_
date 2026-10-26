"""Gemma 4 via a local Ollama server.

This is the adapter that makes SchemaPulse defensible in a room where the
network is off.  Because Gemma is open-weight, the identical engine — same
prompt, same schema, same validator, same shadow experiment — runs against a
model on the judge's own laptop.  Nothing about the verification half changes.

Two practical notes:

* ``format`` is set to the JSON Schema. Ollama's structured-output support is
  grammar-constrained, so a bad shape fails loudly rather than silently.
* CPU-only inference is slow (tens of seconds is normal for a 12B model).
  Latency is reported honestly rather than hidden, and the retry budget is 0 by
  default so a slow local model cannot silently double a demo's runtime.
"""

from __future__ import annotations

from typing import Any

import httpx

from schemapulse.agent.context import DiagnosticContext
from schemapulse.agent.providers.base import BaseProvider, GemmaResponse, Stopwatch
from schemapulse.agent.providers.parsing import parse_candidate_set
from schemapulse.agent.schemas import ExperimentFeedback
from schemapulse.errors import ModelOutputError, ModelUnavailableError

DEFAULT_OLLAMA_MODEL = "gemma4:12b-it-q4"


class OllamaGemmaProvider(BaseProvider):
    """Local Gemma adapter for Ollama's ``/api/chat`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model_id: str = DEFAULT_OLLAMA_MODEL,
        timeout_s: float = 180.0,
        max_retries: int = 0,
        temperature: float = 0.2,
        max_output_tokens: int = 1536,
    ) -> None:
        super().__init__(model_id, timeout_s=timeout_s, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------ core
    def propose(
        self,
        *,
        context: DiagnosticContext,
        response_schema: dict[str, Any],
        attempt: int = 1,
        feedback: list[ExperimentFeedback] | None = None,
        max_candidates: int = 3,
    ) -> GemmaResponse:
        prompt = self.render(context, feedback, max_candidates)
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": response_schema,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        trace: list[dict[str, Any]] = []
        last_error: Exception | None = None
        attempts_made = 0

        for attempt_index in range(self.max_retries + 1):
            attempts_made = attempt_index + 1
            try:
                with Stopwatch() as sw, httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(f"{self.base_url}/api/chat", json=payload)
                trace.append(
                    {
                        "event": "ollama_http",
                        "attempt": attempt_index + 1,
                        "status_code": response.status_code,
                        "latency_ms": round(sw.elapsed_ms, 2),
                    }
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        f"Ollama returned HTTP {response.status_code}: "
                        f"{' '.join(response.text.split())[:300]}"
                    )
                data = response.json()
                text = str(data.get("message", {}).get("content", ""))
                if not text.strip():
                    raise ModelOutputError("Ollama returned an empty message")
                candidates, note = parse_candidate_set(text)
                return GemmaResponse(
                    candidates=candidates,
                    raw_text=text,
                    latency_ms=sw.elapsed_ms,
                    model_id=self.model_id,
                    provider=self.name,
                    is_live=True,
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                    finish_reason=data.get("done_reason"),
                    attempts_made=attempts_made,
                    repair_note=note,
                    trace=trace,
                )
            except (httpx.HTTPError, ModelUnavailableError, ValueError) as exc:
                last_error = exc
                trace.append(
                    {
                        "event": "ollama_error",
                        "attempt": attempt_index + 1,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )

        raise ModelOutputError(
            f"Ollama provider failed after {attempts_made} attempt(s): {last_error}"
        )

    # ---------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "provider": self.name,
            "model": self.model_id,
            "base_url": self.base_url,
            "reachable": False,
            "model_available": None,
            "detail": "",
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                info["detail"] = f"HTTP {response.status_code} from /api/tags"
                return info
            info["reachable"] = True
            models = [
                m.get("name", "") for m in response.json().get("models", []) if isinstance(m, dict)
            ]
            base = self.model_id.split(":")[0]
            exact = self.model_id in models
            same_family = any(m.split(":")[0] == base for m in models)
            info["installed_models"] = models
            info["model_available"] = exact or same_family
            if exact:
                info["detail"] = f"{self.model_id} is installed"
            elif same_family:
                matching = [m for m in models if m.split(":")[0] == base]
                info["detail"] = (
                    f"{self.model_id} not pulled; same family available: {', '.join(matching)}"
                )
            else:
                info["detail"] = f"no '{base}' model installed; run: ollama pull {self.model_id}"
        except httpx.HTTPError as exc:
            info["detail"] = f"{type(exc).__name__}: {exc}"[:200]
        return info
