"""Gemma 4 via the official Google Gemini API — the reference provider.

This is the path the hackathon demo uses when a ``GEMINI_API_KEY`` is present:
``gemma-4-26b-a4b-it`` is a mixture-of-experts model (~25B total / ~3.8B active)
with a very large context window, which is why a whole execution plan, schema and
statistics block can be handed to it in one call.

Three engineering details that matter more than they look:

1. **Structured output.** The response schema is generated from the Pydantic
   models, so the API constrains the reply to valid JSON. We still parse it
   defensively (:mod:`schemapulse.agent.providers.parsing`) because a
   constrained-output guarantee is not the same as a correctness guarantee.
2. **``response_mime_type``/``response_schema``** — not prompt-only JSON
   instructions. Relying on the prompt alone is the single most common cause of
   "the model returned prose instead of JSON".
3. **The key is never logged.** :meth:`GeminiGemmaProvider.health` reports only
   whether a key is *present*.
"""

from __future__ import annotations

from typing import Any

import httpx

from schemapulse.agent.context import DiagnosticContext
from schemapulse.agent.providers.base import BaseProvider, GemmaResponse, Stopwatch
from schemapulse.agent.providers.parsing import parse_candidate_set
from schemapulse.agent.schemas import ExperimentFeedback
from schemapulse.errors import ModelOutputError, ModelUnavailableError

DEFAULT_MODEL = "gemma-4-26b-a4b-it"


class GeminiGemmaProvider(BaseProvider):
    """Structured-output Gemma adapter for the Gemini REST API."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = DEFAULT_MODEL,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 90.0,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_output_tokens: int = 1536,
    ) -> None:
        super().__init__(model_id, timeout_s=timeout_s, max_retries=max_retries)
        if not api_key:
            raise ModelUnavailableError(
                "SCHEMAPULSE_MODEL_PROVIDER=gemini requires GEMINI_API_KEY to be set"
            )
        # keep the key out of any exception message and out of repr()
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"GeminiGemmaProvider(model_id={self.model_id!r}, api_key='***')"

    # ------------------------------------------------------------------ core
    def _endpoint(self) -> str:
        return f"{self.base_url}/models/{self.model_id}:generateContent"

    def _payload(
        self,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a PostgreSQL performance engineer. You propose "
                            "index experiments. You never issue production commands and you "
                            "always answer with a single JSON object that matches the provided "
                            "schema exactly."
                        )
                    }
                ]
            },
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": _clean_schema(response_schema),
            },
        }

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
        payload = self._payload(prompt, response_schema)
        headers = {"Content-Type": "application/json"}
        trace: list[dict[str, Any]] = []
        last_error: Exception | None = None
        attempts_made = 0

        for attempt_index in range(self.max_retries + 1):
            attempts_made = attempt_index + 1
            try:
                with Stopwatch() as sw, httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(
                        self._endpoint(),
                        json=payload,
                        headers={**headers, "x-goog-api-key": self._api_key},
                    )
                trace.append(
                    {
                        "event": "gemini_http",
                        "attempt": attempt_index + 1,
                        "status_code": response.status_code,
                        "latency_ms": round(sw.elapsed_ms, 2),
                    }
                )
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        f"Gemini API returned HTTP {response.status_code}: "
                        f"{_safe_error(response.text)}"
                    )
                data = response.json()
                text, usage, finish = _extract_text(data)
                candidates, note = parse_candidate_set(text)
                return GemmaResponse(
                    candidates=candidates,
                    raw_text=text,
                    latency_ms=sw.elapsed_ms,
                    model_id=self.model_id,
                    provider=self.name,
                    is_live=True,
                    prompt_tokens=_usage_int(usage, "promptTokenCount"),
                    completion_tokens=_usage_int(usage, "candidatesTokenCount"),
                    finish_reason=finish,
                    attempts_made=attempts_made,
                    repair_note=note,
                    trace=trace,
                )
            except (httpx.HTTPError, ModelUnavailableError) as exc:
                last_error = exc
                trace.append(
                    {
                        "event": "gemini_error",
                        "attempt": attempt_index + 1,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )
            except ValueError as exc:
                last_error = exc
                trace.append(
                    {
                        "event": "gemini_parse_error",
                        "attempt": attempt_index + 1,
                        "error": str(exc)[:300],
                    }
                )
                # A structured-output contract means malformed JSON is a bug worth
                # one retry, not a reason to abandon the demo.
                payload["generationConfig"]["responseMimeType"] = "application/json"
                payload["generationConfig"]["temperature"] = min(
                    self.temperature + 0.1 * (attempt_index + 1), 0.8
                )

        raise ModelOutputError(
            f"Gemini provider failed after {attempts_made} attempt(s): "
            f"{type(last_error).__name__ if last_error else 'unknown'}: {last_error}"
        )

    # ---------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        """Verify the key and model without generating anything expensive."""
        info: dict[str, Any] = {
            "provider": self.name,
            "model": self.model_id,
            "api_key_present": True,
            "reachable": False,
            "model_available": None,
            "detail": "",
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(
                    f"{self.base_url}/models/{self.model_id}",
                    headers={"x-goog-api-key": self._api_key},
                )
            info["http_status"] = response.status_code
            if response.status_code == 200:
                info["reachable"] = True
                info["model_available"] = True
                info["detail"] = "model endpoint reachable"
            else:
                info["model_available"] = False
                info["detail"] = _safe_error(response.text)
        except httpx.HTTPError as exc:
            info["detail"] = f"{type(exc).__name__}: {exc}"[:200]
        return info


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _extract_text(
    data: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelOutputError(f"unexpected Gemini response shape: {list(data)[:6]}") from exc
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text.strip():
        raise ModelOutputError(
            f"Gemini returned no text (finishReason={candidate.get('finishReason')})"
        )
    return text, data.get("usageMetadata"), candidate.get("finishReason")


def _usage_int(usage: dict[str, Any] | None, key: str) -> int | None:
    if not usage:
        return None
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) else None


def _safe_error(text: str) -> str:
    """Truncate an API error body without echoing anything sensitive."""
    collapsed = " ".join(text.split())
    return collapsed[:300]


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the Pydantic-generated schema into the subset Gemini accepts.

    Two classes of problem, both of which the API rejects with an opaque HTTP 400
    naming an internal path rather than anything the caller can act on:

    * keywords it does not know (``title``, ``default``, ``description``, ...), and
    * ``const``, which Pydantic emits for a single-valued ``Literal``. Gemini
      supports ``enum`` but not ``const``, so ``Literal["index"]`` has to become
      ``{"enum": ["index"]}`` or every single request fails.

    ``$defs`` is kept during the walk and resolved afterwards, because inlining a
    ``$ref`` needs the definitions table still to be present.
    """
    unsupported = {
        "additionalProperties",
        "title",
        "default",
        "examples",
        "$schema",
        "description",
    }

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node

        const = node.get("const")
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "const":
                # Emitted after the loop so it never lands as a bare `const`.
                continue
            if key in unsupported:
                continue
            out[key] = walk(value)
        if const is not None:
            out["enum"] = [const]
        elif "enum" in node:
            out["enum"] = [walk(v) for v in node["enum"]]
        return out

    cleaned = walk(schema)
    defs = cleaned.pop("$defs", None)
    if defs:
        cleaned = _inline_defs(cleaned, defs)
    return cleaned


def _inline_defs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    """Gemini wants ``$ref`` resolved, not declared."""
    if depth > 12:
        return node
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.split("/")[-1], {})
            merged = {k: v for k, v in node.items() if k != "$ref"}
            target_clean = _inline_defs(target, defs, depth + 1)
            if isinstance(target_clean, dict):
                target_clean = {**target_clean, **merged}
                return target_clean
        return {k: _inline_defs(v, defs, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_defs(v, defs, depth + 1) for v in node]
    return node
