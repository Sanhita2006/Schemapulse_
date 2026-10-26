"""Provider selection.

The configured provider is a *preference order*, not a hard requirement: if the
preferred one is unreachable, SchemaPulse walks down the list and records the
downgrade in the trace.  A demo that dies because a Wi-Fi network blinked is a
worse demo than one that honestly says "live Gemma unavailable, using the
deterministic baseline".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemapulse.agent.providers.base import BaseProvider, GemmaProvider, default_response_schema
from schemapulse.agent.providers.fixture import FixtureGemmaProvider
from schemapulse.agent.providers.gemini import DEFAULT_MODEL, GeminiGemmaProvider
from schemapulse.agent.providers.ollama import DEFAULT_OLLAMA_MODEL, OllamaGemmaProvider
from schemapulse.config import Settings
from schemapulse.errors import ConfigurationError, ModelUnavailableError


@dataclass(slots=True)
class ResolvedProvider:
    """The chosen provider plus an honest account of how it was chosen."""

    provider: GemmaProvider
    requested: str
    downgraded_from: str | None = None
    reason: str = ""
    considered: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return bool(getattr(self.provider, "is_live", False))

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "model": self.provider.model_id,
            "is_live": self.is_live,
            "requested": self.requested,
            "downgraded_from": self.downgraded_from,
            "reason": self.reason,
            "considered": self.considered,
        }


def _build(provider: str, settings: Settings) -> BaseProvider:
    if provider == "gemini":
        key = settings.gemini_api_key
        if key is None or not key.get_secret_value().strip():
            raise ModelUnavailableError("GEMINI_API_KEY is not set")
        return GeminiGemmaProvider(
            api_key=key.get_secret_value(),
            model_id=settings.gemma_model or DEFAULT_MODEL,
            base_url=settings.gemini_base_url,
            timeout_s=settings.model_timeout_s,
            max_retries=settings.model_max_retries,
            temperature=settings.model_temperature,
            max_output_tokens=settings.model_max_tokens,
        )
    if provider == "ollama":
        return OllamaGemmaProvider(
            base_url=settings.ollama_base_url,
            model_id=settings.ollama_model or DEFAULT_OLLAMA_MODEL,
            timeout_s=settings.ollama_timeout_s,
            max_retries=0,
            temperature=settings.model_temperature,
            max_output_tokens=settings.model_max_tokens,
        )
    if provider == "fixture":
        return FixtureGemmaProvider()
    raise ConfigurationError(f"unknown provider {provider!r}")


#: Preference order used when the configured provider is unavailable.
FALLBACK_ORDER: dict[str, tuple[str, ...]] = {
    "gemini": ("ollama", "fixture"),
    "ollama": ("gemini", "fixture"),
    "fixture": ("ollama",),
}


def resolve_provider(
    settings: Settings,
    *,
    allow_fallback: bool = True,
    require_live: bool = False,
) -> ResolvedProvider:
    """Build the configured provider, optionally falling back when unreachable."""
    requested = settings.model_provider
    order = (requested, *FALLBACK_ORDER.get(requested, ()))
    considered: list[dict[str, Any]] = []
    first_error: str = ""

    for position, name in enumerate(order):
        try:
            provider = _build(name, settings)
        except Exception as exc:
            considered.append({"provider": name, "usable": False, "detail": str(exc)[:200]})
            if position == 0:
                first_error = f"{type(exc).__name__}: {exc}"
            if not allow_fallback and position == 0:
                raise
            continue

        if require_live:
            health = provider.health()
            considered.append({"provider": name, "usable": False, "detail": str(health)[:200]})
            if not health.get("model_available"):
                if position == 0:
                    first_error = str(health.get("detail") or health)
                if not allow_fallback:
                    raise ModelUnavailableError(
                        f"provider {name!r} is not usable: {health.get('detail')}"
                    )
                continue

        considered.append({"provider": name, "usable": True})
        return ResolvedProvider(
            provider=provider,
            requested=requested,
            downgraded_from=requested if name != requested else None,
            reason=first_error,
            considered=considered,
        )

    raise ModelUnavailableError(
        f"no usable Gemma provider. Tried {', '.join(order)}. Last error: {first_error or 'unknown'}"
    )


def response_schema() -> dict[str, Any]:
    return default_response_schema()


__all__ = [
    "BaseProvider",
    "FixtureGemmaProvider",
    "GeminiGemmaProvider",
    "OllamaGemmaProvider",
    "ResolvedProvider",
    "default_response_schema",
    "resolve_provider",
    "response_schema",
]
