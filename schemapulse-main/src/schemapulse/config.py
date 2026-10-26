"""Configuration for SchemaPulse.

Everything is environment-driven so the same code runs in CI, in a judge's
notebook and in a laptop demo.  Resolution order for every setting:

1. explicit constructor argument (programmatic API)
2. process environment variable
3. ``.env`` file in the current working directory
4. built-in default

Secrets (``GEMINI_API_KEY``) never have a default and are never logged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from schemapulse.errors import ConfigurationError

ProviderName = Literal["gemini", "ollama", "fixture"]

KNOWN_PROVIDERS: tuple[str, ...] = ("gemini", "ollama", "fixture")

#: Field names whose values must never be printed, even when absent.
SECRET_FIELDS: frozenset[str] = frozenset({"gemini_api_key"})

#: Environment variables that are read but intentionally *not* prefixed.
UNPREFIXED_KEYS: dict[str, str] = {
    "GEMINI_API_KEY": "gemini_api_key",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
}


class Settings(BaseSettings):
    """Runtime configuration for one SchemaPulse run."""

    model_config = SettingsConfigDict(
        env_prefix="SCHEMAPULSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ model
    model_provider: ProviderName = Field(
        default="fixture",
        description="Which Gemma adapter to use: gemini | ollama | fixture.",
    )
    gemma_model: str = Field(
        default="gemma-4-26b-a4b-it",
        description="Model id passed to the Gemini API.",
    )
    # ``env_prefix`` does not apply to ``validation_alias``: without an explicit
    # alias, pydantic-settings would only ever look for SCHEMAPULSE_GEMINI_API_KEY
    # and silently ignore the documented GEMINI_API_KEY.  Each of these accepts
    # both spellings so a copied .env from another tool still works.
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("gemini_api_key", "GEMINI_API_KEY"),
        description="Gemini API key.",
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("ollama_base_url", "OLLAMA_BASE_URL"),
    )
    ollama_model: str = Field(
        # Gemma 3 is the newest Gemma published to the Ollama registry; there is
        # no `gemma4:*` tag, so pointing at one makes `ollama pull` fail with a
        # bare "file does not exist".  Override with OLLAMA_MODEL.
        default="gemma3:4b-it-q4_K_M",
        validation_alias=AliasChoices("ollama_model", "OLLAMA_MODEL"),
    )
    ollama_timeout_s: float = 180.0

    model_timeout_s: float = 90.0
    model_max_tokens: int = 1536
    model_temperature: float = 0.2
    model_max_retries: int = 2

    # ------------------------------------------------------------- databases
    database_url: str = "postgresql://postgres:postgres@localhost:5432/schemapulse"
    shadow_database_url: str = "postgresql://postgres:postgres@localhost:5433/schemapulse"

    # ------------------------------------------------------------ measurement
    query_timeout_ms: int = 30_000
    benchmark_runs: int = 5
    # A freshly built index is cold: the first execution after CREATE INDEX
    # reliably reads far slower than the steady state, which shows up as a huge
    # spread in the candidate arm and (before the noise guard existed) as a
    # spurious extra slowdown.  Three warmups costs a few hundred milliseconds
    # on the demo and buys a measurement that reflects steady state.
    benchmark_warmups: int = 3
    min_speedup: float = 1.20
    max_candidate_attempts: int = 3
    enable_refinement: bool = True

    # ------------------------------------------------------------- demo data
    demo_orders: int = 1_000_000
    demo_customers: int = 20_000

    # ---------------------------------------------------------------- output
    artifacts_dir: Path = Path(".schemapulse")
    verbose: bool = False

    @field_validator("model_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "none", "off"}:
                return "fixture"
            if v not in KNOWN_PROVIDERS:
                raise ValueError(
                    f"unknown model provider {value!r}; expected one of {', '.join(KNOWN_PROVIDERS)}"
                )
            return v
        return value

    @field_validator("min_speedup")
    @classmethod
    def _positive_speedup(cls, value: float) -> float:
        if value <= 1.0:
            raise ValueError("SCHEMAPULSE_MIN_SPEEDUP must be > 1.0 (e.g. 1.20)")
        return value

    @field_validator("benchmark_runs")
    @classmethod
    def _positive_runs(cls, value: int) -> int:
        if value < 1:
            raise ValueError("SCHEMAPULSE_BENCHMARK_RUNS must be >= 1")
        return value

    @field_validator("query_timeout_ms")
    @classmethod
    def _positive_timeout(cls, value: int) -> int:
        if value < 100:
            raise ValueError("SCHEMAPULSE_QUERY_TIMEOUT_MS must be >= 100")
        return value

    # ------------------------------------------------------------- helpers

    @property
    def redacted(self) -> dict[str, str]:
        """Config as a dict safe to print or attach to a report."""
        out: dict[str, str] = {}
        for name in self.model_fields:
            value = getattr(self, name)
            if name in SECRET_FIELDS:
                # An absent key must not stringify to the literal "None", which
                # reads like a value and hides the fact that nothing is set.
                secret = value.get_secret_value() if isinstance(value, SecretStr) else None
                out[name] = "***set***" if secret else "<unset>"
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = str(value)
        return out

    def require_target_url(self) -> str:
        return self.database_url

    def require_shadow_url(self) -> str:
        return self.shadow_database_url

    def assert_production_safe(self, target: str, shadow: str) -> None:
        """Refuse to run when the shadow database is not actually a different database.

        This is a hard invariant of the design: the model never gets a connection
        that could mutate the inspected database.
        """
        if target.strip() == shadow.strip():
            raise ConfigurationError(
                "shadow_database_url must differ from database_url — "
                "SchemaPulse never runs model-proposed DDL against the inspected database."
            )


@lru_cache(maxsize=1)
def get_settings(**overrides: object) -> Settings:
    """Return the process-wide settings singleton, with optional overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]


def reload_settings() -> Settings:
    """Drop the cached settings (used by tests and the CLI after env changes)."""
    get_settings.cache_clear()
    return get_settings()


def load_dotenv_if_present(path: str | os.PathLike[str] = ".env") -> bool:
    """Best-effort ``.env`` load. Returns True when the file existed."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(p, override=False)
    except Exception:  # pragma: no cover - dotenv is a hard dep, this is belt-and-braces
        return False
    get_settings.cache_clear()
    return True
