"""Configuration, secret handling, and the shadow-isolation invariant."""

from __future__ import annotations

from pathlib import Path

import pytest

from schemapulse.config import Settings
from schemapulse.errors import ConfigurationError

# --------------------------------------------------------------------------- #
# env contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make ``Settings()`` see nothing at all, the way a fresh clone would.

    Deleting the process variables is not enough on its own. ``Settings`` also
    reads a ``.env`` file from the current working directory, and anyone who has
    followed the README has one — with ``SCHEMAPULSE_MODEL_PROVIDER=gemini`` and
    a real key in it. Without the chdir below, every "default is offline"
    assertion fails on exactly the machine that has configured the tool, and
    passes on a machine that has not, which is the worst possible split.
    """
    for key in (
        "GEMINI_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "SCHEMAPULSE_MODEL_PROVIDER",
        "SCHEMAPULSE_GEMMA_MODEL",
        "SCHEMAPULSE_MIN_SPEEDUP",
        "SCHEMAPULSE_BENCHMARK_RUNS",
        "SCHEMAPULSE_QUERY_TIMEOUT_MS",
    ):
        monkeypatch.delenv(key, raising=False)
    # ``env_file=".env"`` is resolved relative to the cwd, so an empty directory
    # is a faithful stand-in for a checkout that has no .env yet.
    monkeypatch.chdir(tmp_path)


def test_defaults_are_offline_and_safe(clean_env: None) -> None:
    """A fresh install must not require a key or reach the network."""
    s = Settings()
    assert s.model_provider == "fixture"
    assert s.gemini_api_key is None


def test_prefixed_variables_load(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHEMAPULSE_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("SCHEMAPULSE_GEMMA_MODEL", "gemma-x")
    monkeypatch.setenv("SCHEMAPULSE_MIN_SPEEDUP", "2.5")
    monkeypatch.setenv("SCHEMAPULSE_BENCHMARK_RUNS", "9")
    s = Settings()
    assert s.model_provider == "ollama"
    assert s.gemma_model == "gemma-x"
    assert s.min_speedup == 2.5
    assert s.benchmark_runs == 9


def test_unprefixed_variables_load(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """These three are documented without the prefix, so they must work.

    Regression guard: ``env_prefix`` does not apply to ``validation_alias``, so
    without an explicit alias pydantic-settings would look only for
    ``SCHEMAPULSE_GEMINI_API_KEY`` and silently ignore ``GEMINI_API_KEY``.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-value")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.invalid:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    s = Settings()
    assert s.gemini_api_key is not None
    assert s.gemini_api_key.get_secret_value() == "test-key-value"
    assert s.ollama_base_url == "http://example.invalid:9999"
    assert s.ollama_model == "gemma3:1b"


# --------------------------------------------------------------------------- #
# secrets
# --------------------------------------------------------------------------- #
def test_redacted_never_contains_the_key(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    s = Settings()
    dumped = str(s.redacted)
    assert "super-secret-value" not in dumped
    assert s.redacted["gemini_api_key"] == "***set***"


def test_redacted_marks_an_unset_key(clean_env: None) -> None:
    assert Settings().redacted["gemini_api_key"] == "<unset>"


def test_secret_is_not_in_the_repr(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    s = Settings()
    assert "super-secret-value" not in repr(s)
    assert "super-secret-value" not in s.model_dump_json()


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["", "none", "off", "FIXTURE", " fixture "])
def test_provider_aliases_for_no_model(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SCHEMAPULSE_MODEL_PROVIDER", value)
    assert Settings().model_provider == "fixture"


def test_unknown_provider_is_rejected(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHEMAPULSE_MODEL_PROVIDER", "gpt4")
    with pytest.raises(ValueError, match="unknown model provider"):
        Settings()


def test_speedup_must_exceed_one(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHEMAPULSE_MIN_SPEEDUP", "1.0")
    with pytest.raises(ValueError, match="min_speedup"):
        Settings()


def test_runs_must_be_positive(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHEMAPULSE_BENCHMARK_RUNS", "0")
    with pytest.raises(ValueError, match="BENCHMARK_RUNS"):
        Settings()


# --------------------------------------------------------------------------- #
# the load-bearing safety invariant
# --------------------------------------------------------------------------- #
def test_shadow_url_must_differ_from_target(clean_env: None) -> None:
    """The model must never hold a connection that could mutate the target."""
    s = Settings()
    with pytest.raises(ConfigurationError, match="never runs model-proposed DDL"):
        s.assert_production_safe(s.database_url, s.database_url)


def test_distinct_urls_pass_the_invariant(clean_env: None) -> None:
    s = Settings()
    s.assert_production_safe(s.database_url, s.shadow_database_url)


def test_default_shadow_is_a_separate_port(clean_env: None) -> None:
    s = Settings()
    assert s.database_url != s.shadow_database_url
    assert s.database_url.endswith(":5432/schemapulse")
    assert s.shadow_database_url.endswith(":5433/schemapulse")


def test_shipped_default_ollama_model_exists_in_the_registry(clean_env: None) -> None:
    """Guard against pointing at a tag that `ollama pull` cannot resolve.

    ``gemma4:*`` is not published to the Ollama registry; a default naming it
    fails with a bare "file does not exist" long before any useful error.
    """
    assert not Settings().ollama_model.lower().startswith("gemma4")
