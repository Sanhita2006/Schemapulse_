"""Shared pytest fixtures and markers.

Unit tests (everything in ``tests/unit``) must run with no database, no Docker
and no network.  Integration tests opt in with ``@pytest.mark.integration`` and
are skipped automatically when the demo environment is not up, so a fresh clone
can still say ``pytest`` and get a useful result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"
GOLDEN_QUERY = DEMO_DIR / "slow_orders.sql"


# --------------------------------------------------------------------------- #
# environment probing
# --------------------------------------------------------------------------- #
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _database_reachable(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _default_dsn(which: str) -> str:
    if which == "target":
        return os.environ.get(
            "SCHEMAPULSE_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/schemapulse"
        )
    return os.environ.get(
        "SCHEMAPULSE_SHADOW_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5433/schemapulse",
    )


@pytest.fixture(scope="session")
def demo_env() -> Any:
    """Skip the test unless both demo databases answer.

    Set ``SCHEMAPULSE_SKIP_INTEGRATION=1`` to skip explicitly even when the
    databases happen to be running (useful for a fast offline run).
    """
    if _env_flag("SCHEMAPULSE_SKIP_INTEGRATION"):
        pytest.skip("integration disabled via SCHEMAPULSE_SKIP_INTEGRATION")

    from schemapulse.config import Settings

    settings = Settings()
    for which, dsn in (("target", settings.database_url), ("shadow", settings.shadow_database_url)):
        if not _database_reachable(dsn):
            pytest.skip(
                f"{which} demo database is not reachable at {dsn}\n"
                "start it with:  docker compose up -d --wait"
            )
    if settings.database_url == settings.shadow_database_url:
        pytest.skip("target and shadow URLs are identical; refusing to run an A/B test")
    return settings


@pytest.fixture(scope="session")
def golden_sql() -> str:
    """The demo query, read as utf-8-sig so a BOM from a Windows editor is fine.

    Session-scoped: an integration ``diagnose()`` is expensive and several tests
    want the same result, so they share one run via a module-scoped fixture.
    """
    return GOLDEN_QUERY.read_text(encoding="utf-8-sig")


@pytest.fixture
def settings() -> Any:
    """Settings with environment overrides applied, cache cleared."""
    from schemapulse.config import reload_settings

    return reload_settings()
