"""PostgreSQL connection helpers.

Deliberately thin.  psycopg 3 in autocommit mode plus an explicit
``statement_timeout`` on every session gives us:

* no accidental transaction wrapping of ``CREATE INDEX CONCURRENTLY``-style DDL,
* a hard upper bound on how long any single statement can run, which is what
  stops a pathological generated query from hanging the demo.

The target ("inspected") connection and the shadow ("experiment") connection are
kept as separate objects so it is structurally impossible for one to be used
where the other is expected.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import psycopg
from psycopg.rows import dict_row

from schemapulse.errors import DatabaseError, DatabaseUnavailableError, QueryTimeoutError

#: Every connection SchemaPulse opens uses ``dict_row``, so this is the one
#: connection type in the codebase.  Spelling it out keeps the row type out of
#: the signatures of every db helper.
DbConn = psycopg.Connection[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class DbTarget:
    """A named PostgreSQL endpoint plus the role SchemaPulse plays against it."""

    name: str
    dsn: str
    read_only: bool = True

    @property
    def safe_dsn(self) -> str:
        """DSN with the password redacted, safe to print or store in a report."""
        parsed = urlparse(self.dsn)
        if not parsed.scheme.startswith("postgres"):
            return "<redacted>"
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        if parsed.username:
            netloc = f"{parsed.username}@{netloc}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"

    def describe(self) -> str:
        return f"{self.name}@{self.safe_dsn}"


def target_from_dsn(name: str, dsn: str, *, read_only: bool = True) -> DbTarget:
    return DbTarget(name=name, dsn=dsn, read_only=read_only)


def connect(target: DbTarget, *, timeout_ms: int = 30_000) -> DbConn:
    """Open a connection with a session-level ``statement_timeout``."""
    try:
        conn = psycopg.connect(target.dsn, autocommit=True, row_factory=dict_row)
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailableError(f"could not connect to {target.describe()}: {exc}") from exc
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
            cur.execute("SET lock_timeout = 5000")
            cur.execute("SET idle_in_transaction_session_timeout = 30000")
            cur.execute("SET application_name = 'schemapulse'")
    except psycopg.Error as exc:  # pragma: no cover - server-side config failure
        conn.close()
        raise DatabaseError(f"could not configure session on {target.describe()}: {exc}") from exc
    return conn


@contextlib.contextmanager
def session(target: DbTarget, *, timeout_ms: int = 30_000) -> Iterator[DbConn]:
    """Context manager that always closes the connection."""
    conn = connect(target, timeout_ms=timeout_ms)
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def execute(
    conn: DbConn,
    sql: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one statement and return dict rows, translating driver errors.

    A ``statement_timeout`` is reported as :class:`QueryTimeoutError` so callers
    can treat "the model generated something pathological" as a rejected
    experiment instead of a crash.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return list(cur.fetchall())
    except psycopg.errors.QueryCanceled as exc:
        raise QueryTimeoutError(
            f"statement exceeded the configured timeout: {exc}".strip()
        ) from exc
    except psycopg.Error as exc:
        raise DatabaseError(f"{type(exc).__name__}: {exc}".strip()) from exc


def ping(target: DbTarget, *, timeout_ms: int = 5_000) -> dict[str, Any]:
    """Cheap health check used by ``schemapulse doctor``."""
    info: dict[str, Any] = {
        "target": target.name,
        "dsn": target.safe_dsn,
        "reachable": False,
        "server_version": None,
    }
    try:
        with session(target, timeout_ms=timeout_ms) as conn:
            rows = execute(conn, "SELECT version() AS v, current_database() AS db")
            if rows:
                raw = str(rows[0]["v"])
                info["reachable"] = True
                info["server_version"] = raw.split(" on ")[0]
                info["database"] = rows[0]["db"]
                stats = execute(
                    conn,
                    "SELECT pg_size_pretty(pg_database_size(current_database())) AS size",
                )
                if stats:
                    info["database_size"] = stats[0]["size"]
    except DatabaseError as exc:
        info["error"] = str(exc)
    return info


def same_server(target_a: DbTarget, target_b: DbTarget) -> bool:
    """True when both endpoints point at the same host:port.

    Used by ``doctor`` to warn if someone accidentally points the shadow at the
    same *instance* as the target (a different database on the same server is a
    much weaker guarantee than a different server).
    """

    def hp(t: DbTarget) -> tuple[str, int | None] | None:
        p = urlparse(t.dsn)
        if not p.hostname:
            return None
        return (p.hostname, p.port)

    return hp(target_a) == hp(target_b)


def url_from_parts(**parts: str) -> str:
    """Small helper used by tests/demo bootstrap to build a DSN."""
    user = parts.get("user", "postgres")
    password = parts.get("password", "postgres")
    host = parts.get("host", "localhost")
    port = parts.get("port", "5432")
    database = parts.get("database", "postgres")
    auth = f"{user}:{unquote(password)}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{database}"
