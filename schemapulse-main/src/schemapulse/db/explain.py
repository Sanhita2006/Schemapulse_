"""Capture real PostgreSQL execution evidence.

The only way SchemaPulse learns what a query actually costs is by making
PostgreSQL run it and report.  Nothing here is estimated or simulated.
"""

from __future__ import annotations

from typing import Any

from schemapulse.db.connection import DbConn, execute
from schemapulse.errors import QueryTimeoutError

#: The one statement that produces every fact the rest of the pipeline needs:
#:   ANALYZE  -> real row counts and real per-node times (not planner estimates)
#:   BUFFERS  -> how much of the table was actually touched
#:   VERBOSE  -> filter/index conditions in full, needed for reasoning
#:   SETTINGS -> non-default planner settings that would skew the measurement
#:   FORMAT JSON -> machine-readable tree
EXPLAIN_PREAMBLE = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, FORMAT JSON)"

#: Same, without ANALYZE. Used to explain a candidate *before* we create it,
#: which shows what the planner *thinks* it will do. Cost estimates only.
EXPLAIN_PLAN_ONLY_PREAMBLE = "EXPLAIN (VERBOSE, SETTINGS, FORMAT JSON)"


def explain_json(
    conn: DbConn,
    sql: str,
    *,
    analyze: bool = True,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    """Run ``EXPLAIN`` and return the decoded plan document.

    ``EXPLAIN ANALYZE`` executes the statement, which is why the caller must
    have already validated that ``sql`` is a single read-only ``SELECT`` (see
    :func:`schemapulse.safety.sql_guard.parse_readonly_query`).
    """
    preamble = EXPLAIN_PREAMBLE if analyze else EXPLAIN_PLAN_ONLY_PREAMBLE
    statement = f"{preamble} {sql}"

    if timeout_ms is not None:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
    rows = execute(conn, statement)
    if not rows:
        raise QueryTimeoutError("EXPLAIN returned no rows")
    value = next(iter(rows[0].values()))
    return value


def plan_only_json(conn: DbConn, sql: str, *, timeout_ms: int | None = None) -> dict[str, Any]:
    """Planner cost estimate without executing. Safe for any SELECT."""
    return explain_json(conn, sql, analyze=False, timeout_ms=timeout_ms)


def execution_time_ms(plan_doc: dict[str, Any]) -> float:
    """Extract ``Execution Time`` from an ANALYZE plan document."""
    value = plan_doc.get("Execution Time")
    return float(value) if value is not None else 0.0


def planning_time_ms(plan_doc: dict[str, Any]) -> float:
    value = plan_doc.get("Planning Time")
    return float(value) if value is not None else 0.0


def rowcount_probe(conn: DbConn, sql: str, *, timeout_ms: int = 30_000) -> int:
    """How many rows the query returns, without materialising them twice.

    Wraps the query in ``SELECT count(*) FROM (<sql>)`` which PostgreSQL can
    satisfy without transferring rows to the client. Used to tell the model how
    many rows a query returns, which matters for cost reasoning.
    """
    from schemapulse.safety.sql_guard import parse_readonly_query

    parse_readonly_query(sql)
    if timeout_ms is not None:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
    rows = execute(conn, f"SELECT count(*) AS n FROM ({sql.rstrip().rstrip(';')}) AS _sp_probe")
    return int(rows[0]["n"]) if rows else 0
