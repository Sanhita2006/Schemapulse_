"""The SQL guard is the security boundary of this project.

If :mod:`schemapulse.safety.sql_guard` lets something through, the model has
arbitrary SQL execution.  These tests are therefore written as an allow-list
audit: everything in ``ACCEPTED`` must parse, everything in ``REJECTED`` must
raise, and nothing may raise an unexpected exception type.  When adding a new
statement form, add it to one list deliberately.
"""

from __future__ import annotations

import pytest

from schemapulse.errors import SafetyError, UnsafeQueryError
from schemapulse.safety.sql_guard import (
    identifier_ok,
    parse_index_ddl,
    parse_readonly_query,
    sanitize_index_sql,
)

# --------------------------------------------------------------------------- #
# read-only SELECT gate
# --------------------------------------------------------------------------- #
ACCEPTED_QUERIES = [
    "SELECT 1",
    "SELECT id FROM orders",
    "SELECT * FROM orders WHERE status = 'pending'",
    "select o.id from orders o join customers c on c.id = o.customer_id",
    "SELECT id FROM orders UNION ALL SELECT id FROM archive_orders",
    "SELECT count(*) FROM orders",
    "WITH recent AS (SELECT id FROM orders) SELECT * FROM recent",
    "SELECT id FROM orders ORDER BY created_at DESC LIMIT 50",
    "SELECT id FROM orders;",
    "  \n SELECT  1  \n ",
]

REJECTED_QUERIES = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace only"),
    pytest.param("SELECT 1; DROP TABLE orders", id="two statements"),
    pytest.param("DROP TABLE orders", id="ddl"),
    pytest.param("DELETE FROM orders", id="delete"),
    pytest.param("UPDATE orders SET status = 'x'", id="update"),
    pytest.param("INSERT INTO orders VALUES (1)", id="insert"),
    pytest.param("TRUNCATE orders", id="truncate"),
    pytest.param("CREATE INDEX x ON orders (id)", id="ddl create index"),
    pytest.param("ALTER TABLE orders ADD COLUMN x int", id="alter"),
    pytest.param("GRANT ALL ON orders TO public", id="grant"),
    pytest.param("SELECT * INTO backup FROM orders", id="select into"),
    pytest.param("LOCK TABLE orders", id="lock"),
    pytest.param("SET search_path = public", id="set"),
    pytest.param("SELECT 1; -- comment\n DELETE FROM orders", id="commented statement"),
    pytest.param("WITH d AS (DELETE FROM orders RETURNING id) SELECT * FROM d", id="dml in cte"),
    pytest.param("SELECT pg_sleep(60)", id="side-effecting function"),
    pytest.param("COPY orders TO '/tmp/x'", id="copy"),
    pytest.param("SELECT * FROM orders; SELECT * FROM customers", id="two selects"),
    pytest.param("EXPLAIN SELECT 1", id="explain"),
    pytest.param("SELECT 1 UNION SELECT pg_sleep(60)", id="sneaky union"),
    pytest.param("SELECT lo_import('/etc/passwd')", id="large object import"),
]


@pytest.mark.parametrize("sql", ACCEPTED_QUERIES)
def test_readonly_accepts_safe_selects(sql: str) -> None:
    parsed = parse_readonly_query(sql)
    assert parsed.tables == () or isinstance(parsed.tables, tuple)
    assert parsed.sql.strip().lower().startswith(("select", "with"))


@pytest.mark.parametrize("sql", REJECTED_QUERIES)
def test_readonly_rejects_everything_else(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        parse_readonly_query(sql)


def test_readonly_extracts_both_tables() -> None:
    parsed = parse_readonly_query(
        "SELECT o.id FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'pending'"
    )
    assert set(parsed.tables) == {"orders", "customers"}


def test_readonly_drops_the_schema_qualifier() -> None:
    """``public.orders`` is reported as ``orders``.

    The catalog lookup searches every schema for the bare relation name, which
    finds the table the user meant and avoids hard-coding an assumption that the
    interesting schema is always ``public``.
    """
    parsed = parse_readonly_query("SELECT * FROM public.orders")
    assert parsed.tables == ("orders",)


# --------------------------------------------------------------------------- #
# CREATE INDEX allow-list
# --------------------------------------------------------------------------- #
ACCEPTED_INDEXES = [
    pytest.param("CREATE INDEX ix ON orders (id)", id="plain"),
    pytest.param("CREATE INDEX ix ON orders (status)", id="equality column"),
    pytest.param("CREATE INDEX ix ON orders (created_at DESC)", id="descending"),
    pytest.param("CREATE INDEX ix ON orders (a ASC, b DESC)", id="mixed directions"),
    pytest.param(
        "CREATE INDEX ix ON orders (created_at DESC) WHERE status = 'pending'",
        id="partial",
    ),
    pytest.param("CREATE UNIQUE INDEX ix ON orders (email)", id="unique"),
    pytest.param("CREATE INDEX CONCURRENTLY ix ON orders (id)", id="concurrently"),
    pytest.param("CREATE INDEX IF NOT EXISTS ix ON orders (id)", id="if not exists"),
    pytest.param("CREATE INDEX ix ON public.orders (id)", id="schema qualified"),
    pytest.param("CREATE INDEX ix ON orders (id) INCLUDE (status)", id="include"),
    pytest.param(
        'CREATE INDEX "ix2" ON orders ("status", "created_at" DESC)', id="quoted identifiers"
    ),
    pytest.param(
        "CREATE INDEX ix ON orders (created_at DESC) WHERE status = 'pending' AND amount_cents > 0",
        id="compound predicate",
    ),
]


@pytest.mark.parametrize(
    "sql",
    [
        'CREATE INDEX "Mixed_Case" ON "orders" ("id")',
        'CREATE INDEX "ix2" ON orders ("status", "created_at" DESC)',
    ],
)
def test_index_accepts_quoted_but_legal_identifiers(sql: str) -> None:
    """Quoting is fine; the *contents* must still match our identifier policy.

    A space in an index name is rejected even though PostgreSQL would accept it
    quoted.  The narrow allow-list is deliberate: re-rendering is only auditable
    if the permitted character set is small and obvious.
    """
    parsed = parse_index_ddl(sql)
    assert parsed.columns


def test_index_rejects_quoted_identifier_with_a_space() -> None:
    with pytest.raises(UnsafeQueryError, match="identifier"):
        parse_index_ddl('CREATE INDEX "Odd Name" ON orders (id)')


REJECTED_INDEXES = [
    pytest.param("DROP INDEX ix", id="drop"),
    pytest.param("DROP TABLE orders", id="drop table"),
    pytest.param("DELETE FROM orders", id="delete"),
    pytest.param("UPDATE pg_index SET reltuples = 0", id="update"),
    pytest.param("ALTER TABLE orders DROP COLUMN id", id="alter"),
    pytest.param("TRUNCATE orders", id="truncate"),
    pytest.param("CREATE TABLE t (id int)", id="create table"),
    pytest.param("CREATE INDEX ix ON orders (lower(status))", id="expression key"),
    pytest.param("CREATE INDEX ix ON orders ((a + b))", id="arithmetic key"),
    pytest.param("CREATE INDEX ix ON orders (id) TABLESPACE fast", id="tablespace"),
    pytest.param("CREATE INDEX ix ON orders (id) WITH (fillfactor = 10)", id="storage parameter"),
    pytest.param("CREATE INDEX ix ON orders (id) USING gin", id="non default am"),
    pytest.param("CREATE INDEX ix ON orders (id) WHERE id IN (SELECT id FROM x)", id="subquery"),
    pytest.param("CREATE INDEX ix ON orders (id); DROP TABLE orders", id="two statements"),
    pytest.param("CREATE INDEX ix ON orders (id) INCLUDE (1)", id="non identifier include"),
    pytest.param('CREATE INDEX "ix""; DROP TABLE orders; --" ON orders (id)', id="quote escape"),
    pytest.param(
        "CREATE INDEX ix ON orders (id) WHERE nextval('s') IS NOT NULL",
        id="sequence function in predicate",
    ),
    pytest.param(
        "CREATE INDEX ix ON orders (id) WHERE clock_timestamp() IS NOT NULL",
        id="volatile function in predicate",
    ),
    pytest.param(
        "CREATE INDEX ix ON orders (id) WHERE dblink('x', 'y') IS NOT NULL",
        id="remote connection in predicate",
    ),
    pytest.param("", id="empty"),
]


@pytest.mark.parametrize("sql", ACCEPTED_INDEXES)
def test_index_accepts_allowlisted_ddl(sql: str) -> None:
    parsed = parse_index_ddl(sql)
    assert parsed.table
    assert parsed.columns


@pytest.mark.parametrize("sql", REJECTED_INDEXES)
def test_index_rejects_everything_else(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        parse_index_ddl(sql)


def test_index_roundtrips_columns_direction_and_predicate() -> None:
    parsed = parse_index_ddl(
        "CREATE INDEX ix ON orders (status, created_at DESC) WHERE status = 'pending'"
    )
    assert parsed.table == "orders"
    assert parsed.columns == ("status", "created_at")
    assert parsed.column_order == ("ASC", "DESC")
    assert parsed.predicate == "status = 'pending'"


def test_index_suggested_name_is_identifier_safe() -> None:
    parsed = parse_index_ddl("CREATE INDEX ON orders (status, created_at)")
    name = parsed.suggest_name()
    assert identifier_ok(name)
    assert name.startswith("sp_orders_")


def test_index_suggested_name_marks_partial_indexes() -> None:
    parsed = parse_index_ddl("CREATE INDEX ON orders (created_at DESC) WHERE status = 'pending'")
    assert parsed.suggest_name().endswith("_partial")


def test_sanitizer_rebuilds_sql_from_parsed_values() -> None:
    """The executed SQL must come from the AST, not from the model's text."""
    parsed = parse_index_ddl(
        "CREATE INDEX ix ON orders (status, created_at DESC) WHERE status = 'pending'"
    )
    sql = sanitize_index_sql(parsed, "sp_test")
    assert sql.startswith('CREATE INDEX "sp_test" ON "orders" (')
    # every identifier re-quoted, ASC left implicit (identical semantics)
    assert '"status", "created_at" DESC' in sql
    # the string literal survived the round trip -- dropping it would silently
    # turn `status = 'pending'` into `status = pending` and index nothing
    assert "status = 'pending'" in sql


def test_sanitizer_rejects_unsafe_index_name() -> None:
    parsed = parse_index_ddl("CREATE INDEX ix ON orders (id)")
    with pytest.raises(UnsafeQueryError):
        sanitize_index_sql(parsed, 'bad"; DROP TABLE orders; --')


@pytest.mark.parametrize(
    "name",
    [
        "orders",
        "ix_orders_status",
        "_leading_underscore",
        "Mixed_Case_123",
        "x" * 63,
    ],
)
def test_identifier_ok_accepts_legal_names(name: str) -> None:
    assert identifier_ok(name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "1abc",
        "has space",
        'has"quote',
        "has;semicolon",
        "has-dash",
        "schéma",
        "x" * 64,
    ],
)
def test_identifier_ok_rejects_illegal_names(name: str) -> None:
    assert not identifier_ok(name)


def test_guard_never_leaks_raw_db_errors() -> None:
    """Guard errors must be readable, not raw driver output."""
    with pytest.raises(SafetyError) as excinfo:
        parse_index_ddl("SELECT 1")
    assert isinstance(excinfo.value, UnsafeQueryError)
    assert "SELECT 1" in str(excinfo.value) or "allow-list" in str(excinfo.value)
