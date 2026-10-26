"""SQL safety guard — the boundary between "text the model produced" and SQL.

Design rules (see ``docs/safety.md``):

1. The **analysed query** must be exactly one read-only ``SELECT``.  This is not
   paranoia: ``EXPLAIN ANALYZE`` *executes* the statement it explains, so a write
   query would be executed for real during diagnosis.
2. The **generated DDL** must be exactly one ``CREATE INDEX`` statement whose
   keys are plain column references on a table that exists in the catalog.
3. Anything we cannot parse is rejected.  Rejection is the default, not the
   exception path.

Every check runs against ``sqlglot``'s PostgreSQL AST, not on regexes over a
string, so comment tricks and statement smuggling do not get through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

import sqlglot
from sqlglot import exp

from schemapulse.errors import UnsafeQueryError

#: Identifiers are normalised by PostgreSQL, so restrict what we accept.
IDENT_RE = re.compile(r"^[a-z_][a-z0-9_$]{0,62}$")

#: Side-effecting statements. Present as a *defence in depth* walk over the whole
#: AST, so a DML node nested inside a CTE or subquery is still caught.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Copy,
    exp.Set,
    exp.Use,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Lock,
    exp.Create,
    exp.Merge,
)

#: ``sqlglot`` gives up on some exotic DDL and falls back to a generic
#: ``Command`` node.  We treat "we could not really parse this" as "no".
_OPAQUE_NODES: tuple[type[exp.Expression], ...] = (exp.Command,)

#: Functions that must never appear anywhere in generated DDL: either volatile
#: (an index over them is meaningless) or privileged (file/network access).
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "pg_sleep",
        "nextval",
        "currval",
        "setval",
        "random",
        "rand",
        "clock_timestamp",
        "timeofday",
        "version",
        "current_setting",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "pg_terminate_backend",
        "pg_cancel_backend",
    }
)

#: Row-level-security / privilege machinery is never allowed in a predicate.
FORBIDDEN_PREDICATE_KEYWORDS: frozenset[str] = frozenset(
    {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "COPY",
        "CALL",
        "DO",
        "EXECUTE",
        "MERGE",
    }
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _strip_comments(sql: str) -> str:
    """Remove ``--`` and ``/* */`` comments so they cannot hide a second statement."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _strip_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def identifier_ok(name: str) -> bool:
    return bool(IDENT_RE.match(name.lower()))


def quote_ident(name: str) -> str:
    """Quote an identifier for safe interpolation into generated DDL."""
    if not identifier_ok(name):
        raise UnsafeQueryError(f"cannot quote unsafe identifier: {name!r}")
    return f'"{name.lower()}"'


def _parse_one(sql: str, label: str) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except Exception as exc:
        raise UnsafeQueryError(f"{label} is not parseable as PostgreSQL SQL: {exc}") from exc
    if not statements:
        raise UnsafeQueryError(f"{label} parsed to nothing")
    if len(statements) > 1:
        kinds = ", ".join(type(s).__name__ for s in statements)
        raise UnsafeQueryError(
            f"{label} must be a single statement; found {len(statements)} ({kinds})"
        )
    # sqlglot types the statement list as ``list[Expr]`` but every node it
    # yields is an ``Expression``; narrow it once here rather than at each caller.
    return cast("exp.Expression", statements[0])


def _reject_unsafe_nodes(root: exp.Expression, label: str, *, allow_create: bool = False) -> None:
    for node in root.walk():
        for bad in _OPAQUE_NODES:
            if isinstance(node, bad):
                raise UnsafeQueryError(
                    f"{label} contains unparseable/opaque SQL ({type(node).__name__}); refusing"
                )
        for bad in FORBIDDEN_NODES:
            if bad is exp.Create and allow_create:
                continue
            if isinstance(node, bad):
                raise UnsafeQueryError(
                    f"{label} contains a forbidden operation ({type(node).__name__})"
                )


def _normalise_function_name(name: str) -> str:
    """Fold a function name to a comparable form.

    sqlglot turns ``pg_sleep`` into an ``Anonymous('pg_sleep')`` but turns
    ``clock_timestamp`` into a dedicated ``exp.ClockTimestamp``, whose class name
    lower-cases to ``clocktimestamp``.  Comparing those against a literal list
    silently misses half the entries, so compare with underscores and spacing
    removed on both sides.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


_FORBIDDEN_FUNCTIONS_NORMALISED = frozenset(
    _normalise_function_name(f) for f in FORBIDDEN_FUNCTIONS
)


def _reject_forbidden_functions(root: exp.Expression) -> None:
    for node in root.walk():
        name: str | None = None
        if isinstance(node, exp.Anonymous):
            name = str(node.this)
        elif isinstance(node, exp.Func):
            # ``sql_name`` returns the source-level name (``clock_timestamp``)
            # rather than the CamelCase class name.
            try:
                name = node.sql_name()
            except Exception:  # pragma: no cover - defensive
                name = type(node).__name__
        if name and _normalise_function_name(name) in _FORBIDDEN_FUNCTIONS_NORMALISED:
            raise UnsafeQueryError(f"SQL references forbidden function {name}()")


# --------------------------------------------------------------------------- #
# 1. the query under analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """A query that passed the guard, plus the tables and columns it touches."""

    sql: str
    tables: tuple[str, ...]
    referenced_columns: dict[str, tuple[str, ...]]
    aliases: tuple[str, ...] = ()
    has_where: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_join: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "tables": list(self.tables),
            "referenced_columns": {k: list(v) for k, v in self.referenced_columns.items()},
            "aliases": list(self.aliases),
            "has_where": self.has_where,
            "has_order_by": self.has_order_by,
            "has_limit": self.has_limit,
            "has_join": self.has_join,
        }


def parse_readonly_query(sql: str) -> ParsedQuery:
    """Validate ``sql`` as a single read-only ``SELECT`` and extract its shape.

    Raises :class:`~schemapulse.errors.UnsafeQueryError` on anything else.
    """
    cleaned = _strip_comments(sql).strip()
    if not cleaned:
        raise UnsafeQueryError("query is empty")

    root = _parse_one(_strip_trailing_semicolon(cleaned), "query")

    if not isinstance(root, exp.Select | exp.Union | exp.Subquery):
        raise UnsafeQueryError(
            f"only SELECT statements may be analysed (SchemaPulse's MVP is SELECT-only); "
            f"got {type(root).__name__}"
        )

    _reject_unsafe_nodes(root, "query")
    _reject_forbidden_functions(root)

    if root.find(exp.Into) is not None:
        raise UnsafeQueryError("SELECT ... INTO is not allowed (it creates a table)")

    # Row-locking clauses change side-effect semantics under EXPLAIN ANALYZE.
    if root.find(exp.Lock) is not None:
        raise UnsafeQueryError("SELECT ... FOR UPDATE/SHARE is not allowed")

    tables: list[str] = []
    for tbl in root.find_all(exp.Table):
        name = tbl.name
        if name and name not in tables:
            tables.append(name)

    columns: dict[str, set[str]] = {}
    for col in root.find_all(exp.Column):
        key = col.table or (tables[0] if tables else "")
        columns.setdefault(key, set()).add(col.name)

    return ParsedQuery(
        sql=cleaned,
        tables=tuple(tables),
        referenced_columns={k: tuple(sorted(v)) for k, v in columns.items()},
        aliases=tuple(t.alias_or_name for t in root.find_all(exp.Table) if t.alias_or_name),
        has_where=root.find(exp.Where) is not None,
        has_order_by=root.find(exp.Order) is not None,
        has_limit=root.find(exp.Limit) is not None,
        has_join=any(root.find_all(exp.Join)),
    )


# --------------------------------------------------------------------------- #
# 2. generated DDL
# --------------------------------------------------------------------------- #

#: The only access method a candidate may name.  It is also the default, so
#: ``sanitize_index_sql`` renders the canonical form without the clause.
ALLOWED_ACCESS_METHOD = "btree"


@dataclass(frozen=True, slots=True)
class IndexCandidate:
    """A structurally valid, allow-listed ``CREATE INDEX`` statement."""

    sql: str
    table: str
    columns: tuple[str, ...]
    index_name: str = ""
    unique: bool = False
    predicate: str | None = None
    concurrently: bool = False
    include_columns: tuple[str, ...] = ()
    column_order: tuple[str, ...] = ()

    def key(self) -> tuple[object, ...]:
        """Identity used to reject duplicates across loop attempts."""
        return (self.table, self.columns, self.unique, (self.predicate or "").strip().lower())

    def identity(self) -> str:
        pred = f" WHERE {self.predicate}" if self.predicate else ""
        kind = "UNIQUE " if self.unique else ""
        cols = ", ".join(
            f"{c} {d}" if d == "DESC" else c
            for c, d in zip(
                self.columns, self.column_order or ("ASC",) * len(self.columns), strict=False
            )
        )
        inc = f" INCLUDE ({', '.join(self.include_columns)})" if self.include_columns else ""
        return f"CREATE {kind}INDEX ON {self.table} ({cols}){inc}{pred}"

    def to_production_sql(self, index_name: str | None = None) -> str:
        """A CONCURRENTLY variant a human can review and run themselves.

        SchemaPulse never executes this.  ``CREATE INDEX CONCURRENTLY`` cannot run
        inside a transaction block and takes a share lock that blocks writes, which
        is exactly why it needs a human deployment decision.
        """
        name = quote_ident(index_name or self.index_name or self.suggest_name())
        cols = ", ".join(
            f"{quote_ident(c)} {d}" if d == "DESC" else quote_ident(c)
            for c, d in zip(
                self.columns, self.column_order or ("ASC",) * len(self.columns), strict=False
            )
        )
        inc = (
            f" INCLUDE ({', '.join(quote_ident(c) for c in self.include_columns)})"
            if self.include_columns
            else ""
        )
        kind = "UNIQUE " if self.unique else ""
        pred = f" WHERE {self.predicate}" if self.predicate else ""
        return f"CREATE {kind}INDEX CONCURRENTLY IF NOT EXISTS {name} ON {quote_ident(self.table)} ({cols}){inc}{pred};"

    def suggest_name(self) -> str:
        base = f"sp_{self.table}_" + "_".join(c[:12] for c in self.columns)
        if self.predicate:
            base += "_partial"
        return base[:63]

    def to_dict(self) -> dict[str, object]:
        return {
            "index_name": self.index_name,
            "table": self.table,
            "columns": list(self.columns),
            "unique": self.unique,
            "predicate": self.predicate,
            "concurrently": self.concurrently,
            "include_columns": list(self.include_columns),
            "identity": self.identity(),
        }


def _columns_from_params(params: exp.IndexParameters) -> list[exp.Expression]:
    cols = params.args.get("columns")
    out: list[exp.Expression] = []
    if isinstance(cols, list):
        out.extend(c for c in cols if c is not None)
    elif cols is not None:
        out.append(cols)
    return out


def _plain_column_name(node: exp.Expression) -> str:
    """Extract a plain column name from a key node, rejecting expressions.

    Index keys must be bare columns.  A key like ``lower(name)`` or
    ``(a + b)`` creates an index the model can no longer reason about, and it is
    a well-known source of confidently wrong recommendations — so we refuse.
    """
    if isinstance(node, exp.Ordered):
        node = node.this
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    raise UnsafeQueryError(
        f"index key must be a plain column reference, got {type(node).__name__} "
        f"({node.sql(dialect='postgres')!r}). Expression indexes are intentionally "
        f"not supported by SchemaPulse."
    )


def _column_direction(node: exp.Expression) -> str:
    """Return ``ASC``/``DESC`` for a key node (ordering matters for index usefulness)."""
    if isinstance(node, exp.Ordered):
        return "DESC" if node.args.get("desc") else "ASC"
    return "ASC"


def parse_index_ddl(sql: str) -> IndexCandidate:
    """Validate generated DDL as a single safe ``CREATE INDEX`` statement.

    ``CONCURRENTLY`` is parsed but flagged: the shadow experiment needs a plain
    ``CREATE INDEX`` it can time inside a transaction.
    """
    cleaned = _strip_comments(sql).strip()
    if not cleaned:
        raise UnsafeQueryError("candidate SQL is empty")
    if ";" in _strip_trailing_semicolon(cleaned):
        raise UnsafeQueryError("candidate SQL must be a single statement with no ';'")

    root = _parse_one(_strip_trailing_semicolon(cleaned), "candidate SQL")

    if isinstance(root, exp.Command):
        raise UnsafeQueryError(
            "candidate SQL could not be fully parsed; refusing to execute unparseable DDL"
        )
    if not isinstance(root, exp.Create):
        raise UnsafeQueryError(
            f"candidate must be CREATE INDEX; got {type(root).__name__} "
            f"(only CREATE INDEX is on the allow-list)"
        )

    kind = (root.kind or "").upper()
    if kind != "INDEX":
        raise UnsafeQueryError(f"only CREATE INDEX is allowed; got CREATE {kind or '?'}")

    if root.args.get("replace"):
        raise UnsafeQueryError("CREATE OR REPLACE INDEX is not allowed")

    index_node = root.this
    if not isinstance(index_node, exp.Index):
        raise UnsafeQueryError(f"unexpected CREATE INDEX shape: {type(index_node).__name__}")

    _reject_unsafe_nodes(root, "candidate SQL", allow_create=True)
    _reject_forbidden_functions(root)

    name_node = index_node.args.get("this")
    index_name = name_node.name if isinstance(name_node, exp.Identifier) else ""
    if index_name and not identifier_ok(index_name):
        raise UnsafeQueryError(f"unsafe index identifier: {index_name!r}")

    table_node = index_node.args.get("table")
    if not isinstance(table_node, exp.Table):
        raise UnsafeQueryError("could not determine which table the candidate index targets")
    table = table_node.name
    if not identifier_ok(table):
        raise UnsafeQueryError(f"unsafe table identifier: {table!r}")

    params = index_node.args.get("params")
    if not isinstance(params, exp.IndexParameters):
        raise UnsafeQueryError("malformed index parameters")

    for prop in params.args.get("with_storage") or []:
        raise UnsafeQueryError(
            f"index storage parameters are not allowed ({prop.sql(dialect='postgres')})"
        )
    if params.args.get("tablespace") is not None:
        raise UnsafeQueryError("index TABLESPACE is not allowed")
    if params.args.get("using") is not None:
        # ``btree`` is PostgreSQL's default access method, so naming it explicitly
        # is semantically identical to omitting the clause -- rejecting it would
        # fail a correct candidate on a technicality.  Every other method changes
        # what the index does (operator classes, ordering assumptions, lossy
        # equality) in ways nothing downstream reasons about, so they stay blocked.
        using = params.args["using"]
        name = using.name if isinstance(using, exp.Identifier) else str(using)
        if name.lower() != ALLOWED_ACCESS_METHOD:
            raise UnsafeQueryError(
                f"USING {name} access method is not allowed; only "
                f"{ALLOWED_ACCESS_METHOD} is supported"
            )
    if params.args.get("partition_by"):
        raise UnsafeQueryError("partitioned indexes are not supported")

    columns = tuple(_plain_column_name(c) for c in _columns_from_params(params))
    column_order = tuple(_column_direction(c) for c in _columns_from_params(params))
    if not columns:
        raise UnsafeQueryError("candidate index has no columns")
    for c in columns:
        if not identifier_ok(c):
            raise UnsafeQueryError(f"unsafe column identifier: {c!r}")
    if len(set(columns)) != len(columns):
        raise UnsafeQueryError(f"duplicate column in index definition: {list(columns)}")

    include: list[str] = []
    for inc in params.args.get("include") or []:
        include.append(_plain_column_name(inc))
    for c in include:
        if not identifier_ok(c):
            raise UnsafeQueryError(f"unsafe INCLUDE identifier: {c!r}")
    if set(include) & set(columns):
        raise UnsafeQueryError("INCLUDE columns must not repeat index key columns")

    predicate: str | None = None
    where = params.args.get("where")
    if where is not None:
        # ``Where.sql()`` re-emits the keyword; the model should only see the
        # predicate itself because SchemaPulse re-attaches "WHERE" when rendering.
        pred_sql = (where.this if isinstance(where, exp.Where) else where).sql(dialect="postgres")
        upper = pred_sql.upper()
        for keyword in FORBIDDEN_PREDICATE_KEYWORDS:
            if re.search(rf"\b{keyword}\b", upper):
                raise UnsafeQueryError(
                    f"partial-index predicate contains forbidden keyword {keyword}"
                )
        for node in where.walk():
            if isinstance(node, exp.Select | exp.Subquery):
                raise UnsafeQueryError("partial-index predicate may not contain a subquery")
            if isinstance(node, exp.Anonymous):
                raise UnsafeQueryError(
                    f"partial-index predicate may not call a function ({node.this})"
                )
        predicate = pred_sql

    return IndexCandidate(
        sql=cleaned,
        index_name=index_name,
        table=table,
        columns=columns,
        unique=bool(root.args.get("unique")),
        predicate=predicate,
        concurrently=bool(root.args.get("concurrently")),
        include_columns=tuple(include),
        column_order=column_order,
    )


def sanitize_index_sql(candidate: IndexCandidate, index_name: str | None = None) -> str:
    """Render the DDL SchemaPulse will actually execute on the shadow database.

    Every identifier is re-quoted from the *parsed* structure.  The model's
    original string is never interpolated into a statement verbatim — only the
    AST-derived values are, which is why comment injection cannot survive here.
    """
    name = quote_ident(index_name or candidate.index_name or candidate.suggest_name())
    cols = ", ".join(
        f"{quote_ident(c)} {d}" if d == "DESC" else quote_ident(c)
        for c, d in zip(
            candidate.columns,
            candidate.column_order or ("ASC",) * len(candidate.columns),
            strict=False,
        )
    )
    inc = (
        f" INCLUDE ({', '.join(quote_ident(c) for c in candidate.include_columns)})"
        if candidate.include_columns
        else ""
    )
    kind = "UNIQUE " if candidate.unique else ""
    pred = f" WHERE {candidate.predicate}" if candidate.predicate else ""
    return f"CREATE {kind}INDEX {name} ON {quote_ident(candidate.table)} ({cols}){inc}{pred}"


def drop_index_sql(index_name: str) -> str:
    return f"DROP INDEX IF EXISTS {quote_ident(index_name)}"
