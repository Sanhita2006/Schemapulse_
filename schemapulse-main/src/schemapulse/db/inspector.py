"""PostgreSQL catalog inspection.

The model does not get the raw catalog.  It gets a *filtered, sized* summary:
only the tables the query touches, only the columns it references or that could
plausibly be indexed, and only the statistics that inform those decisions.
Sending the whole catalog would waste the context window and bury the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemapulse.db.connection import DbConn, execute


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool = True
    position: int = 0
    default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.data_type,
            "nullable": self.is_nullable,
            "position": self.position,
        }


@dataclass(frozen=True, slots=True)
class IndexInfo:
    name: str
    table: str
    definition: str
    is_unique: bool = False
    is_primary: bool = False
    method: str = "btree"
    size_bytes: int = 0

    @property
    def columns(self) -> tuple[str, ...]:
        """Column list parsed back out of ``pg_get_indexdef``."""
        try:
            inner = self.definition.split(" USING ", 1)[1]
            inner = inner[inner.index("(") + 1 : inner.rindex(")")]
            parts: list[str] = []
            depth = 0
            current = ""
            for ch in inner:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append(current)
                    current = ""
                else:
                    current += ch
            parts.append(current)
            return tuple(p.strip().strip('"').split()[0] for p in parts if p.strip())
        except (ValueError, IndexError):
            return ()

    def covers(self, columns: tuple[str, ...]) -> bool:
        """True when this index's leading columns match ``columns`` in order."""
        return self.columns[: len(columns)] == columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "columns": list(self.columns),
            "unique": self.is_unique,
            "primary": self.is_primary,
            "method": self.method,
            "size_bytes": self.size_bytes,
            "definition": self.definition,
        }


@dataclass(frozen=True, slots=True)
class TableInfo:
    name: str
    columns: tuple[ColumnInfo, ...]
    indexes: tuple[IndexInfo, ...]
    row_estimate: int = 0
    size_bytes: int = 0
    primary_key: tuple[str, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> ColumnInfo | None:
        lowered = name.lower()
        for c in self.columns:
            if c.name.lower() == lowered:
                return c
        return None

    def has_column(self, name: str) -> bool:
        return self.column(name) is not None

    def to_dict(self, *, include_definition: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "row_estimate": self.row_estimate,
            "size_bytes": self.size_bytes,
            "primary_key": list(self.primary_key),
            "columns": [c.to_dict() for c in self.columns],
            "indexes": [i.to_dict() for i in self.indexes],
        }
        if include_definition:
            out["index_definitions"] = [i.definition for i in self.indexes]
        return out


@dataclass(frozen=True, slots=True)
class SchemaSnapshot:
    """Everything SchemaPulse knows about the database before the model is called."""

    database: str
    server_version: str
    tables: dict[str, TableInfo] = field(default_factory=dict)
    relations: tuple[str, ...] = ()
    total_size_bytes: int = 0

    def table(self, name: str) -> TableInfo | None:
        lowered = name.lower()
        for key, value in self.tables.items():
            if key.lower() == lowered:
                return value
        return None

    def table_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.tables))

    def to_prompt_dict(self, tables: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in tables:
            t = self.table(name)
            if t is None:
                out[name] = {"error": "relation not found in catalog"}
            else:
                out[name] = t.to_dict()
        return out


_COLUMNS_SQL = """
SELECT
    c.table_name,
    c.column_name,
    c.ordinal_position,
    c.data_type,
    c.is_nullable,
    c.column_default
FROM information_schema.columns c
JOIN information_schema.tables t
  ON t.table_schema = c.table_schema AND t.table_name = c.table_name
WHERE t.table_type = 'BASE TABLE'
  AND t.table_schema = ANY(%s)
ORDER BY c.table_name, c.ordinal_position
"""

_INDEXES_SQL = """
SELECT
    t.relname                                        AS table_name,
    i.relname                                        AS index_name,
    ix.indisunique                                   AS is_unique,
    ix.indisprimary                                  AS is_primary,
    pg_get_indexdef(ix.indexrelid)                   AS definition,
    am.amname                                        AS method,
    pg_relation_size(ix.indexrelid)                  AS size_bytes
FROM pg_index ix
JOIN pg_class i      ON i.oid = ix.indexrelid
JOIN pg_class t      ON t.oid = ix.indrelid
JOIN pg_namespace n  ON n.oid = t.relnamespace
JOIN pg_am am        ON am.oid = i.relam
WHERE n.nspname = ANY(%s)
  AND t.relkind = 'r'
ORDER BY t.relname, i.relname
"""

_TABLES_SQL = """
SELECT
    c.relname                                        AS table_name,
    c.reltuples::bigint                              AS row_estimate,
    pg_total_relation_size(c.oid)                   AS size_bytes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = ANY(%s)
  AND c.relkind = 'r'
ORDER BY c.relname
"""

_PK_SQL = """
SELECT t.relname AS table_name, a.attname AS column_name, k.ord
FROM pg_index ix
JOIN pg_class t ON t.oid = ix.indrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
WHERE ix.indisprimary AND n.nspname = ANY(%s)
ORDER BY t.relname, k.ord
"""


def list_relations(conn: DbConn, schemas: tuple[str, ...] = ("public",)) -> tuple[str, ...]:
    rows = execute(
        conn,
        """
        SELECT c.relname AS table_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = ANY(%s) AND c.relkind IN ('r', 'p', 'v', 'm')
        ORDER BY c.relname
        """,
        (list(schemas),),
    )
    return tuple(r["table_name"] for r in rows)


def inspect_relations(
    conn: DbConn,
    relations: tuple[str, ...] | None = None,
    *,
    schemas: tuple[str, ...] = ("public",),
) -> SchemaSnapshot:
    """Read columns, indexes, PKs and row estimates for the requested relations."""
    db_rows = execute(
        conn, "SELECT current_database() AS db, current_setting('server_version') AS v"
    )
    database = str(db_rows[0]["db"]) if db_rows else "unknown"
    server_version = str(db_rows[0]["v"]) if db_rows else "unknown"

    if relations is None:
        relations = list_relations(conn, schemas)

    wanted = list(relations)
    if not wanted:
        return SchemaSnapshot(
            database=database,
            server_version=server_version,
            tables={},
            relations=(),
            total_size_bytes=0,
        )

    col_rows = execute(conn, _COLUMNS_SQL, (list(schemas),))
    idx_rows = execute(conn, _INDEXES_SQL, (list(schemas),))
    tbl_rows = execute(conn, _TABLES_SQL, (list(schemas),))
    pk_rows = execute(conn, _PK_SQL, (list(schemas),))

    lowered_wanted = {r.lower() for r in wanted}

    columns: dict[str, list[ColumnInfo]] = {}
    for r in col_rows:
        if str(r["table_name"]).lower() not in lowered_wanted:
            continue
        columns.setdefault(str(r["table_name"]).lower(), []).append(
            ColumnInfo(
                name=str(r["column_name"]),
                data_type=str(r["data_type"]),
                is_nullable=str(r["is_nullable"]) == "YES",
                position=int(r["ordinal_position"]),
                default=r["column_default"],
            )
        )

    indexes: dict[str, list[IndexInfo]] = {}
    for r in idx_rows:
        key = str(r["table_name"]).lower()
        if key not in lowered_wanted:
            continue
        indexes.setdefault(key, []).append(
            IndexInfo(
                name=str(r["index_name"]),
                table=str(r["table_name"]),
                definition=str(r["definition"]),
                is_unique=bool(r["is_unique"]),
                is_primary=bool(r["is_primary"]),
                method=str(r["method"]),
                size_bytes=int(r["size_bytes"]),
            )
        )

    sizes: dict[str, tuple[int, int]] = {}
    for r in tbl_rows:
        key = str(r["table_name"]).lower()
        if key in lowered_wanted:
            sizes[key] = (max(int(r["row_estimate"]), 0), max(int(r["size_bytes"]), 0))

    pks: dict[str, list[tuple[int, str]]] = {}
    for r in pk_rows:
        key = str(r["table_name"]).lower()
        if key in lowered_wanted:
            pks.setdefault(key, []).append((int(r["ord"]), str(r["column_name"])))

    tables: dict[str, TableInfo] = {}
    for key in lowered_wanted:
        cols = tuple(sorted(columns.get(key, []), key=lambda c: c.position))
        tables[key] = TableInfo(
            name=key,
            columns=cols,
            indexes=tuple(indexes.get(key, [])),
            row_estimate=sizes.get(key, (0, 0))[0],
            size_bytes=sizes.get(key, (0, 0))[1],
            primary_key=tuple(name for _, name in sorted(pks.get(key, []))),
        )

    return SchemaSnapshot(
        database=database,
        server_version=server_version,
        tables=tables,
        relations=relations,
        total_size_bytes=sum(t.size_bytes for t in tables.values()),
    )


def redundant_index(
    snapshot: SchemaSnapshot, table: str, columns: tuple[str, ...]
) -> IndexInfo | None:
    """An existing index whose leading columns already match ``columns``.

    This is the deterministic guard that stops Gemma from "recommending" an
    index that already exists — a failure mode that is embarrassing and easy to
    demo, so it is worth catching in Python rather than in a prompt.
    """
    info = snapshot.table(table)
    if info is None:
        return None
    for index in info.indexes:
        if index.columns[: len(columns)] == columns:
            return index
    return None


def index_definitions(snapshot: SchemaSnapshot, tables: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for name in tables:
        info = snapshot.table(name)
        if info is None:
            continue
        for index in info.indexes:
            out.append(index.definition)
    return out
