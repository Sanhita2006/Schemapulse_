"""``pg_stats`` access and relation statistics.

Statistics are how the model learns *why* the planner chose what it chose.  A
planner that thinks 200 rows match when 20 000 do is a different problem from a
missing index, and conflating the two produces confidently wrong advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemapulse.db.connection import DbConn, execute


@dataclass(frozen=True, slots=True)
class ColumnStats:
    column: str
    null_frac: float
    n_distinct: float
    most_common_vals: tuple[str, ...] = ()
    most_common_freqs: tuple[float, ...] = ()
    histogram_bounds: tuple[str, ...] = ()
    correlation: float | None = None

    @property
    def most_common(self) -> str | None:
        return self.most_common_vals[0] if self.most_common_vals else None

    @property
    def is_low_selectivity_constant(self) -> bool:
        """A column with one dominant value: the textbook partial-index predicate."""
        if not self.most_common_freqs:
            return False
        return self.most_common_freqs[0] >= 0.5

    @property
    def is_high_cardinality(self) -> bool:
        """``n_distinct < 0`` means "fraction of total rows" — a high-cardinality column."""
        return self.n_distinct < 0

    @property
    def n_distinct_abs(self) -> float:
        return abs(self.n_distinct)

    def frequency_of(self, literal: str) -> float | None:
        """Best available estimate of the frequency of one specific value.

        ``most_common_freqs[0]`` is the frequency of the single *most common*
        value, which is usually not the value a query filters on -- for our demo
        column it reports ~79% (``shipped``) when the query actually asks for
        ``pending``, ~2%.  That is the difference between proposing a partial
        index and missing the obvious win, so we look the literal up properly.

        Returns ``None`` when we have no MCV list at all.  When the literal is
        absent from the list it is rarer than every listed value, so we return
        the residual share (an upper bound) instead of guessing.
        """
        if not self.most_common_vals:
            return None
        for value, freq in zip(self.most_common_vals, self.most_common_freqs, strict=False):
            if value.strip("'") == literal or value == literal:
                return freq
        remaining = 1.0 - sum(self.most_common_freqs)
        if remaining <= 0:
            return 0.0
        # pg_stats truncates the MCV list at 100 entries; without the true
        # n_distinct we cannot split the residual any further, so report it
        # conservatively rather than pretending to precision.
        return remaining / max(len(self.most_common_vals) + 1, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "null_fraction": round(self.null_frac, 5),
            "n_distinct": self.n_distinct,
            "most_common_values": list(self.most_common_vals[:8]),
            "most_common_frequencies": [round(f, 5) for f in self.most_common_freqs[:8]],
            "histogram_bounds_sample": list(self.histogram_bounds[:8]),
            "correlation": self.correlation,
        }


_STATS_SQL = """
SELECT
    schemaname,
    tablename,
    attname,
    null_frac,
    n_distinct,
    most_common_vals::text   AS most_common_vals,
    most_common_freqs::text  AS most_common_freqs,
    histogram_bounds::text   AS histogram_bounds,
    correlation
FROM pg_stats
WHERE schemaname = ANY(%s)
  AND (%s::text[] IS NULL OR tablename = ANY(%s))
"""

_SIZE_SQL = """
SELECT
    c.relname                                        AS table_name,
    c.reltuples::bigint                              AS row_estimate,
    c.relpages                                       AS pages,
    pg_relation_size(c.oid)                          AS heap_bytes,
    pg_indexes_size(c.oid)                           AS index_bytes,
    pg_total_relation_size(c.oid)                    AS total_bytes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = ANY(%s)
  AND c.relkind = 'r'
  AND (%s::text[] IS NULL OR c.relname = ANY(%s))
"""


def _as_list(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    inner = text.strip().lstrip("{").rstrip("}")
    if not inner:
        return ()
    out: list[str] = []
    current = ""
    in_quotes = False
    escaped = False
    for ch in inner:
        if escaped:
            current += ch
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if ch == "," and not in_quotes:
            out.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        out.append(current.strip())
    return tuple(out)


def fetch_stats(
    conn: DbConn,
    relations: tuple[str, ...] | None = None,
    *,
    schemas: tuple[str, ...] = ("public",),
) -> dict[str, dict[str, ColumnStats]]:
    """``pg_stats`` keyed by ``table -> column -> ColumnStats``."""
    # An empty array means "no filter" in these catalog queries, so ``None`` has
    # to become ``[]`` rather than reaching the driver as NULL.
    names = list(relations or ())
    rows = execute(conn, _STATS_SQL, (list(schemas), names, names))
    out: dict[str, dict[str, ColumnStats]] = {}
    for r in rows:
        table = str(r["tablename"]).lower()
        corr = r["correlation"]
        out.setdefault(table, {})[str(r["attname"])] = ColumnStats(
            column=str(r["attname"]),
            null_frac=float(r["null_frac"] or 0.0),
            n_distinct=float(r["n_distinct"] or 0.0),
            most_common_vals=_as_list(r["most_common_vals"]),
            most_common_freqs=tuple(float(x) for x in _as_list(r["most_common_freqs"])),
            histogram_bounds=_as_list(r["histogram_bounds"]),
            correlation=float(corr) if corr is not None else None,
        )
    return out


def fetch_sizes(
    conn: DbConn,
    relations: tuple[str, ...] | None = None,
    *,
    schemas: tuple[str, ...] = ("public",),
) -> dict[str, dict[str, int]]:
    names = list(relations or ())
    rows = execute(conn, _SIZE_SQL, (list(schemas), names, names))
    return {
        str(r["table_name"]).lower(): {
            "row_estimate": max(int(r["row_estimate"]), 0),
            "pages": int(r["pages"]),
            "heap_bytes": int(r["heap_bytes"]),
            "index_bytes": int(r["index_bytes"]),
            "total_bytes": int(r["total_bytes"]),
        }
        for r in rows
    }


def refresh_statistics(conn: DbConn, tables: tuple[str, ...] = ()) -> None:
    """Run ``ANALYZE`` on the shadow database after a schema change.

    Without this the planner would still be reasoning from pre-index statistics
    and the candidate would be measured under a stale plan.
    """
    if not tables:
        execute(conn, "ANALYZE")
        return
    from schemapulse.safety.sql_guard import quote_ident

    targets = ", ".join(quote_ident(t) for t in tables)
    execute(conn, f"ANALYZE {targets}")
