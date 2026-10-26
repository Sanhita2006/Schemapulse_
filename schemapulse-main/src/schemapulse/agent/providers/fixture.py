"""Deterministic offline candidate generator — the fixture provider.

Two jobs, and it is important they are the same job:

1. **Demo insurance.** If the network dies or no API key exists, the golden path
   still runs end to end.  The output is *always* labelled ``is_live=False`` and
   printed as ``FIXTURE``, so a fixture result can never be mistaken for a model
   result.

2. **The research control.** This is a small, honest heuristic index advisor:
   "find the dominant equality predicate, look up its selectivity in
   ``pg_stats``, then propose either a partial index (if the predicate is
   low-selectivity) or a composite index with the sort column last".  Comparing
   Gemma's candidates against this baseline is how you answer the question
   *"what did the model actually contribute?"* with data instead of adjectives.

It reads the :class:`DiagnosticContext` directly rather than parsing a prompt,
which is why it needs no model.
"""

from __future__ import annotations

import re
from typing import Any

from schemapulse.agent.context import DiagnosticContext
from schemapulse.agent.providers.base import BaseProvider, GemmaResponse
from schemapulse.agent.providers.parsing import extract_json_blob
from schemapulse.agent.schemas import CandidateSet, ExperimentFeedback, OptimizationCandidate
from schemapulse.db.statistics import ColumnStats
from schemapulse.errors import ModelOutputError

#: A predicate matching more than this share of the table is not worth a partial index.
PARTIAL_INDEX_MAX_FREQUENCY = 0.35

#: Minimum selectivity for a plain composite index to be plausible.
MIN_USEFUL_SELECTIVITY = 0.0005


class FixtureGemmaProvider(BaseProvider):
    """Deterministic, offline, clearly-labelled candidate generator."""

    name = "fixture"

    def __init__(self, *, model_id: str = "schemapulse-heuristic-v1") -> None:
        super().__init__(model_id, timeout_s=1.0, max_retries=0)

    @property
    def is_live(self) -> bool:
        """Always False. This provider never contacts a model."""
        return False

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model_id,
            "reachable": True,
            "detail": "deterministic offline heuristic — no model is contacted",
        }

    # ------------------------------------------------------------------ core
    def propose(
        self,
        *,
        context: DiagnosticContext,
        response_schema: dict[str, Any],
        attempt: int = 1,
        feedback: list[ExperimentFeedback] | None = None,
        max_candidates: int = 3,
    ) -> GemmaResponse:
        if not context.parsed:
            raise ModelOutputError("fixture provider needs a parsed query")

        # Anything already measured is off the table for this round, whatever the
        # verdict.  Re-proposing a failed candidate wastes an attempt and makes
        # the loop look stuck; a real model gets told the same thing in the
        # feedback block.
        already_tried: set[str] = set()
        for fb in feedback or []:
            already_tried.add(_normalise_sql(fb.candidate_sql))

        candidates = self._heuristic_candidates(context)
        fresh = [c for c in candidates if _normalise_sql(c.candidate_sql) not in already_tried]
        if not fresh and feedback:
            raise ModelOutputError(
                "heuristic baseline exhausted its candidate space after feedback; "
                "no further deterministic proposal is available"
            )
        candidates = fresh

        candidate_set = CandidateSet(
            analysis_summary=(
                "Deterministic heuristic baseline (no language model involved). "
                "Read the plan hotspots, took the dominant equality predicate, looked "
                "its frequency up in pg_stats, and combined it with the ORDER BY key."
            ),
            root_cause=_root_cause(context),
            candidates=candidates[:max_candidates],
        )
        return GemmaResponse(
            candidates=candidate_set,
            raw_text=extract_json_blob(candidate_set.model_dump_json()),
            latency_ms=0.0,
            model_id=self.model_id,
            provider=self.name,
            is_live=False,
            finish_reason="heuristic",
            attempts_made=1,
            repair_note="fixture provider: output is generated locally, not by a model",
        )

    # ------------------------------------------------------------- heuristics
    def _heuristic_candidates(self, context: DiagnosticContext) -> list[OptimizationCandidate]:
        parsed = context.parsed
        assert parsed is not None

        out: list[OptimizationCandidate] = []
        for predicate in equality_predicates(parsed.sql):
            for table in predicate["tables"]:
                brief = context.relation(table)
                if brief is None:
                    continue
                column = predicate["columns"][0]
                if column not in brief.column_types:
                    continue
                existing = _existing_leading_index(context, table, (column,))
                if existing is not None:
                    continue

                stats = context.column_stats.get(brief.name, {}).get(column)
                sort_columns = _order_by_columns(parsed.sql)

                candidate = self._build_candidate(
                    context,
                    table=brief.name,
                    column=column,
                    literal=predicate["literal"],
                    sort_columns=sort_columns,
                    stats=stats,
                    table_stats=brief.row_estimate,
                )
                if candidate is not None:
                    out.append(candidate)

                # Fallback: composite (filter_col, sort_col) with no predicate.
                if sort_columns and sort_columns[0] != column:
                    composite = _composite_candidate(context, brief.name, column, sort_columns[0])
                    if composite is not None and not _is_duplicate(out, composite):
                        out.append(composite)
        return out

    def _build_candidate(
        self,
        context: DiagnosticContext,
        *,
        table: str,
        column: str,
        literal: str | None,
        sort_columns: tuple[str, ...],
        stats: ColumnStats | None,
        table_stats: int,
    ) -> OptimizationCandidate | None:
        sort_col = sort_columns[0] if sort_columns else None
        # Frequency of *this* literal, not of the column's most common value.
        frequency = (
            stats.frequency_of(literal)
            if literal is not None and stats is not None
            else (stats.most_common_freqs[0] if stats and stats.most_common_freqs else None)
        )

        if (
            literal is not None
            and frequency is not None
            and frequency <= PARTIAL_INDEX_MAX_FREQUENCY
        ):
            name = f"sp_{table}_{column}_{sort_col or 'x'}_partial"[:63]
            cols = f'"{sort_col}" DESC' if sort_col and sort_col != column else f'"{column}"'
            sql = f'CREATE INDEX "{name}" ON "{table}" ({cols}) WHERE "{column}" = \'{literal}\''
            matched = int((frequency or 0.0) * table_stats)
            hypothesis = (
                f"The filter \"{column}\" = '{literal}' matches only {frequency:.1%} of "
                f"{table_stats:,} rows, but a sequential scan reads the whole table. "
                f"A partial index over just those rows can be scanned in order and stop "
                f"after LIMIT."
            )
            expected = f"Parallel Seq Scan on {table} -> Index Scan using {name}" + (
                " (explicit Sort eliminated)" if sort_col else ""
            )
            evidence = (
                f"plan read {context.rows_examined:,} rows to return "
                f"{context.rows_returned if context.rows_returned is not None else 'a few'}; "
                f"pg_stats reports most_common_freqs[0]={frequency:.5f} for {column}; "
                f"ORDER BY is on {sort_col or 'n/a'}."
            )
            risks = (
                f"Index holds roughly {matched:,} entries; every INSERT/UPDATE touching "
                f'"{column}" must also maintain it. Write cost is small because the '
                f"predicate is selective."
            )
            return OptimizationCandidate(
                candidate_type="index",
                hypothesis=hypothesis,
                candidate_sql=sql,
                expected_plan_change=expected,
                reasoning_summary=evidence,
                risk_notes=risks,
                confidence=0.85 if frequency < 0.05 else 0.6,
            )

        if sort_col and sort_col != column:
            name = f"sp_{table}_{column}_{sort_col}"[:63]
            sql = f'CREATE INDEX "{name}" ON "{table}" ("{column}", "{sort_col}" DESC)'
            hypothesis = (
                f'The query filters on "{column}" and orders by "{sort_col}" with a LIMIT. '
                f"A composite index with the filter column first and the sort column "
                f"second satisfies both."
            )
            return OptimizationCandidate(
                candidate_type="index",
                hypothesis=hypothesis,
                candidate_sql=sql,
                expected_plan_change=f"Seq Scan on {table} -> Index Scan using {name}",
                reasoning_summary=(
                    f"equality on {column} + ORDER BY {sort_col} DESC LIMIT; "
                    f"{context.rows_examined:,} rows examined in the baseline plan"
                ),
                risk_notes="Full-table index: larger, and every write must maintain it.",
                confidence=0.55,
            )
        return None


# --------------------------------------------------------------------------- #
# tiny deterministic query analysis helpers
# --------------------------------------------------------------------------- #

_COL = r'(?:"[^"]+"|\w+)(?:\.(?:"[^"]+"|\w+))?'
_DIR = r"(?:\s+(?:ASC|DESC))?"

_EQUALITY_RE = re.compile(
    rf"(?P<col>{_COL})\s*=\s*'(?P<lit>[^']*)'",
    re.IGNORECASE,
)
#: Column references may be qualified (``o.created_at``), so the pattern has to
#: swallow the qualifier too -- otherwise ``ORDER BY o.created_at DESC`` yields
#: the alias ``o`` as if it were a column name.
_ORDER_RE = re.compile(
    rf"ORDER\s+BY\s+(?P<cols>{_COL}{_DIR}(?:\s*,\s*{_COL}{_DIR})*)",
    re.IGNORECASE,
)

#: Words that can follow a table name but are not aliases.  Without this,
#: ``FROM orders JOIN customers c`` would bind the alias ``o`` to ``JOIN``.
_NOT_AN_ALIAS = frozenset(
    {
        "where",
        "join",
        "inner",
        "left",
        "right",
        "full",
        "cross",
        "lateral",
        "on",
        "using",
        "group",
        "order",
        "limit",
        "offset",
        "having",
        "union",
        "intersect",
        "except",
        "and",
        "or",
        "returning",
        "window",
        "for",
    }
)

_FROM_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:\"?(\w+)\"?\.)?\"?(\w+)\"?(?:\s+(?:AS\s+)?\"?(\w+)\"?)",
    re.IGNORECASE,
)


def strip_sql_comments(sql: str) -> str:
    """Remove ``--`` and ``/* */`` comments while leaving string literals alone.

    The heuristic below is regex based, and prose in a comment can look exactly
    like SQL -- one of our own demo files contains ``from "column X has no
    index"`` in its header, which parsed as a table named ``column``.  Analysing
    comments is never useful, so they go first.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            # Copy a single-quoted literal verbatim, honouring '' escapes.
            out.append(ch)
            i += 1
            while i < n:
                out.append(sql[i])
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        out.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if sql.startswith("--", i):
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def equality_predicates(sql: str) -> list[dict[str, Any]]:
    """Find ``col = 'literal'`` predicates and attribute them to a table.

    A deliberately small parser. Its output is always re-validated downstream by
    the SQL guard, so a mistake here can only produce a *rejected* candidate,
    never an unsafe one.
    """
    aliases = _table_aliases(sql)
    out: list[dict[str, Any]] = []
    body = strip_sql_comments(sql)
    for match in _EQUALITY_RE.finditer(body):
        raw = match.group("col")
        literal = match.group("lit")
        if "." in raw:
            qualifier, column = raw.split(".", 1)
            table = aliases.get(qualifier.strip('"').lower(), qualifier.strip('"').lower())
        else:
            column = raw
            qualifier = ""
            table = aliases.get("", "") or (next(iter(aliases.values())) if aliases else "")
        out.append(
            {
                "columns": [column.strip('"')],
                "literal": literal,
                "qualifier": qualifier.strip('"'),
                "tables": [table] if table else [],
            }
        )
    return out


def _table_aliases(sql: str) -> dict[str, str]:
    """Map ``alias -> table`` (plus ``"" -> first table``) for a simple query."""
    out: dict[str, str] = {}
    first: str = ""
    body = strip_sql_comments(sql)
    for match in _FROM_JOIN_RE.finditer(body):
        table, alias = match.group(2), match.group(3)
        name = table.lower()
        if alias and alias.lower() in _NOT_AN_ALIAS:
            alias = None
        out[(alias or name).lower()] = name
        if not first:
            first = name
    out.setdefault("", first)
    return out


def _order_by_columns(sql: str) -> tuple[str, ...]:
    match = _ORDER_RE.search(strip_sql_comments(sql))
    if not match:
        return ()
    cols: list[str] = []
    for part in match.group("cols").split(","):
        token = part.strip().split()[0] if part.strip() else ""
        bare = token.strip('"').split(".")[-1].strip('"')
        if bare:
            cols.append(bare)
    return tuple(cols)


def _existing_leading_index(
    context: DiagnosticContext, table: str, columns: tuple[str, ...]
) -> Any:
    brief = context.relation(table)
    if brief is None:
        return None
    for index in brief.indexes:
        leading = tuple(index.get("columns", [])[: len(columns)])
        if leading == columns:
            return index
    return None


def _composite_candidate(
    context: DiagnosticContext, table: str, filter_col: str, sort_col: str
) -> OptimizationCandidate | None:
    if _existing_leading_index(context, table, (filter_col, sort_col)):
        return None
    name = f"sp_{table}_{filter_col}_{sort_col}"[:63]
    sql = f'CREATE INDEX "{name}" ON "{table}" ("{filter_col}", "{sort_col}" DESC)'
    return OptimizationCandidate(
        candidate_type="index",
        hypothesis=(
            f"Composite index ({filter_col}, {sort_col} DESC): equality filter first, "
            f"sort key last."
        ),
        candidate_sql=sql,
        expected_plan_change=f"Seq Scan on {table} -> Index Scan using {name}",
        reasoning_summary=f"equality on {filter_col}; ORDER BY {sort_col} DESC LIMIT",
        risk_notes="Full-table index: larger write cost than a partial index.",
        confidence=0.55,
    )


def _is_duplicate(existing: list[OptimizationCandidate], candidate: OptimizationCandidate) -> bool:
    def norm(c: OptimizationCandidate) -> str:
        return _normalise_sql(c.candidate_sql)

    return norm(candidate) in {norm(c) for c in existing}


def _normalise_sql(sql: str) -> str:
    return " ".join(sql.lower().replace('"', "").split()).rstrip(";")


def _root_cause(context: DiagnosticContext) -> str:
    hotspots = context.hotspots
    for h in hotspots:
        if h.get("code") == "seq_scan":
            return (
                f"sequential scan on {h.get('relation')} with no usable index ({h.get('detail')})"
            )
    for h in hotspots:
        if h.get("code") == "row_estimate_error":
            return f"planner misestimation: {h.get('title')}"
    return f"plan reads {context.rows_examined:,} rows for this query shape"
