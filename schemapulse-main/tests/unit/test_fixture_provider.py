"""The deterministic offline "model".

``fixture.py`` has to behave like a language model for everything downstream to
be testable: it must return the same answer every run, it must claim to be
non-live, and it must be allowed to be wrong only in ways the validator catches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from schemapulse.agent.context import DiagnosticContext, RelationBrief
from schemapulse.agent.providers.fixture import (
    FixtureGemmaProvider,
    _order_by_columns,
    _table_aliases,
    equality_predicates,
    strip_sql_comments,
)
from schemapulse.agent.schemas import ExperimentFeedback
from schemapulse.db.statistics import ColumnStats
from schemapulse.safety.sql_guard import ParsedQuery, parse_readonly_query

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SQL = REPO_ROOT / "demo" / "slow_orders.sql"

ORDERS_SQL = """
SELECT o.id
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.status = 'pending'
ORDER BY o.created_at DESC
LIMIT 50
"""


def stats(**kwargs: Any) -> ColumnStats:
    base: dict[str, Any] = {
        "column": "status",
        "null_frac": 0.0,
        "n_distinct": 5.0,
        "most_common_vals": ("shipped", "pending", "cancelled"),
        "most_common_freqs": (0.79, 0.0198, 0.09),
    }
    base.update(kwargs)
    return ColumnStats(**base)


def context(
    sql: str,
    *,
    column_stats: dict[str, dict[str, ColumnStats]] | None = None,
    orders_indexes: list[dict[str, Any]] | None = None,
) -> DiagnosticContext:
    return DiagnosticContext(
        query=sql,
        database="schemapulse",
        server_version="PostgreSQL 16.3",
        query_shape={"operation": "SELECT", "tables": ["orders", "customers"]},
        relations=[
            RelationBrief(
                name="orders",
                row_estimate=1_000_000,
                size_bytes=104_000_000,
                primary_key=["id"],
                columns=["id", "customer_id", "status", "amount_cents", "created_at"],
                column_types={
                    "id": "bigint",
                    "customer_id": "bigint",
                    "status": "text",
                    "amount_cents": "bigint",
                    "created_at": "timestamp with time zone",
                },
                indexes=orders_indexes
                if orders_indexes is not None
                else [{"name": "orders_pkey", "columns": ["id"]}],
            ),
            RelationBrief(
                name="customers",
                row_estimate=20_000,
                size_bytes=2_000_000,
                primary_key=["id"],
                columns=["id", "email"],
                column_types={"id": "bigint", "email": "text"},
                indexes=[{"name": "customers_pkey", "columns": ["id"]}],
            ),
        ],
        plan_summary={"nodes": []},
        plan_text="Parallel Seq Scan on orders",
        hotspots=[{"code": "filter_waste", "title": "94% discarded"}],
        baseline_execution_ms=147.16,
        baseline_planning_ms=0.9,
        rows_examined=346_576,
        rows_returned=50,
        plan_node_types=["Limit", "Nested Loop", "Parallel Seq Scan"],
        parsed=parse_readonly_query(sql),
        column_stats=column_stats if column_stats is not None else {"orders": {"status": stats()}},
    )


def propose(sql: str = ORDERS_SQL, feedback: list[ExperimentFeedback] | None = None) -> list[Any]:
    response = FixtureGemmaProvider().propose(
        context=context(sql),
        response_schema={},
        feedback=feedback,
    )
    return list(response.candidates.candidates)


def feedback_for(candidate: Any) -> ExperimentFeedback:
    return ExperimentFeedback(
        attempt=1,
        candidate_sql=candidate.candidate_sql,
        verdict="inconclusive",
        baseline_median_ms=147.16,
        candidate_median_ms=1.52,
        speedup=96.0,
        min_speedup=1.2,
        baseline_plan_shape="Limit->NestedLoop->SeqScan",
        candidate_plan_shape="Limit->IndexScan",
    )


# --------------------------------------------------------------------------- #
# feedback rendering must not invent measurements
# --------------------------------------------------------------------------- #
def test_blocked_feedback_reports_no_measurement() -> None:
    """A refused candidate never ran, so the prompt must not print fake timings.

    Rendering ``0.00 ms -> 0.00 ms (speedup 0.00x)`` for a candidate that was
    never benchmarked puts invented numbers in front of the model.
    """
    from schemapulse.agent.prompts import render_prompt

    blocked = ExperimentFeedback(
        attempt=1,
        candidate_sql="CREATE INDEX ix ON orders USING gin (status)",
        verdict="blocked",
        baseline_median_ms=0.0,
        candidate_median_ms=0.0,
        speedup=0.0,
        min_speedup=0.0,
        baseline_plan_shape="not executed (blocked before benchmarking)",
        candidate_plan_shape="not executed (blocked before benchmarking)",
        notes="safety guard rejected it: USING gin access method is not allowed",
    )
    text = render_prompt(context(ORDERS_SQL), feedback=[blocked])
    assert "not measured" in text
    assert "0.00 ms" not in text
    assert "0.00x" not in text
    assert "USING gin" in text, "the model must be shown what it actually wrote"


def test_measured_feedback_still_reports_timings() -> None:
    """The measured path is unchanged: real numbers still reach the prompt."""
    from schemapulse.agent.prompts import render_prompt

    text = render_prompt(context(ORDERS_SQL), feedback=[feedback_for(_Fake())])
    assert "147.16 ms" in text
    assert "96.00x" in text
    assert "required >= 1.20x" in text


class _Fake:
    candidate_sql = "CREATE INDEX ix ON orders (status)"


# --------------------------------------------------------------------------- #
# provider contract
# --------------------------------------------------------------------------- #
def test_provider_declares_itself_offline() -> None:
    """The report is only honest if this flag is False."""
    assert FixtureGemmaProvider().is_live is False


def test_health_does_not_require_credentials() -> None:
    health = FixtureGemmaProvider().health()
    assert health["reachable"] is True
    assert health["provider"] == "fixture"


def test_proposal_is_byte_for_byte_reproducible() -> None:
    first = propose()
    second = propose()
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_response_carries_a_root_cause() -> None:
    response = FixtureGemmaProvider().propose(context=context(ORDERS_SQL), response_schema={})
    assert response.candidates.root_cause
    assert "no language model" in response.candidates.analysis_summary.lower()
    assert response.provider == "fixture"
    assert response.is_live is False
    assert "not by a model" in response.repair_note


# --------------------------------------------------------------------------- #
# the headline recommendation
# --------------------------------------------------------------------------- #
def test_proposes_the_partial_index_for_a_selective_predicate() -> None:
    cands = propose()
    assert cands, "the golden query must yield at least one candidate"
    partials = [c for c in cands if "WHERE" in c.candidate_sql.upper()]
    assert partials, [c.candidate_sql for c in cands]
    sql = partials[0].candidate_sql
    assert '"created_at" DESC' in sql
    assert "status" in sql and "pending" in sql
    assert "orders_pkey" not in sql


def test_uses_the_queried_literal_frequency_not_the_column_most_common() -> None:
    """``most_common_freqs[0]`` is ``shipped`` at 79%, not ``pending`` at ~2%.

    Using the column's most common frequency suppresses the partial index (79%
    is far too common to be worth one) and misses the actual win.
    """
    s = stats()
    assert s.most_common == "shipped"
    assert s.frequency_of("pending") == pytest.approx(0.0198)
    cands = propose()
    partial = next(c for c in cands if "WHERE" in c.candidate_sql.upper())
    # the evidence must quote *pending's* 1.98%, never shipped's 79%
    assert "0.01980" in partial.reasoning_summary
    assert "79" not in partial.reasoning_summary
    assert "2.0%" in partial.hypothesis
    assert partial.confidence >= 0.8


def test_does_not_propose_an_index_on_an_already_indexed_column() -> None:
    ctx = context(
        ORDERS_SQL,
        orders_indexes=[
            {"name": "orders_pkey", "columns": ["id"]},
            {"name": "orders_status_idx", "columns": ["status"]},
        ],
    )
    response = FixtureGemmaProvider().propose(context=ctx, response_schema={})
    assert all('"status"' not in c.candidate_sql for c in response.candidates.candidates), [
        c.candidate_sql for c in response.candidates.candidates
    ]


def test_already_tried_candidates_are_not_repeated() -> None:
    """Feedback is measurement, so a measured candidate must never come back."""
    first = propose()
    assert first
    second = propose(feedback=[feedback_for(first[0])])
    sqls = {c.candidate_sql for c in second}
    assert first[0].candidate_sql not in sqls


def test_every_candidate_is_valid_sql() -> None:
    for c in propose():
        assert c.candidate_sql.strip().upper().startswith("CREATE INDEX")


def test_golden_query_produces_the_documented_recommendation() -> None:
    cands = propose(GOLDEN_SQL.read_text(encoding="utf-8"))
    assert any("sp_orders_status_created_at_partial" in c.candidate_sql for c in cands), [
        c.candidate_sql for c in cands
    ]


def test_golden_query_is_read_only() -> None:
    assert parse_readonly_query(GOLDEN_SQL.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# parsing helpers: regressions behind the wrong first recommendation
# --------------------------------------------------------------------------- #
def test_comments_do_not_become_tables_or_predicates() -> None:
    """Our own demo header contains ``from "column X has no index"``."""
    sql = """
    -- pending orders are slow; the plan shows a Seq Scan
    /* status has no index */
    SELECT o.id FROM orders o WHERE o.status = 'pending'
    """
    cleaned = strip_sql_comments(sql)
    assert "Seq Scan" not in cleaned
    assert "column" not in set(_table_aliases(sql).values())
    assert [p["literal"] for p in equality_predicates(sql)] == ["pending"]


def test_string_literals_survive_comment_stripping() -> None:
    sql = "SELECT * FROM orders WHERE status = 'a--b' -- trailing"
    cleaned = strip_sql_comments(sql)
    assert "'a--b'" in cleaned
    assert "trailing" not in cleaned


def test_block_comment_becomes_whitespace_not_joined_text() -> None:
    out = strip_sql_comments("SELECT a/* x */FROM t")
    assert "x" not in out
    assert "a" in out and "FROM" in out


def test_order_by_alias_is_not_treated_as_a_column() -> None:
    """``ORDER BY o.created_at`` used to yield the alias ``o`` as a column."""
    assert _order_by_columns(ORDERS_SQL) == ("created_at",)
    assert _order_by_columns("SELECT 1 FROM orders o ORDER BY o.id DESC, o.created_at ASC") == (
        "id",
        "created_at",
    )


def test_order_by_direction_survives_into_the_candidate() -> None:
    for c in propose():
        if "created_at" in c.candidate_sql:
            assert '"created_at" DESC' in c.candidate_sql


def test_table_alias_resolution() -> None:
    aliases = _table_aliases(ORDERS_SQL)
    assert aliases["o"] == "orders"
    assert aliases["c"] == "customers"
    assert aliases[""] == "orders"


def test_join_keywords_are_not_captured_as_aliases() -> None:
    """``FROM orders JOIN customers`` must not bind an alias named ``join``."""
    aliases = _table_aliases("SELECT * FROM orders JOIN customers ON 1=1")
    assert "join" not in aliases
    assert "on" not in aliases
    assert aliases[""] == "orders"


def test_predicates_are_attributed_to_the_right_table() -> None:
    p = next(p for p in equality_predicates(ORDERS_SQL) if p["columns"] == ["status"])
    assert p["tables"] == ["orders"]
    assert p["qualifier"] == "o"


def test_frequency_of_unknown_literal_is_an_upper_bound() -> None:
    residual = stats().frequency_of("never-seen")
    assert residual is not None
    assert 0.0 < residual < 1.0


def test_frequency_of_without_mcv_data_returns_none() -> None:
    assert stats(most_common_vals=(), most_common_freqs=()).frequency_of("pending") is None


def test_low_selectivity_constant_detection() -> None:
    assert stats(most_common_freqs=(0.79,)).is_low_selectivity_constant is True
    assert stats(most_common_freqs=(0.02,)).is_low_selectivity_constant is False


def test_high_cardinality_detection_uses_absolute_value() -> None:
    assert stats(n_distinct=-1000.0).is_high_cardinality is True
    assert stats(n_distinct=1000.0).is_high_cardinality is False
    assert stats(n_distinct=-1000.0).n_distinct_abs == 1000.0


# --------------------------------------------------------------------------- #
# context plumbing
# --------------------------------------------------------------------------- #
def test_parsed_query_is_attached_to_the_context() -> None:
    ctx = context(ORDERS_SQL)
    assert isinstance(ctx.parsed, ParsedQuery)
    assert set(ctx.parsed.tables) == {"orders", "customers"}
    assert ctx.parsed.has_where and ctx.parsed.has_order_by and ctx.parsed.has_limit


def test_prompt_dict_carries_the_numbers_the_model_needs() -> None:
    d = context(ORDERS_SQL).to_prompt_dict()
    assert d["baseline"]["rows_examined"] == 346_576
    assert d["baseline"]["execution_time_ms"] == 147.16
    assert {r["name"] for r in d["relations"]} == {"orders", "customers"}
    assert d["deterministic_hotspots"], "the model needs the finding, not just the plan"


def test_relations_are_addressable_by_name() -> None:
    ctx = context(ORDERS_SQL)
    assert ctx.relation("orders") is not None
    assert ctx.relation("ORDERS") is not None
    assert ctx.relation("nope") is None
