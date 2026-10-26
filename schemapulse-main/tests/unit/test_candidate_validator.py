"""The grounding gate: a candidate must be real *for this schema*.

Two failures these tests exist to prevent, both of which a language model
produces routinely:

* a column that does not exist (or a table *alias* mistaken for a column --
  the bug that shipped an index on a column named ``o``)
* an index that duplicates one already present, so the run wastes a benchmark
  slot and reports a redundant change as a win

The fake catalog is built from the real ``SchemaSnapshot``/``TableInfo`` types
rather than a stub, so a change to their shape breaks these tests too.
"""

from __future__ import annotations

from typing import Any

import pytest

from schemapulse.agent.context import RelationBrief
from schemapulse.agent.schemas import OptimizationCandidate
from schemapulse.db.inspector import ColumnInfo, IndexInfo, SchemaSnapshot, TableInfo
from schemapulse.errors import UnsafeQueryError
from schemapulse.safety.candidate_validator import CandidateValidator


def table(name: str, columns: tuple[str, ...], indexes: tuple[IndexInfo, ...]) -> TableInfo:
    return TableInfo(
        name=name,
        columns=tuple(
            ColumnInfo(name=c, data_type="text", position=i) for i, c in enumerate(columns)
        ),
        indexes=indexes,
        row_estimate=1_000_000 if name == "orders" else 20_000,
    )


@pytest.fixture
def snapshot() -> SchemaSnapshot:
    return SchemaSnapshot(
        database="schemapulse",
        server_version="PostgreSQL 16.3",
        tables={
            "orders": table(
                "orders",
                ("id", "customer_id", "status", "amount_cents", "created_at"),
                (
                    # Definitions mirror real ``pg_get_indexdef`` output, which
                    # always includes the ``USING method`` clause -- that is what
                    # ``IndexInfo.columns`` parses the key list back out of.
                    IndexInfo(
                        name="orders_pkey",
                        table="orders",
                        definition="CREATE UNIQUE INDEX orders_pkey ON public.orders USING btree (id)",
                        is_unique=True,
                        is_primary=True,
                    ),
                    IndexInfo(
                        name="orders_amount_idx",
                        table="orders",
                        definition="CREATE INDEX orders_amount_idx ON public.orders USING btree (amount_cents)",
                    ),
                ),
            ),
            "customers": table("customers", ("id", "email"), ()),
        },
    )


@pytest.fixture
def context(snapshot: SchemaSnapshot) -> Any:
    """A ``DiagnosticContext`` carrying the same catalog, as a live run would."""

    class _Ctx:
        def relation(self, name: str) -> RelationBrief | None:
            info = snapshot.table(name)
            if info is None:
                return None
            return RelationBrief(
                name=info.name,
                row_estimate=info.row_estimate,
                size_bytes=info.size_bytes,
                columns=[c.name for c in info.columns],
                column_types={c.name: c.data_type for c in info.columns},
                indexes=[
                    {"name": i.name, "columns": [], "definition": i.definition}
                    for i in info.indexes
                ],
            )

    return _Ctx()


def candidate(sql: str, *, confidence: float = 0.99) -> OptimizationCandidate:
    return OptimizationCandidate(
        candidate_type="index",
        hypothesis="a hypothesis long enough to satisfy the schema",
        candidate_sql=sql,
        expected_plan_change="Seq Scan -> Index Scan",
        reasoning_summary="supported by the measured baseline",
        risk_notes="write amplification on insert",
        confidence=confidence,
    )


def check(sql: str, snapshot: SchemaSnapshot | None, **kwargs: Any) -> Any:
    return CandidateValidator(snapshot=snapshot).validate(candidate(sql), **kwargs)


# --------------------------------------------------------------------------- #
# grounding against a live snapshot
# --------------------------------------------------------------------------- #
def test_accepts_a_valid_index(snapshot: SchemaSnapshot) -> None:
    rep = check("CREATE INDEX ix ON orders (status, created_at DESC)", snapshot)
    assert rep.ok, rep.reason_text
    assert rep.index_name == "ix"
    assert 'CREATE INDEX "ix" ON "orders"' in rep.execution_sql


def test_rejects_unknown_table(snapshot: SchemaSnapshot) -> None:
    rep = check("CREATE INDEX ix ON invoices (status)", snapshot)
    assert not rep.ok
    assert "does not exist" in rep.reason_text


def test_rejects_unknown_column(snapshot: SchemaSnapshot) -> None:
    rep = check('CREATE INDEX ix ON orders ("nope")', snapshot)
    assert not rep.ok
    assert "column orders.nope does not exist" in rep.reason_text


def test_rejects_a_table_alias_mistaken_for_a_column(snapshot: SchemaSnapshot) -> None:
    """The exact failure mode where ``o.created_at`` was parsed as column ``o``."""
    rep = check('CREATE INDEX ix ON orders ("status", "o" DESC)', snapshot)
    assert not rep.ok
    assert "column orders.o does not exist" in rep.reason_text


def test_rejects_duplicate_of_primary_key(snapshot: SchemaSnapshot) -> None:
    rep = check("CREATE INDEX ix ON orders (id)", snapshot)
    assert not rep.ok
    assert "redundant" in rep.reason_text


def test_rejects_duplicate_of_an_existing_index(snapshot: SchemaSnapshot) -> None:
    rep = check("CREATE INDEX ix ON orders (amount_cents)", snapshot)
    assert not rep.ok
    assert "already covers leading columns" in rep.reason_text


def test_rejects_reusing_an_existing_index_name(snapshot: SchemaSnapshot) -> None:
    rep = check("CREATE INDEX orders_amount_idx ON orders (status)", snapshot)
    assert not rep.ok
    assert "already exists" in rep.reason_text


def test_partial_index_over_an_equal_key_is_not_a_duplicate(snapshot: SchemaSnapshot) -> None:
    """``WHERE status='pending'`` serves a subset, so it is not redundant."""
    rep = check("CREATE INDEX ix ON orders (created_at DESC) WHERE status = 'pending'", snapshot)
    assert rep.ok, rep.reason_text


def test_model_confidence_does_not_affect_grounding(snapshot: SchemaSnapshot) -> None:
    """A 0.99-confidence hallucination is still a hallucination."""
    sure = CandidateValidator(snapshot=snapshot).validate(
        candidate("CREATE INDEX ix ON orders (nope)", confidence=0.99)
    )
    unsure = CandidateValidator(snapshot=snapshot).validate(
        candidate("CREATE INDEX ix ON orders (nope)", confidence=0.01)
    )
    assert sure.ok is unsure.ok is False
    assert sure.reason_text == unsure.reason_text


def test_key_column_limit(snapshot: SchemaSnapshot) -> None:
    assert check("CREATE INDEX ix ON orders (id, status, amount_cents, created_at)", snapshot).ok


def test_too_many_key_columns_is_refused() -> None:
    """Over the limit there is no catalog to check, so the DDL itself must fail."""
    rep = CandidateValidator().validate(candidate("CREATE INDEX ix ON orders (a, b, c, d, e, f)"))
    assert not rep.ok
    assert "key columns" in rep.reason_text


# --------------------------------------------------------------------------- #
# grounding from the diagnostic context (no snapshot passed)
# --------------------------------------------------------------------------- #
def test_grounds_from_context_when_no_snapshot_is_supplied(context: Any) -> None:
    ok = CandidateValidator().validate(
        candidate("CREATE INDEX ix ON orders (status, created_at DESC)"), context=context
    )
    assert ok.ok, ok.reason_text

    bad = CandidateValidator().validate(
        candidate('CREATE INDEX ix ON orders ("status", "o" DESC)'), context=context
    )
    assert not bad.ok
    assert "column orders.o does not exist" in bad.reason_text


def test_unknown_catalog_does_not_block_the_candidate() -> None:
    """Without a catalog we cannot claim grounding, but we must not fake a failure."""
    rep = CandidateValidator().validate(candidate("CREATE INDEX ix ON orders (whatever)"))
    assert rep.ok, rep.reason_text


def test_a_table_with_no_columns_and_no_indexes_is_not_treated_as_known() -> None:
    """An empty catalog entry means the lookup failed, not that the table is empty."""

    class _Empty:
        def table(self, name: str) -> Any:
            return table(name, (), ())

    rep = CandidateValidator(snapshot=_Empty()).validate(  # type: ignore[arg-type]
        candidate("CREATE INDEX ix ON orders (whatever)")
    )
    assert not rep.ok
    assert "does not exist" in rep.reason_text


# --------------------------------------------------------------------------- #
# interaction with the safety guard
# --------------------------------------------------------------------------- #
def test_safety_guard_runs_before_grounding(snapshot: SchemaSnapshot) -> None:
    rep = check("DROP TABLE orders", snapshot)
    assert not rep.ok
    assert "safety guard rejected it" in rep.reason_text


def test_partial_indexes_can_be_disabled(snapshot: SchemaSnapshot) -> None:
    rep = check(
        "CREATE INDEX ix ON orders (created_at) WHERE status = 'pending'",
        snapshot,
        allow_partial=False,
    )
    assert not rep.ok
    assert "partial indexes are disabled" in rep.reason_text


# --------------------------------------------------------------------------- #
# access methods
# --------------------------------------------------------------------------- #
def test_accepts_the_canonical_statement_without_a_using_clause(snapshot: SchemaSnapshot) -> None:
    rep = check("CREATE INDEX ix ON orders (status, created_at DESC)", snapshot)
    assert rep.ok, rep.reason_text
    assert "USING" not in rep.execution_sql.upper()


def test_accepts_an_explicit_using_btree(snapshot: SchemaSnapshot) -> None:
    """``btree`` is PostgreSQL's default, so naming it changes nothing.

    A model that writes it is not doing anything unsafe, and rejecting it burned
    real attempts during the Ollama rehearsal.
    """
    rep = check("CREATE INDEX ix ON orders USING btree (status, created_at DESC)", snapshot)
    assert rep.ok, rep.reason_text
    # Normalised to the canonical form, so the executed DDL never varies by model.
    assert "USING" not in rep.execution_sql.upper()
    assert '"created_at" DESC' in rep.execution_sql


@pytest.mark.parametrize(
    "method",
    ["gin", "gist", "brin", "hash", "spgist", "GIST", "GIN"],
)
def test_rejects_any_non_btree_access_method(snapshot: SchemaSnapshot, method: str) -> None:
    """Other methods change index semantics in ways nothing downstream reasons about."""
    rep = check(f"CREATE INDEX ix ON orders USING {method} (status)", snapshot)
    assert not rep.ok
    assert "access method is not allowed" in rep.reason_text
    assert "only btree is supported" in rep.reason_text


def test_rejects_using_after_the_column_list(snapshot: SchemaSnapshot) -> None:
    """That ordering is a genuine PostgreSQL syntax error, not a variant we accept.

    PostgreSQL requires ``ON <table> USING <method> (<cols>)``; this is the exact
    shape a model emitted during the Ollama rehearsal.
    """
    rep = check("CREATE INDEX ix ON orders (status) USING btree", snapshot)
    assert not rep.ok
    assert "could not be fully parsed" in rep.reason_text


def test_using_btree_survives_with_predicate_and_include(snapshot: SchemaSnapshot) -> None:
    rep = check(
        "CREATE INDEX ix ON orders USING btree (created_at DESC) "
        "INCLUDE (status) WHERE status = 'pending'",
        snapshot,
    )
    assert rep.ok, rep.reason_text
    assert "INCLUDE" in rep.execution_sql
    assert "WHERE" in rep.execution_sql


def test_execution_sql_is_never_the_model_string(snapshot: SchemaSnapshot) -> None:
    """Whatever the model wrote, the executed DDL is rebuilt from the AST."""
    rep = check("CREATE INDEX ix ON orders (status, created_at DESC)", snapshot)
    assert rep.ok
    assert rep.execution_sql.startswith('CREATE INDEX "ix" ON "orders" (')
    assert '"created_at" DESC' in rep.execution_sql


def test_report_rejects_candidate_that_is_too_short() -> None:
    """Pydantic schema validation happens before we ever touch the database."""
    with pytest.raises(ValueError):
        OptimizationCandidate(
            candidate_type="index",
            hypothesis="x",
            candidate_sql="CREATE INDEX ix ON orders (status)",
            expected_plan_change="y",
            reasoning_summary="z",
            risk_notes="w",
            confidence=0.5,
        )


def test_unsafe_error_type_is_shared_with_the_guard() -> None:
    assert issubclass(UnsafeQueryError, Exception)
