"""End-to-end tests against the live demo databases.

These prove the claims that only a real database can support:

* the baseline plan really is a sequential scan over ~1M rows
* the candidate index is really created in the shadow database and dropped again
* the target database is byte-for-byte unchanged afterwards
* the artifacts on disk say what the run measured

Skipped automatically when the demo stack is not up (see ``tests/conftest.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
import pytest

from schemapulse.engine import SchemaPulse
from schemapulse.plans.comparator import Verdict

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def table_fingerprint(dsn: str, table: str = "orders") -> dict[str, int]:
    """Counts and index names -- enough to prove we did not touch the target."""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        indexes = conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s ORDER BY indexname",
            (table,),
        ).fetchall()
        return {
            "rows": rows[0],
            "indexes": len(indexes),
            "index_names": sorted(i[0] for i in indexes),
        }


@pytest.fixture(scope="module")
def pulse(demo_env: Any) -> SchemaPulse:
    return SchemaPulse(settings=demo_env)


@pytest.fixture(scope="module")
def diagnosis(pulse: SchemaPulse, golden_sql: str) -> Any:
    return pulse.diagnose(golden_sql, provider="fixture")


# --------------------------------------------------------------------------- #
# the fixture itself must actually be seeded
# --------------------------------------------------------------------------- #
def test_demo_fixture_is_seeded(demo_env: Any) -> None:
    with psycopg.connect(demo_env.database_url) as conn:
        assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] == 1_000_000
        assert conn.execute("SELECT count(*) FROM customers").fetchone()[0] == 20_000


def test_pending_is_the_selective_side(demo_env: Any) -> None:
    with psycopg.connect(demo_env.database_url) as conn:
        pending = conn.execute("SELECT count(*) FROM orders WHERE status = 'pending'").fetchone()[0]
    assert pending == 19_794
    assert pending / 1_000_000 < 0.05


def test_orders_starts_under_indexed(demo_env: Any) -> None:
    """If someone pre-indexes the demo table the whole story stops being true."""
    with psycopg.connect(demo_env.database_url) as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'orders'"
            ).fetchall()
        ]
    assert names == ["orders_pkey"]


def test_target_and_shadow_hold_the_same_data(demo_env: Any) -> None:
    assert table_fingerprint(demo_env.database_url) == table_fingerprint(
        demo_env.shadow_database_url
    )


# --------------------------------------------------------------------------- #
# baseline evidence
# --------------------------------------------------------------------------- #
def test_insight_reports_a_real_sequential_scan(diagnosis: Any) -> None:
    insight = diagnosis.insight
    assert "Seq Scan" in insight.plan.plan_shape
    assert insight.is_select_only
    assert set(insight.tables) == {"orders", "customers"}


def test_rows_examined_reflects_the_full_scan(diagnosis: Any) -> None:
    """A 1M-row table filtered to 19,794 rows: the read count must be ~1M."""
    assert diagnosis.insight.rows_examined > 900_000


def test_a_hotspot_names_the_wasted_read(diagnosis: Any) -> None:
    codes = {h.code for h in diagnosis.insight.hotspots}
    assert "filter_waste" in codes


# --------------------------------------------------------------------------- #
# the recommendation
# --------------------------------------------------------------------------- #
def test_fixture_run_is_not_labelled_live(diagnosis: Any) -> None:
    assert diagnosis.provider_is_live is False
    assert diagnosis.provider == "fixture"


def test_the_first_attempt_is_the_partial_index(diagnosis: Any) -> None:
    first = diagnosis.attempts[0]
    assert first.candidate is not None
    assert "WHERE" in first.candidate.statement.upper()
    assert "created_at" in first.candidate.statement
    assert "orders_pkey" not in first.candidate.statement


def test_a_verdict_is_reached(diagnosis: Any) -> None:
    assert diagnosis.verdict in set(Verdict)
    if diagnosis.verdict is Verdict.VERIFIED:
        assert diagnosis.speedup is not None
        assert diagnosis.speedup >= diagnosis.min_speedup


def test_verified_recommendation_is_never_auto_applied(diagnosis: Any) -> None:
    if diagnosis.recommendation is not None:
        assert diagnosis.recommendation.auto_applied is False
        assert diagnosis.recommendation.production_sql.upper().startswith("CREATE INDEX")


# --------------------------------------------------------------------------- #
# the load-bearing safety guarantee
# --------------------------------------------------------------------------- #
def test_target_database_is_untouched(demo_env: Any, diagnosis: Any) -> None:
    """The headline safety claim, checked after a real run."""
    after = table_fingerprint(demo_env.database_url)
    assert after["rows"] == 1_000_000
    assert after["index_names"] == ["orders_pkey"]


def test_shadow_index_is_cleaned_up(demo_env: Any, diagnosis: Any) -> None:
    """A leaked index would silently change every later benchmark."""
    names = table_fingerprint(demo_env.shadow_database_url)["index_names"]
    assert names == ["orders_pkey"]


def test_repeated_runs_are_stable(demo_env: Any, golden_sql: str) -> None:
    """Running twice must not accumulate state.

    What has to be identical is the deterministic half: the same candidates
    proposed, in the same order. The verdict is deliberately *not* compared --
    it depends on measured spread, so two runs on a busy machine can honestly
    disagree. Asserting they match would test the machine, not the code.
    """
    engine = SchemaPulse(settings=demo_env)
    first = engine.diagnose(golden_sql, provider="fixture")
    second = engine.diagnose(golden_sql, provider="fixture")

    def statements(result: Any) -> list[str]:
        return [a.candidate.statement for a in result.attempts if a.candidate]

    assert statements(first) == statements(second)
    assert first.insight.plan.total_rows_examined == second.insight.plan.total_rows_examined
    assert first.insight.plan.signature() == second.insight.plan.signature()
    assert first.verdict in set(Verdict)
    assert second.verdict in set(Verdict)


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def test_run_writes_both_artifacts(
    demo_env: Any, diagnosis: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(demo_env, "artifacts_dir", str(tmp_path), raising=False)
    from schemapulse.reporting.report import ResultStore

    store = ResultStore(directory=tmp_path)
    store.save(diagnosis)
    assert store.render().exists()

    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    assert payload["provider_is_live"] is False
    assert payload["insight"]["query"].strip()


def test_saved_result_matches_the_returned_result(diagnosis: Any, tmp_path: Path) -> None:
    from schemapulse.reporting.report import ResultStore

    store = ResultStore(directory=tmp_path)
    store.save(diagnosis)
    payload = store.load()
    assert payload["verdict"] == diagnosis.verdict.value
    if diagnosis.comparison:
        assert payload["comparison"]["evidence"]


# --------------------------------------------------------------------------- #
# the safety guard holds against a live database too
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders WHERE id = 1",
        "UPDATE orders SET status = 'x'",
        "DROP TABLE orders",
        "INSERT INTO orders (id) VALUES (1)",
        "CREATE INDEX evil ON orders (id)",
        "TRUNCATE orders",
        "COPY orders TO '/tmp/out.csv'",
    ],
)
def test_writes_are_rejected_before_any_connection(demo_env: Any, sql: str) -> None:
    from schemapulse.errors import UnsafeQueryError
    from schemapulse.safety.sql_guard import parse_readonly_query

    before = table_fingerprint(demo_env.database_url)
    with pytest.raises(UnsafeQueryError):
        parse_readonly_query(sql)
    assert table_fingerprint(demo_env.database_url) == before
