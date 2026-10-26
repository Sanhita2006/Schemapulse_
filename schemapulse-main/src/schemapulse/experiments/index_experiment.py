"""One shadow experiment: measure whether a single index candidate actually helps.

This is the heart of the product, and the whole design rests on three decisions:

**1. Both arms are measured on the same connection.**
The baseline arm runs on the shadow database with the candidate index *absent*,
then the index is created and the identical query is measured again. Same server,
same cache, same session settings, back to back. Any difference is the index.

**2. The shadow database is mutated, the target database never is.**
The target connection is only ever used for read-only catalog reads and
``EXPLAIN``.  The index this function creates lives on a URL that
:meth:`Settings.assert_production_safe` has already guaranteed is a different
database.

**3. Cleanup is unconditional.**
Whatever happens — success, rejection, exception, timeout — the candidate index
is dropped.  A schema-polluted shadow database would corrupt the next run's
baseline, which is exactly the kind of quiet bug that makes a benchmark lie.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from schemapulse.db.connection import DbConn, execute
from schemapulse.db.statistics import refresh_statistics
from schemapulse.errors import DatabaseError
from schemapulse.experiments.benchmark import BenchmarkResult, benchmark
from schemapulse.plans.comparator import PlanComparison, Verdict, compare
from schemapulse.plans.models import PlanSummary
from schemapulse.safety.sql_guard import IndexCandidate, drop_index_sql

log = logging.getLogger("schemapulse.experiments")


@dataclass(slots=True)
class ExperimentResult:
    """Everything measured for one candidate."""

    candidate: IndexCandidate
    verdict: Verdict
    comparison: PlanComparison | None
    baseline: BenchmarkResult
    candidate_benchmark: BenchmarkResult | None = None
    create_ms: float = 0.0
    drop_ms: float = 0.0
    ddl_error: str | None = None
    index_used: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def speedup(self) -> float | None:
        return self.comparison.speedup if self.comparison else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "verdict": self.verdict.value,
            "speedup": None if self.comparison is None else round(self.comparison.speedup, 4),
            "index_used": self.index_used,
            "create_ms": round(self.create_ms, 2),
            "drop_ms": round(self.drop_ms, 2),
            "ddl_error": self.ddl_error,
            "notes": list(self.notes),
            "baseline_benchmark": self.baseline.to_dict(),
            "candidate_benchmark": self.candidate_benchmark.to_dict()
            if self.candidate_benchmark
            else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


ProgressFn = Callable[[str, dict[str, Any]], None]


def _noop(_stage: str, _info: dict[str, Any]) -> None:  # pragma: no cover
    return None


def index_exists(conn: DbConn, index_name: str) -> bool:
    rows = execute(
        conn,
        "SELECT 1 FROM pg_class WHERE relkind = 'i' AND relname = %s",
        (index_name,),
    )
    return bool(rows)


def drop_index(conn: DbConn, index_name: str) -> float | None:
    """Drop an index if present.

    Returns the elapsed milliseconds when something was actually dropped, or
    ``None`` when there was nothing to remove -- so callers can report a real
    cleanup instead of a misleading "dropped" line on a clean slate.
    Never raises.
    """
    start = time.perf_counter()
    try:
        if not index_exists(conn, index_name):
            return None
        execute(conn, drop_index_sql(index_name))
    except DatabaseError as exc:
        log.warning("could not drop %s: %s", index_name, exc)
        return (time.perf_counter() - start) * 1000.0
    return (time.perf_counter() - start) * 1000.0


def create_index(conn: DbConn, ddl: str, index_name: str) -> float:
    start = time.perf_counter()
    execute(conn, ddl)
    return (time.perf_counter() - start) * 1000.0


def run_index_experiment(
    conn: DbConn,
    *,
    query: str,
    candidate: IndexCandidate,
    execution_sql: str,
    runs: int = 5,
    warmups: int = 1,
    timeout_ms: int = 30_000,
    min_speedup: float = 1.2,
    progress: ProgressFn | None = None,
    cleanup: bool = True,
) -> ExperimentResult:
    """Paired A/B measurement of one index candidate on the shadow connection.

    Sequence
    --------
    1. drop the index (clean slate), warm up, time N runs            -> baseline arm
    2. ``CREATE INDEX``, ``ANALYZE``, warm up, time N runs            -> candidate arm
    3. compare medians, decide the verdict
    4. drop the index again                                            -> always
    """
    emit = progress or _noop
    index_name = candidate.index_name or candidate.suggest_name()
    notes: list[str] = []

    # ``drop_index`` returns None when the index did not exist, which is the
    # normal first-run case.  Accumulate into a float that always has a value so
    # the cleanup path below can add to it unconditionally.
    preexisting_drop_ms = drop_index(conn, index_name)
    if preexisting_drop_ms is not None:
        emit(
            "index_dropped",
            {
                "index": index_name,
                "action": "dropped any previous experiment index",
                "drop_ms": round(preexisting_drop_ms, 1),
            },
        )
    drop_ms: float = preexisting_drop_ms or 0.0

    # ---------------- arm 1: baseline (index absent) --------------------
    emit("baseline_start", {"runs": runs})
    baseline = benchmark(
        conn,
        query,
        runs=runs,
        warmups=warmups,
        timeout_ms=timeout_ms,
        label="baseline",
        on_run=lambda i, total, s: emit(
            "baseline_run", {"run": i, "total": total, "ms": s.execution_ms, "error": s.error}
        ),
    )
    if not baseline.ok_samples:
        return ExperimentResult(
            candidate=candidate,
            verdict=Verdict.FAILED,
            comparison=None,
            baseline=baseline,
            ddl_error="baseline query could not be executed on the shadow database",
        )
    emit(
        "baseline_done",
        {
            "median_ms": round(baseline.median_ms, 3),
            "spread_pct": round(baseline.spread_pct, 1),
            "plan": baseline.plan.plan_shape if baseline.plan else None,
        },
    )

    # ---------------- arm 2: candidate (index present) ------------------
    try:
        create_ms = create_index(conn, execution_sql, index_name)
    except DatabaseError as exc:
        return ExperimentResult(
            candidate=candidate,
            verdict=Verdict.FAILED,
            comparison=None,
            baseline=baseline,
            ddl_error=f"{type(exc).__name__}: {exc}",
            notes=[f"CREATE INDEX failed on the shadow database: {exc}"],
        )
    emit("index_created", {"index": index_name, "create_ms": round(create_ms, 1)})

    try:
        refresh_statistics(conn, (candidate.table,))
        emit("analyzed", {"table": candidate.table})
    except DatabaseError as exc:  # pragma: no cover - ANALYZE rarely fails
        notes.append(f"ANALYZE failed ({exc}); measuring with stale statistics")

    try:
        candidate_bench = benchmark(
            conn,
            query,
            runs=runs,
            warmups=warmups,
            timeout_ms=timeout_ms,
            label="candidate",
            on_run=lambda i, total, s: emit(
                "candidate_run", {"run": i, "total": total, "ms": s.execution_ms, "error": s.error}
            ),
        )
    finally:
        if cleanup:
            # ``drop_index`` returns None when there was nothing to drop, which
            # is the normal case on a first run: the accumulator starts at 0.0,
            # not None, so the addition below stays valid either way.
            cleanup_ms = drop_index(conn, index_name)
            if cleanup_ms is not None:
                drop_ms += cleanup_ms
            emit("index_dropped", {"index": index_name})

    if not candidate_bench.ok_samples:
        return ExperimentResult(
            candidate=candidate,
            verdict=Verdict.FAILED,
            comparison=None,
            baseline=baseline,
            candidate_benchmark=candidate_bench,
            create_ms=create_ms,
            drop_ms=drop_ms,
            ddl_error="candidate query could not be executed after CREATE INDEX",
            notes=notes,
        )

    baseline_plan: PlanSummary = baseline.plan
    candidate_plan: PlanSummary = candidate_bench.plan

    used = next((i for i in candidate_plan.indexes_used if i.lower() == index_name.lower()), None)
    if used is None:
        notes.append(
            "the index was created but the planner did not choose it "
            "(plan unchanged in shape); the speedup, if any, came from noise"
        )

    # ``planner_used_index`` goes into the comparison so the verdict is decided
    # once, in one place.  Deciding it here and rebuilding the comparison to match
    # put two sources of truth in play, and the headline verdict could disagree
    # with the attempt that produced it -- which is exactly what happened: an
    # attempt recorded as REJECTED while the summary banner read VERIFIED.
    comparison = compare(
        baseline_runs=baseline.times_ms,
        baseline_plan=baseline_plan,
        candidate_runs=candidate_bench.times_ms,
        candidate_plan=candidate_plan,
        min_speedup=min_speedup,
        added_index=index_name,
        planner_used_index=used is not None,
        notes=tuple(notes),
    )

    return ExperimentResult(
        candidate=candidate,
        verdict=comparison.verdict,
        comparison=comparison,
        baseline=baseline,
        candidate_benchmark=candidate_bench,
        create_ms=create_ms,
        drop_ms=drop_ms,
        index_used=used,
        notes=notes,
    )
