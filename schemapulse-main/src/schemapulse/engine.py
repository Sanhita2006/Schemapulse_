"""``SchemaPulse`` — the programmatic API.

Everything the CLI does, this object does. The CLI is a thin rendering layer on
top of exactly these two calls:

    agent = SchemaPulse(database_url=..., shadow_database_url=...)
    insight = agent.inspect(query_sql)      # no model involved
    result = agent.diagnose(query_sql)      # full loop, model involved

The separation is intentional for the hackathon: the library is the product, and
the CLI is a demo surface.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from schemapulse.agent.context import DiagnosticContext, build_context
from schemapulse.agent.engine import AgentConfig, OptimizationAgent, build_query_insight
from schemapulse.agent.providers import resolve_provider
from schemapulse.config import Settings, get_settings
from schemapulse.db.connection import ping, session, target_from_dsn
from schemapulse.db.explain import explain_json, rowcount_probe
from schemapulse.db.inspector import index_definitions, inspect_relations
from schemapulse.db.statistics import fetch_sizes, fetch_stats
from schemapulse.errors import DatabaseUnavailableError, UnsafeQueryError
from schemapulse.models import DiagnosisResult, QueryInsight
from schemapulse.plans.comparator import Verdict
from schemapulse.plans.parser import find_hotspots, parse_plan, summarize_for_prompt
from schemapulse.safety.candidate_validator import CandidateValidator
from schemapulse.safety.sql_guard import parse_readonly_query

log = logging.getLogger("schemapulse")

ProgressFn = Callable[[str, dict[str, Any]], None]


def _noop(_stage: str, _info: dict[str, Any]) -> None:  # pragma: no cover
    return None


class SchemaPulse:
    """Execution-verified PostgreSQL optimization.

    Parameters
    ----------
    database_url:
        The database to inspect. Only ever receives read-only statements:
        catalog reads and ``EXPLAIN (ANALYZE)`` of a validated ``SELECT``.
    shadow_database_url:
        The disposable database where candidate DDL is applied and measured.
        Must differ from ``database_url``; this is enforced, not assumed.
    settings:
        Optional pre-built configuration (tests and the CLI use this).
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        shadow_database_url: str | None = None,
        settings: Settings | None = None,
        **overrides: Any,
    ) -> None:
        base = settings or (get_settings(**overrides) if overrides else get_settings())
        self.settings = base.model_copy(
            update={
                k: v
                for k, v in {
                    "database_url": database_url,
                    "shadow_database_url": shadow_database_url,
                }.items()
                if v is not None
            }
        )
        self.settings.assert_production_safe(
            self.settings.database_url, self.settings.shadow_database_url
        )
        self.target = target_from_dsn("target", self.settings.database_url, read_only=True)
        self.shadow = target_from_dsn("shadow", self.settings.shadow_database_url, read_only=False)
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ info
    def health(self) -> dict[str, Any]:
        return {
            "target": ping(self.target),
            "shadow": ping(self.shadow),
            "settings": self.settings.redacted,
        }

    # --------------------------------------------------------------- inspect
    def inspect(
        self,
        query_sql: str,
        *,
        count_rows: bool = True,
    ) -> QueryInsight:
        """Deterministic baseline analysis. **No model is involved.**

        Returns the real execution plan, the real timings and the deterministic
        hotspot list. This is what ``schemapulse inspect`` prints and what the
        model is given as evidence.
        """
        parsed = parse_readonly_query(query_sql)
        timeout = self.settings.query_timeout_ms

        with session(self.target, timeout_ms=timeout) as conn:
            snapshot = inspect_relations(conn, parsed.tables)
            stats = fetch_stats(conn, parsed.tables)
            sizes = fetch_sizes(conn, parsed.tables)
            plan_doc = explain_json(conn, parsed.sql, analyze=True, timeout_ms=timeout)
            plan = parse_plan(plan_doc, parsed.sql)
            rows_returned = (
                rowcount_probe(conn, parsed.sql, timeout_ms=timeout) if count_rows else None
            )

        hotspots = find_hotspots(plan)
        relation_stats = {
            name: {
                **(sizes.get(name.lower(), {})),
                **{"stats_columns": sorted(stats.get(name.lower(), {}))},
            }
            for name in parsed.tables
        }

        insight = build_query_insight(
            query=parsed.sql,
            database=snapshot.database,
            server_version=snapshot.server_version,
            tables=parsed.tables,
            plan=plan,
            hotspots=hotspots,
            existing_indexes=index_definitions(snapshot, parsed.tables),
            relation_stats=relation_stats,
            rows_returned=rows_returned,
        )
        self.trace.append(
            {
                "event": "inspect",
                "tables": list(parsed.tables),
                "execution_time_ms": round(plan.execution_time_ms, 3),
                "plan_shape": plan.plan_shape,
                "hotspots": [h.code for h in hotspots],
            }
        )
        return insight

    # -------------------------------------------------------------- diagnose
    def diagnose(
        self,
        query_sql: str,
        *,
        provider: str | None = None,
        require_live_model: bool = False,
        allow_fallback: bool = True,
        progress: ProgressFn | None = None,
    ) -> DiagnosisResult:
        """Run the full optimize -> validate -> verify loop.

        ``provider`` overrides ``SCHEMAPULSE_MODEL_PROVIDER`` for this call.
        ``require_live_model=True`` refuses to run on the fixture baseline, which
        is what the live-provider test uses to prove a real model was called.
        """
        emit = progress or _noop
        started = time.perf_counter()

        parsed = parse_readonly_query(query_sql)
        settings = self.settings
        if provider:
            settings = settings.model_copy(update={"model_provider": provider})

        emit("stage", {"stage": "inspect", "detail": "capturing real execution evidence"})
        context, insight = self._build_context(parsed.sql)

        emit(
            "stage",
            {
                "stage": "provider",
                "detail": f"resolving Gemma provider ({settings.model_provider})",
            },
        )
        resolved = resolve_provider(
            settings, allow_fallback=allow_fallback, require_live=require_live_model
        )
        if require_live_model and not resolved.is_live:
            raise DatabaseUnavailableError(
                f"a live model was required but provider {resolved.provider.name!r} is not live"
            )

        config = AgentConfig(
            max_attempts=settings.max_candidate_attempts,
            min_speedup=settings.min_speedup,
            benchmark_runs=settings.benchmark_runs,
            benchmark_warmups=settings.benchmark_warmups,
            query_timeout_ms=settings.query_timeout_ms,
            enable_refinement=settings.enable_refinement,
        )

        agent = OptimizationAgent(
            provider=resolved.provider,
            config=config,
            shadow=self.shadow,
            validator=CandidateValidator(),
        )

        emit(
            "stage",
            {
                "stage": "optimize",
                "detail": f"up to {config.max_attempts} experiment(s) on the shadow database",
            },
        )
        run = agent.optimize(query=parsed.sql, context=context, progress=emit)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        comparison = run.best_experiment.comparison if run.best_experiment else None

        result = DiagnosisResult(
            insight=insight,
            candidate_set=run.candidate_set,
            attempts=run.attempts,
            best_attempt=run.verified.attempt if run.verified else None,
            comparison=comparison,
            verified=run.verified,
            recommendation=run.recommendation,
            provider=resolved.provider.name,
            provider_is_live=resolved.is_live,
            model_id=resolved.provider.model_id,
            min_speedup=config.min_speedup,
            elapsed_ms=elapsed_ms,
            trace=[*self.trace, *run.trace, *resolved.describe()["considered"]],
        )
        emit(
            "stage",
            {
                "stage": "done",
                "verdict": result.verdict.value,
                "speedup": result.speedup,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
        return result

    # ------------------------------------------------------------- internals
    def _build_context(self, query_sql: str) -> tuple[DiagnosticContext, QueryInsight]:
        """One pass over the database producing both the context and the insight.

        Only one ``EXPLAIN ANALYZE`` is issued, so the numbers the user sees and
        the numbers the model reasons about are guaranteed identical.
        """
        parsed = parse_readonly_query(query_sql)
        timeout = self.settings.query_timeout_ms

        with session(self.target, timeout_ms=timeout) as conn:
            snapshot = inspect_relations(conn, parsed.tables)
            stats = fetch_stats(conn, parsed.tables)
            sizes = fetch_sizes(conn, parsed.tables)
            plan_doc = explain_json(conn, parsed.sql, analyze=True, timeout_ms=timeout)
            plan = parse_plan(plan_doc, parsed.sql)
            rows_returned = rowcount_probe(conn, parsed.sql, timeout_ms=timeout)

        hotspots = find_hotspots(plan)
        context = build_context(
            query=parsed.sql,
            parsed=parsed,
            snapshot=snapshot,
            stats=stats,
            plan=plan,
            hotspots=hotspots,
            rows_returned=rows_returned,
            plan_text=summarize_for_prompt(plan),
        )
        relation_stats = {name: sizes.get(name.lower(), {}) for name in parsed.tables}
        insight = build_query_insight(
            query=parsed.sql,
            database=snapshot.database,
            server_version=snapshot.server_version,
            tables=parsed.tables,
            plan=plan,
            hotspots=hotspots,
            existing_indexes=index_definitions(snapshot, parsed.tables),
            relation_stats=relation_stats,
            rows_returned=rows_returned,
        )
        return context, insight


def safe_inspect(query_sql: str, settings: Settings | None = None) -> QueryInsight:
    """Convenience wrapper used by the CLI and docs examples."""
    s = settings or get_settings()
    return SchemaPulse(
        database_url=s.database_url, shadow_database_url=s.shadow_database_url, settings=s
    ).inspect(query_sql)


__all__ = [
    "SchemaPulse",
    "Settings",
    "UnsafeQueryError",
    "Verdict",
    "safe_inspect",
]
