"""The bounded optimization loop.

    for attempt in 1..max_attempts:
        candidates  <- Gemma proposes (with measured feedback if attempt > 1)
        validated   <- deterministic guard + catalog grounding
        experiment  <- shadow-database A/B benchmark
        if verified: stop
        if refinement enabled: feed the measurement back and loop

Two properties are worth defending in a Q&A:

**Why is the loop bounded?**  Because an unbounded LLM search loop over a
real database is an expensive way to burn a laptop, and because a loop that keeps
"trying" until something passes is a loop that will eventually find noise.  Three
attempts is enough for a refinement to react to a measurement and small enough
that a bad run ends quickly.

**Why does Gemma get to refine at all?**  Because the feedback it receives is
*measurement* — median milliseconds, the actual before/after plan shapes, and a
verdict. That is the research question: can a language model use execution
feedback to search an optimization space better than a fixed heuristic? The
answer has to be demonstrated, not asserted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from schemapulse.agent.context import DiagnosticContext
from schemapulse.agent.providers.base import GemmaProvider, GemmaResponse, default_response_schema
from schemapulse.agent.schemas import ExperimentFeedback
from schemapulse.agent.schemas import OptimizationCandidate as RawCandidate
from schemapulse.db.connection import DbTarget
from schemapulse.errors import (
    ModelError,
    QueryTimeoutError,
    SafetyError,
)
from schemapulse.experiments.index_experiment import ExperimentResult, run_index_experiment
from schemapulse.experiments.scoring import build_recommendation, rejection_reason
from schemapulse.models import (
    AttemptOutcome,
    OptimizationCandidate,
    QueryInsight,
    Recommendation,
)
from schemapulse.plans.comparator import Verdict
from schemapulse.safety.candidate_validator import CandidateValidator

log = logging.getLogger("schemapulse.agent")

ProgressFn = Callable[[str, dict[str, Any]], None]


def _noop(_stage: str, _info: dict[str, Any]) -> None:  # pragma: no cover
    return None


@dataclass(slots=True)
class AgentConfig:
    max_attempts: int = 3
    min_speedup: float = 1.2
    benchmark_runs: int = 5
    benchmark_warmups: int = 1
    query_timeout_ms: int = 30_000
    enable_refinement: bool = True
    max_candidates_per_attempt: int = 3


@dataclass(slots=True)
class AgentRun:
    """Internal accumulator for one ``optimize()`` call."""

    provider: GemmaProvider
    config: AgentConfig
    shadow: DbTarget
    attempts: list[AttemptOutcome] = field(default_factory=list)
    feedback: list[ExperimentFeedback] = field(default_factory=list)
    candidate_set: Any = None
    verified: OptimizationCandidate | None = None
    recommendation: Recommendation | None = None
    best_experiment: ExperimentResult | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    validator: CandidateValidator | None = None
    seen_identities: set[tuple[object, ...]] = field(default_factory=set)


class OptimizationAgent:
    """Drives one query from baseline plan to a verified (or rejected) candidate."""

    def __init__(
        self,
        *,
        provider: GemmaProvider,
        config: AgentConfig,
        shadow: DbTarget,
        validator: CandidateValidator | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.shadow = shadow
        self.validator = validator or CandidateValidator()

    # ------------------------------------------------------------------ main
    def optimize(
        self,
        *,
        query: str,
        context: DiagnosticContext,
        progress: ProgressFn | None = None,
    ) -> AgentRun:
        from schemapulse.db.connection import session

        emit = progress or _noop
        run = AgentRun(
            provider=self.provider,
            config=self.config,
            shadow=self.shadow,
            validator=self.validator,
        )
        schema = default_response_schema()
        started = time.perf_counter()

        with session(self.shadow, timeout_ms=self.config.query_timeout_ms) as conn:
            for attempt in range(1, self.config.max_attempts + 1):
                emit(
                    "model_request",
                    {
                        "attempt": attempt,
                        "provider": self.provider.name,
                        "model": self.provider.model_id,
                        "feedback_items": len(run.feedback),
                    },
                )
                try:
                    response = self.provider.propose(
                        context=context,
                        response_schema=schema,
                        attempt=attempt,
                        feedback=list(run.feedback) or None,
                        max_candidates=self.config.max_candidates_per_attempt,
                    )
                except ModelError as exc:
                    run.trace.append(
                        {
                            "event": "model_error",
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}"[:400],
                        }
                    )
                    run.attempts.append(
                        AttemptOutcome(
                            attempt=attempt,
                            verdict=Verdict.FAILED,
                            reason=f"model provider failed: {exc}",
                            provider=self.provider.name,
                        )
                    )
                    if attempt >= self.config.max_attempts:
                        break
                    continue

                run.trace.extend(response.trace)
                run.candidate_set = response.candidates
                emit(
                    "model_response",
                    {
                        "attempt": attempt,
                        "latency_ms": round(response.latency_ms, 1),
                        "candidates": len(response.candidates.candidates),
                        "is_live": response.is_live,
                        "repair_note": response.repair_note,
                        "root_cause": response.candidates.root_cause,
                    },
                )

                if not response.candidates.candidates:
                    run.attempts.append(
                        AttemptOutcome(
                            attempt=attempt,
                            verdict=Verdict.FAILED,
                            reason="model returned no candidates",
                            provider=self.provider.name,
                            model_latency_ms=response.latency_ms,
                        )
                    )
                    break

                stop = False
                for raw in response.candidates.candidates:
                    validated = self.validator.validate(raw, context=context)
                    emit(
                        "candidate_validated",
                        {
                            "attempt": attempt,
                            "ok": validated.ok,
                            "candidate_sql": raw.candidate_sql,
                            "reasons": list(validated.reasons),
                        },
                    )
                    if not validated.ok or validated.statement is None:
                        # The candidate is recorded so each blocked row stays
                        # identifiable in the report, and fed back to the model
                        # so it does not re-propose the same refused SQL.
                        run.attempts.append(
                            AttemptOutcome(
                                attempt=attempt,
                                candidate=_recorded_proposal(attempt, raw),
                                verdict=Verdict.BLOCKED,
                                reason=validated.reason_text,
                                provider=self.provider.name,
                                model_latency_ms=response.latency_ms,
                                safety_rejections=list(validated.reasons),
                            )
                        )
                        run.feedback.append(_blocked_feedback(attempt, raw, validated.reason_text))
                        continue

                    statement = validated.statement
                    if statement.key() in run.seen_identities:
                        run.attempts.append(
                            AttemptOutcome(
                                attempt=attempt,
                                candidate=_recorded_proposal(attempt, raw),
                                verdict=Verdict.BLOCKED,
                                reason="duplicate of an already-tested candidate",
                                provider=self.provider.name,
                            )
                        )
                        continue
                    run.seen_identities.add(statement.key())

                    promoted = OptimizationCandidate(
                        hypothesis=raw.hypothesis,
                        candidate_sql=raw.candidate_sql,
                        statement=validated.execution_sql,
                        index_name=validated.index_name,
                        table=statement.table,
                        columns=list(statement.columns),
                        predicate=statement.predicate,
                        unique=statement.unique,
                        include_columns=list(statement.include_columns),
                        expected_plan_change=raw.expected_plan_change,
                        reasoning_summary=raw.reasoning_summary,
                        risk_notes=raw.risk_notes,
                        model_confidence=raw.confidence,
                        attempt=attempt,
                    )

                    try:
                        experiment = run_index_experiment(
                            conn,
                            query=query,
                            candidate=statement,
                            execution_sql=validated.execution_sql,
                            runs=self.config.benchmark_runs,
                            warmups=self.config.benchmark_warmups,
                            timeout_ms=self.config.query_timeout_ms,
                            min_speedup=self.config.min_speedup,
                            progress=emit,
                        )
                    except QueryTimeoutError as exc:
                        run.attempts.append(
                            AttemptOutcome(
                                attempt=attempt,
                                candidate=promoted,
                                verdict=Verdict.FAILED,
                                reason=f"experiment timed out: {exc}",
                                provider=self.provider.name,
                            )
                        )
                        continue
                    except SafetyError as exc:  # pragma: no cover - defensive
                        run.attempts.append(
                            AttemptOutcome(
                                attempt=attempt,
                                candidate=promoted,
                                verdict=Verdict.BLOCKED,
                                reason=f"safety error during experiment: {exc}",
                                provider=self.provider.name,
                            )
                        )
                        continue

                    outcome = self._to_outcome(attempt, promoted, experiment, response)
                    run.attempts.append(outcome)
                    emit(
                        "experiment_complete",
                        {
                            "attempt": attempt,
                            "verdict": experiment.verdict.value,
                            "speedup": None
                            if experiment.comparison is None
                            else round(experiment.comparison.speedup, 3),
                            "reason": outcome.reason,
                        },
                    )

                    run.feedback.append(_feedback_from(attempt, promoted, experiment))

                    if experiment.verdict is Verdict.VERIFIED:
                        run.verified = promoted
                        run.recommendation = build_recommendation(
                            experiment=experiment,
                            candidate=promoted,
                            min_speedup=self.config.min_speedup,
                        )
                        run.best_experiment = experiment
                        stop = True
                        break

                    # Keep the best-measured experiment even when nothing verified,
                    # so the report can show the closest attempt.
                    if experiment.comparison is not None and (
                        run.best_experiment is None
                        or run.best_experiment.comparison is None
                        or experiment.comparison.speedup > run.best_experiment.comparison.speedup
                    ):
                        run.best_experiment = experiment

                if stop:
                    break
                if not self.config.enable_refinement:
                    break

        run.trace.append(
            {
                "event": "loop_finished",
                "attempts": len(run.attempts),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "verified": run.verified is not None,
            }
        )
        return run

    # ------------------------------------------------------------- helpers
    def _to_outcome(
        self,
        attempt: int,
        candidate: OptimizationCandidate,
        experiment: ExperimentResult,
        response: GemmaResponse,
    ) -> AttemptOutcome:
        comparison = experiment.comparison
        return AttemptOutcome(
            attempt=attempt,
            candidate=candidate,
            verdict=experiment.verdict,
            reason=rejection_reason(experiment),
            speedup=None if comparison is None else round(comparison.speedup, 4),
            baseline_median_ms=comparison.baseline_median_ms if comparison else None,
            candidate_median_ms=comparison.candidate_median_ms if comparison else None,
            index_used=experiment.index_used,
            evidence=list(comparison.evidence_lines) if comparison else list(experiment.notes),
            model_latency_ms=round(response.latency_ms, 1),
            provider=self.provider.name,
        )


def _feedback_from(
    attempt: int, candidate: OptimizationCandidate, experiment: ExperimentResult
) -> ExperimentFeedback:
    comparison = experiment.comparison
    baseline_plan = experiment.baseline.plan
    candidate_plan = experiment.candidate_benchmark.plan if experiment.candidate_benchmark else None
    return ExperimentFeedback(
        attempt=attempt,
        candidate_sql=candidate.statement,
        verdict=experiment.verdict.value,
        baseline_median_ms=round(comparison.baseline_median_ms, 4) if comparison else 0.0,
        candidate_median_ms=round(comparison.candidate_median_ms, 4) if comparison else 0.0,
        speedup=round(comparison.speedup, 4) if comparison else 0.0,
        min_speedup=comparison.min_speedup if comparison else 1.2,
        baseline_plan_shape=baseline_plan.plan_shape if baseline_plan else "n/a",
        candidate_plan_shape=candidate_plan.plan_shape if candidate_plan else "n/a",
        notes=rejection_reason(experiment),
    )


def _recorded_proposal(attempt: int, raw: RawCandidate) -> OptimizationCandidate:
    """The recorded view of a proposal that never became an executable statement.

    Used for candidates the validator refused or recognised as a repeat.  Only the
    text the model actually produced is carried over; ``statement`` stays empty
    because no statement was ever built from it.
    """
    return OptimizationCandidate(
        hypothesis=raw.hypothesis,
        candidate_sql=raw.candidate_sql,
        expected_plan_change=raw.expected_plan_change,
        reasoning_summary=raw.reasoning_summary,
        risk_notes=raw.risk_notes,
        model_confidence=raw.confidence,
        attempt=attempt,
    )


def _blocked_feedback(
    attempt: int,
    candidate: RawCandidate,
    reason: str,
) -> ExperimentFeedback:
    """Tell the model exactly which SQL was refused, and why.

    A blocked candidate is never executed, so there is no measurement to report.
    Leaving it out of the feedback loop left the model with no record of the
    refusal and invited it to re-propose the same statement on the next attempt.
    """
    return ExperimentFeedback(
        attempt=attempt,
        candidate_sql=candidate.candidate_sql,
        verdict=Verdict.BLOCKED.value,
        baseline_median_ms=0.0,
        candidate_median_ms=0.0,
        speedup=0.0,
        min_speedup=0.0,
        baseline_plan_shape="not executed (blocked before benchmarking)",
        candidate_plan_shape="not executed (blocked before benchmarking)",
        notes=reason,
    )


def build_query_insight(
    *,
    query: str,
    database: str,
    server_version: str,
    tables: tuple[str, ...],
    plan: Any,
    hotspots: tuple[Any, ...],
    existing_indexes: list[str],
    relation_stats: dict[str, Any],
    rows_returned: int | None,
) -> QueryInsight:
    return QueryInsight(
        query=query,
        database=database,
        server_version=server_version,
        tables=list(tables),
        plan=plan,
        hotspots=list(hotspots),
        existing_indexes=existing_indexes,
        relation_stats=relation_stats,
        rows_returned=rows_returned,
        captured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
