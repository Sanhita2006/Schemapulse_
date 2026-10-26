"""How a refused or repeated candidate is recorded, and how the report shows it.

An Ollama rehearsal returned three candidates per model attempt, all of which the
safety guard refused.  The run reported six ``BLOCKED`` rows that were *not*
duplicate bookkeeping -- they were six genuinely distinct candidate evaluations.
They read as duplicates because a blocked outcome recorded no candidate at all,
so every row rendered identically with nothing to tell them apart.

These tests pin the behaviour that fixes that: every recorded attempt identifies
which candidate it was, an exact repeat is labelled as such, and the HTML report
shows the SQL the model proposed for a blocked row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from schemapulse.agent.engine import AgentConfig, OptimizationAgent
from schemapulse.agent.providers.base import GemmaResponse
from schemapulse.agent.schemas import CandidateSet, OptimizationCandidate
from schemapulse.db.connection import target_from_dsn
from schemapulse.engine import SchemaPulse
from schemapulse.models import DiagnosisResult
from schemapulse.plans.comparator import Verdict
from schemapulse.safety.candidate_validator import CandidateValidator

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pulse(demo_env: Any) -> SchemaPulse:
    return SchemaPulse(settings=demo_env)


#: Three genuinely different candidates, none of which the guard will accept.
#: ``USING`` sits before the column list, as PostgreSQL requires.
BLOCKED_SQL = (
    "CREATE INDEX sp_b1 ON orders USING gin (status)",
    "CREATE INDEX sp_b2 ON orders USING gist (created_at)",
    "CREATE INDEX sp_b3 ON orders USING brin (customer_id)",
)


class _RefusingProvider:
    """Returns the same three candidates on every attempt."""

    name = "refusing"
    model_id = "refusing-model"

    @property
    def is_live(self) -> bool:
        return True

    def propose(
        self,
        *,
        context: Any,
        response_schema: dict[str, Any],
        attempt: int = 1,
        **kwargs: Any,
    ) -> GemmaResponse:
        return GemmaResponse(
            candidates=CandidateSet(
                analysis_summary="three candidates that will all be refused",
                root_cause="sequential scan over the orders table",
                candidates=[
                    OptimizationCandidate(
                        hypothesis=f"index candidate number {i}",
                        candidate_sql=sql,
                        expected_plan_change="Seq Scan -> Index Scan",
                        reasoning_summary="supported by the measured baseline plan",
                        risk_notes="write amplification on insert",
                        confidence=0.9,
                    )
                    for i, sql in enumerate(BLOCKED_SQL, start=1)
                ],
            ),
            raw_text="refusing",
            latency_ms=1.0,
            model_id=self.model_id,
            provider=self.name,
            is_live=True,
        )


@pytest.fixture(scope="module")
def context(pulse: Any, golden_sql: str) -> Any:
    built, _insight = pulse._build_context(golden_sql)
    return built


@pytest.fixture(scope="module")
def blocked_run(demo_env: Any, context: Any) -> Any:
    agent = OptimizationAgent(
        provider=_RefusingProvider(),
        config=AgentConfig(max_attempts=2, max_candidates_per_attempt=3),
        shadow=target_from_dsn("shadow", demo_env.shadow_database_url, read_only=False),
        validator=CandidateValidator(),
    )
    return agent.optimize(query=context.query, context=context)


@pytest.fixture(scope="module")
def blocked_result(pulse: Any, golden_sql: str, blocked_run: Any) -> Any:
    """The refusing run as a real ``DiagnosisResult``, so the report can render it."""
    _context, insight = pulse._build_context(golden_sql)
    return DiagnosisResult(
        insight=insight,
        candidate_set=blocked_run.candidate_set,
        attempts=blocked_run.attempts,
        provider=blocked_run.provider.name,
        provider_is_live=blocked_run.provider.is_live,
        model_id=blocked_run.provider.model_id,
        min_speedup=1.2,
        elapsed_ms=1.0,
    )


# --------------------------------------------------------------------------- #
# every recorded attempt identifies its candidate
# --------------------------------------------------------------------------- #
def test_every_attempt_records_which_candidate_it_was(blocked_run: Any) -> None:
    """The defect: a blocked outcome used to carry no candidate at all."""
    assert blocked_run.attempts, "the refusing provider should have produced attempts"
    for outcome in blocked_run.attempts:
        assert outcome.candidate is not None, (
            f"attempt {outcome.attempt} has no candidate identity, so its report row "
            "is indistinguishable from every other row"
        )
        assert outcome.candidate.candidate_sql


def test_each_distinct_candidate_gets_its_own_row(blocked_run: Any) -> None:
    """Six rows over two model attempts -- one per candidate evaluation.

    This is the behaviour that looked like duplicate recording: it is not. Each
    candidate the model returned is validated separately and recorded separately.
    """
    assert len(blocked_run.attempts) == len(BLOCKED_SQL) * 2
    assert {a.attempt for a in blocked_run.attempts} == {1, 2}


def test_distinct_candidates_are_distinguishable_in_the_data(blocked_run: Any) -> None:
    """Every row must be identifiable from the recorded data alone."""
    rows = [(a.candidate.candidate_sql, a.reason) for a in blocked_run.attempts]
    for sql in BLOCKED_SQL:
        assert sum(1 for r in rows if r[0] == sql) == 2  # once per model attempt


def test_all_outcomes_are_blocked(blocked_run: Any) -> None:
    assert {a.verdict for a in blocked_run.attempts} == {Verdict.BLOCKED}


def test_the_refusal_reason_names_the_access_method(blocked_run: Any) -> None:
    reasons = " ".join(a.reason for a in blocked_run.attempts)
    for method in ("gin", "gist", "brin"):
        assert f"USING {method} access method is not allowed" in reasons


def test_blocked_attempts_have_no_measured_speedup(blocked_run: Any) -> None:
    """A refused candidate was never benchmarked, so it must not claim a number."""
    for outcome in blocked_run.attempts:
        assert outcome.speedup is None


# --------------------------------------------------------------------------- #
# the model is told what it got wrong, so it does not repeat itself
# --------------------------------------------------------------------------- #
def test_refused_candidates_are_fed_back_to_the_model(blocked_run: Any) -> None:
    """Without this the model re-proposes the same refused SQL every attempt."""
    blocked = [fb for fb in blocked_run.feedback if fb.verdict == Verdict.BLOCKED.value]
    assert blocked, "a refused candidate must produce feedback"
    assert {fb.candidate_sql for fb in blocked} == set(BLOCKED_SQL)


def test_refusal_feedback_states_it_was_never_executed(blocked_run: Any) -> None:
    for fb in blocked_run.feedback:
        if fb.verdict == Verdict.BLOCKED.value:
            assert "not executed" in fb.baseline_plan_shape
            assert fb.notes


# --------------------------------------------------------------------------- #
# an exact repeat is labelled a repeat, not silently dropped or re-measured
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def repeated_run(demo_env: Any, context: Any) -> Any:
    """A provider that proposes the *same valid but unhelpful* candidate twice.

    The candidate has to be valid (so it is not refused) and unhelpful (so the
    first attempt does not verify and end the loop before the repeat happens).
    An index on ``amount_cents`` serves none of the query's filters, so the
    planner ignores it and the run proceeds to a second, identical attempt.
    """

    class _RepeatingProvider(_RefusingProvider):
        name = "repeating"
        model_id = "repeating-model"

        def propose(
            self,
            *,
            context: Any,
            response_schema: dict[str, Any],
            attempt: int = 1,
            **kwargs: Any,
        ) -> GemmaResponse:
            return GemmaResponse(
                candidates=CandidateSet(
                    analysis_summary="the same candidate twice",
                    root_cause="sequential scan over the orders table",
                    candidates=[
                        OptimizationCandidate(
                            hypothesis="an index on the amount column",
                            candidate_sql="CREATE INDEX sp_rep ON orders (amount_cents)",
                            expected_plan_change="Seq Scan -> Index Scan",
                            reasoning_summary="supported by the measured baseline plan",
                            risk_notes="write amplification on insert",
                            confidence=0.9,
                        )
                    ],
                ),
                raw_text="repeating",
                latency_ms=1.0,
                model_id=self.model_id,
                provider=self.name,
                is_live=True,
            )

    agent = OptimizationAgent(
        provider=_RepeatingProvider(),
        config=AgentConfig(max_attempts=2, max_candidates_per_attempt=1),
        shadow=target_from_dsn("shadow", demo_env.shadow_database_url, read_only=False),
        validator=CandidateValidator(),
    )
    return agent.optimize(query=context.query, context=context)


def test_an_exact_repeat_is_recorded_and_labelled(repeated_run: Any) -> None:
    """The repeat is kept in the record and named as a repeat."""
    reasons = [a.reason for a in repeated_run.attempts]
    assert any("duplicate of an already-tested candidate" in r for r in reasons), reasons


def test_an_exact_repeat_is_measured_only_once(repeated_run: Any) -> None:
    measured = [a for a in repeated_run.attempts if a.verdict is not Verdict.BLOCKED]
    assert len(measured) == 1, "the same index must not be benchmarked twice"


def test_a_repeat_row_still_identifies_its_candidate(repeated_run: Any) -> None:
    duplicate = next(a for a in repeated_run.attempts if "duplicate" in a.reason)
    assert duplicate.candidate is not None
    assert "sp_rep" in duplicate.candidate.candidate_sql


# --------------------------------------------------------------------------- #
# the report shows the SQL behind a blocked row
# --------------------------------------------------------------------------- #
def test_report_shows_the_sql_for_a_blocked_row(blocked_result: Any, tmp_path: Path) -> None:
    from schemapulse.reporting.report import ResultStore

    store = ResultStore(directory=tmp_path)
    store.save(blocked_result)
    html = store.render().read_text(encoding="utf-8")
    for sql in BLOCKED_SQL:
        assert sql in html, "a blocked row must show the SQL the model proposed"


def test_report_marks_blocked_rows_as_never_benchmarked(
    blocked_result: Any, tmp_path: Path
) -> None:
    from schemapulse.reporting.report import ResultStore

    store = ResultStore(directory=tmp_path)
    store.save(blocked_result)
    html = store.render().read_text(encoding="utf-8")
    assert "refused before execution" in html


def test_saved_json_keeps_the_refused_sql(blocked_result: Any, tmp_path: Path) -> None:
    from schemapulse.reporting.report import ResultStore

    store = ResultStore(directory=tmp_path)
    store.save(blocked_result)
    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    recorded = [a["candidate"]["candidate_sql"] for a in payload["attempts"]]
    assert set(BLOCKED_SQL).issubset(set(recorded))


def test_no_blocked_row_claims_a_speedup(blocked_result: Any, tmp_path: Path) -> None:
    from schemapulse.reporting.report import ResultStore

    store = ResultStore(directory=tmp_path)
    store.save(blocked_result)
    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    assert all(a["speedup"] is None for a in payload["attempts"])
