"""Turn a measured experiment into a reviewable recommendation.

The recommendation SchemaPulse emits is *never* something to run automatically.
It is a ``CREATE INDEX CONCURRENTLY`` statement plus the measured evidence and an
explicit list of the operational risks that a single-query benchmark cannot
settle.  Auto-apply is intentionally absent from the MVP: an index that is 40x
faster for one read can still cost write throughput, disk, and lock time that no
single-query measurement observes.
"""

from __future__ import annotations

from schemapulse.experiments.index_experiment import ExperimentResult
from schemapulse.models import OptimizationCandidate, Recommendation
from schemapulse.plans.comparator import Verdict
from schemapulse.safety.sql_guard import IndexCandidate


def build_recommendation(
    *,
    experiment: ExperimentResult,
    candidate: OptimizationCandidate,
    min_speedup: float,
) -> Recommendation | None:
    """Produce a human-reviewable suggestion, or ``None`` if not verified."""
    if experiment.verdict is not Verdict.VERIFIED or experiment.comparison is None:
        return None

    parsed: IndexCandidate = experiment.candidate
    comparison = experiment.comparison

    rationale = (
        f"Measured on an isolated shadow database with identical data: median execution "
        f"time fell from {comparison.baseline_median_ms:.2f} ms to "
        f"{comparison.candidate_median_ms:.2f} ms across "
        f"{len(comparison.baseline_runs_ms)} timed runs per arm — a "
        f"{comparison.speedup:.1f}x improvement "
        f"(acceptance threshold {min_speedup:.2f}x). "
        f"Plan changed from '{comparison.baseline_plan.plan_shape}' to "
        f"'{comparison.candidate_plan.plan_shape}', and rows examined fell from "
        f"{comparison.baseline_plan.total_rows_examined:,} to "
        f"{comparison.candidate_plan.total_rows_examined:,}."
    )

    return Recommendation(
        # Keep the name that was actually measured on the shadow database. Naming
        # it differently here would tell the reader to create one index while the
        # evidence above them describes another.
        production_sql=parsed.to_production_sql(parsed.index_name or None),
        measured_speedup=round(comparison.speedup, 3),
        rationale=rationale,
        warnings=default_warnings(parsed, experiment),
        auto_applied=False,
    )


def default_warnings(parsed: IndexCandidate, experiment: ExperimentResult) -> list[str]:
    """Risks a single-query benchmark structurally cannot measure."""
    warnings = [
        "Measured on a synthetic shadow database with this one query's workload. "
        "Real workloads mix reads and writes; validate on a production-sized copy.",
        "SchemaPulse never applies this automatically. CREATE INDEX takes a lock "
        "that blocks writes on the table while it runs.",
        "CREATE INDEX CONCURRENTLY avoids blocking writes but takes longer, cannot "
        "run inside a transaction, and can leave an INVALID index if it fails.",
    ]
    if parsed.predicate:
        warnings.append(
            f"This is a PARTIAL index (WHERE {parsed.predicate}). It is only used by the "
            f"planner when it can prove the same predicate, so its benefit is specific "
            f"to queries with that predicate — and it silently stops being used if the "
            f"predicate's shape changes."
        )
    if parsed.unique:
        warnings.append(
            "UNIQUE index: creation will FAIL if duplicate rows exist, and it enforces a "
            "constraint, not just a performance change. Treat it as a migration."
        )
    if parsed.include_columns:
        warnings.append(
            "INCLUDE columns are stored in the index leaf pages, so this index is larger "
            "than the key alone."
        )
    if experiment.comparison and experiment.comparison.planning_time_delta_ms > 0:
        warnings.append(
            f"Planning time rose by {experiment.comparison.planning_time_delta_ms:.2f} ms "
            f"with the index present. Negligible for this query, but it applies to every "
            f"parse of the statement."
        )
    return warnings


def rejection_reason(experiment: ExperimentResult) -> str:
    """One honest sentence about why something was not accepted."""
    if experiment.ddl_error:
        return f"DDL failed on the shadow database: {experiment.ddl_error}"
    comparison = experiment.comparison
    if comparison is None:
        return "no measurement was produced"
    if experiment.verdict is Verdict.REJECTED:
        slower = (
            f"median time did not improve: {comparison.baseline_median_ms:.2f} ms -> "
            f"{comparison.candidate_median_ms:.2f} ms ({comparison.speedup:.2f}x)"
        )
        if comparison.speedup > 1.0 and experiment.index_used is None:
            # Distinguished from "median time did not improve": the median did
            # improve.  What did not happen is the planner choosing the index, so
            # the difference is noise rather than a result.  Only worth saying
            # when there was an apparent gain to explain away -- on a slowdown
            # the timing is already the whole story.
            return (
                f"the planner did not select the new index, so the "
                f"{comparison.speedup:.2f}x median difference is not attributable to it "
                f"({comparison.baseline_median_ms:.2f} ms -> "
                f"{comparison.candidate_median_ms:.2f} ms)"
            )
        return slower
    if experiment.verdict is Verdict.INCONCLUSIVE:
        return (
            f"improved but below the acceptance threshold: "
            f"{comparison.baseline_median_ms:.2f} ms -> "
            f"{comparison.candidate_median_ms:.2f} ms ({comparison.speedup:.2f}x, "
            f"required >= {comparison.min_speedup:.2f}x)"
        )
    return experiment.verdict.value


def summary_line(experiment: ExperimentResult) -> str:
    """Single-line verdict for CLI output and logs."""
    comparison = experiment.comparison
    if comparison is None:
        return f"{experiment.verdict.value.upper()} — {experiment.ddl_error or 'no measurement'}"
    return (
        f"{experiment.verdict.value.upper()} — {comparison.baseline_median_ms:.2f} ms -> "
        f"{comparison.candidate_median_ms:.2f} ms ({comparison.speedup:.2f}x)"
    )
