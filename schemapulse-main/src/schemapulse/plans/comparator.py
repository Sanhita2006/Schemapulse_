"""Deterministic comparison of a baseline plan against a candidate plan.

This module decides the verdict.  It never asks the model whether it worked.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from schemapulse.plans.models import PlanSummary


class Verdict(StrEnum):
    """Outcome of a shadow experiment."""

    VERIFIED = "verified"
    """Measured median improvement >= ``SCHEMAPULSE_MIN_SPEEDUP``."""

    REJECTED = "rejected"
    """Measured and the candidate made the query slower or equal."""

    INCONCLUSIVE = "inconclusive"
    """Measured, but the improvement did not clear the acceptance threshold."""

    FAILED = "failed"
    """The candidate could not be executed at all (invalid DDL, DDL error, timeout)."""

    BLOCKED = "blocked"
    """Rejected before execution by the deterministic safety validator."""


#: Human-facing, one-line explanation per verdict.
VERDICT_EXPLANATION: dict[Verdict, str] = {
    Verdict.VERIFIED: "measured median execution time improved beyond the required threshold",
    Verdict.REJECTED: "measured median execution time did not improve",
    Verdict.INCONCLUSIVE: "measured improvement was positive but below the required threshold",
    Verdict.FAILED: "candidate could not be executed against the shadow database",
    Verdict.BLOCKED: "candidate was refused by the deterministic safety validator",
}


@dataclass(frozen=True, slots=True)
class PlanComparison:
    """Before/after evidence for one candidate."""

    baseline_median_ms: float
    candidate_median_ms: float
    baseline_runs_ms: tuple[float, ...]
    candidate_runs_ms: tuple[float, ...]
    baseline_plan: PlanSummary
    candidate_plan: PlanSummary
    min_speedup: float
    added_index: str | None = None
    #: Whether the planner actually chose the candidate index, when that is
    #: known.  ``None`` means the caller did not check (or there was no candidate
    #: index), and the verdict then rests on timing alone.
    #:
    #: ``False`` is decisive and must not be reported as a win: a median can
    #: improve while the plan is byte-for-byte identical, which means the
    #: improvement was noise rather than the index.  Keeping this on the
    #: comparison -- rather than re-deciding it in the caller -- is what keeps
    #: ``PlanComparison.verdict`` and the headline verdict from disagreeing.
    planner_used_index: bool | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ---------------------------------------------------------------- metrics
    @property
    def speedup(self) -> float:
        if self.candidate_median_ms <= 0:
            return float("inf") if self.baseline_median_ms > 0 else 1.0
        return self.baseline_median_ms / self.candidate_median_ms

    @property
    def candidate_spread_pct(self) -> float:
        return spread_pct(list(self.candidate_runs_ms))

    @property
    def verdict(self) -> Verdict:
        if self.planner_used_index is False:
            # The index exists and the query may well have got faster, but the
            # planner never chose it.  Whatever changed was not this index.
            return Verdict.REJECTED
        speedup = self.speedup
        if speedup <= 1.0:
            return Verdict.REJECTED
        if speedup < self.min_speedup:
            return Verdict.INCONCLUSIVE
        if self.candidate_spread_pct > MAX_TRUSTWORTHY_SPREAD_PCT:
            # The number clears the bar but the measurements do not.  Refusing
            # to certify here is the whole point of the tool.
            return Verdict.INCONCLUSIVE
        return Verdict.VERIFIED

    @property
    def median_delta_ms(self) -> float:
        return self.candidate_median_ms - self.baseline_median_ms

    @property
    def improvement_pct(self) -> float:
        if self.baseline_median_ms <= 0:
            return 0.0
        return (self.speedup - 1.0) * 100.0

    @property
    def planning_time_delta_ms(self) -> float:
        return self.candidate_plan.planning_time_ms - self.baseline_plan.planning_time_ms

    @property
    def plan_shape_changed(self) -> bool:
        return self.baseline_plan.signature() != self.candidate_plan.signature()

    @property
    def seq_scan_removed(self) -> bool:
        return self.baseline_plan.has_seq_scan() and not self.candidate_plan.has_seq_scan()

    @property
    def index_added(self) -> str | None:
        if self.added_index and self.planner_used_index is False:
            # The planner ignored it, so it is not an index the plan added.
            return None
        new = [
            i for i in self.candidate_plan.indexes_used if i not in self.baseline_plan.indexes_used
        ]
        return new[0] if new else self.added_index

    @property
    def buffers_delta(self) -> int:
        return self.candidate_plan.total_buffers - self.baseline_plan.total_buffers

    @property
    def rows_examined_delta(self) -> int:
        return self.candidate_plan.total_rows_examined - self.baseline_plan.total_rows_examined

    @property
    def sort_removed(self) -> bool:
        return self.baseline_plan.has_sort() and not self.candidate_plan.has_sort()

    @property
    def evidence_lines(self) -> list[str]:
        """Compact, judge-readable proof list. Every number here is measured."""
        lines = [
            f"median execution time: {self.baseline_median_ms:.2f} ms -> "
            f"{self.candidate_median_ms:.2f} ms",
            f"speedup: {self.speedup:.2f}x (required >= {self.min_speedup:.2f}x)",
            f"plan shape: {self.baseline_plan.plan_shape} -> {self.candidate_plan.plan_shape}",
        ]
        idx = self.index_added
        if idx:
            lines.append(f"index used by candidate plan: {idx}")
        if self.seq_scan_removed:
            lines.append("sequential scan removed")
        if self.sort_removed:
            lines.append("explicit sort eliminated")
        lines.append(
            f"rows examined: {self.baseline_plan.total_rows_examined:,} -> "
            f"{self.candidate_plan.total_rows_examined:,}"
        )
        lines.append(
            f"buffers touched: {self.baseline_plan.total_buffers:,} -> "
            f"{self.candidate_plan.total_buffers:,}"
        )
        lines.append(
            f"planning time: {self.baseline_plan.planning_time_ms:.2f} ms -> "
            f"{self.candidate_plan.planning_time_ms:.2f} ms"
        )
        lines.append(
            f"median of {len(self.baseline_runs_ms)} timed runs per arm "
            f"(baseline {list(self.baseline_runs_ms)}, candidate {list(self.candidate_runs_ms)})"
        )
        lines.extend(self.notes)
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "speedup": None if self.speedup == float("inf") else round(self.speedup, 4),
            "improvement_pct": round(self.improvement_pct, 3),
            "min_speedup": self.min_speedup,
            "baseline_median_ms": round(self.baseline_median_ms, 4),
            "candidate_median_ms": round(self.candidate_median_ms, 4),
            "baseline_runs_ms": [round(v, 4) for v in self.baseline_runs_ms],
            "candidate_runs_ms": [round(v, 4) for v in self.candidate_runs_ms],
            "median_delta_ms": round(self.median_delta_ms, 4),
            "planning_time_delta_ms": round(self.planning_time_delta_ms, 4),
            "plan_shape_changed": self.plan_shape_changed,
            "seq_scan_removed": self.seq_scan_removed,
            "sort_removed": self.sort_removed,
            "rows_examined_delta": self.rows_examined_delta,
            "buffers_delta": self.buffers_delta,
            "added_index": self.added_index,
            "evidence": self.evidence_lines,
        }


def median_ms(samples: list[float]) -> float:
    """Median of execution-time samples.

    Median rather than mean: a single scheduler hiccup or autovacuum pass should
    not be able to decide whether an index "worked".
    """
    if not samples:
        raise ValueError("cannot take the median of an empty sample set")
    return float(statistics.median(samples))


def spread_pct(samples: list[float]) -> float:
    """Relative spread (max-min)/median — a cheap noise indicator."""
    if len(samples) < 2:
        return 0.0
    med = median_ms(samples)
    if med <= 0:
        return 0.0
    return (max(samples) - min(samples)) / med * 100.0


#: Above this relative spread the timings are not trustworthy enough to certify.
#:
#: 100% means the slowest sample is more than twice the fastest.  On a real
#: result that pattern usually means interference rather than a genuine bimodal
#: cost — another process, an autovacuum pass, or the first touch of a brand-new
#: index.  A median is robust to a couple of outliers, but it is not evidence
#: when the samples are split into two clusters: certifying "90x faster" from a
#: median of a bimodal distribution would be exactly the kind of overclaim this
#: project exists to avoid.
MAX_TRUSTWORTHY_SPREAD_PCT = 100.0

#: Spread above which we still mention it, but do not block the verdict.
NOISY_SPREAD_PCT = 50.0


def compare(
    *,
    baseline_runs: list[float],
    baseline_plan: PlanSummary,
    candidate_runs: list[float],
    candidate_plan: PlanSummary,
    min_speedup: float,
    added_index: str | None = None,
    planner_used_index: bool | None = None,
    notes: tuple[str, ...] = (),
) -> PlanComparison:
    """Build the verdict-bearing comparison for one experiment."""
    all_notes = list(notes)
    base_spread = spread_pct(baseline_runs)
    cand_spread = spread_pct(candidate_runs)

    if base_spread > NOISY_SPREAD_PCT:
        all_notes.append(f"WARNING: baseline timing spread was {base_spread:.0f}% — noisy machine")
    if cand_spread > MAX_TRUSTWORTHY_SPREAD_PCT:
        all_notes.append(
            f"WARNING: candidate timing spread was {cand_spread:.0f}% — the fastest and "
            f"slowest runs differ by more than 2x, so this arm is not measuring one thing. "
            f"Re-run with more samples or a quieter machine before trusting the number."
        )
    elif cand_spread > NOISY_SPREAD_PCT:
        all_notes.append(f"WARNING: candidate timing spread was {cand_spread:.0f}% — noisy machine")

    return PlanComparison(
        baseline_median_ms=median_ms(baseline_runs),
        candidate_median_ms=median_ms(candidate_runs),
        baseline_runs_ms=tuple(baseline_runs),
        candidate_runs_ms=tuple(candidate_runs),
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        min_speedup=min_speedup,
        added_index=added_index,
        planner_used_index=planner_used_index,
        notes=tuple(all_notes),
    )
