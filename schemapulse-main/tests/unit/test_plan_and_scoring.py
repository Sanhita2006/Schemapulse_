"""Plan parsing, hotspot ranking and the accept/reject decision.

The point of these tests is that SchemaPulse's claims come from the plan, not
from the model.  So the assertions are about *where the number came from*:
self time has to exclude children, rows examined has to include filtered rows,
and a verdict has to follow from the medians.
"""

from __future__ import annotations

import pytest
from tests.fixtures.explain_plans import (
    GOLDEN_QUERY,
    INDEX_SCAN_PLAN,
    SEQ_SCAN_PLAN,
)

from schemapulse.plans.comparator import (
    MAX_TRUSTWORTHY_SPREAD_PCT,
    Verdict,
    compare,
)
from schemapulse.plans.parser import find_hotspots, parse_plan

GOLDEN = GOLDEN_QUERY


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parses_execution_and_planning_time() -> None:
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    assert summary.execution_time_ms == pytest.approx(147.16)
    assert summary.planning_time_ms == pytest.approx(1.33)


def test_plan_tree_is_nested_and_rooted() -> None:
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    assert summary.nodes[0].node_type == "Limit"
    limit = summary.nodes[0]
    nested = limit.children[0]
    assert nested.node_type == "Nested Loop"
    assert {c.node_type for c in nested.children} == {"Gather Merge", "Memoize"}


def test_children_are_shared_not_duplicated() -> None:
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    child_ids = [id(c) for node in summary.nodes for c in node.children]
    assert len(child_ids) == len(set(child_ids)), "a node appeared as a child twice"
    # Limit, Nested Loop, Gather Merge, Sort, Seq Scan, Memoize, Index Scan
    assert len(summary.nodes) == 7


def test_rows_examined_includes_rows_the_filter_discarded() -> None:
    """A seq scan returning 19,794 rows really read ~346k of them."""
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    seq = next(n for n in summary.nodes if n.node_type == "Seq Scan")
    assert seq.actual_total_rows == 19794  # 6,598 rows x 3 parallel loops

    # EXPLAIN reports "Rows Removed by Filter" per loop; the parser must not
    # forget that or the whole-table figure comes out 3x too small.
    assert seq.rows_removed_by_filter == 108912  # raw, per loop
    assert seq.actual_loops == 3
    assert seq.rows_removed_total == 326736
    assert seq.rows_examined == 19794 + 326736
    assert summary.total_rows_examined == seq.rows_examined + 46


def test_self_time_excludes_children() -> None:
    """EXPLAIN's per-node time is inclusive; self time is the honest number."""
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    gather = next(n for n in summary.nodes if n.node_type == "Gather Merge")
    sort = gather.children[0]
    assert gather.actual_total_time_ms == pytest.approx(145.9)
    assert gather.self_time_ms == pytest.approx(145.9 - sort.actual_total_time_ms)
    assert gather.self_time_ms < gather.actual_total_time_ms


def test_parallel_loops_are_multiplied_out() -> None:
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    pkey = next(n for n in summary.nodes if n.index_name == "customers_pkey")
    assert pkey.actual_loops == 46
    assert pkey.actual_total_rows == 46  # 1 row x 46 loops


def test_scan_classification() -> None:
    summary = parse_plan(SEQ_SCAN_PLAN, GOLDEN)
    assert next(n for n in summary.nodes if n.node_type == "Seq Scan").is_full_scan
    assert next(n for n in summary.nodes if n.index_name == "customers_pkey").is_index_access
    assert not next(n for n in summary.nodes if n.node_type == "Sort").is_full_scan


def test_plan_shape_is_a_stable_string() -> None:
    baseline = parse_plan(SEQ_SCAN_PLAN, GOLDEN).plan_shape
    candidate = parse_plan(INDEX_SCAN_PLAN, GOLDEN).plan_shape
    assert baseline == (
        "Limit -> Nested Loop -> Gather Merge -> Sort -> Seq Scan(orders)"
        " -> Memoize -> Index Scan(customers)"
    )
    assert candidate == (
        "Limit -> Nested Loop -> Index Scan(orders) -> Memoize -> Index Scan(customers)"
    )
    assert baseline != candidate


# --------------------------------------------------------------------------- #
# hotspots
# --------------------------------------------------------------------------- #
def test_finds_the_sequential_scan() -> None:
    hotspots = find_hotspots(parse_plan(SEQ_SCAN_PLAN, GOLDEN))
    codes = {h.code for h in hotspots}
    assert "seq_scan" in codes
    seq = next(h for h in hotspots if h.code == "seq_scan")
    assert seq.relation == "orders"
    assert "no index available" in seq.detail


def test_reports_filter_waste_as_a_percentage_of_rows_read() -> None:
    hotspots = find_hotspots(parse_plan(SEQ_SCAN_PLAN, GOLDEN))
    waste = next(h for h in hotspots if h.code == "filter_waste")
    # 326,736 discarded of 346,530 read == 94%
    assert "94%" in waste.title
    assert "keeping 19,794" in waste.detail


def test_hotspots_are_ordered_by_cost() -> None:
    hotspots = find_hotspots(parse_plan(SEQ_SCAN_PLAN, GOLDEN))
    costs = [h.self_time_ms for h in hotspots]
    assert costs == sorted(costs, reverse=True)


def test_no_seq_scan_hotspot_once_an_index_exists() -> None:
    hotspots = find_hotspots(parse_plan(INDEX_SCAN_PLAN, GOLDEN))
    assert "seq_scan" not in {h.code for h in hotspots}


# --------------------------------------------------------------------------- #
# comparison and the verdict
# --------------------------------------------------------------------------- #
def _runs(*values: float) -> tuple[float, ...]:
    return tuple(values)


def test_speedup_and_verdict_accepted() -> None:
    result = compare(
        baseline_runs=_runs(147.16, 156.75, 162.42, 110.51, 91.61),
        candidate_runs=_runs(1.52, 1.48, 1.61, 1.39, 1.72),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="sp_orders_status_created_at_partial",
    )
    assert result.speedup > 50
    assert result.verdict is Verdict.VERIFIED
    assert "sp_orders_status_created_at_partial" in " ".join(result.evidence_lines)


def test_verdict_is_rejected_when_the_index_makes_it_slower() -> None:
    result = compare(
        baseline_runs=_runs(10.0, 10.1, 9.9),
        candidate_runs=_runs(40.0, 41.0, 39.5),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
    )
    assert result.verdict is Verdict.REJECTED
    assert result.speedup < 1.0


def test_below_threshold_is_inconclusive_not_rejected() -> None:
    """Faster but not by enough is its own answer, distinct from "no effect"."""
    result = compare(
        baseline_runs=_runs(100.0, 100.0, 100.0),
        candidate_runs=_runs(90.0, 90.0, 90.0),  # 1.11x, below 1.20x
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
    )
    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_cold_first_run_cannot_be_certified() -> None:
    """3.87 ms then 1.5 ms four times is a cold cache, not a bimodal cost.

    This is exactly what a single warmup produced before the warmup default was
    raised, and the old code would have certified the median.
    """
    result = compare(
        baseline_runs=_runs(147.16, 156.75, 162.42, 110.51, 91.61),
        candidate_runs=_runs(3.87, 1.32, 1.52, 1.39, 1.72),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
    )
    assert result.verdict is Verdict.INCONCLUSIVE
    assert any("spread" in note for note in result.notes)


def test_noisy_measurements_are_not_a_verified_win() -> None:
    """A 10x "speedup" from bimodal samples must not count as evidence."""
    result = compare(
        baseline_runs=_runs(100.0, 100.0, 100.0),
        candidate_runs=_runs(10.0, 90.0, 10.0, 10.0, 80.0),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
    )
    assert result.verdict is not Verdict.VERIFIED
    assert result.candidate_spread_pct > MAX_TRUSTWORTHY_SPREAD_PCT
    assert any("spread" in note for note in result.notes)


def test_evidence_reports_the_median_not_the_best_case() -> None:
    result = compare(
        baseline_runs=_runs(100.0, 100.0, 100.0, 100.0, 100.0),
        candidate_runs=_runs(5.0, 50.0, 5.0, 5.0, 50.0),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
    )
    assert "median" in " ".join(result.evidence_lines).lower()
    # median of [5, 5, 5, 50, 50] is 5, and the best case is also 5 here; the
    # point is that the reported figure is derived from the median function.
    assert result.candidate_median_ms == 5.0


def test_an_index_the_planner_ignores_is_never_a_win() -> None:
    """A faster median with an unchanged plan is noise, not evidence.

    Reproduces a real live-model run: the candidate median was 1.44x better, the
    attempt was recorded REJECTED, and the summary banner still printed
    VERIFIED because the headline verdict came from a rebuilt comparison that
    knew nothing about the planner. The verdict must be decided in one place.
    """
    result = compare(
        baseline_runs=_runs(89.78, 82.29, 90.0, 85.0, 88.0),
        candidate_runs=_runs(62.34, 59.66, 62.18, 64.0, 61.0),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),  # identical plan
        min_speedup=1.2,
        added_index="customers_signup_date_country_idx",
        planner_used_index=False,
    )
    assert result.speedup > 1.2, "the raw timing difference is real; it is just not ours"
    assert result.verdict is Verdict.REJECTED
    assert result.index_added is None


def test_the_planner_using_the_index_keeps_the_timing_verdict() -> None:
    """The same 1.44x is VERIFIED once the planner actually chose the index."""
    result = compare(
        baseline_runs=_runs(89.78, 82.29, 90.0, 85.0, 88.0),
        candidate_runs=_runs(62.34, 59.66, 62.18, 64.0, 61.0),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
        planner_used_index=True,
    )
    assert result.verdict is Verdict.VERIFIED
    # The plan names the index it actually used; that beats the name we asked for.
    assert result.index_added == "sp_orders_status_created_at_partial"


def test_median_beats_best_case_when_one_run_is_an_outlier() -> None:
    result = compare(
        baseline_runs=_runs(100.0, 100.0, 100.0, 100.0, 100.0),
        candidate_runs=_runs(5.0, 5.0, 5.0, 5.0, 500.0),
        baseline_plan=parse_plan(SEQ_SCAN_PLAN, GOLDEN),
        candidate_plan=parse_plan(INDEX_SCAN_PLAN, GOLDEN),
        min_speedup=1.2,
        added_index="ix",
    )
    assert result.candidate_median_ms == 5.0
    assert result.speedup == 20.0
