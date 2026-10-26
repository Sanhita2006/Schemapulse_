"""Turn PostgreSQL ``EXPLAIN (FORMAT JSON)`` output into typed :class:`PlanSummary`.

This is intentionally a *pure* function of the JSON document.  Given the same
plan it always produces the same summary, which is what makes the before/after
comparison in :mod:`schemapulse.plans.comparator` trustworthy.
"""

from __future__ import annotations

from typing import Any

from schemapulse.errors import DatabaseError
from schemapulse.plans.models import Hotspot, PlanNode, PlanSummary

#: A scan that discarded more than this share of the rows it read is reported.
FILTER_WASTE_RATIO = 0.5

#: ``actual / estimated`` above this is reported as a statistics problem.
ROW_ESTIMATE_ERROR_RATIO = 10.0


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return (str(value),)


def unwrap_explain(result: Any) -> dict[str, Any]:
    """Return the plan document from a psycopg ``EXPLAIN ... FORMAT JSON`` result.

    PostgreSQL returns ``[{ "Plan": {...}, "Planning Time": ..., ... }]``.  Some
    drivers hand back a JSON string instead of a decoded object; handle both.
    """
    if isinstance(result, str):
        import json

        result = json.loads(result)
    if isinstance(result, list):
        if not result:
            raise DatabaseError("EXPLAIN returned an empty result set")
        result = result[0]
    if isinstance(result, str):  # nested JSON string
        import json

        result = json.loads(result)
    if not isinstance(result, dict):
        raise DatabaseError(f"unexpected EXPLAIN payload type: {type(result).__name__}")
    if "Plan" not in result:
        raise DatabaseError("EXPLAIN payload has no 'Plan' key")
    return result


def parse_plan(
    explain_result: Any,
    query: str = "",
    *,
    max_nodes: int = 400,
) -> PlanSummary:
    """Flatten an ``EXPLAIN`` JSON document into a :class:`PlanSummary`.

    Parameters
    ----------
    explain_result:
        Raw driver output of ``EXPLAIN (..., FORMAT JSON) <query>``.
    query:
        The SQL that produced the plan, kept for reporting.
    max_nodes:
        Safety valve for pathological plans. Truncation is recorded by lowering
        ``node_count``; the report shows what was used.
    """
    doc = unwrap_explain(explain_result)
    plan_doc = doc["Plan"]

    nodes: list[PlanNode] = []

    def walk(node: dict[str, Any], depth: int) -> None:
        if len(nodes) >= max_nodes:
            return
        nodes.append(_to_node(node, depth))
        for child in node.get("Plans") or []:
            walk(child, depth + 1)

    walk(plan_doc, 0)

    # ``PlanNode.self_time_ms`` reads ``self.children``; attach children lazily by
    # reconstructing parentage on a mutable side table to keep the dataclass frozen.
    _attach_children(nodes, plan_doc)

    return PlanSummary(
        query=query,
        planning_time_ms=_as_float(doc.get("Planning Time")),
        execution_time_ms=_as_float(doc.get("Execution Time")),
        trigger_time_ms=_as_float(doc.get("Trigger Time")),
        jit_time_ms=_as_float(doc.get("JIT Time")),
        settings=doc.get("Settings", {}) or {},
        nodes=tuple(nodes),
    )


def _to_node(node: dict[str, Any], depth: int) -> PlanNode:
    return PlanNode(
        depth=depth,
        node_type=str(node.get("Node Type", "Unknown")),
        relation=node.get("Relation Name"),
        alias=node.get("Alias"),
        index_name=node.get("Index Name"),
        startup_cost=_as_float(node.get("Startup Cost")),
        total_cost=_as_float(node.get("Total Cost")),
        plan_rows=_as_int(node.get("Plan Rows")),
        plan_width=_as_int(node.get("Plan Width")),
        actual_rows=_as_int(node.get("Actual Rows")),
        actual_loops=max(_as_int(node.get("Actual Loops")) or 1, 1),
        actual_total_time_ms=_as_float(node.get("Actual Total Time")),
        actual_startup_ms=_as_float(node.get("Actual Startup Time")),
        rows_removed_by_filter=_as_int(node.get("Rows Removed by Filter")),
        filter=node.get("Filter"),
        index_condition=node.get("Index Cond"),
        recheck_condition=node.get("Recheck Cond"),
        join_filter=node.get("Join Filter"),
        hash_condition=node.get("Hash Cond"),
        sort_key=_as_str_tuple(node.get("Sort Key")),
        sort_method=node.get("Sort Method"),
        group_key=_as_str_tuple(node.get("Group Key")),
        shared_hit_blocks=_as_int(node.get("Shared Hit Blocks")),
        shared_read_blocks=_as_int(node.get("Shared Read Blocks")),
        shared_dirtied_blocks=_as_int(node.get("Shared Dirtied Blocks")),
        temp_read_blocks=_as_int(node.get("Temp Read Blocks")),
        temp_written_blocks=_as_int(node.get("Temp Written Blocks")),
        workers_launched=_as_int(node.get("Workers Launched")),
        raw=node,
    )


def _attach_children(nodes: list[PlanNode], plan_doc: dict[str, Any]) -> None:
    """Give every frozen :class:`PlanNode` a direct ``children`` tuple.

    ``PlanNode.self_time_ms`` needs parentage to subtract child cost, but the
    dataclass is frozen and slotted.  ``object.__setattr__`` plus a side table
    keeps ``PlanNode`` hashable and comparable without a mutable public field.
    """
    by_raw: dict[int, PlanNode] = {id(n.raw): n for n in nodes}

    for node in nodes:
        direct = tuple(
            child
            for child_doc in (node.raw.get("Plans") or [])
            if (child := by_raw.get(id(child_doc))) is not None
        )
        object.__setattr__(node, "children", direct)


def find_hotspots(summary: PlanSummary) -> tuple[Hotspot, ...]:
    """Deterministic plan analysis — the checks Gemma is *not* trusted to do.

    These are cheap, explainable rules over the plan tree.  If they were removed
    the product would still work, which is intentional: they are the *floor* of
    the analysis, and Gemma's job is to go beyond them.
    """
    out: list[Hotspot] = []

    # 1. Sequential scans, ranked by self time.
    for node in summary.by_node_type("Seq Scan") + summary.by_node_type("Parallel Seq Scan"):
        out.append(
            Hotspot(
                code="seq_scan",
                title=f"Sequential Scan on {node.relation or 'relation'}",
                detail=(
                    f"{node.rows_examined:,} rows read"
                    + (
                        f", {node.rows_removed_total:,} discarded by filter"
                        if node.rows_removed_total
                        else ""
                    )
                    + (
                        f", index used: {node.index_name}"
                        if node.index_name
                        else ", no index available"
                    )
                ),
                self_time_ms=node.self_time_ms,
                relation=node.relation,
                severity="warn" if node.self_time_ms > 50 else "info",
            )
        )

    # 2. Filter waste: the DB read far more than it returned.
    for node in summary.scan_nodes():
        read = node.rows_examined
        if read <= 0 or node.rows_removed_total <= 0:
            continue
        ratio = node.rows_removed_total / read
        if ratio >= FILTER_WASTE_RATIO:
            out.append(
                Hotspot(
                    code="filter_waste",
                    title=(
                        f"{ratio:.0%} of rows read on "
                        f"{node.relation or node.node_type} were discarded"
                    ),
                    detail=(
                        f"filter {node.filter or node.index_condition or 'n/a'} "
                        f"discarded {node.rows_removed_total:,} of "
                        f"{read:,} rows, keeping {node.actual_total_rows:,}"
                    ),
                    self_time_ms=node.self_time_ms,
                    relation=node.relation,
                    severity="warn",
                )
            )

    # 3. Cardinality misestimation -> statistics problem, not an index problem.
    for node in summary.scan_nodes():
        err = node.row_estimate_error
        if err is None:
            continue
        if err >= ROW_ESTIMATE_ERROR_RATIO or err <= 1 / ROW_ESTIMATE_ERROR_RATIO:
            direction = "under" if err >= ROW_ESTIMATE_ERROR_RATIO else "over"
            out.append(
                Hotspot(
                    code="row_estimate_error",
                    title=(
                        f"Planner {direction}-estimated rows on {node.relation or node.node_type} "
                        f"by {max(err, 1 / err):.0f}x"
                    ),
                    detail=(
                        f"estimated {node.plan_rows:,} rows, actual {node.actual_rows:,} rows "
                        f"— consider ANALYZE / higher statistics target before adding an index"
                    ),
                    self_time_ms=node.self_time_ms,
                    relation=node.relation,
                    severity="warn",
                )
            )

    # 4. Explicit sorts that an index could eliminate.
    for node in summary.nodes:
        if node.sort_key:
            out.append(
                Hotspot(
                    code="explicit_sort",
                    title=f"Explicit Sort on {'+'.join(node.sort_key)}",
                    detail=(f"method={node.sort_method or 'unknown'}, loops={node.actual_loops}"),
                    self_time_ms=node.self_time_ms,
                    relation=node.relation,
                    severity="info",
                )
            )

    # 5. Disk spill.
    for node in summary.nodes:
        if node.temp_written_blocks:
            out.append(
                Hotspot(
                    code="temp_spill",
                    title=f"Temporary file spill ({node.temp_written_blocks:,} blocks)",
                    detail="work_mem is too small for this operator",
                    self_time_ms=node.self_time_ms,
                    relation=node.relation,
                    severity="warn",
                )
            )

    order = {"warn": 0, "info": 1}
    out.sort(key=lambda h: (order[h.severity], -h.self_time_ms, h.code))
    return tuple(out)


def summarize_for_prompt(summary: PlanSummary, max_nodes: int = 30) -> str:
    """Compact, token-efficient rendering of a plan for the model context.

    Sending raw EXPLAIN JSON wastes the context window on fields a reasoning
    step cannot use.  This keeps the fields that carry signal and drops the rest.
    """
    lines: list[str] = []
    for n in summary.nodes[:max_nodes]:
        pad = "  " * n.depth
        parts = [f"{n.node_type}"]
        if n.relation:
            parts.append(f"on {n.relation}")
        if n.index_name:
            parts.append(f"using {n.index_name}")
        lines.append(f"{pad}- " + " ".join(parts))
        detail: list[str] = []
        detail.append(f"self={n.self_time_ms:.2f}ms")
        detail.append(f"total={n.actual_total_time_ms:.2f}ms")
        detail.append(f"est_rows={n.plan_rows:,}")
        detail.append(f"actual_rows={n.actual_rows:,}")
        if n.actual_loops > 1:
            detail.append(f"loops={n.actual_loops}")
        if n.filter:
            detail.append(f"filter=({n.filter})")
        if n.index_condition:
            detail.append(f"index_cond=({n.index_condition})")
        if n.rows_removed_total:
            detail.append(f"removed_by_filter={n.rows_removed_total:,}")
        if n.sort_key:
            detail.append(f"sort_key={list(n.sort_key)}")
        if n.shared_hit_blocks or n.shared_read_blocks:
            detail.append(f"buffers=hit:{n.shared_hit_blocks}/read:{n.shared_read_blocks}")
        lines.append(f"{pad}    " + "; ".join(detail))
    return "\n".join(lines)
