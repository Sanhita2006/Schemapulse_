"""Deterministic, typed representation of a PostgreSQL execution plan.

Nothing in this module talks to a model.  It exists so that the rest of the
pipeline can reason about plans as data instead of as nested JSON dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Node types that mean "the database read the whole relation".
FULL_SCAN_NODES = frozenset({"Seq Scan", "Parallel Seq Scan"})

#: Node types that mean "the database used an index".
INDEX_SCAN_NODES = frozenset(
    {
        "Index Scan",
        "Index Only Scan",
        "Bitmap Index Scan",
        "Tid Scan",
    }
)


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One normalized node of an EXPLAIN tree.

    Only the fields that matter for optimization decisions are kept; raw plan
    JSON stays available through :attr:`raw` for the report.
    """

    depth: int
    node_type: str
    relation: str | None = None
    alias: str | None = None
    index_name: str | None = None
    startup_cost: float = 0.0
    total_cost: float = 0.0
    plan_rows: int = 0
    plan_width: int = 0
    actual_rows: int = 0
    actual_loops: int = 1
    actual_total_time_ms: float = 0.0
    actual_startup_ms: float = 0.0
    rows_removed_by_filter: int = 0
    filter: str | None = None
    index_condition: str | None = None
    recheck_condition: str | None = None
    join_filter: str | None = None
    hash_condition: str | None = None
    sort_key: tuple[str, ...] = ()
    sort_method: str | None = None
    group_key: tuple[str, ...] = ()
    shared_hit_blocks: int = 0
    shared_read_blocks: int = 0
    shared_dirtied_blocks: int = 0
    temp_read_blocks: int = 0
    temp_written_blocks: int = 0
    workers_launched: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    #: Filled in by the parser after the whole tree is known. Needed to compute
    #: self time (EXPLAIN's per-node time is inclusive of children).
    children: tuple[PlanNode, ...] = field(default=(), repr=False, compare=False)

    # ---------------------------------------------------------------- derived
    @property
    def is_full_scan(self) -> bool:
        return self.node_type in FULL_SCAN_NODES

    @property
    def is_index_access(self) -> bool:
        return self.node_type in INDEX_SCAN_NODES

    @property
    def actual_total_rows(self) -> int:
        """Rows actually produced across all loops."""
        return self.actual_rows * max(self.actual_loops, 1)

    @property
    def rows_removed_total(self) -> int:
        """Rows the filter discarded, summed across loops.

        EXPLAIN reports ``Rows Removed by Filter`` **per loop**, exactly like
        ``Actual Rows``.  On a parallel scan of 1M rows with 3 workers, each
        worker reports roughly a third of the total, so reading the raw field as
        a whole-table number under-reports by a factor of ``loops`` -- which is
        precisely the figure used to argue "this scan threw away 94% of what it
        read".  The raw field stays untouched for fidelity to EXPLAIN; everything
        that does arithmetic uses this.
        """
        return self.rows_removed_by_filter * max(self.actual_loops, 1)

    @property
    def rows_examined(self) -> int:
        """Rows this node read, counting the ones a filter later threw away.

        ``actual_total_rows`` is post-filter, so on a selective scan it looks
        tiny.  ``rows_examined`` is the number that reflects the work done.
        """
        return self.actual_total_rows + self.rows_removed_total

    @property
    def filter_selectivity(self) -> float:
        """Share of rows read that survived the filter, in ``[0, 1]``."""
        if self.rows_examined <= 0:
            return 1.0
        return self.actual_total_rows / self.rows_examined

    @property
    def self_time_ms(self) -> float:
        """Time attributed to this node minus its children.

        EXPLAIN's ``Actual Total Time`` is inclusive of children, so a nested
        scan looks slow even when it did nothing.  Self time is what we rank on.
        """
        child_ms = sum(c.actual_total_time_ms for c in self.children)
        return max(self.actual_total_time_ms - child_ms, 0.0)

    @property
    def total_buffers(self) -> int:
        return (
            self.shared_hit_blocks
            + self.shared_read_blocks
            + self.shared_dirtied_blocks
            + self.temp_read_blocks
            + self.temp_written_blocks
        )

    @property
    def row_estimate_error(self) -> float | None:
        """Ratio ``actual / estimated``; ``None`` when either side is 0.

        A large value is a statistics problem, not an index problem, and the two
        call for different fixes — so we keep them distinguishable.
        """
        if self.plan_rows <= 0 or self.actual_rows <= 0:
            return None
        return round(self.actual_rows / self.plan_rows, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "node_type": self.node_type,
            "relation": self.relation,
            "alias": self.alias,
            "index_name": self.index_name,
            "startup_cost": self.startup_cost,
            "total_cost": self.total_cost,
            "plan_rows": self.plan_rows,
            "actual_rows": self.actual_rows,
            "actual_loops": self.actual_loops,
            "actual_total_time_ms": self.actual_total_time_ms,
            "self_time_ms": round(self.self_time_ms, 4),
            "rows_removed_by_filter": self.rows_removed_by_filter,
            "filter": self.filter,
            "index_condition": self.index_condition,
            "join_filter": self.join_filter,
            "sort_key": list(self.sort_key),
            "sort_method": self.sort_method,
            "group_key": list(self.group_key),
            "shared_hit_blocks": self.shared_hit_blocks,
            "shared_read_blocks": self.shared_read_blocks,
            "temp_read_blocks": self.temp_read_blocks,
            "is_full_scan": self.is_full_scan,
            "is_index_access": self.is_index_access,
            "row_estimate_error": self.row_estimate_error,
        }


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """Flattened, comparable summary of a whole execution plan."""

    query: str
    planning_time_ms: float
    execution_time_ms: float
    nodes: tuple[PlanNode, ...]
    trigger_time_ms: float = 0.0
    jit_time_ms: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- lookup
    def by_node_type(self, node_type: str) -> tuple[PlanNode, ...]:
        return tuple(n for n in self.nodes if n.node_type == node_type)

    @property
    def relations(self) -> tuple[str, ...]:
        seen: list[str] = []
        for n in self.nodes:
            if n.relation and n.relation not in seen:
                seen.append(n.relation)
        return tuple(seen)

    @property
    def indexes_used(self) -> tuple[str, ...]:
        return tuple(n.index_name for n in self.nodes if n.index_name)

    @property
    def total_buffers(self) -> int:
        return sum(n.total_buffers for n in self.nodes)

    @property
    def shared_hit_blocks(self) -> int:
        return sum(n.shared_hit_blocks for n in self.nodes)

    @property
    def shared_read_blocks(self) -> int:
        return sum(n.shared_read_blocks for n in self.nodes)

    @property
    def total_rows_produced(self) -> int:
        """Rows returned by scan nodes across all loops."""
        return sum(n.actual_total_rows for n in self.nodes if n.is_full_scan or n.is_index_access)

    @property
    def total_rows_examined(self) -> int:
        """Rows a scan node actually *read*, including ones a filter then discarded.

        This is the number that explains a sequential scan: a 1M-row scan that
        returns 20k rows "examined" a million rows, not twenty thousand.
        """
        return sum(n.rows_examined for n in self.nodes if n.is_full_scan or n.is_index_access)

    @property
    def rows_removed_by_filter(self) -> int:
        """Rows discarded by filters across all nodes, summed over loops."""
        return sum(n.rows_removed_total for n in self.nodes)

    @property
    def root(self) -> PlanNode | None:
        return self.nodes[0] if self.nodes else None

    @property
    def plan_shape(self) -> str:
        """Compact ``A -> B -> C`` signature used for before/after comparison."""
        return " -> ".join(
            f"{n.node_type}({n.relation})" if n.relation else n.node_type for n in self.nodes
        )

    @property
    def node_type_signature(self) -> tuple[str, ...]:
        return tuple(n.node_type for n in self.nodes)

    # ---------------------------------------------------------------- ranking
    def hot_nodes(self, limit: int = 3) -> tuple[PlanNode, ...]:
        """Highest self-time nodes, deterministic tie-break by depth then type."""
        ranked = sorted(
            self.nodes,
            key=lambda n: (-n.self_time_ms, n.depth, n.node_type),
        )
        return tuple(ranked[:limit])

    def scan_nodes(self) -> tuple[PlanNode, ...]:
        return tuple(n for n in self.nodes if n.is_full_scan or n.is_index_access)

    def has_seq_scan(self) -> bool:
        return bool(self.by_node_type("Seq Scan")) or bool(self.by_node_type("Parallel Seq Scan"))

    def has_sort(self) -> bool:
        return any(n.sort_key for n in self.nodes)

    def signature(self) -> str:
        """Stable fingerprint of the plan *shape* (cost-insensitive)."""
        parts = [f"{n.node_type}:{n.relation or '-'}:{n.index_name or '-'}" for n in self.nodes]
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "planning_time_ms": round(self.planning_time_ms, 4),
            "execution_time_ms": round(self.execution_time_ms, 4),
            "plan_shape": self.plan_shape,
            "relations": list(self.relations),
            "indexes_used": list(self.indexes_used),
            "total_buffers": self.total_buffers,
            "shared_hit_blocks": self.shared_hit_blocks,
            "shared_read_blocks": self.shared_read_blocks,
            "total_rows_examined": self.total_rows_examined,
            "total_rows_produced": self.total_rows_produced,
            "rows_removed_by_filter": self.rows_removed_by_filter,
            "node_count": len(self.nodes),
            "nodes": [n.to_dict() for n in self.nodes],
            "settings": self.settings,
        }


@dataclass(frozen=True, slots=True)
class Hotspot:
    """A deterministic, model-free finding about a plan.

    These are the checks SchemaPulse runs itself.  Gemma is given the same
    evidence but is *not* trusted to notice these — they are computed in Python.
    """

    code: str
    title: str
    detail: str
    self_time_ms: float = 0.0
    relation: str | None = None
    severity: str = "info"  # info | warn

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "self_time_ms": round(self.self_time_ms, 4),
            "relation": self.relation,
            "severity": self.severity,
        }
