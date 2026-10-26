"""Deterministic execution-plan representation, parsing and comparison."""

from schemapulse.plans.comparator import PlanComparison, Verdict, compare, median_ms
from schemapulse.plans.models import Hotspot, PlanNode, PlanSummary
from schemapulse.plans.parser import find_hotspots, parse_plan, summarize_for_prompt

__all__ = [
    "Hotspot",
    "PlanComparison",
    "PlanNode",
    "PlanSummary",
    "Verdict",
    "compare",
    "find_hotspots",
    "median_ms",
    "parse_plan",
    "summarize_for_prompt",
]
