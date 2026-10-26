"""The Gemma optimization agent: providers, context building, bounded search loop.

Re-exports are resolved lazily (PEP 562).  Eager imports here created an import
cycle: :mod:`schemapulse.models` needs :mod:`schemapulse.agent.schemas`, but
importing anything under ``schemapulse.agent`` runs this module first, and
``agent.engine`` reaches back into :mod:`schemapulse.models` through
``experiments.scoring``.  Deferring the lookups means each submodule is imported
on first use, so the cycle has no entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from schemapulse.agent.context import DiagnosticContext, RelationBrief, build_context
    from schemapulse.agent.engine import AgentConfig, OptimizationAgent
    from schemapulse.agent.prompts import render_prompt, system_prompt
    from schemapulse.agent.providers import resolve_provider
    from schemapulse.agent.schemas import CandidateSet, ExperimentFeedback, OptimizationCandidate

__all__ = [
    "AgentConfig",
    "CandidateSet",
    "DiagnosticContext",
    "ExperimentFeedback",
    "OptimizationAgent",
    "OptimizationCandidate",
    "RelationBrief",
    "build_context",
    "render_prompt",
    "resolve_provider",
    "system_prompt",
]

#: public name -> module that defines it
_EXPORTS: dict[str, str] = {
    "DiagnosticContext": "schemapulse.agent.context",
    "RelationBrief": "schemapulse.agent.context",
    "build_context": "schemapulse.agent.context",
    "AgentConfig": "schemapulse.agent.engine",
    "OptimizationAgent": "schemapulse.agent.engine",
    "render_prompt": "schemapulse.agent.prompts",
    "system_prompt": "schemapulse.agent.prompts",
    "resolve_provider": "schemapulse.agent.providers",
    "CandidateSet": "schemapulse.agent.schemas",
    "ExperimentFeedback": "schemapulse.agent.schemas",
    "OptimizationCandidate": "schemapulse.agent.schemas",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value  # cache so the next lookup skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
