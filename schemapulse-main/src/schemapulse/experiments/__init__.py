"""Shadow-database experiments: apply a candidate, measure it, decide, clean up.

Lazy re-exports (PEP 562) -- see the note in :mod:`schemapulse.agent` about the
import cycle that eager re-exports created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from schemapulse.experiments.benchmark import BenchmarkResult, benchmark, run_once
    from schemapulse.experiments.index_experiment import ExperimentResult, run_index_experiment
    from schemapulse.experiments.scoring import build_recommendation, rejection_reason

__all__ = [
    "BenchmarkResult",
    "ExperimentResult",
    "benchmark",
    "build_recommendation",
    "rejection_reason",
    "run_index_experiment",
    "run_once",
]

_EXPORTS: dict[str, str] = {
    "BenchmarkResult": "schemapulse.experiments.benchmark",
    "benchmark": "schemapulse.experiments.benchmark",
    "run_once": "schemapulse.experiments.benchmark",
    "ExperimentResult": "schemapulse.experiments.index_experiment",
    "run_index_experiment": "schemapulse.experiments.index_experiment",
    "build_recommendation": "schemapulse.experiments.scoring",
    "rejection_reason": "schemapulse.experiments.scoring",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
