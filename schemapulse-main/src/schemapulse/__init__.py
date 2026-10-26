"""SchemaPulse — Execution-Verified PostgreSQL Optimization Agent.

The product loop is:

    slow SQL
      -> real EXPLAIN (ANALYZE, BUFFERS) evidence from PostgreSQL
      -> normalized diagnostic context
      -> Gemma optimization *hypothesis*
      -> independent deterministic SQL validation
      -> isolated shadow-database experiment
      -> repeated benchmark (median)
      -> accept / reject on measured evidence only
      -> evidence-backed report

The model is a probabilistic candidate generator. PostgreSQL is the verifier.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "DiagnosisResult",
    "ExperimentOutcome",
    "OptimizedIndex",
    "SchemaPulse",
    "Verdict",
    "__version__",
]


def __getattr__(name: str) -> object:
    """Lazy re-exports so that ``import schemapulse`` stays cheap for the CLI."""
    if name in {"DiagnosisResult", "ExperimentOutcome", "OptimizedIndex", "Verdict"}:
        from schemapulse import models as _models

        return getattr(_models, name)
    if name == "SchemaPulse":
        from schemapulse.engine import SchemaPulse

        return SchemaPulse
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
