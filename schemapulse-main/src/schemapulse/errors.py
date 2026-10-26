"""Typed error hierarchy for SchemaPulse.

Every failure mode that the demo must survive has a named error so the CLI can
render an actionable message instead of a traceback.
"""

from __future__ import annotations


class SchemaPulseError(Exception):
    """Base class for every error raised by SchemaPulse."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(SchemaPulseError):
    """Configuration is missing or internally inconsistent."""


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
class DatabaseError(SchemaPulseError):
    """Any failure originating from a PostgreSQL connection or statement."""


class DatabaseUnavailableError(DatabaseError):
    """The configured PostgreSQL server could not be reached."""


class QueryTimeoutError(DatabaseError):
    """A statement exceeded ``SCHEMAPULSE_QUERY_TIMEOUT_MS``."""


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #
class SafetyError(SchemaPulseError):
    """Model output violated a deterministic safety rule. Never retried as-is."""


class UnsafeQueryError(SafetyError):
    """The analyzed query is not a single read-only ``SELECT``."""


class UnsafeDDLError(SafetyError):
    """Generated DDL is outside the allow-list or references unknown objects."""


# --------------------------------------------------------------------------- #
# model layer
# --------------------------------------------------------------------------- #
class ModelError(SchemaPulseError):
    """The Gemma provider failed to return a usable candidate set."""


class ModelUnavailableError(ModelError):
    """No Gemma provider is reachable (no API key, no Ollama, etc.)."""


class ModelOutputError(ModelError):
    """Model responded but the payload could not be parsed / validated."""


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
class ExperimentError(SchemaPulseError):
    """A shadow-database experiment could not be completed."""


class ExperimentRejectedError(ExperimentError):
    """The candidate was measured and did not clear the acceptance threshold."""
