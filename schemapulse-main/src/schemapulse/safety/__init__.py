"""The safety boundary: what SQL SchemaPulse is willing to run, and on which connection."""

from schemapulse.safety.candidate_validator import CandidateValidator, ValidationReport
from schemapulse.safety.sql_guard import (
    IndexCandidate,
    ParsedQuery,
    drop_index_sql,
    parse_index_ddl,
    parse_readonly_query,
    quote_ident,
    sanitize_index_sql,
)

__all__ = [
    "CandidateValidator",
    "IndexCandidate",
    "ParsedQuery",
    "ValidationReport",
    "drop_index_sql",
    "parse_index_ddl",
    "parse_readonly_query",
    "quote_ident",
    "sanitize_index_sql",
]
