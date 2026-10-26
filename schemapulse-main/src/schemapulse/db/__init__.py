"""Database access: connections, catalog inspection, EXPLAIN capture, statistics."""

from schemapulse.db.connection import DbTarget, ping, session, target_from_dsn
from schemapulse.db.explain import explain_json, plan_only_json, rowcount_probe
from schemapulse.db.inspector import SchemaSnapshot, TableInfo, inspect_relations
from schemapulse.db.statistics import ColumnStats, fetch_stats, refresh_statistics

__all__ = [
    "ColumnStats",
    "DbTarget",
    "SchemaSnapshot",
    "TableInfo",
    "explain_json",
    "fetch_stats",
    "inspect_relations",
    "ping",
    "plan_only_json",
    "refresh_statistics",
    "rowcount_probe",
    "session",
    "target_from_dsn",
]
