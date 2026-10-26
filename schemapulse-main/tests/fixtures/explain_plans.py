"""Captured PostgreSQL fixtures shared by the plan/scoring tests.

These are verbatim ``EXPLAIN (ANALYZE, FORMAT JSON)`` documents, trimmed to the
fields the parser reads.  Keeping them as real captured output (rather than
hand-written dicts) means a change in PostgreSQL's JSON shape shows up as a
test failure instead of silently producing a wrong plan summary.
"""

from __future__ import annotations

#: Baseline plan for demo/slow_orders.sql on 1M rows, no index on orders.
#: The query both captured plans were taken from.  Shared so every test refers
#: to literally the same string the report will print.
GOLDEN_QUERY = (
    "SELECT o.id FROM orders o JOIN customers c ON c.id = o.customer_id "
    "WHERE o.status = 'pending' ORDER BY o.created_at DESC LIMIT 50"
)

SEQ_SCAN_PLAN: list[dict[str, object]] = [
    {
        "Plan": {
            "Node Type": "Limit",
            "Parallel Aware": False,
            "Startup Cost": 17277.43,
            "Total Cost": 17278.36,
            "Plan Rows": 50,
            "Plan Width": 41,
            "Actual Startup Time": 0.021,
            "Actual Total Time": 147.16,
            "Actual Rows": 50,
            "Actual Loops": 1,
            "Shared Hit Blocks": 53413,
            "Plans": [
                {
                    "Node Type": "Nested Loop",
                    "Parent Relationship": "Outer",
                    "Startup Cost": 17277.43,
                    "Total Cost": 17278.36,
                    "Plan Rows": 50,
                    "Actual Startup Time": 0.02,
                    "Actual Total Time": 147.15,
                    "Actual Rows": 50,
                    "Actual Loops": 1,
                    "Plans": [
                        {
                            "Node Type": "Gather Merge",
                            "Startup Cost": 17177.41,
                            "Total Cost": 17178.34,
                            "Plan Rows": 50,
                            "Actual Total Time": 145.9,
                            "Actual Rows": 50,
                            "Actual Loops": 1,
                            "Workers Launched": 2,
                            "Plans": [
                                {
                                    "Node Type": "Sort",
                                    "Startup Cost": 17176.38,
                                    "Total Cost": 17178.34,
                                    "Plan Rows": 2355,
                                    "Actual Total Time": 145.2,
                                    "Actual Rows": 2355,
                                    "Actual Loops": 1,
                                    "Sort Key": ["o.created_at DESC"],
                                    "Sort Method": "quicksort",
                                    "Plans": [
                                        {
                                            "Node Type": "Seq Scan",
                                            "Parent Relationship": "Outer",
                                            "Parallel Aware": True,
                                            "Relation Name": "orders",
                                            "Alias": "o",
                                            "Startup Cost": 0.0,
                                            "Total Cost": 5109.44,
                                            "Plan Rows": 2355,
                                            "Actual Startup Time": 0.011,
                                            "Actual Total Time": 118.7,
                                            "Actual Rows": 6598,
                                            "Actual Loops": 3,
                                            "Filter": "(o.status = 'pending'::text)",
                                            "Rows Removed by Filter": 108912,
                                            "Shared Hit Blocks": 53390,
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "Node Type": "Memoize",
                            "Startup Cost": 27.5,
                            "Total Cost": 27.55,
                            "Plan Rows": 1,
                            "Actual Total Time": 0.031,
                            "Actual Rows": 46,
                            "Actual Loops": 1,
                            "Plans": [
                                {
                                    "Node Type": "Index Scan",
                                    "Parent Relationship": "Outer",
                                    "Index Name": "customers_pkey",
                                    "Relation Name": "customers",
                                    "Alias": "c",
                                    "Index Cond": "(c.id = o.customer_id)",
                                    "Actual Total Time": 0.004,
                                    "Actual Rows": 1,
                                    "Actual Loops": 46,
                                    "Shared Hit Blocks": 23,
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        "Planning Time": 1.33,
        "Triggers": [],
        "Execution Time": 147.16,
    }
]

#: Same query after ``CREATE INDEX ... ON orders (created_at DESC) WHERE status='pending'``
INDEX_SCAN_PLAN: list[dict[str, object]] = [
    {
        "Plan": {
            "Node Type": "Limit",
            "Startup Cost": 0.43,
            "Total Cost": 8.61,
            "Plan Rows": 50,
            "Actual Startup Time": 0.03,
            "Actual Total Time": 1.52,
            "Actual Rows": 50,
            "Actual Loops": 1,
            "Shared Hit Blocks": 708,
            "Plans": [
                {
                    "Node Type": "Nested Loop",
                    "Parent Relationship": "Outer",
                    "Startup Cost": 0.43,
                    "Total Cost": 8.61,
                    "Actual Total Time": 1.5,
                    "Actual Rows": 50,
                    "Actual Loops": 1,
                    "Plans": [
                        {
                            "Node Type": "Index Scan",
                            "Parent Relationship": "Outer",
                            "Index Name": "sp_orders_status_created_at_partial",
                            "Relation Name": "orders",
                            "Alias": "o",
                            "Actual Total Time": 1.34,
                            "Actual Rows": 50,
                            "Actual Loops": 1,
                            "Index Cond": "(o.status = 'pending'::text)",
                            "Shared Hit Blocks": 655,
                        },
                        {
                            "Node Type": "Memoize",
                            "Actual Total Time": 0.02,
                            "Actual Rows": 46,
                            "Actual Loops": 1,
                            "Plans": [
                                {
                                    "Node Type": "Index Scan",
                                    "Parent Relationship": "Outer",
                                    "Index Name": "customers_pkey",
                                    "Relation Name": "customers",
                                    "Alias": "c",
                                    "Index Cond": "(c.id = o.customer_id)",
                                    "Actual Total Time": 0.003,
                                    "Actual Rows": 1,
                                    "Actual Loops": 46,
                                    "Shared Hit Blocks": 53,
                                }
                            ],
                        },
                    ],
                }
            ],
        },
        "Planning Time": 0.71,
        "Triggers": [],
        "Execution Time": 1.52,
    }
]
