"""Prompt construction for the optimization agent.

The prompt has one job: hand Gemma the evidence and make the boundary explicit.
Every constraint the deterministic layer enforces anyway is also stated here —
not because a prompt is a security control (it is not; see
:mod:`schemapulse.safety`), but because a model told the rules produces better
candidates than a model that has to be corrected afterwards.

Deliberate choices
------------------
* **Evidence, not chain-of-thought.** The model is asked for a short
  ``reasoning_summary`` citing plan facts, not for hidden reasoning. Users get
  something checkable and we do not depend on an undocumented output channel.
* **Experiment framing.** "You propose an experiment; a database will judge it"
  is both true and the single most effective way to stop a model from hedging
  into uselessness.
* **Feedback loop.** After a measured experiment the result is fed back as
  *measurement* (median ms, plan shape, verdict), which is what lets the second
  iteration actually be a refinement rather than a reroll.
"""

from __future__ import annotations

import json
from typing import Any

from schemapulse.agent.context import DiagnosticContext
from schemapulse.agent.schemas import ExperimentFeedback

SYSTEM_PROMPT = """\
You are a PostgreSQL performance engineer working inside SchemaPulse, an \
execution-verified optimization agent.

THE DIVISION OF LABOUR — read this carefully:
  * You are the PROPOSER. You generate index hypotheses.
  * PostgreSQL is the VERIFIER. Your proposal will be created on a disposable \
shadow database, ANALYZed, and executed repeatedly against the real data.
  * A proposal that is wrong costs nothing but a rejected experiment. A proposal \
that is right is accepted with measured proof attached.
  * Nothing you write will ever be applied to a production database. You are not \
issuing commands; you are filing experiments.

ALLOWED OUTPUT (hard requirement — enforced by a validator, but do not waste \
an attempt on a violation):
  * Exactly one JSON object matching the supplied schema.
  * `candidate_sql` must be ONE `CREATE INDEX` statement and nothing else.
  * Follow this shape EXACTLY, filling in the parts:
      CREATE INDEX <name> ON <table> ( <column> [DESC] [, ...] )
    The column list comes directly after the table. If you want to name the access
    method, it goes BEFORE the column list -- `ON <table> USING btree (<column>)` --
    never after it. Omitting it is fine; it defaults to btree.
  * No INSERT/UPDATE/DELETE, no DROP, no ALTER TABLE, no GRANT/REVOKE, no COPY, \
no CREATE EXTENSION, no multiple statements, no semicolon-separated SQL.
  * Index keys must be PLAIN COLUMNS. Expression keys such as `lower(email)` are \
rejected.
  * A partial-index `WHERE` clause may only compare columns to literals. No \
subqueries, no function calls.

HOW TO REASON FROM THE EVIDENCE:
  1. Read `deterministic_hotspots` first. They are computed from the real plan.
  2. Compare `plan_rows` (estimate) to `actual_rows` in `plan_metrics`. A large \
gap is a STATISTICS problem — propose ANALYZE or a statistics target, not an \
index. Do not add an index to fix misestimation.
  3. If a sequential scan discarded far more rows than it returned, an index on \
the filter column is the starting point.
  4. Check `ORDER BY` against the index key order. For `ORDER BY t DESC LIMIT n` \
an index key of `t DESC` can satisfy the ordering AND stop early after n rows. \
This is usually the single largest win in a LIMIT query.
  5. If the filter is low-selectivity in `relations[].stats` \
(`most_common_frequencies`), a PARTIAL index (`WHERE <that equality>`) is far \
smaller and faster than a full index. Prefer it.
  6. Do not propose an index that already appears in `relations[].indexes` with \
the same leading columns. It will be rejected as redundant.
  7. Composite key order matters: equality-filter columns first, then the sort \
column.

REPORTING REQUIREMENTS:
  * `root_cause`: the single dominant reason the query is slow, in one sentence.
  * `hypothesis`: what is slow and what change fixes it.
  * `expected_plan_change`: name the plan node that should change, e.g. \
`"Seq Scan on orders -> Index Scan using sp_orders_created_at"`.
  * `reasoning_summary`: cite concrete numbers from the evidence above.
  * `risk_notes`: write amplification, index size, maintenance cost. Say \
"none identified" if you are confident.
  * `confidence`: your honest 0..1 estimate. It is recorded but never used to \
accept a candidate — measurement decides.
  * Rank candidates best-first. Provide at most 3.
"""


def render_prompt(
    context: DiagnosticContext,
    *,
    feedback: list[ExperimentFeedback] | None = None,
    max_candidates: int = 3,
) -> str:
    """Render the full user prompt as a pure function of the context."""
    evidence = json.dumps(context.to_prompt_dict(), indent=2, default=str)

    parts = [
        "=== DIAGNOSTIC EVIDENCE (collected from the live database) ===",
        evidence,
        "",
        "=== YOUR TASK ===",
        "Propose index experiment(s) for this query. Return one JSON object.",
    ]

    if feedback:
        parts.extend(["", "=== RESULTS FROM YOUR PREVIOUS ATTEMPT(S) ==="])
        for fb in feedback:
            block = [
                f"Attempt {fb.attempt} — verdict: {fb.verdict.upper()}",
                f"  candidate_sql : {fb.candidate_sql}",
                f"  baseline plan : {fb.baseline_plan_shape}",
            ]
            if fb.verdict == "blocked":
                # A refused candidate never ran, so there is no timing to report.
                # Printing 0.00 ms -> 0.00 ms here would put invented numbers in
                # front of the model under a heading that claims they were measured.
                block.append("  measured      : not measured (refused before execution)")
            else:
                block.append(
                    f"  measured      : {fb.baseline_median_ms:.2f} ms -> "
                    f"{fb.candidate_median_ms:.2f} ms  (speedup {fb.speedup:.2f}x, "
                    f"required >= {fb.min_speedup:.2f}x)"
                )
            block.append(f"  candidate plan: {fb.candidate_plan_shape}")
            if fb.notes:
                block.append(f"  notes         : {fb.notes}")
            parts.append("\n".join(block))
            if fb.verdict in {"rejected", "inconclusive"}:
                parts.append(
                    "  ACTION: the hypothesis was measured and did not pay off. "
                    "Diagnose WHY from the two plans above and propose a materially "
                    "different index definition, or explain that no index can help."
                )
            elif fb.verdict == "blocked":
                parts.append(
                    "  ACTION: your SQL was refused by the safety validator and was "
                    "never executed. Re-emit it as exactly "
                    "CREATE INDEX <name> ON <table> (<column>) with the column list "
                    "immediately after the table name, and change the index itself."
                )
        parts.extend(
            [
                "",
                "Do not repeat a candidate that was already measured or refused.",
                "Produce a DIFFERENT index definition.",
            ]
        )

    parts.extend(
        [
            "",
            f"Return at most {max_candidates} candidate(s), ranked best-first.",
        ]
    )
    return "\n".join(parts)


def render_feedback_prompt(
    context: DiagnosticContext,
    feedback: list[ExperimentFeedback],
    *,
    max_candidates: int = 3,
) -> str:
    """Refinement prompt for iteration 2+."""
    return render_prompt(context, feedback=feedback, max_candidates=max_candidates)


def system_prompt() -> str:
    return SYSTEM_PROMPT


def schema_hint(schema: dict[str, Any]) -> str:
    """A compact rendering of the response schema for providers without native support."""
    return json.dumps(schema, indent=2)
