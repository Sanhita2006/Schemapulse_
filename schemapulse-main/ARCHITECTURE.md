schemapulse/
├── __init__.py
├── __main__.py         # `python -m schemapulse`
├── config.py           # environment-driven Settings + safety assertions
├── errors.py           # SchemaPulseError hierarchy -> CLI exit codes
├── models.py           # cross-cutting Pydantic result models
├── engine.py           # programmatic API facade: SchemaPulse.diagnose(...)
├── cli.py              # doctor / inspect / diagnose / demo / report / provider
├── db/
│   ├── connection.py    # psycopg 3 helpers + statement timeouts (DbConn alias)
│   ├── inspector.py     # tables / columns / indexes / constraints / sizes
│   ├── explain.py       # EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, FORMAT JSON)
│   └── statistics.py    # pg_stats + relation sizes
├── plans/
│   ├── models.py        # PlanNode / PlanSummary / Hotspot
│   ├── parser.py        # EXPLAIN JSON -> PlanSummary + deterministic hotspots
│   └── comparator.py    # before/after measurement -> Verdict + noise guard
├── agent/
│   ├── prompts.py       # diagnostic context -> prompt + JSON schema
│   ├── schemas.py       # OptimizationCandidate / CandidateSet (Pydantic)
│   ├── context.py       # DiagnosticContext / RelationBrief normalization
│   ├── engine.py        # bounded propose -> validate -> refine loop
│   └── providers/
│       ├── __init__.py  # resolve_provider(): selection + honest fallback
│       ├── base.py      # BaseProvider + GemmaProvider protocol
│       ├── gemini.py    # Google Gemini API (gemma-4-26b-a4b-it)
│       ├── ollama.py    # local Ollama (gemma3:4b-it-q4_K_M)
│       ├── parsing.py   # model text -> CandidateSet, shared by both adapters
│       └── fixture.py   # deterministic offline candidate generator
├── safety/
│   ├── sql_guard.py        # SELECT-only allow-list; CREATE INDEX AST + rebuild
│   └── candidate_validator.py  # grounding: catalog, redundancy, name reuse
├── experiments/
│   ├── benchmark.py     # warmups + repeated timed execution, median, spread
│   ├── index_experiment.py  # paired A/B on the shadow database, cleanup
│   └── scoring.py       # verdict + human-reviewable recommendation synthesis
├── reporting/
│   ├── report.py        # ResultStore: JSON persistence + HTML rendering
│   └── templates/report.html
└── demo/
    └── runner.py        # docker/env bootstrap + seed + golden query

tests/
├── unit/                # no database, no network
│   ├── test_plan_and_scoring.py
│   ├── test_candidate_validator.py
│   ├── test_config.py
│   ├── test_fixture_provider.py
│   ├── test_reporting.py
│   └── test_sql_guard.py
├── integration/         # marked; skipped when demo databases are unreachable
│   ├── test_end_to_end.py     # live target/shadow safety + diagnosis
│   └── test_golden_result.py  # timing-independent golden assertions
├── conftest.py
└── fixtures/explain_plans.py

demo/
├── init.sql             # customers + under-indexed orders
├── seed.sql             # deterministic 1M orders / 20k customers
├── slow_orders.sql      # golden query
└── expected/golden_result.json

## Request flow

`cli.diagnose` -> `engine.SchemaPulse.diagnose`:

1. `safety.sql_guard.ensure_select_only` — refuse anything but one `SELECT`.
2. `db.inspector` + `db.statistics` -> `SchemaSnapshot`, `pg_stats`, sizes.
3. `db.explain` -> `plans.parser.parse_plan` -> `PlanSummary` + hotspots.
4. `agent.prompts` -> provider `propose` -> `agent.parsing` -> `CandidateSet`.
5. For each candidate: `safety.sql_guard.parse_index_ddl` (allow-list, AST
   rebuild) then `safety.candidate_validator` (catalog grounding).
6. `experiments.index_experiment` — create on shadow, `ANALYZE`, warmup, N timed
   runs, drop. Baseline arm measured identically.
7. `plans.comparator.compare` -> `Verdict`, with a spread guard.
8. `experiments.scoring.build_recommendation` -> suggestion + warnings, never
   applied.
9. `reporting.ResultStore` -> `.schemapulse/last-result.json` + `report.html`.

Steps 1-3 and 5-7 involve no model output. The model contributes only step 4.