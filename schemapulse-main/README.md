# SchemaPulse

**Execution-verified PostgreSQL optimization agent.**
Gemma proposes. PostgreSQL decides.

SchemaPulse takes one slow `SELECT`, measures it for real, asks a Gemma model for
index hypotheses, refuses the ones that do not survive contact with the catalog,
and then **proves or disproves each remaining candidate by running it** on a
disposable shadow database. Nothing is recommended on the strength of a model's
confidence.

```
one SELECT
   │
   ├─► read-only EXPLAIN (ANALYZE, BUFFERS) ──► plan, rows, hotspots   [deterministic]
   │
   ├─► catalog + pg_stats ──► normalized diagnostic context            [deterministic]
   │
   ├─► Gemma proposes N index candidates
   │
   ├─► SQLGlot allow-list ──► parsed, sanitized AST                   [deterministic]
   ├─► catalog grounding ──► real columns, no duplicates, no DDL
   │
   ├─► shadow database: CREATE INDEX ──► ANALYZE ──► 3 warmups ──► 5 timed runs
   ├─► DROP INDEX ──► baseline arm measured the same way
   │
   └─► median-to-median comparison + noise guard ──► VERIFIED / REJECTED / INCONCLUSIVE
                                                          │
                                     HTML + JSON evidence report, never auto-applied
```

## Why this exists

Most "AI database tuning" demos show a plausible `CREATE INDEX` statement and call
it a result. Three things are usually missing:

1. **The model's word that it helps.** A confident claim is not a measurement.
2. **Evidence the index is even usable.** A model that invents a column name, or
   proposes an index the primary key already covers, produces a plan that cannot
   execute at all.
3. **Any accounting for noise.** One timing on a shared laptop is a coin flip.

SchemaPulse treats the model as a hypothesis generator and the database as the
judge. Every claim in the output report traces back to a number SchemaPulse
measured itself.

## Install

Requires Python 3.11+, and for the demo two PostgreSQL servers (Docker Compose
provides them).

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Three-minute demo

```bash
docker compose up -d --wait      # baseline :5432, shadow :5433
schemapulse demo --seed-only     # schema + 1,000,000 orders / 20,000 customers
schemapulse inspect demo/slow_orders.sql          # no model, no network
schemapulse diagnose demo/slow_orders.sql         # full loop + report
```

The fixture provider is deterministic and offline, so the demo runs with no API
key and no network. It is labelled `FIXTURE` in the CLI and in the report, and
`provider_is_live` is `false` in the JSON — a fixture result is never presented
as a model result.

A run on a typical laptop:

```
baseline median 75.30 ms (spread 29%)   →   candidate median 1.09 ms
VERIFIED · 69.21x faster
rows examined: 1,000,045 → 96
buffers touched: 53,413 → 708
```

The exact numbers depend on your hardware. A run that does not clear the
acceptance threshold says `INCONCLUSIVE` or `REJECTED` rather than flattering the
proposal.

## Commands

| Command | What it does | Needs a model? |
| --- | --- | --- |
| `schemapulse doctor` | Checks Python, Docker, both databases, and the provider | no |
| `schemapulse inspect <query.sql>` | Baseline plan, hotspots, catalog facts | no |
| `schemapulse diagnose <query.sql>` | Full loop: propose → validate → shadow experiment → verdict | yes, unless `--provider fixture` |
| `schemapulse demo` | Brings the environment up, seeds, then runs the golden path | optional |
| `schemapulse report [--open]` | Re-renders the HTML report from the last result | no |
| `schemapulse provider [--probe]` | Which adapter is active, and is it live | no |

Useful flags on `diagnose`: `--provider gemini|ollama|fixture`,
`--require-live/--allow-fixture`, `--min-speedup`, `--runs`, `--attempts`,
`--json`, `--no-report`.

Exit codes: `0` verified, `2` not verified, `1` error. This makes
`schemapulse diagnose` usable as a CI gate.

## Providers

| Provider | Model | Notes |
| --- | --- | --- |
| `gemini` | `gemma-4-26b-a4b-it` (override `SCHEMAPULSE_GEMMA_MODEL`) | Needs `GEMINI_API_KEY` |
| `ollama` | `gemma3:4b-it-q4_K_M` (override `OLLAMA_MODEL`) | Local, no key; `ollama pull` first |
| `fixture` | `schemapulse-heuristic-v1` | Deterministic, offline, no model call |

Configuration is environment-driven. See `.env.example`; copy it to `.env` and
edit. Anything prefixed `SCHEMAPULSE_` maps to a setting, and
`GEMINI_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL` are also read unprefixed so a
copied `.env` from another tool still works.

```bash
export GEMINI_API_KEY=...                     # never logged, never in a report
export SCHEMAPULSE_MODEL_PROVIDER=gemini
export SCHEMAPULSE_MIN_SPEEDUP=1.5
export SCHEMAPULSE_DATABASE_URL=postgresql://...
export SCHEMAPULSE_SHADOW_DATABASE_URL=postgresql://...
```

## Safety model

These are invariants, not features. There are tests for each.

- **Read-only target.** The analysed query must be a single `SELECT`. It is
  parsed with SQLGlot and rejected otherwise; no `;`-chained statement, no DML,
  no CTE that writes.
- **Model-proposed DDL never touches the target.** Candidates are executed only
  on the shadow database, and `assert_production_safe` refuses to start if the two
  URLs are the same string.
- **Executed SQL is rebuilt from the parsed AST,** never interpolated from the
  model's text. Quoting, `CONCURRENTLY`, `IF NOT EXISTS`, and `INCLUDE` are set
  by the tool.
- **Every candidate must be grounded** in the live catalog: the table exists, the
  key and `INCLUDE` columns exist, the name is free, and no existing index
  already covers the leading columns (a partial index over a subset is not
  redundant, and is allowed).
- **Model confidence is not evidence.** Confidence never enters the grounding or
  the verdict.
- **The shadow index is always dropped,** including on exception.
- **Nothing is auto-applied.** The output is a `CREATE INDEX CONCURRENTLY`
  suggestion plus warnings, for a human to run during a deployment.

The only write SchemaPulse ever performs is `CREATE INDEX` and `DROP INDEX` on
the shadow database.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `verified` | Median improved by at least `min_speedup`, spread within the noise guard, and the planner used the new index |
| `rejected` | No improvement, or the planner ignored the index |
| `inconclusive` | Improved but below threshold, or the measurements were too noisy to trust |
| `failed` | The DDL or a benchmark run errored on the shadow database |
| `blocked` | Configuration or environment problem stopped the run |

Faster-but-below-threshold is `inconclusive`, not `inconclusive`-as-success: the
report will not round a 1.05x gain up into a win.

## Reports

Every `diagnose` run writes two files to `.schemapulse/` (override with
`SCHEMAPULSE_ARTIFACTS_DIR`):

- `last-result.json` — the full machine-readable result, including every attempt
  and the reasons for each rejection
- `report.html` — a self-contained page: plan before and after, hotspot evidence,
  per-run timings, the verdict, and the recommendation with its warnings

The JSON deliberately contains no timings in the golden fixture file
(`demo/expected/golden_result.json`). A golden file that hard-codes a
millisecond count either rots or gets rubber-stamped until it does.

## Development

```bash
make check        # ruff + mypy + unit tests
make test-all     # adds integration tests (needs demo-up && seed)
make build        # wheel + sdist
```

Or directly:

```bash
ruff check . && ruff format --check .
mypy src
pytest tests -q
```

Integration tests are marked and skipped automatically when the demo databases
are not reachable, so `pytest tests -q` is safe to run anywhere.

See [docs/DEMO.md](docs/DEMO.md) for the demo runbook and troubleshooting,
[ARCHITECTURE.md](ARCHITECTURE.md) for the module map,
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, and
[ROADMAP.md](ROADMAP.md) for what is deliberately not built yet.

## Status

Alpha. Version 0.1.0. Apache-2.0.

Known limits, stated plainly: single-statement `SELECT` only; index candidates
only (no statistics tuning, no query rewriting); one shadow database on one
machine, so the numbers are a smoke test rather than a capacity plan; and the
model adapter is exercised end-to-end only against the fixture provider unless
you supply a live key.

## License

Apache-2.0. See [LICENSE](LICENSE).