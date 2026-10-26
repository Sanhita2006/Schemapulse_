# Contributing to SchemaPulse

Thanks for looking. This project is small enough that a good contribution is
usually a bug report, a test, or a fix — not a new subsystem.

## Getting set up

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
docker compose up -d --wait
schemapulse demo --seed-only
```

The two demo PostgreSQL servers are required for integration tests only. Unit
tests need nothing.

## Before you open a pull request

```bash
make check        # ruff check + ruff format --check + mypy + unit tests
make test-all     # also runs integration tests against the demo databases
```

All four must be clean. If you touch `src/schemapulse/safety/`, `src/schemapulse/plans/`,
or the experiment loop, add or update tests in the same change.

## What this project will and will not accept

The safety invariants in the README are the product. Changes that weaken them
will be declined even if they are shorter or faster:

- Executing model-proposed SQL on the target database.
- Interpolating model text into a statement instead of rebuilding from the AST.
- Letting model confidence influence the verdict.
- Auto-applying a recommendation.

If you think an invariant is wrong, open an issue describing the failure it
prevents. That is a better conversation than a patch that removes it quietly.

## Style

- `ruff format` decides layout. Do not hand-format around it.
- Type annotations on new public functions. `mypy` is configured with
  `check_untyped_defs`, so an untyped body still gets checked.
- Comments explain *why*. A comment restating the next line is noise.
- Errors derive from `SchemaPulseError` in `src/schemapulse/errors.py` so the CLI
  can map them to exit codes and hints.

## Adding a provider

1. Implement the interface in `src/schemapulse/agent/providers/base.py`.
2. Parse the response into `CandidateSet` via `providers/parsing.py`. Do not
   hand-roll JSON extraction.
3. Register it in `providers/__init__.py` and add the name to `ProviderName` in
   `config.py`.
4. A provider must set `is_live` honestly. A deterministic fallback that reports
   itself as live is a bug, not a feature.

## Tests

- `tests/unit/` — no database, no network. Keep them that way.
- `tests/integration/` — marked `integration`; skipped when the demo databases
  are unreachable.

Never assert a specific millisecond value. Assert the plan shape, the row counts,
the verdict, or the safety invariants — those are stable; timings are not. This
is why `demo/expected/golden_result.json` has no timings in it.

## Commit messages

One logical change per commit. Say what changed and why:

```
Reject index candidates whose key columns do not exist

A candidate naming column `o` parsed from `o.created_at` passed validation and
produced a plan that could not execute. Ground every key and INCLUDE column
against the live catalog before benchmarking.
```

## Reporting bugs

Include the command you ran, the `schemapulse doctor` output, your PostgreSQL
version, and the relevant part of `.schemapulse/last-result.json`. Redact
connection strings; they contain passwords.

Security issues: see [SECURITY.md](SECURITY.md), not a public issue.

## License

Contributions are licensed under Apache-2.0, the same as the project. See
[LICENSE](LICENSE) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).